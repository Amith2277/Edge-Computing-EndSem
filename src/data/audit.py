"""Dataset audit (spec section 5). Format-agnostic: it reports what is actually on disk and never assumes
a MetaPKLot layout. Python 3.8 compatible.

Outputs (under out_dir):
  manifests/image_inventory.csv, manifests/annotation_inventory.csv, manifests/dataset_hashes.json
  audit_summary.json, duplicate_report.json, corrupt_report.json, tree.txt
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.hashing import sha256_file

DEFAULT_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_ANN_EXTS = {".json", ".xml", ".txt", ".csv", ".yaml", ".yml", ".mat", ".npy", ".pkl"}
DEFAULT_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

DATE_PATTERNS = [
    re.compile(r"(19|20)\d{2}[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])"),
]
TIME_PATTERNS = [re.compile(r"(?<!\d)([01]\d|2[0-3])[-_:]?[0-5]\d[-_:]?[0-5]\d(?!\d)")]


def list_files(root: Path) -> List[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.is_file())


def sample_evenly(items: List[Any], max_items: Optional[int]) -> List[Any]:
    """Deterministic, evenly spaced sample (so smoke mode is not biased to the first folder)."""
    if not max_items or len(items) <= max_items:
        return items
    step = len(items) / float(max_items)
    return [items[int(i * step)] for i in range(max_items)]


def tree_text(root: Path, depth: int = 3, max_entries: int = 25) -> str:
    """Directory tree with file counts per folder; truncates wide folders."""
    lines = [str(root)]

    def walk(d: Path, level: int, prefix: str) -> None:
        if level > depth:
            return
        try:
            entries = sorted(d.iterdir())
        except OSError:
            return
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        if files:
            exts = Counter(f.suffix.lower() or "<none>" for f in files)
            lines.append("{}[{} files: {}]".format(prefix, len(files), dict(exts)))
        for i, sub in enumerate(dirs):
            if i >= max_entries:
                lines.append("{}... ({} more folders)".format(prefix, len(dirs) - max_entries))
                break
            lines.append("{}{}/".format(prefix, sub.name))
            walk(sub, level + 1, prefix + "  ")

    walk(Path(root), 1, "  ")
    return "\n".join(lines)


def average_hash(img, hash_size: int = 8) -> int:
    g = img.convert("L").resize((hash_size, hash_size))
    px = list(g.tobytes())  # "L" mode: one byte per pixel; avoids deprecated getdata()
    mean = sum(px) / float(len(px))
    bits = 0
    for v in px:
        bits = (bits << 1) | (1 if v >= mean else 0)
    return bits


def inspect_image(path: Path, hash_size: Optional[int] = None) -> Dict[str, Any]:
    from PIL import Image
    rec = {"width": None, "height": None, "format": None, "mode": None, "ahash": None, "error": ""}
    try:
        with Image.open(str(path)) as im:
            rec["width"], rec["height"] = im.size
            rec["format"], rec["mode"] = im.format, im.mode
            im.load()  # forces a full decode so truncated files are caught
            if hash_size:
                rec["ahash"] = average_hash(im, hash_size)
    except Exception as exc:
        rec["error"] = "{}: {}".format(type(exc).__name__, exc)
    return rec


def summarize_annotation(path: Path, max_bytes: int = 50_000_000) -> Dict[str, Any]:
    """Cheap structural summary of an annotation file. Facts only, no interpretation."""
    rec = {"kind": "", "detail": "", "error": ""}
    ext = path.suffix.lower()
    try:
        if path.stat().st_size > max_bytes and ext in (".json", ".xml", ".csv", ".txt"):
            rec["kind"], rec["detail"] = ext[1:], "skipped parse: file larger than {} bytes".format(max_bytes)
        elif ext == ".json":
            with open(str(path), "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                rec["kind"] = "json:dict"
                rec["detail"] = "keys=" + ",".join(list(map(str, data.keys()))[:25])
                for k, v in list(data.items())[:25]:
                    if isinstance(v, list):
                        rec["detail"] += " | {}=list[{}]".format(k, len(v))
            elif isinstance(data, list):
                rec["kind"] = "json:list"
                rec["detail"] = "len={}".format(len(data))
                if data and isinstance(data[0], dict):
                    rec["detail"] += " first_keys=" + ",".join(list(map(str, data[0].keys()))[:25])
            else:
                rec["kind"] = "json:" + type(data).__name__
        elif ext == ".xml":
            import xml.etree.ElementTree as ET
            root = ET.parse(str(path)).getroot()
            rec["kind"] = "xml"
            rec["detail"] = "root={} children={}".format(
                root.tag, ",".join(sorted({c.tag for c in root})[:25]))
        elif ext in (".txt", ".csv"):
            with open(str(path), "r", encoding="utf-8", errors="ignore") as f:
                first = [next(f, "") for _ in range(3)]
            n_lines = sum(1 for _ in open(str(path), "r", encoding="utf-8", errors="ignore"))
            cols = len(first[0].replace(",", " ").split()) if first[0] else 0
            rec["kind"] = ext[1:]
            rec["detail"] = "lines={} first_line_fields={} first_line={!r}".format(n_lines, cols, first[0].strip()[:80])
        else:
            rec["kind"] = ext[1:] or "none"
            rec["detail"] = "not parsed"
    except Exception as exc:
        rec["error"] = "{}: {}".format(type(exc).__name__, exc)
    return rec


def metadata_hints(rel_paths: List[str]) -> Dict[str, Any]:
    """Hints about camera/site/date/sequence structure from path names. Hints only, not ground truth."""
    depth_names = defaultdict(Counter)
    date_hits = time_hits = 0
    for rp in rel_paths:
        parts = Path(rp).parts
        for i, part in enumerate(parts[:-1][:4]):
            depth_names[i][part] += 1
        stem = Path(rp).stem
        if any(p.search(rp) for p in DATE_PATTERNS):
            date_hits += 1
        if any(p.search(stem) for p in TIME_PATTERNS):
            time_hits += 1
    n = max(len(rel_paths), 1)
    return {
        "folder_names_by_depth": {str(d): {"distinct": len(c), "top": c.most_common(10)}
                                  for d, c in sorted(depth_names.items())},
        "paths_with_date_like_token": date_hits,
        "filenames_with_time_like_token": time_hits,
        "date_like_fraction": round(date_hits / float(n), 4),
        "time_like_fraction": round(time_hits / float(n), 4),
        "caveat": "Pattern hints only. Real timestamps/sequence ids must come from annotation or metadata files.",
    }


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_near_duplicates(hashes: List[Tuple[str, int]], max_hamming: int, limit: int = 2000) -> List[Dict[str, Any]]:
    """O(n^2) on a bounded set; pairs capped at `limit`. Bucket by hash prefix to cut cost on large sets."""
    pairs = []  # type: List[Dict[str, Any]]
    buckets = defaultdict(list)  # type: Dict[int, List[Tuple[str, int]]]
    for rel, h in hashes:
        buckets[h >> 48].append((rel, h))  # top 16 bits of a 64-bit hash
    for items in buckets.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                d = hamming(items[i][1], items[j][1])
                if d <= max_hamming:
                    pairs.append({"a": items[i][0], "b": items[j][0], "hamming": d})
                    if len(pairs) >= limit:
                        return pairs
    return pairs


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_audit(root, out_dir, smoke: bool = False, smoke_max_files: int = 200,
              compute_hashes: bool = True, near_dup: bool = True, hash_size: int = 8,
              max_hamming: int = 2, tree_depth: int = 3,
              image_exts=None, ann_exts=None, video_exts=None) -> Dict[str, Any]:
    root, out_dir = Path(str(root)), Path(str(out_dir))
    if not root.is_dir():
        raise FileNotFoundError("Dataset root not found: {}. Set dataset.root in configs/dataset.yaml "
                                "or ADAPTIVEPARK_DATA_ROOT.".format(root))
    image_exts = set(image_exts or DEFAULT_IMAGE_EXTS)
    ann_exts = set(ann_exts or DEFAULT_ANN_EXTS)
    video_exts = set(video_exts or DEFAULT_VIDEO_EXTS)

    all_files = list_files(root)
    images_all = [p for p in all_files if p.suffix.lower() in image_exts]
    anns_all = [p for p in all_files if p.suffix.lower() in ann_exts]
    videos_all = [p for p in all_files if p.suffix.lower() in video_exts]
    other_exts = Counter(p.suffix.lower() or "<none>" for p in all_files
                         if p.suffix.lower() not in image_exts | ann_exts | video_exts)

    images = sample_evenly(images_all, smoke_max_files) if smoke else images_all
    anns = sample_evenly(anns_all, smoke_max_files) if smoke else anns_all

    img_rows, corrupt, ahashes = [], [], []  # type: List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, int]]
    sha_to_paths = defaultdict(list)  # type: Dict[str, List[str]]
    for p in images:
        rel = p.relative_to(root).as_posix()
        info = inspect_image(p, hash_size if near_dup else None)
        sha = sha256_file(p) if compute_hashes else ""
        row = {"path": rel, "bytes": p.stat().st_size, "sha256": sha, "width": info["width"],
               "height": info["height"], "format": info["format"], "mode": info["mode"], "error": info["error"]}
        img_rows.append(row)
        if info["error"]:
            corrupt.append({"path": rel, "error": info["error"]})
        else:
            if sha:
                sha_to_paths[sha].append(rel)
            if info["ahash"] is not None:
                ahashes.append((rel, info["ahash"]))

    ann_rows = []
    for p in anns:
        rel = p.relative_to(root).as_posix()
        s = summarize_annotation(p)
        ann_rows.append({"path": rel, "bytes": p.stat().st_size,
                         "sha256": sha256_file(p) if compute_hashes else "",
                         "kind": s["kind"], "detail": s["detail"], "error": s["error"]})

    exact_dups = {h: ps for h, ps in sha_to_paths.items() if len(ps) > 1}
    near_pairs = find_near_duplicates(ahashes, max_hamming) if near_dup else []

    sizes = Counter((r["width"], r["height"]) for r in img_rows if r["width"])
    fmts = Counter(r["format"] for r in img_rows if r["format"])
    stem_index = Counter(Path(r["path"]).stem for r in img_rows)
    ann_stems = {Path(r["path"]).stem for r in ann_rows}
    imgs_with_same_stem_annotation = sum(1 for s in stem_index if s in ann_stems)

    summary = {
        "mode": "smoke" if smoke else "full",
        "root": str(root),
        "total_files": len(all_files),
        "image_files": len(images_all),
        "annotation_like_files": len(anns_all),
        "video_files": len(videos_all),
        "other_extensions": dict(other_exts.most_common(15)),
        "images_inspected": len(images),
        "annotation_files_summarized": len(anns),
        "corrupt_or_unreadable_images": len(corrupt),
        "exact_duplicate_groups": len(exact_dups),
        "near_duplicate_pairs_found": len(near_pairs),
        "near_duplicate_note": "Pairs capped at 2000; hash buckets use top-16-bit prefix, so this is a lower bound.",
        "image_size_distribution_top": [{"size": list(k), "count": v} for k, v in sizes.most_common(10)],
        "image_formats": dict(fmts),
        "images_with_same_stem_annotation_file": imgs_with_same_stem_annotation,
        "annotation_kinds": dict(Counter(r["kind"] for r in ann_rows)),
        "metadata_hints": metadata_hints([p.relative_to(root).as_posix() for p in images_all]),
        "not_determined_here": [
            "Vehicle bounding boxes vs masks, class mapping: read the annotation_inventory details.",
            "Slot polygons and slot occupancy ground truth: confirm in notebook 02 by inspecting annotation files.",
            "Real timestamps/sequence ids: only if present in annotations or metadata.",
        ],
        "smoke_warning": "Counts for inspected images/duplicates cover a sample only." if smoke else "",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "manifests" / "image_inventory.csv", img_rows,
              ["path", "bytes", "sha256", "width", "height", "format", "mode", "error"])
    write_csv(out_dir / "manifests" / "annotation_inventory.csv", ann_rows,
              ["path", "bytes", "sha256", "kind", "detail", "error"])
    hashes = {"mode": summary["mode"], "annotation_sha256": {r["path"]: r["sha256"] for r in ann_rows},
              "image_inventory_rows": len(img_rows)}
    for name, obj in [("manifests/dataset_hashes.json", hashes), ("audit_summary.json", summary),
                      ("corrupt_report.json", corrupt),
                      ("duplicate_report.json", {"exact": exact_dups, "near_pairs": near_pairs})]:
        with open(str(out_dir / name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    (out_dir / "tree.txt").write_text(tree_text(root, tree_depth), encoding="utf-8")
    return summary
