"""MetaPKLot annotation loading and validation. Python 3.8 compatible, no pandas/shapely needed.

Facts established by notebook 02a (annotation inspection) on the Kaggle mirror dsbd-research/metapklot-dataset:
  annotations/{original,cropped}/{cars,spots}/{PKLot,CNRPark-EXT,PLds}/<name>.tar.xz, each holding one COCO-style JSON.
  cars:  annotations have segmentation, bbox [x, y, w, h], car_id (same car keeps its id across images).
  spots: annotations have polygon segmentation, category_id (0 empty, 1 occupied), car_id (-1 none, -2 non-car),
         best_rotated_rect.
  images: file_name (POSIX, relative to dataset root), width, height, annotationsRectangle [x, y, w, h], date, time,
          climate (id into the file's own "climates" table).
The dataset README differs from the actual files (car category id, climate ids), so ids are always resolved from
each file's own tables, never from the README.
"""
import io
import json
import os
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SOURCES = ("PKLot", "CNRPark-EXT", "PLds")
KINDS = ("cars", "spots")
VARIANTS = ("original", "cropped")


# ----------------------------------------------------------------------------------------------- loading
def load_tar_json(path) -> Dict[str, Any]:
    """Load the single JSON member of a .tar.xz (or any tarfile) annotation file."""
    with tarfile.open(str(path), "r:*") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".json")]
        if len(members) != 1:
            raise ValueError("{}: expected exactly one JSON member, found {}".format(path, len(members)))
        f = tf.extractfile(members[0])
        return json.load(io.TextIOWrapper(f, encoding="utf-8"))


def annotation_files(root, variant: str = "original", kind: str = "spots",
                     sources: Iterable[str] = ("PKLot", "CNRPark-EXT")) -> List[Path]:
    if variant not in VARIANTS or kind not in KINDS:
        raise ValueError("variant must be in {} and kind in {}".format(VARIANTS, KINDS))
    base = Path(str(root)) / "annotations" / variant / kind
    out = []
    for src in sources:
        d = base / src
        if d.is_dir():
            out.extend(sorted(d.glob("*.tar.xz")))
    return out


def camera_key(file_name: str) -> str:
    """'PKLot/PUCPR/Sunny/...' -> 'PKLot/PUCPR'; 'CNRPark-EXT/camera1/SUNNY/...' -> 'CNRPark-EXT/camera1'."""
    parts = file_name.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else file_name


def image_timestamp(img: Dict[str, Any]) -> Optional[datetime]:
    try:
        y, mo, d = [int(v) for v in img["date"]]
        t = [int(v) for v in img.get("time", [0, 0, 0])] + [0, 0, 0]
        return datetime(y, mo, d, t[0], t[1], t[2])
    except Exception:
        return None


def polygon_area(flat: List[float]) -> float:
    """Shoelace area of a flat [x1, y1, x2, y2, ...] polygon."""
    xs, ys = flat[0::2], flat[1::2]
    n = len(xs)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) / 2.0


def polygon_key(flat: List[float], ndigits: int = 0) -> Tuple[float, ...]:
    return tuple(round(float(v), ndigits) for v in flat)


# ----------------------------------------------------------------------------------------------- tables
def parse_file(data: Dict[str, Any], kind: str, source_file: str = "") -> Dict[str, List[Dict[str, Any]]]:
    """Flatten one cars/spots JSON into image rows and annotation rows (plain dicts)."""
    climates = {c["id"]: str(c["name"]).lower() for c in data.get("climates", [])}
    categories = {c["id"]: str(c["name"]).lower() for c in data.get("categories", [])}
    images = []
    for img in data.get("images", []):
        fn = img.get("file_name", "")
        ts = image_timestamp(img)
        rect = img.get("annotationsRectangle")
        images.append({
            "image_id": img.get("id"), "file_name": fn, "camera": camera_key(fn),
            "source": fn.split("/")[0] if fn else "", "width": img.get("width"), "height": img.get("height"),
            "rect": list(rect) if rect else None, "timestamp": ts.isoformat() if ts else None,
            "day": ts.strftime("%Y-%m-%d") if ts else None,
            "climate": climates.get(img.get("climate"), "unknown:{}".format(img.get("climate"))),
            "source_file": source_file,
        })
    anns = []
    for a in data.get("annotations", []):
        seg = a.get("segmentation") or []
        poly = seg[0] if seg and isinstance(seg[0], list) else []
        row = {"ann_id": a.get("id"), "image_id": a.get("image_id"), "car_id": a.get("car_id"),
               "category_id": a.get("category_id"), "category": categories.get(a.get("category_id"), "unknown"),
               "bbox": a.get("bbox"), "n_polygons": len(seg) if isinstance(seg, list) else 0, "polygon": poly}
        if kind == "spots":
            row["best_rotated_rect"] = a.get("best_rotated_rect")
        anns.append(row)
    key_union = {
        "image_keys": sorted({k for img in data.get("images", []) for k in img.keys()}),
        "annotation_keys": sorted({k for a in data.get("annotations", []) for k in a.keys()}),
        "categories": categories, "climates": climates,
    }
    return {"images": images, "annotations": anns, "schema": [key_union]}


def load_set(root, variant: str = "original", kind: str = "spots",
             sources: Iterable[str] = ("PKLot", "CNRPark-EXT")) -> Dict[str, List[Dict[str, Any]]]:
    out = {"images": [], "annotations": [], "schema": []}  # type: Dict[str, List[Dict[str, Any]]]
    for f in annotation_files(root, variant, kind, sources):
        parsed = parse_file(load_tar_json(f), kind, f.name)
        for k in out:
            out[k].extend(parsed[k])
    return out


# ----------------------------------------------------------------------------------------------- checks
def check_images_on_disk(root, images: List[Dict[str, Any]], sample: Optional[int] = None) -> Dict[str, Any]:
    """Does each annotated file exist, and do JSON width/height match the real image? (sizes on a sample)."""
    from PIL import Image
    root = Path(str(root))
    missing, size_mismatch, checked = [], [], 0
    step = max(1, len(images) // sample) if sample else 1
    for i, img in enumerate(images):
        p = root / img["file_name"]
        if not p.is_file():
            missing.append(img["file_name"])
            continue
        if i % step == 0:
            checked += 1
            with Image.open(str(p)) as im:
                if (im.width, im.height) != (img["width"], img["height"]):
                    size_mismatch.append({"file": img["file_name"], "json": [img["width"], img["height"]],
                                          "disk": [im.width, im.height]})
    return {"images": len(images), "missing_on_disk": len(missing), "missing_sample": missing[:5],
            "sizes_checked": checked, "size_mismatch": len(size_mismatch), "size_mismatch_sample": size_mismatch[:5]}


def _inside(b: List[float], w: float, h: float, tol: float = 1.0) -> bool:
    x, y, bw, bh = b
    return x >= -tol and y >= -tol and x + bw <= w + tol and y + bh <= h + tol and bw > 0 and bh > 0


def validate_cars(images: List[Dict[str, Any]], cars: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_id = {i["image_id"]: i for i in images}
    bad_bbox, outside_rect, orphan = 0, 0, 0
    for a in cars:
        img = by_id.get(a["image_id"])
        if img is None:
            orphan += 1
            continue
        b = a["bbox"]
        if not b or len(b) != 4 or not _inside(b, img["width"], img["height"]):
            bad_bbox += 1
            continue
        r = img["rect"]
        if r:
            cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
            if not (r[0] <= cx <= r[0] + r[2] and r[1] <= cy <= r[1] + r[3]):
                outside_rect += 1
    widths = sorted(a["bbox"][2] for a in cars if a["bbox"] and len(a["bbox"]) == 4)
    heights = sorted(a["bbox"][3] for a in cars if a["bbox"] and len(a["bbox"]) == 4)
    pct = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None  # noqa: E731
    return {"car_annotations": len(cars), "images_with_cars": len({a["image_id"] for a in cars}),
            "orphan_annotations": orphan, "invalid_or_out_of_image_bbox": bad_bbox,
            "center_outside_annotation_rectangle": outside_rect,
            "distinct_car_ids": len({a["car_id"] for a in cars}),
            "bbox_width_p5_p50_p95": [pct(widths, .05), pct(widths, .5), pct(widths, .95)],
            "bbox_height_p5_p50_p95": [pct(heights, .05), pct(heights, .5), pct(heights, .95)]}


def validate_spots(images: List[Dict[str, Any]], spots: List[Dict[str, Any]],
                   cars: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    by_id = {i["image_id"]: i for i in images}
    cat = Counter(s["category"] for s in spots)
    car_id_codes = Counter("none(-1)" if s["car_id"] == -1 else "non_car(-2)" if s["car_id"] == -2
                           else "car" for s in spots)
    bad_poly, out_of_image = 0, 0
    for s in spots:
        img = by_id.get(s["image_id"])
        p = s["polygon"]
        if len(p) < 6 or polygon_area(p) <= 0:
            bad_poly += 1
        elif img is not None:
            xs_ok = all(-1 <= x <= img["width"] + 1 for x in p[0::2])
            ys_ok = all(-1 <= y <= img["height"] + 1 for y in p[1::2])
            if not (xs_ok and ys_ok):
                out_of_image += 1
    # label consistency: occupied <-> car_id, and car_id present in the cars JSON for that image
    inconsistent = sum(1 for s in spots if (s["category_id"] == 1) != (s["car_id"] != -1))
    linked = unlinked = 0
    if cars is not None:
        car_pairs = {(c["image_id"], c["car_id"]) for c in cars}
        for s in spots:
            if s["car_id"] is not None and s["car_id"] >= 0:
                if (s["image_id"], s["car_id"]) in car_pairs:
                    linked += 1
                else:
                    unlinked += 1
    return {"spot_annotations": len(spots), "images_with_spots": len({s["image_id"] for s in spots}),
            "category_counts": dict(cat), "car_id_codes": dict(car_id_codes),
            "invalid_polygons": bad_poly, "polygon_outside_image": out_of_image,
            "occupied_vs_car_id_inconsistent": inconsistent,
            "occupied_spots_linked_to_car_annotation": linked if cars is not None else "NOT CHECKED",
            "occupied_spots_without_car_annotation": unlinked if cars is not None else "NOT CHECKED"}


def slot_identity(images: List[Dict[str, Any]], spots: List[Dict[str, Any]], ndigits: int = 0) -> Dict[str, Any]:
    """Are spot polygons fixed per camera (so a persistent slot id can be derived from geometry)?"""
    cam_of = {i["image_id"]: i["camera"] for i in images}
    per_cam = defaultdict(Counter)  # type: Dict[str, Counter]
    spots_per_image = defaultdict(Counter)  # type: Dict[str, Counter]
    per_img = Counter(s["image_id"] for s in spots)
    for s in spots:
        per_cam[cam_of.get(s["image_id"], "?")][polygon_key(s["polygon"], ndigits)] += 1
    for img_id, n in per_img.items():
        spots_per_image[cam_of.get(img_id, "?")][n] += 1
    out = {}
    for cam in sorted(per_cam):
        c = per_cam[cam]
        out[cam] = {"distinct_polygons": len(c),
                    "spots_per_image_top": spots_per_image[cam].most_common(3),
                    "images": sum(spots_per_image[cam].values()),
                    "polygons_seen_once": sum(1 for v in c.values() if v == 1)}
    return out


def temporal_structure(images: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per camera: days, frames per day, and the gap between consecutive frames within a day (minutes)."""
    by_cam_day = defaultdict(list)  # type: Dict[Tuple[str, str], List[datetime]]
    no_ts = 0
    for img in images:
        if img["timestamp"] is None:
            no_ts += 1
            continue
        by_cam_day[(img["camera"], img["day"])].append(datetime.fromisoformat(img["timestamp"]))
    per_cam = defaultdict(lambda: {"days": 0, "frames": 0, "gaps": []})  # type: Dict[str, Dict[str, Any]]
    for (cam, _day), stamps in by_cam_day.items():
        stamps.sort()
        d = per_cam[cam]
        d["days"] += 1
        d["frames"] += len(stamps)
        d["gaps"].extend((b - a).total_seconds() / 60.0 for a, b in zip(stamps, stamps[1:]))
    out = {"images_without_timestamp": no_ts, "cameras": {}}
    for cam in sorted(per_cam):
        g = sorted(per_cam[cam]["gaps"])
        q = lambda p: round(g[min(len(g) - 1, int(p * len(g)))], 1) if g else None  # noqa: E731
        out["cameras"][cam] = {"days": per_cam[cam]["days"], "frames": per_cam[cam]["frames"],
                               "gap_min_p10_p50_p90": [q(.1), q(.5), q(.9)],
                               "duplicate_timestamps": sum(1 for x in g if x == 0)}
    return out


def climate_counts(images: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out = defaultdict(Counter)  # type: Dict[str, Counter]
    for img in images:
        out[img["camera"]][img["climate"]] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def read_day_lists(root) -> Dict[str, List[str]]:
    """non_working_days / non_representative_days hold extension-less text files listing days."""
    root = Path(str(root))
    out = {}
    for sub in ("non_working_days", "non_representative_days"):
        for p in sorted((root / sub).rglob("*")):
            if p.is_file():
                lines = [l.strip() for l in p.read_text(errors="ignore").splitlines() if l.strip()]
                out[p.relative_to(root).as_posix()] = lines
    return out
