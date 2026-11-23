"""Look inside annotation files, including compressed ones (.xz/.gz/.tar.*), and report facts only.

Used by notebook 02 after the audit showed the annotations are not plain text files. Python 3.8 compatible.
"""
import gzip
import json
import lzma
import os
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def _open_stream(path: Path):
    s = str(path)
    if s.endswith(".xz"):
        return lzma.open(s, "rb")
    if s.endswith(".gz"):
        return gzip.open(s, "rb")
    return open(s, "rb")


def _decode(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


def _json_shape(text: str) -> Optional[str]:
    try:
        data = json.loads(text)
    except Exception:
        return None
    if isinstance(data, dict):
        parts = ["dict keys=" + ",".join(map(str, list(data.keys())[:20]))]
        for k, v in list(data.items())[:20]:
            if isinstance(v, list):
                parts.append("{}=list[{}]".format(k, len(v)))
                if v and isinstance(v[0], dict):
                    parts.append("{}[0] keys={}".format(k, ",".join(map(str, list(v[0].keys())[:20]))))
        return " | ".join(parts)
    if isinstance(data, list):
        s = "list[{}]".format(len(data))
        if data and isinstance(data[0], dict):
            s += " first keys=" + ",".join(map(str, list(data[0].keys())[:20]))
        return s
    return type(data).__name__


def probe_file(path, head_bytes: int = 800, max_members: int = 12) -> Dict[str, Any]:
    """Return what the file is and a short head of its content. Never interprets meaning."""
    p = Path(str(path))
    rec = {"path": str(p), "size_bytes": p.stat().st_size, "container": "", "members_total": None,
           "members_sample": [], "head": "", "json_shape": None, "error": ""}  # type: Dict[str, Any]
    try:
        if tarfile.is_tarfile(str(p)):
            rec["container"] = "tar"
            with tarfile.open(str(p), "r:*") as tf:
                names = []
                first_member = None
                for m in tf:
                    if m.isfile():
                        names.append(m.name)
                        if first_member is None:
                            f = tf.extractfile(m)
                            first_member = (m.name, f.read(head_bytes) if f else b"")
                rec["members_total"] = len(names)
                rec["members_sample"] = names[:max_members]
                ext_counts = Counter(os.path.splitext(n)[1].lower() or "<none>" for n in names)
                rec["member_extensions"] = dict(ext_counts.most_common(8))
                if first_member:
                    rec["head"] = "[{}]\n{}".format(first_member[0], _decode(first_member[1]))
        else:
            rec["container"] = "xz" if str(p).endswith(".xz") else "gz" if str(p).endswith(".gz") else "plain"
            with _open_stream(p) as f:
                raw = f.read(head_bytes)
            rec["head"] = _decode(raw)
            if rec["size_bytes"] < 20_000_000:
                with _open_stream(p) as f:
                    text = _decode(f.read(20_000_000))
                rec["json_shape"] = _json_shape(text)
                rec["lines"] = text.count("\n")
    except Exception as exc:
        rec["error"] = "{}: {}".format(type(exc).__name__, exc)
    return rec


def dir_summary(root, max_names: int = 6) -> List[Dict[str, Any]]:
    """For each directory under root that holds files: file count, extension counts, total MB, sample names."""
    rows = []
    for cur, _dirs, files in os.walk(str(root)):
        if not files:
            continue
        exts = Counter(os.path.splitext(f)[1].lower() or "<none>" for f in files)
        total = sum(os.path.getsize(os.path.join(cur, f)) for f in files)
        rows.append({"dir": cur, "files": len(files), "extensions": dict(exts),
                     "total_mb": round(total / 1e6, 2), "sample": sorted(files)[:max_names]})
    return sorted(rows, key=lambda r: r["dir"])


def first_file_per_dir(root, exts=(".xz", ".gz", ".tar", ".json", ".txt", ".csv", ".xml")) -> List[Path]:
    """One representative file per directory (the alphabetically first with a matching extension)."""
    picks = []
    for cur, _dirs, files in sorted(os.walk(str(root))):
        match = sorted(f for f in files if f.lower().endswith(exts))
        if match:
            picks.append(Path(cur) / match[0])
    return picks


def basename_index(root) -> set:
    """Set of every file basename under root (one walk, reused for many lookups)."""
    return {f for _c, _d, files in os.walk(str(root)) for f in files}


def check_listed_paths(list_file, index, limit: int = 5000) -> Dict[str, Any]:
    """For a text file listing image paths (removed_images/*.txt): how many of their file names exist on disk?

    Matches on basename because list entries may use a different folder order than the mounted dataset.
    Basenames can repeat across cameras, so a match is an upper bound.
    """
    lines = [l.strip() for l in open(str(list_file), "r", encoding="utf-8", errors="ignore") if l.strip()]
    lines = lines[:limit]
    names = [os.path.basename(l) for l in lines]
    found = sum(1 for n in names if n in index)
    missing = [l for l, n in zip(lines, names) if n not in index][:3]
    return {"list": str(list_file), "entries": len(lines), "basename_found_on_disk": found,
            "not_found_sample": missing}
