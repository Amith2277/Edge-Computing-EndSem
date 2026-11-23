"""Locate the attached dataset folder under /kaggle/input without hard-coding a slug.

Kaggle mounts datasets as /kaggle/input/<slug> or /kaggle/input/datasets/<owner>/<slug> depending on
the notebook environment, so the path is detected from what is actually on disk. Python 3.8 compatible.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

WRAPPERS = {"datasets", "competitions", "models"}


def _file_dirs(root: Path) -> List[str]:
    """Directories (under root) that directly contain at least one file."""
    out = []
    for cur, _dirs, files in os.walk(str(root)):
        if files:
            out.append(cur)
    return out


def common_data_root(root: Path) -> Optional[Path]:
    """Deepest folder that still contains every file under `root` (strips wrapper-only folders)."""
    dirs = _file_dirs(root)
    if not dirs:
        return None
    return Path(os.path.commonpath(dirs))


def count_files(root: Path) -> int:
    return sum(len(f) for _c, _d, f in os.walk(str(root)))


def candidates(base) -> List[Path]:
    """One candidate folder per attached top-level entry (wrapper folders like 'datasets' are expanded)."""
    base = Path(str(base))
    if not base.is_dir():
        return []
    tops = sorted(p for p in base.iterdir() if p.is_dir())
    expanded = []  # type: List[Path]
    for t in tops:
        if t.name in WRAPPERS:
            expanded.extend(sorted(p for p in t.iterdir() if p.is_dir()))
        else:
            expanded.append(t)
    found, seen = [], set()
    for e in expanded:
        root = common_data_root(e)
        if root is not None and str(root) not in seen:
            seen.add(str(root))
            found.append(root)
    return found


def describe(path: Path, max_children: int = 12) -> Dict[str, object]:
    try:
        kids = sorted(p.name for p in Path(path).iterdir())
    except OSError:
        kids = []
    return {"path": str(path), "files": count_files(Path(path)), "children": kids[:max_children],
            "n_children": len(kids)}


def find_dataset_root(base="/kaggle/input", hint: str = "") -> Path:
    """Return the dataset folder, or raise with an actionable list when it is ambiguous or missing.

    hint: case-insensitive substring of the path (e.g. part of the dataset slug) to disambiguate.
    """
    cands = candidates(base)
    if hint:
        cands = [c for c in cands if hint.lower() in str(c).lower()]
    if not cands:
        raise FileNotFoundError(
            "No dataset with files found under {}{}. Attach the dataset with Add Input first.".format(
                base, " matching hint '{}'".format(hint) if hint else ""))
    if len(cands) > 1:
        lines = ["  {}  ({} files)".format(c, count_files(c)) for c in cands]
        raise RuntimeError("Several datasets are attached; set DATASET_HINT to part of the right path "
                           "or set DATA_ROOT yourself:\n" + "\n".join(lines))
    return cands[0]
