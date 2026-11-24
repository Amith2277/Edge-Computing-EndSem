"""Helpers for notebook 06: evaluate the trained detectors on the frozen test split, overall and per group.

Pure Python + numpy (ultralytics is only used in the notebook), Python 3.8 compatible.
Groups come from the notebook 03 split manifest: site, camera, climate and flagged/normal day.
Nothing is estimated: a group that is too small is reported with its size and flagged, never filled in.
"""
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.data.yolo_export import safe_name  # the flat file name notebook 03 gave each exported image

NOT_MEASURED = "NOT MEASURED"
GROUP_KINDS = ("site", "camera", "climate", "day_type")


def read_manifest(path) -> List[Dict[str, str]]:
    with open(str(path), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_keys(row: Dict[str, str]) -> List[str]:
    """Group labels of one manifest row, e.g. ['site:PUCPR', 'camera:PKLot/PUCPR', 'climate:sunny', 'day_type:normal']."""
    return ["site:" + row["site"], "camera:" + row["camera"], "climate:" + (row.get("climate") or "unknown"),
            "day_type:" + ("flagged" if (row.get("day_flag") or "").strip() else "normal")]


def test_groups(rows: List[Dict[str, str]], available: Sequence[str], split: str = "test") -> Dict[str, Dict[str, Any]]:
    """{group: {"images": [exported names], "days": n distinct days}} for rows of `split` whose image was exported.

    `available` = file names present in images/<split> of the export. Includes the group "all".
    """
    have = set(available)
    imgs = defaultdict(list)  # type: Dict[str, List[str]]
    days = defaultdict(set)
    for r in rows:
        if r["split"] != split:
            continue
        name = safe_name(r["file_name"])
        if name not in have:
            continue
        for g in ["all"] + group_keys(r):
            imgs[g].append(name)
            days[g].add((r["site"], r["day"]))
    return {g: {"images": sorted(v), "days": len(days[g])} for g, v in sorted(imgs.items())}


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copyfile(str(src), str(dst))


def build_group_dataset(local_root, dest, names: Sequence[str], split: str = "test") -> Path:
    """Make dest/images/<split> + dest/labels/<split> holding only `names` (hard links, copy as fallback) and a
    data.yaml. Each group gets its own folder so ultralytics keeps one label cache per group."""
    local_root, dest = Path(str(local_root)), Path(str(dest))
    for kind in ("images", "labels"):
        (dest / kind / split).mkdir(parents=True, exist_ok=True)
    for n in names:
        _link_or_copy(local_root / "images" / split / n, dest / "images" / split / n)
        lbl = os.path.splitext(n)[0] + ".txt"
        src_lbl = local_root / "labels" / split / lbl
        if src_lbl.exists():
            _link_or_copy(src_lbl, dest / "labels" / split / lbl)
    y = dest / "data.yaml"
    # ultralytics requires train and val keys in every data.yaml; all keys point at this group's images (eval only)
    keys = sorted({"train", "val", split})
    y.write_text("path: {}\n{}names:\n  0: car\n".format(
        dest.as_posix(), "".join("{}: images/{}\n".format(k, split) for k in keys)))
    return y


def count_boxes(label_dir, names: Sequence[str]) -> int:
    label_dir = Path(str(label_dir))
    n = 0
    for name in names:
        p = label_dir / (os.path.splitext(name)[0] + ".txt")
        if p.exists():
            n += sum(1 for line in p.read_text().splitlines() if line.strip())
    return n


def stride_sample(names: Sequence[str], k: Optional[int]) -> List[str]:
    """Evenly spaced sample of at most k names (smoke mode). k=None keeps all."""
    names = sorted(names)
    if not k or len(names) <= k:
        return list(names)
    step = len(names) / float(k)
    return [names[int(i * step)] for i in range(k)]


def best_f1_conf(px, f1_curve) -> Dict[str, float]:
    """Confidence with the highest class-mean F1 (unsmoothed). px: (N,), f1_curve: (nc, N)."""
    import numpy as np
    px, f1 = np.asarray(px, dtype=float), np.asarray(f1_curve, dtype=float)
    if f1.ndim == 2:
        f1 = f1.mean(0)
    if f1.size == 0:
        return {"conf": NOT_MEASURED, "f1": NOT_MEASURED}
    i = int(f1.argmax())
    return {"conf": round(float(px[i]), 4), "f1": round(float(f1[i]), 4)}


def metrics_at_conf(px, p_curve, r_curve, conf: float) -> Dict[str, float]:
    """Class-mean precision, recall and F1 at one confidence threshold, interpolated on the curves."""
    import numpy as np
    px = np.asarray(px, dtype=float)
    p, r = np.asarray(p_curve, dtype=float), np.asarray(r_curve, dtype=float)
    if p.ndim == 2:
        p, r = p.mean(0), r.mean(0)
    if p.size == 0:
        return {"precision": NOT_MEASURED, "recall": NOT_MEASURED, "f1": NOT_MEASURED}
    pv, rv = float(np.interp(conf, px, p)), float(np.interp(conf, px, r))
    f1 = 2 * pv * rv / (pv + rv) if pv + rv > 0 else 0.0
    return {"precision": round(pv, 4), "recall": round(rv, 4), "f1": round(f1, 4)}


def box_summary(box, conf: Optional[float] = None) -> Dict[str, Any]:
    """Headline numbers of an ultralytics DetMetrics.box (duck-typed so tests can pass a stub)."""
    out = {"precision_at_best_f1": round(float(box.mp), 4), "recall_at_best_f1": round(float(box.mr), 4),
           "mAP50": round(float(box.map50), 4), "mAP50-95": round(float(box.map), 4)}
    px, f1c = getattr(box, "px", None), getattr(box, "f1_curve", None)
    if px is not None and f1c is not None and len(px):
        out["best_f1_on_this_set"] = best_f1_conf(px, f1c)
        if conf is not None:
            out["at_operating_conf"] = dict(metrics_at_conf(px, box.p_curve, box.r_curve, conf), conf=conf)
    return out


def small_group_note(n_images: int, n_days: int, min_images: int = 100, min_days: int = 3) -> str:
    notes = []
    if n_images < min_images:
        notes.append("fewer than {} images".format(min_images))
    if n_days < min_days:
        notes.append("fewer than {} days".format(min_days))
    return "; ".join(notes)
