"""Vehicle-to-slot association and slot-level occupancy metrics. Python 3.8 + numpy (no shapely).

Three association methods (spec section 9), each with an overlap threshold t:
  box_iou                 IoU(vehicle box, slot polygon) >= t
  intersection_over_slot  area(box & slot) / area(slot) >= t
  center_in_polygon       centre of the vehicle box lies inside the slot polygon (no t)
Option `exclusive`: each detection supports only the slot it overlaps most (under that method), so one large box
cannot fill two neighbouring slots.

Slot score = highest confidence among the detections that support the slot (0 when none). The slot is occupied when
score >= c, where c is the detector-confidence threshold. This score is an UNCALIBRATED detector score, so c is a
"score threshold", not a probability threshold (calibration and cost ratios are notebook 11).

`assign_image` is the runtime path (one frame); `image_pairs` + `slot_scores` is the batch path used to sweep
thresholds. Unit tests check that the two agree.
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.occupancy import geometry as geo

METHODS = ("box_iou", "intersection_over_slot", "center_in_polygon")
_COL = {"box_iou": "iou", "intersection_over_slot": "ios", "center_in_polygon": "center"}


def prepare_slots(polygons: Sequence[Sequence[float]]) -> Dict[str, Any]:
    pts = [geo.to_points(p) for p in polygons]
    return {"pts": pts, "area": np.array([geo.area(p) for p in pts]),
            "bounds": np.array([geo.bounds(p) for p in pts]).reshape(-1, 4)}


def image_pairs(slots: Dict[str, Any], dets: np.ndarray) -> np.ndarray:
    """All (slot, detection) pairs whose extents overlap, with their association measures.

    dets: (N, 5+) rows [x0, y0, x1, y1, conf, ...] in image pixels.
    Returns a structured array with fields slot, det, conf, iou, ios, center.
    """
    dt = np.dtype([("slot", np.int32), ("det", np.int32), ("conf", np.float32),
                   ("iou", np.float32), ("ios", np.float32), ("center", np.float32)])
    dets = np.asarray(dets, dtype=float)
    if dets.ndim != 2 or not len(dets) or not len(slots["pts"]):
        return np.zeros(0, dtype=dt)
    b = slots["bounds"]
    ov = ((b[:, None, 0] < dets[None, :, 2]) & (b[:, None, 2] > dets[None, :, 0]) &
          (b[:, None, 1] < dets[None, :, 3]) & (b[:, None, 3] > dets[None, :, 1]))
    si, di = np.nonzero(ov)
    out = np.zeros(len(si), dtype=dt)
    for k, (s, d) in enumerate(zip(si, di)):
        x0, y0, x1, y1, conf = dets[d, :5]
        inter = geo.rect_intersection_area(slots["pts"][s], x0, y0, x1, y1)
        sa = slots["area"][s]
        ba = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = sa + ba - inter
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        out[k] = (s, d, conf, inter / union if union > 0 else 0.0, inter / sa if sa > 0 else 0.0,
                  1.0 if geo.point_in_polygon(cx, cy, slots["pts"][s]) else 0.0)
    return out


def exclusive_mask(pairs: np.ndarray, method: str, det_field: str = "det") -> np.ndarray:
    """True for the one pair per detection with the largest measure under `method` (ties: lowest slot)."""
    m = pairs[_COL[method]].astype(float)
    if not len(pairs):
        return np.zeros(0, dtype=bool)
    order = np.lexsort((pairs["slot"], -m, pairs[det_field]))
    first = np.ones(len(order), dtype=bool)
    first[1:] = pairs[det_field][order][1:] != pairs[det_field][order][:-1]
    keep = np.zeros(len(pairs), dtype=bool)
    keep[order[first]] = True
    return keep & (m > 0)


def supporting(pairs: np.ndarray, method: str, overlap_thr: Optional[float], exclusive: bool,
               excl: Optional[np.ndarray] = None, det_field: str = "det") -> np.ndarray:
    m = pairs[_COL[method]]
    ok = m >= 0.5 if method == "center_in_polygon" else m >= overlap_thr
    if exclusive:
        ok &= excl if excl is not None else exclusive_mask(pairs, method, det_field)
    return ok


def slot_scores(n_slots: int, pairs: np.ndarray, method: str, overlap_thr: Optional[float], exclusive: bool,
                excl: Optional[np.ndarray] = None, slot_field: str = "slot", det_field: str = "det") -> np.ndarray:
    """Per-slot score = max confidence of supporting detections (0 if none)."""
    s = np.zeros(n_slots, dtype=float)
    if len(pairs):
        ok = supporting(pairs, method, overlap_thr, exclusive, excl, det_field)
        np.maximum.at(s, pairs[slot_field][ok], pairs["conf"][ok].astype(float))
    return s


def assign_image(slots: Dict[str, Any], dets: np.ndarray, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Runtime association for one frame. cfg: method, overlap_thr, exclusive, conf_thr.
    Returns per slot: score, occupied, supporting detection indices (only detections with conf >= conf_thr)."""
    dets = np.asarray(dets, dtype=float)
    pairs = image_pairs(slots, dets)
    ok = supporting(pairs, cfg["method"], cfg.get("overlap_thr"), cfg["exclusive"]) if len(pairs) else np.zeros(0, bool)
    out = [{"score": 0.0, "occupied": False, "support": []} for _ in slots["pts"]]
    for p in pairs[ok]:
        o = out[int(p["slot"])]
        o["score"] = max(o["score"], float(p["conf"]))
        if p["conf"] >= cfg["conf_thr"]:
            o["support"].append(int(p["det"]))
    for o in out:
        o["occupied"] = o["score"] >= cfg["conf_thr"]
    return out


# ----------------------------------------------------------------------------------------------- metrics
def confusion(gt: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    """Slot-level counts with 'occupied' as the positive class. FV = occupied slot reported vacant (the costly error
    for a driver), FO = vacant slot reported occupied."""
    gt, pred = np.asarray(gt, bool), np.asarray(pred, bool)
    tp = int((gt & pred).sum()); tn = int((~gt & ~pred).sum())
    fo = int((~gt & pred).sum()); fv = int((gt & ~pred).sum())
    return summarize(tp, tn, fo, fv)


def summarize(tp: int, tn: int, fo: int, fv: int) -> Dict[str, Any]:
    n = tp + tn + fo + fv
    div = lambda a, b: a / b if b else 0.0  # noqa: E731
    p_o, r_o = div(tp, tp + fo), div(tp, tp + fv)
    p_v, r_v = div(tn, tn + fv), div(tn, tn + fo)
    f_o, f_v = div(2 * p_o * r_o, p_o + r_o), div(2 * p_v * r_v, p_v + r_v)
    return {"slots": n, "TP": tp, "TN": tn, "FO": fo, "FV": fv,
            "accuracy": round(div(tp + tn, n), 4), "balanced_accuracy": round((r_o + r_v) / 2, 4),
            "occupied_precision": round(p_o, 4), "occupied_recall": round(r_o, 4), "occupied_f1": round(f_o, 4),
            "vacant_precision": round(p_v, 4), "vacant_recall": round(r_v, 4), "vacant_f1": round(f_v, 4),
            "macro_f1": round((f_o + f_v) / 2, 4),
            "false_vacant_rate": round(div(fv, tp + fv), 4), "false_occupied_rate": round(div(fo, tn + fo), 4)}


def overlap_grid(method: str) -> List[Optional[float]]:
    if method == "box_iou":
        return [round(x, 2) for x in np.arange(0.05, 0.71, 0.05)]
    if method == "intersection_over_slot":
        return [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    return [None]


def conf_grid(lo: float = 0.05) -> List[float]:
    return [round(x, 3) for x in np.arange(lo, 0.951, 0.025)]


def sweep(n_slots: int, pairs: np.ndarray, gt: np.ndarray, valid: np.ndarray, confs: Sequence[float],
          slot_field: str = "slot", det_field: str = "det") -> List[Dict[str, Any]]:
    """Every method x exclusive x overlap threshold x score threshold, scored on the valid slots."""
    gt = np.asarray(gt, bool)[valid]
    cg = np.asarray(confs, dtype=float)
    rows = []
    for method in METHODS:
        for exclusive in (False, True):
            excl = exclusive_mask(pairs, method, det_field) if exclusive else None
            for t in overlap_grid(method):
                s = slot_scores(n_slots, pairs, method, t, exclusive, excl, slot_field, det_field)[valid]
                pred = s[:, None] >= cg[None, :]
                tp = (pred & gt[:, None]).sum(0); fo = (pred & ~gt[:, None]).sum(0)
                fv = gt.sum() - tp; tn = (~gt).sum() - fo
                for j, c in enumerate(cg):
                    r = summarize(int(tp[j]), int(tn[j]), int(fo[j]), int(fv[j]))
                    r.update(method=method, exclusive=exclusive, overlap_thr=t, conf_thr=float(c))
                    rows.append(r)
    return rows


def best(rows: List[Dict[str, Any]], key: str = "macro_f1", method: Optional[str] = None) -> Dict[str, Any]:
    """Highest `key`; ties go to higher accuracy, then the non-exclusive and lower thresholds (simplest rule)."""
    cand = [r for r in rows if method is None or r["method"] == method]
    return max(cand, key=lambda r: (r[key], r["accuracy"], not r["exclusive"], -(r["overlap_thr"] or 0), -r["conf_thr"]))


def uncertainty(score: float, conf_thr: float) -> float:
    """1 at the decision threshold, 0 at the far end of the score range. A margin measure, not a probability."""
    span = max(conf_thr, 1.0 - conf_thr)
    return round(max(0.0, 1.0 - abs(score - conf_thr) / span), 4) if span > 0 else 1.0
