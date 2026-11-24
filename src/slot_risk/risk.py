"""Per-slot transition risk R (spec section 11). Python 3.8 + numpy (runs on the Orin).

R_i,t = alpha*U + beta*Delta + gamma*A + delta*O, every component in [0,1], computed BEFORE frame t's inference
from what the slot tracker already holds (last processed frames), so it can decide whether frame t needs a refresh.

  U      uncertainty of the last score: 1 at the score threshold, 0 at score 0 or 1 (same margin as notebook 07/08)
  Delta  |last score - the score before it| (change in occupancy score), in [0,1]
  A      seconds since the last reliable observation / max_age_s, clipped to [0,1]
  O      slot occlusion evidence: the largest share of the slot covered by a detection that is NOT its own
         supporting detection (a neighbouring vehicle spilling over), from the last processed frame

Missing components (first frames, no detection) are re-normalised away, never set to 0.
Global aggregation over slots: max, or mean of the top-k risks (spec: compare, choose on val).
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

COMPONENTS = ("uncertainty", "delta", "age", "occlusion")
SHORT = {"uncertainty": "U", "delta": "Delta", "age": "A", "occlusion": "O"}


def uncertainty(score: np.ndarray, thr: float) -> np.ndarray:
    score = np.asarray(score, float)
    span = np.where(score < thr, thr, 1.0 - thr)
    return np.clip(1.0 - np.abs(score - thr) / np.where(span > 0, span, 1.0), 0.0, 1.0)


def age_component(age_s: np.ndarray, max_age_s: float) -> np.ndarray:
    return np.clip(np.asarray(age_s, float) / float(max_age_s), 0.0, 1.0)


def slot_occlusion(n_slots: int, pairs: np.ndarray, score_thr: float, overlap_thr: float) -> np.ndarray:
    """pairs: association pairs of one frame (fields slot, det, conf, iou, ios). For each slot, the largest
    intersection-over-slot of a confident detection that is not the slot's own supporting detection
    (exclusive best-IoU assignment, IoU >= overlap_thr)."""
    out = np.zeros(n_slots)
    if not len(pairs):
        return out
    p = pairs[pairs["conf"] >= score_thr]
    if not len(p):
        return out
    # exclusive assignment (as chosen in notebook 07): a detection belongs only to the slot it overlaps most
    home = {}
    for r in p:
        d = int(r["det"])
        if d not in home or r["iou"] > home[d][1]:
            home[d] = (int(r["slot"]), float(r["iou"]))
    own = np.full(n_slots, -1)
    best = np.zeros(n_slots)
    for d, (s, iou) in home.items():
        if iou >= overlap_thr and iou > best[s]:
            best[s] = iou; own[s] = d
    for r in p:
        if r["det"] != own[r["slot"]]:
            out[r["slot"]] = max(out[r["slot"]], float(r["ios"]))
    return np.clip(out, 0.0, 1.0)


def aggregate(R: np.ndarray, method: str, k: Optional[int] = None) -> float:
    """Global risk of one frame from its slot risks (NaN ignored). None-safe: returns NaN if no slot has a risk."""
    r = np.asarray(R, float)
    r = r[~np.isnan(r)]
    if not len(r):
        return float("nan")
    if method == "max":
        return float(r.max())
    if method == "topk_mean":
        kk = max(1, min(int(k), len(r)))
        return float(np.sort(r)[-kk:].mean())
    raise ValueError("aggregation must be 'max' or 'topk_mean'")


def aggregation_options() -> List[Dict[str, Any]]:
    return [{"method": "max"}] + [{"method": "topk_mean", "k": k} for k in (3, 5, 10)]


def roc_auc(score: np.ndarray, label: np.ndarray) -> Optional[float]:
    """ROC AUC with average ranks for ties (Mann-Whitney). NaN scores take the median score."""
    s = np.asarray(score, float).copy(); y = np.asarray(label).astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg:
        return None
    if np.isnan(s).any():
        s[np.isnan(s)] = np.nanmedian(s) if (~np.isnan(s)).any() else 0.0
    order = np.argsort(s, kind="stable")
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, weights=r) / cnt)[inv]
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))
