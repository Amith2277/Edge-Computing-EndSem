"""Global scene difficulty D (spec section 10). Python 3.8 + numpy (runs on the Orin).

D_t = w1*B + w2*M + w3*C + w4*V + w5*O + w6*S, every component in [0,1], larger = harder.

Timing (spec): B and M come from the CURRENT image (available before inference). C, V, O and S come from the
PREVIOUS processed frame's detections and slot states (feedback); using them for frame t never uses frame t's
inference. A feature that cannot be computed (first frame of a sequence, no detections for C) is MISSING, never 0:
`weighted_score` re-normalises the weights over the available features and the availability is recorded.

Components
  B brightness   1 - mean gray level (0..255) inside the slot region box; dark = hard
  M motion       mean |gray_t - gray_prev| inside the region box, divided by a val-fitted scale, clipped to [0,1]
  C confidence   1 - mean confidence of the previous frame's detections at or above the score threshold
  V density      detections / slots in the previous frame, clipped to [0,1]
  O occlusion    mean over previous detections of the max IoU with any other detection, / val-fitted scale, clipped
  S instability  fraction of slots whose state changed over the last `n` processed frames
"""
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FEATURES = ("brightness", "motion", "confidence", "density", "occlusion", "instability")
SHORT = {"brightness": "B", "motion": "M", "confidence": "C", "density": "V", "occlusion": "O", "instability": "S"}
SCALED = ("motion", "occlusion")          # need a val-fitted scale; the others are naturally in [0,1]


def to_gray_small(img, box: Sequence[float], width: int = 320) -> np.ndarray:
    """PIL image -> float gray array of the region box, resized to `width` px wide (same size every frame)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    g = img.convert("L").crop((x0, y0, x1, y1))
    h = max(1, int(round(g.height * width / float(max(1, g.width)))))
    return np.asarray(g.resize((width, h)), dtype=np.float32)


def brightness(gray: np.ndarray) -> float:
    return float(np.clip(1.0 - gray.mean() / 255.0, 0.0, 1.0))


def motion_raw(gray: np.ndarray, prev: Optional[np.ndarray]) -> Optional[float]:
    if prev is None or prev.shape != gray.shape:
        return None
    return float(np.abs(gray - prev).mean() / 255.0)


def box_iou_matrix(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, float).reshape(-1, 4)
    x0 = np.maximum(b[:, None, 0], b[None, :, 0]); y0 = np.maximum(b[:, None, 1], b[None, :, 1])
    x1 = np.minimum(b[:, None, 2], b[None, :, 2]); y1 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area[:, None] + area[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def detection_features(dets: Optional[np.ndarray], score_thr: float, n_slots: int) -> Dict[str, Optional[float]]:
    """C, V, raw O from one processed frame's detections (rows x0,y0,x1,y1,conf). None = missing."""
    if dets is None:
        return {"confidence": None, "density": None, "occlusion_raw": None}
    d = np.asarray(dets, float).reshape(-1, 5)
    d = d[d[:, 4] >= score_thr]
    dens = min(1.0, len(d) / float(n_slots)) if n_slots else None
    if not len(d):
        return {"confidence": None, "density": dens, "occlusion_raw": 0.0}   # empty lot: no confidence to judge
    occ = 0.0
    if len(d) > 1:
        m = box_iou_matrix(d[:, :4]); np.fill_diagonal(m, 0.0)
        occ = float(m.max(1).mean())
    return {"confidence": float(1.0 - d[:, 4].mean()), "density": dens, "occlusion_raw": occ}


def instability(states: Sequence[np.ndarray]) -> Optional[float]:
    """Fraction of slots that changed state across the given consecutive state vectors (-1 = unobserved)."""
    if len(states) < 2:
        return None
    s = np.stack(states)
    both = (s[1:] >= 0) & (s[:-1] >= 0)
    ch = ((s[1:] != s[:-1]) & both).any(0)
    obs = both.any(0)
    return float(ch[obs].mean()) if obs.any() else None


def fit_scales(raw: Dict[str, np.ndarray], q: float = 99.0) -> Dict[str, float]:
    """Val-fitted scale for the unbounded features: the q-th percentile of the available values."""
    out = {}
    for f, x in raw.items():
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        p = float(np.percentile(x, q)) if len(x) else 1.0
        out[f] = p if p > 0 else 1.0
    return out


def scale(x, s: float):
    x = np.asarray(x, float)
    return np.clip(x / s, 0.0, 1.0)


def weighted_score(F: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """F: (n, k) components in [0,1] with NaN = missing; w: (k,) or (m, k) nonnegative weights summing to 1.
    Returns (score, weight_available): score re-normalised over available components, NaN if none has weight."""
    F = np.asarray(F, float)
    w = np.asarray(w, float)
    avail = ~np.isnan(F)
    Ff = np.where(avail, F, 0.0)
    if w.ndim == 1:
        num, den = Ff @ w, avail.astype(float) @ w
    else:
        num, den = Ff @ w.T, avail.astype(float) @ w.T            # (n, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return score, den


def simplex_grid(k: int, step: float = 0.1) -> np.ndarray:
    """All weight vectors of length k with entries in {0, step, ..., 1} summing to 1."""
    n = int(round(1.0 / step))
    out = []
    for bars in combinations(range(n + k - 1), k - 1):        # stars and bars
        prev, w = -1, []
        for b in bars:
            w.append(b - prev - 1); prev = b
        w.append(n + k - 2 - prev)
        out.append(w)
    return np.array(out, float) / n


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n (1 = largest), ties get their average rank."""
    x = np.asarray(x, float)
    order = np.argsort(-x, kind="stable")
    r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
    _, inv, cnt = np.unique(-x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return sums[inv] / cnt[inv]


def capture_auc(score: np.ndarray, target: np.ndarray) -> Optional[float]:
    """Area under the capture curve: escalate items in descending score; the curve is the share of the total
    target captured vs the share of items escalated. 0.5 = random order. Ties are averaged; NaN scores are put
    at the median score (no information)."""
    s = np.asarray(score, float).copy()
    t = np.asarray(target, float)
    if not len(s) or t.sum() <= 0:
        return None
    if np.isnan(s).any():
        s[np.isnan(s)] = np.nanmedian(s) if (~np.isnan(s)).any() else 0.0
    r = average_ranks(s)
    return float((t * (1.0 - (r - 0.5) / len(s))).sum() / t.sum())


def capture_auc_many(S: np.ndarray, target: np.ndarray) -> np.ndarray:
    """capture_auc for every column of S (n, m) at once."""
    return np.array([capture_auc(S[:, j], target) or np.nan for j in range(S.shape[1])])


def capture_at(score: np.ndarray, target: np.ndarray, budgets=(0.1, 0.2, 0.3)) -> Dict[str, Optional[float]]:
    """Share of the target captured when escalating the top `b` share of items (ties broken by order)."""
    s = np.asarray(score, float).copy(); t = np.asarray(target, float)
    if not len(s) or t.sum() <= 0:
        return {"top{}%".format(int(b * 100)): None for b in budgets}
    if np.isnan(s).any():
        s[np.isnan(s)] = np.nanmedian(s) if (~np.isnan(s)).any() else 0.0
    order = np.argsort(-s, kind="stable")
    cum = np.cumsum(t[order]) / t.sum()
    return {"top{}%".format(int(b * 100)): round(float(cum[max(0, int(np.ceil(b * len(s))) - 1)]), 4) for b in budgets}


def spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 3:
        return None
    rx, ry = average_ranks(x[m]), average_ranks(y[m])
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])
