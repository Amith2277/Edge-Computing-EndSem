"""Paired bootstrap over camera-day sequences (notebook 13).

Slot decisions inside one sequence are strongly correlated (same camera, same day, consecutive frames), so the
resampling unit is the sequence, not the slot. Two strategies are compared on the SAME resampled sequences.
numpy only, Python 3.8.
"""
from typing import Any, Callable, Dict, List

import numpy as np

COUNT_KEYS = ("TP", "TN", "FO", "FV", "frames", "cost")


def counts(seq_metrics: List[Dict[str, Any]]) -> np.ndarray:
    """(n_sequences, 6) array of TP, TN, FO, FV, frames, cost from per-sequence metric dicts."""
    return np.array([[float(m[k]) for k in COUNT_KEYS] for m in seq_metrics], float)


def macro_f1(c: np.ndarray) -> float:
    tp, tn, fo, fv = c[0], c[1], c[2], c[3]
    f_occ = 2 * tp / (2 * tp + fo + fv) if tp + fo + fv else 1.0
    f_vac = 2 * tn / (2 * tn + fv + fo) if tn + fv + fo else 1.0
    return float((f_occ + f_vac) / 2.0)


def accuracy(c: np.ndarray) -> float:
    n = c[0] + c[1] + c[2] + c[3]
    return float((c[0] + c[1]) / n) if n else float("nan")


def cost_per_1000(c_fv: float, c_fo: float) -> Callable[[np.ndarray], float]:
    def f(c: np.ndarray) -> float:
        n = c[0] + c[1] + c[2] + c[3]
        return float(1000.0 * (c_fv * c[3] + c_fo * c[2]) / n) if n else float("nan")
    return f


def compute_per_frame(c: np.ndarray) -> float:
    return float(c[5] / c[4]) if c[4] else float("nan")


def paired(a: np.ndarray, b: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int = 2000,
           seed: int = 0) -> Dict[str, Any]:
    """stat(sum of a's rows) - stat(sum of b's rows), with a 95% percentile interval over sequence resamples.
    `share_above_0`: share of resamples where the difference is > 0."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, "paired bootstrap needs the same sequences"
    obs = stat(a.sum(0)) - stat(b.sum(0))
    rng = np.random.RandomState(seed)
    n = len(a)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        d[i] = stat(a[idx].sum(0)) - stat(b[idx].sum(0))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"difference": float(obs), "ci95": [float(lo), float(hi)], "share_above_0": float(np.mean(d > 0)),
            "sequences": n, "resamples": n_boot}
