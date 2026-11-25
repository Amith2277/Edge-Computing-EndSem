"""Drift monitoring (spec section 16): rolling distributions of scene / detection signals against a per-camera
baseline, with severity, alert history and a human-readable explanation.

Alerts are for human review only. Unlabelled distribution shift is not proof that accuracy dropped, and nothing
here retrains or changes a model.

- Baseline: per camera and signal, 101 quantiles (0..100th percentile) of the validation values. The same compact
  baseline is used in the notebook and at runtime on the Orin, so both compute identical scores.
- Current: the last `window` frames of that camera (values that are missing, NaN, are dropped).
- Score per signal: PSI (population stability index over the baseline deciles) or KS (largest gap between the
  two cumulative distributions).
- Thresholds: per-signal warning threshold and one overall critical threshold, set from clean validation windows
  so the false-alarm rate on clean data is fixed (see `calibrate`).
numpy only, Python 3.8.
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

SIGNALS = ("brightness", "confidence", "density", "occlusion", "instability", "difficulty", "slot_risk")
DESCRIPTION = {
    "brightness": "image darkness (1 - mean gray level of the slot region)",
    "confidence": "detection uncertainty (1 - mean confidence of accepted detections)",
    "density": "vehicle density (accepted detections per slot)",
    "occlusion": "vehicle-box overlap (occlusion proxy)",
    "instability": "share of slots that changed state over the last frames",
    "difficulty": "global scene difficulty D",
    "slot_risk": "global slot transition risk R (top-10 mean)",
}
TESTS = ("psi", "ks")
SEVERITY = ("ok", "warning", "critical")


def baseline_quantiles(x: np.ndarray) -> Optional[np.ndarray]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return np.quantile(x, np.linspace(0, 1, 101)) if len(x) else None


def psi(base_q: np.ndarray, cur: np.ndarray, eps: float = 1e-3) -> float:
    """PSI of `cur` against the baseline deciles (baseline share is 0.1 per bin by construction; tied deciles merge)."""
    cur = np.asarray(cur, float); cur = cur[~np.isnan(cur)]
    if not len(cur) or base_q is None:
        return float("nan")
    edges = np.unique(base_q[10:100:10])
    # baseline share per bin from the quantile grid (handles ties: a value shared by many quantiles)
    bq = base_q[1:]                                  # 100 points, each 1% of the baseline
    b = np.bincount(np.searchsorted(edges, bq, side="right"), minlength=len(edges) + 1) / float(len(bq))
    c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / float(len(cur))
    b, c = np.maximum(b, eps), np.maximum(c, eps)
    return float(np.sum((c - b) * np.log(c / b)))


def ks(base_q: np.ndarray, cur: np.ndarray) -> float:
    cur = np.asarray(cur, float); cur = cur[~np.isnan(cur)]
    if not len(cur) or base_q is None:
        return float("nan")
    grid = np.unique(np.concatenate([base_q, cur]))
    fb = np.searchsorted(np.sort(base_q), grid, side="right") / float(len(base_q))
    fc = np.searchsorted(np.sort(cur), grid, side="right") / float(len(cur))
    return float(np.max(np.abs(fb - fc)))


def score(test: str, base_q: Optional[np.ndarray], cur: np.ndarray) -> float:
    if test == "psi":
        return psi(base_q, cur)
    if test == "ks":
        return ks(base_q, cur)
    raise ValueError("unknown drift test {}".format(test))


def window_scores(test: str, baseline: Dict[str, np.ndarray], window: Dict[str, np.ndarray],
                  min_values: int = 4) -> Dict[str, float]:
    """Score of every signal; NaN when the signal has no baseline or fewer than `min_values` values in the window."""
    out = {}
    for s in SIGNALS:
        cur = np.asarray(window.get(s, []), float); cur = cur[~np.isnan(cur)]
        out[s] = score(test, baseline.get(s), cur) if len(cur) >= min_values else float("nan")
    return out


def calibrate(clean_scores: List[Dict[str, float]], warn_rate: float = 0.05, crit_rate: float = 0.01) -> Dict[str, Any]:
    """Thresholds from clean (no-drift) windows. Warning: per signal at the 1 - warn_rate / n_signals quantile
    (Bonferroni, so about warn_rate of clean windows raise a warning on any signal). Critical: the 1 - crit_rate
    quantile of the overall normalised score max_s(score_s / warning_s)."""
    k = len(SIGNALS)
    warn = {}
    for s in SIGNALS:
        x = np.array([w[s] for w in clean_scores], float); x = x[~np.isnan(x)]
        warn[s] = float(np.quantile(x, 1 - warn_rate / k)) if len(x) else float("inf")
        warn[s] = max(warn[s], 1e-6)
    norm = np.array([overall(w, warn) for w in clean_scores], float); norm = norm[~np.isnan(norm)]
    crit = float(max(1.0, np.quantile(norm, 1 - crit_rate))) if len(norm) else float("inf")
    return {"warning": warn, "critical_overall": crit, "warn_rate": warn_rate, "crit_rate": crit_rate,
            "clean_windows": len(clean_scores)}


def normalised(scores: Dict[str, float], warn: Dict[str, float]) -> Dict[str, float]:
    return {s: (scores[s] / warn[s] if not np.isnan(scores[s]) else float("nan")) for s in SIGNALS}


def overall(scores: Dict[str, float], warn: Dict[str, float]) -> float:
    n = [x for x in normalised(scores, warn).values() if not np.isnan(x)]
    return float(max(n)) if n else float("nan")


def severity(scores: Dict[str, float], thr: Dict[str, Any]) -> str:
    o = overall(scores, thr["warning"])
    if np.isnan(o) or o < 1.0:
        return "ok"
    return "critical" if o >= thr["critical_overall"] else "warning"


def explain(scores: Dict[str, float], thr: Dict[str, Any], baseline: Dict[str, np.ndarray],
            window: Dict[str, np.ndarray], top: int = 3) -> str:
    n = normalised(scores, thr["warning"])
    hot = sorted([s for s in SIGNALS if not np.isnan(n[s]) and n[s] >= 1.0], key=lambda s: -n[s])[:top]
    if not hot:
        return "No signal outside its baseline range."
    parts = []
    for s in hot:
        cur = np.asarray(window[s], float); cur = cur[~np.isnan(cur)]
        bm, cm = float(np.mean(baseline[s])), float(np.mean(cur))
        how = "distributed differently from" if abs(cm - bm) < 1e-3 else ("higher than" if cm > bm else "lower than")
        parts.append("{} is {} baseline (mean {:.3f} vs {:.3f}; {:.1f}x the warning level)".format(DESCRIPTION[s], how, cm, bm, n[s]))
    return ("; ".join(parts) + ". Review the camera images and recent accuracy; the input distribution changed, "
            "which does not by itself prove that accuracy dropped. No model is retrained automatically.")


class DriftMonitor(object):
    """Rolling drift check for one camera stream. `update` takes one frame's signal values."""

    def __init__(self, camera: str, baseline: Dict[str, np.ndarray], thr: Dict[str, Any], test: str,
                 window: int, min_values: int = 4):
        self.camera, self.baseline, self.thr, self.test = camera, baseline, thr, test
        self.window, self.min_values = int(window), int(min_values)
        self.buf = {s: [] for s in SIGNALS}
        self.state, self.history = "ok", []

    def reset(self) -> None:
        self.buf = {s: [] for s in SIGNALS}

    def update(self, values: Dict[str, float], frame: Any = None, timestamp: Any = None) -> Dict[str, Any]:
        for s in SIGNALS:
            v = values.get(s)
            self.buf[s].append(np.nan if v is None else float(v))
            self.buf[s] = self.buf[s][-self.window:]
        full = len(self.buf[SIGNALS[0]]) >= self.window
        win = {s: np.array(self.buf[s]) for s in SIGNALS}
        sc = window_scores(self.test, self.baseline, win, self.min_values) if full else {s: float("nan") for s in SIGNALS}
        sev = severity(sc, self.thr) if full else "ok"
        out = {"camera": self.camera, "frame": frame, "timestamp": timestamp, "severity": sev, "scores": sc,
               "overall": overall(sc, self.thr["warning"]) if full else float("nan"), "window_full": full}
        if SEVERITY.index(sev) > SEVERITY.index(self.state):          # raised or escalated: new alert event
            ev = dict(out, explanation=explain(sc, self.thr, self.baseline, win), acknowledged=False,
                      action="human review (no automatic retraining)")
            self.history.append(ev)
        self.state = sev
        return out
