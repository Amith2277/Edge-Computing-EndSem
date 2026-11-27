"""Score calibration and cost-sensitive occupancy decisions (spec section 14).

- Calibrators map the uncalibrated slot score (max confidence of supporting detections) to P(occupied):
  Platt (logistic on the score, Newton/IRLS) and isotonic (pool-adjacent-violators, step function).
  Both are fitted on validation only. numpy only, Python 3.8 (runs on the Orin).
- With a calibrated P(O): occupied when P(O) > C_FO / (C_FO + C_FV) (Bayes rule; no threshold tuning).
- Without calibration: a score threshold chosen on validation to minimise the cost-weighted error, reported as a
  SCORE threshold, never as a probability threshold.
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

CALIBRATORS = ("platt", "isotonic")


class Platt(object):
    name = "platt"

    def __init__(self, l2: float = 1e-6):
        self.a, self.b, self.l2 = 0.0, 0.0, float(l2)

    def fit(self, score: np.ndarray, y: np.ndarray) -> "Platt":
        x, y = np.asarray(score, float), np.asarray(y, float)
        # Platt's target smoothing avoids infinite weights on separable data
        n1, n0 = y.sum(), len(y) - y.sum()
        t = np.where(y > 0.5, (n1 + 1.0) / (n1 + 2.0), 1.0 / (n0 + 2.0))
        a, b = 0.0, float(np.log((n1 + 1.0) / (n0 + 1.0)))
        for _ in range(100):
            p = 1.0 / (1.0 + np.exp(-(a * x + b)))
            w = np.maximum(p * (1.0 - p), 1e-12)
            g = np.array([np.sum((p - t) * x) + self.l2 * a, np.sum(p - t)])
            H = np.array([[np.sum(w * x * x) + self.l2, np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
            step = np.linalg.solve(H, g)
            a, b = a - step[0], b - step[1]
            if np.max(np.abs(step)) < 1e-9:
                break
        self.a, self.b = float(a), float(b)
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * np.asarray(score, float) + self.b)))

    def params(self) -> Dict[str, Any]:
        return {"method": "platt", "a": round(self.a, 6), "b": round(self.b, 6)}


class Isotonic(object):
    """Non-decreasing step function of the score (PAV). Scores between fitted points take the value of the
    block on their left (a step, no interpolation), below the first point the first value."""
    name = "isotonic"

    def __init__(self):
        self.x, self.y = np.zeros(1), np.full(1, 0.5)

    def fit(self, score: np.ndarray, y: np.ndarray) -> "Isotonic":
        x, y = np.asarray(score, float), np.asarray(y, float)
        ux, inv = np.unique(x, return_inverse=True)
        s = np.bincount(inv, weights=y)
        n = np.bincount(inv).astype(float)
        vals, wts, starts = [], [], []
        for i in range(len(ux)):
            vals.append(s[i] / n[i]); wts.append(n[i]); starts.append(i)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
                w = wts[-2] + wts[-1]
                vals.pop(); wts.pop(); starts.pop()
                vals[-1], wts[-1] = v, w
        self.x = ux[np.array(starts, int)]
        self.y = np.array(vals, float)
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        i = np.searchsorted(self.x, np.asarray(score, float), side="right") - 1
        return self.y[np.clip(i, 0, len(self.y) - 1)]

    def params(self) -> Dict[str, Any]:
        return {"method": "isotonic", "x": [round(float(v), 6) for v in self.x], "y": [round(float(v), 6) for v in self.y]}


def score_threshold_for(cal, p_thr: float) -> float:
    """Score t such that the calibrated rule P(O) > p_thr equals score >= t (both calibrators are non-decreasing).
    -inf: every score is occupied; inf: none is."""
    if getattr(cal, "name", None) == "platt" or isinstance(cal, Platt):
        if cal.a <= 0:
            return float("inf") if cal.predict(np.array([0.0]))[0] <= p_thr else float("-inf")
        return float(np.nextafter((np.log(p_thr / (1.0 - p_thr)) - cal.b) / cal.a, np.inf))
    above = np.nonzero(cal.y > p_thr)[0]
    if not len(above):
        return float("inf")
    return float("-inf") if above[0] == 0 else float(cal.x[above[0]])


def make(method: str):
    if method == "platt":
        return Platt()
    if method == "isotonic":
        return Isotonic()
    raise ValueError("unknown calibrator {}".format(method))


def from_params(p: Dict[str, Any]):
    c = make(p["method"])
    if p["method"] == "platt":
        c.a, c.b = float(p["a"]), float(p["b"])
    else:
        c.x, c.y = np.asarray(p["x"], float), np.asarray(p["y"], float)
    return c


def group_folds(groups: Sequence[str], k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per row; whole groups (e.g. camera + day) stay in one fold."""
    g = np.asarray(groups)
    u = np.unique(g)
    perm = np.random.RandomState(seed).permutation(len(u))
    fold_of = {u[j]: i % k for i, j in enumerate(perm)}
    return np.array([fold_of[x] for x in g], int)


def cross_fit(method: str, score: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold probabilities: each row predicted by a calibrator that never saw its group."""
    p = np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f
        p[te] = make(method).fit(score[~te], y[~te]).predict(score[te])
    return p


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)) if len(y) else float("nan")


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else float("nan")


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> List[Dict[str, Any]]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    out = []
    for i in range(bins):
        m = b == i
        if m.any():
            out.append({"bin": i, "lo": i / bins, "hi": (i + 1) / bins, "n": int(m.sum()),
                        "mean_p": float(p[m].mean()), "frac_occupied": float(y[m].mean())})
    return out


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error, equal-width bins, weighted by bin size."""
    n = len(y)
    return float(sum(r["n"] / n * abs(r["frac_occupied"] - r["mean_p"]) for r in reliability(p, y, bins))) if n else float("nan")


def bayes_threshold(c_fv: float, c_fo: float) -> float:
    """Occupied when P(O) > C_FO / (C_FO + C_FV)."""
    return float(c_fo) / (float(c_fo) + float(c_fv))


def cost_counts(gt: np.ndarray, pred: np.ndarray) -> Dict[str, int]:
    gt, pred = np.asarray(gt, bool), np.asarray(pred, bool)
    return {"TP": int((gt & pred).sum()), "TN": int((~gt & ~pred).sum()), "FO": int((~gt & pred).sum()),
            "FV": int((gt & ~pred).sum())}


def cost_per_1000(fv: int, fo: int, n: int, c_fv: float, c_fo: float) -> float:
    """Cost-weighted error per 1000 slot decisions (C_FO = 1 unit when the ratio is quoted)."""
    return 1000.0 * (c_fv * fv + c_fo * fo) / n if n else float("nan")


def score_threshold_sweep(score: np.ndarray, gt: np.ndarray, grid: np.ndarray) -> Dict[str, np.ndarray]:
    """FV and FO at every score threshold of `grid` (occupied when score >= thr), in one pass."""
    score, gt = np.asarray(score, float), np.asarray(gt, bool)
    so, sv = np.sort(score[gt]), np.sort(score[~gt])
    fv = np.searchsorted(so, grid, side="left")              # occupied with score < thr
    fo = len(sv) - np.searchsorted(sv, grid, side="left")    # vacant with score >= thr
    return {"thr": np.asarray(grid, float), "FV": fv, "FO": fo}


def best_score_threshold(score: np.ndarray, gt: np.ndarray, c_fv: float, c_fo: float,
                         grid: Optional[np.ndarray] = None, prefer: Optional[float] = None) -> Dict[str, Any]:
    """Score threshold minimising C_FV*FV + C_FO*FO on the given (validation) data. Ties: closest to `prefer`
    (the cost-agnostic threshold), then the lower threshold."""
    grid = np.round(np.arange(0.05, 0.9501, 0.005), 4) if grid is None else np.asarray(grid, float)
    sw = score_threshold_sweep(score, gt, grid)
    cost = c_fv * sw["FV"] + c_fo * sw["FO"]
    ok = np.nonzero(cost == cost.min())[0]
    if prefer is not None:
        ok = ok[np.argsort(np.abs(grid[ok] - prefer), kind="stable")]
    i = int(ok[0])
    return {"thr": float(grid[i]), "FV": int(sw["FV"][i]), "FO": int(sw["FO"][i]), "cost": float(cost[i])}
