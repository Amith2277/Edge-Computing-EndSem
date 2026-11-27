"""Per-slot temporal validation (spec: consecutive support / window / weighted vote, no future leakage).

One implementation is used everywhere: `TemporalFilter.step` takes the current frame only and keeps its own
per-slot state, so the batch replay (`run_sequence`) and the runtime path on the Orin are the same code and
a frame's output can never depend on later frames.

Methods
- none:          state = the per-frame decision (score >= score threshold).
- consecutive:   the state flips only after `k` consecutive observations of the other state.
- window:        majority of the last `w` per-frame decisions of that slot; a tie keeps the previous state.
- weighted_vote: exponentially weighted average of the slot score (weight `alpha` on the newest frame),
                 compared with the same score threshold. Weights come from the score itself, so a confident
                 detection counts more than a marginal one.
- gated:         confidence-gated consecutive support. A decision whose score is far from the threshold
                 (normalised margin >= `gate`, 0 at the threshold, 1 at score 0 or 1) flips the state at once;
                 a marginal one needs `k` consecutive disagreeing frames. Clear arrivals and departures are not
                 delayed, while near-threshold flicker is held back.
- asymmetric:    cost-aware consecutive support. A flip to occupied needs `k_occ` disagreeing frames, a flip to
                 vacant `k_vac`. k_occ = 1 < k_vac holds back false vacants (the costlier error for a driver)
                 without delaying arrivals. Optional `gate`: a flip in either direction happens at once when the
                 margin is >= gate.

State codes: 1 occupied, 0 vacant, -1 no observation of that slot in this frame (not a vacant claim).
Python 3.8 compatible, numpy only (runs on the Orin).
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

METHODS = ("none", "consecutive", "window", "weighted_vote", "gated", "asymmetric")
PARAMS = ("k", "w", "alpha", "gate", "k_occ", "k_vac")


def param_grid() -> List[Dict[str, Any]]:
    g = [{"method": "none"}]
    g += [{"method": "consecutive", "k": k} for k in (2, 3, 4)]
    g += [{"method": "window", "w": w} for w in (3, 5, 7)]
    g += [{"method": "weighted_vote", "alpha": a} for a in (0.3, 0.5, 0.7)]
    g += [{"method": "gated", "k": 2, "gate": x} for x in (0.25, 0.5, 0.75)]
    return g


def margin(score: np.ndarray, thr: float) -> np.ndarray:
    """Normalised distance from the threshold: 0 at thr, 1 at score 0 (below) or 1 (above)."""
    score = np.asarray(score, float)
    span = np.where(score < thr, thr, 1.0 - thr)
    return np.clip(np.abs(score - thr) / np.where(span > 0, span, 1.0), 0.0, 1.0)


def label(cfg: Dict[str, Any]) -> str:
    p = [(n, cfg.get(n)) for n in PARAMS]
    return cfg["method"] + "".join(" {}={}".format(n, v) for n, v in p if v is not None)


class TemporalFilter(object):
    """Causal per-slot filter for one camera sequence. `step` sees one frame at a time."""

    def __init__(self, n_slots: int, cfg: Dict[str, Any], score_thr: float):
        if cfg["method"] not in METHODS:
            raise ValueError("unknown temporal method {}".format(cfg["method"]))
        self.n, self.cfg, self.thr = n_slots, cfg, float(score_thr)
        self.state = np.full(n_slots, -1, np.int8)          # last emitted state (-1: never observed)
        self.run = np.zeros(n_slots, np.int32)               # consecutive: observations disagreeing with state
        w = int(cfg.get("w", 1))
        self.buf = np.full((w, n_slots), -1, np.int8)        # window: last w decisions per slot
        self.pos = np.zeros(n_slots, np.int64)
        self.ema = np.full(n_slots, np.nan)                  # weighted_vote

    def step(self, score: np.ndarray, observed: Optional[np.ndarray] = None, thr: Optional[float] = None) -> np.ndarray:
        """One frame. `thr` overrides the score threshold for this frame only (the adaptive controller switches
        models, and each model has its own threshold); the per-slot state carries over."""
        score = np.asarray(score, float)
        obs = np.ones(self.n, bool) if observed is None else np.asarray(observed, bool)
        th = self.thr if thr is None else float(thr)
        raw = (score >= th).astype(np.int8)
        m, new = self.cfg["method"], self.state.copy()
        first = obs & (self.state < 0)
        if m == "none":
            new[obs] = raw[obs]
        elif m in ("consecutive", "gated", "asymmetric"):
            if m == "asymmetric":
                k = np.where(raw == 1, int(self.cfg["k_occ"]), int(self.cfg["k_vac"]))
            else:
                k = int(self.cfg["k"])
            new[first] = raw[first]
            o = obs & ~first
            agree = o & (raw == self.state)
            disagree = o & (raw != self.state)
            self.run[agree] = 0
            self.run[disagree] += 1
            flip = disagree & (self.run >= k)
            if m == "gated" or (m == "asymmetric" and self.cfg.get("gate") is not None):
                flip |= disagree & (margin(score, th) >= float(self.cfg["gate"]))
            new[flip] = raw[flip]
            self.run[flip] = 0
        elif m == "window":
            w = self.buf.shape[0]
            idx = np.nonzero(obs)[0]
            self.buf[self.pos[idx] % w, idx] = raw[idx]
            self.pos[idx] += 1
            occ = (self.buf == 1).sum(0)
            vac = (self.buf == 0).sum(0)
            new[obs & (occ > vac)] = 1
            new[obs & (vac > occ)] = 0
            tie = obs & (occ == vac)
            new[tie & first] = raw[tie & first]
        else:  # weighted_vote
            a = float(self.cfg["alpha"])
            init = obs & np.isnan(self.ema)
            upd = obs & ~init
            self.ema[init] = score[init]
            self.ema[upd] = a * score[upd] + (1.0 - a) * self.ema[upd]
            new[obs] = (self.ema[obs] >= th).astype(np.int8)
        self.state = new
        out = new.copy()
        out[~obs] = -1
        return out


def run_sequence(scores: np.ndarray, observed: np.ndarray, cfg: Dict[str, Any], score_thr: float) -> np.ndarray:
    """Replay one sequence (frames x slots) frame by frame. Returns states (frames x slots), -1 where unobserved."""
    f = TemporalFilter(scores.shape[1], cfg, score_thr)
    return np.stack([f.step(scores[t], observed[t]) for t in range(scores.shape[0])]) if len(scores) else \
        np.zeros((0, scores.shape[1]), np.int8)


def split_sequences(cameras: Sequence[str], seconds: np.ndarray, days: Sequence[str],
                    max_gap_factor: float = 3.0) -> Dict[str, Any]:
    """Group frames into sequences: same camera, same day, and no gap above max_gap_factor x the camera's median
    frame spacing. Returns {"seq": sequence id per frame, "order": frame order (camera, time), "spacing_s": per camera}."""
    cameras, days = np.asarray(cameras), np.asarray(days)
    seconds = np.asarray(seconds, float)
    order = np.lexsort((seconds, cameras))
    spacing = {}
    for cam in np.unique(cameras):
        t = np.sort(seconds[cameras == cam])
        d = np.diff(t)
        same_day = np.asarray(days[cameras == cam])[np.argsort(seconds[cameras == cam])]
        d = d[same_day[1:] == same_day[:-1]]
        d = d[d > 0]
        spacing[str(cam)] = float(np.median(d)) if len(d) else float("nan")
    seq = np.zeros(len(seconds), np.int64)
    sid = -1
    prev = None
    for i in order:
        cam, day, t = cameras[i], days[i], seconds[i]
        gap_limit = max_gap_factor * spacing[str(cam)]
        if prev is None or cam != prev[0] or day != prev[1] or not (t - prev[2] <= gap_limit):
            sid += 1
        seq[i] = sid
        prev = (cam, day, t)
    return {"seq": seq, "order": order, "spacing_s": spacing}


def _events(col: np.ndarray, t_idx: np.ndarray):
    """State changes along one slot's observed frames: list of (frame index, new state)."""
    if len(col) < 2:
        return []
    ch = np.nonzero(col[1:] != col[:-1])[0] + 1
    return [(int(t_idx[c]), int(col[c])) for c in ch]


def transition_metrics(gt: np.ndarray, pred: np.ndarray, seconds: np.ndarray, tol_frames: int) -> Dict[str, Any]:
    """Transitions of one sequence (frames x slots; -1 = unobserved; frames in time order).

    A true transition (slot turns to state X at frame t) is detected if the output shows X at some observed frame
    in [t, t + tol_frames] before the next true transition; delay = frames (and seconds) until then. Otherwise it
    is missed. A predicted transition to X at frame t is false if the truth did not turn to X in [t - tol_frames, t]
    and the truth at t is not X (spurious flips and flicker)."""
    out = {"true_transitions": 0, "detected": 0, "missed": 0, "pred_transitions": 0, "false_transitions": 0,
           "delay_frames": [], "delay_seconds": []}
    T, S = gt.shape
    for s in range(S):
        o = (gt[:, s] >= 0) & (pred[:, s] >= 0)
        t_idx = np.nonzero(o)[0]
        if len(t_idx) < 2:
            continue
        g, p = gt[t_idx, s], pred[t_idx, s]
        gev, pev = _events(g, t_idx), _events(p, t_idx)
        out["true_transitions"] += len(gev)
        out["pred_transitions"] += len(pev)
        for e, (t, x) in enumerate(gev):
            end = min(t + tol_frames, gev[e + 1][0] - 1 if e + 1 < len(gev) else T - 1)
            a, b = np.searchsorted(t_idx, t), np.searchsorted(t_idx, end, side="right")
            w = np.nonzero(p[a:b] == x)[0]
            if len(w):
                u = int(t_idx[a + w[0]])
                out["detected"] += 1
                out["delay_frames"].append(u - t)
                out["delay_seconds"].append(float(seconds[u] - seconds[t]))
            else:
                out["missed"] += 1
        gset = {}
        for t, x in gev:
            gset.setdefault(x, []).append(t)
        for t, x in pev:
            recent = any(t - tol_frames <= u <= t for u in gset.get(x, []))
            if not recent and g[np.searchsorted(t_idx, t)] != x:
                out["false_transitions"] += 1
    return out


def merge_transition_metrics(parts: List[Dict[str, Any]], slot_frames: int, slot_hours: float) -> Dict[str, Any]:
    tot = {k: sum(p[k] for p in parts) for k in ("true_transitions", "detected", "missed", "pred_transitions",
                                                 "false_transitions")}
    df = [d for p in parts for d in p["delay_frames"]]
    ds = [d for p in parts for d in p["delay_seconds"]]
    tot.update({
        "detection_rate": round(tot["detected"] / tot["true_transitions"], 4) if tot["true_transitions"] else None,
        "missed_rate": round(tot["missed"] / tot["true_transitions"], 4) if tot["true_transitions"] else None,
        "false_per_1000_slot_frames": round(1000.0 * tot["false_transitions"] / slot_frames, 4) if slot_frames else None,
        "pred_transitions_per_slot_hour": round(tot["pred_transitions"] / slot_hours, 4) if slot_hours else None,
        "true_transitions_per_slot_hour": round(tot["true_transitions"] / slot_hours, 4) if slot_hours else None,
        "mean_delay_frames": round(float(np.mean(df)), 3) if df else None,
        "p90_delay_frames": float(np.percentile(df, 90)) if df else None,
        "mean_delay_minutes": round(float(np.mean(ds)) / 60.0, 2) if ds else None,
        "slot_frames": int(slot_frames), "slot_hours": round(float(slot_hours), 2)})
    return tot


def choose(rows: List[Dict[str, Any]], f1_tolerance: float) -> Dict[str, Any]:
    """Selection rule (fixed before test): among configs within f1_tolerance of the best val macro-F1, fewest false
    transitions per 1000 slot-frames, then shortest mean delay, then the simplest (grid order)."""
    best = max(r["macro_f1"] for r in rows)
    ok = [r for r in rows if r["macro_f1"] >= best - f1_tolerance]
    return min(ok, key=lambda r: (r["false_per_1000_slot_frames"], r["mean_delay_frames"] or 0.0, r["grid_index"]))
