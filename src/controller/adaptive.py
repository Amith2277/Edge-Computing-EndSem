"""Adaptive inference controller (spec sections 12-13). Python 3.8 + numpy (runs on the Orin).

Modes
  reduced         YOLO11n every `cadence` frames; frames in between are SKIPPED (last state reused, marked stale)
  standard        YOLO11n every frame
  enhanced        YOLO11s every frame (verification: none configured; the bigger model is the escalation)
  forced_refresh  a frame that would be skipped is run with YOLO11n because a safety condition fired:
                  sequence start, a slot reached max observation age, or a single slot's risk is critical

Decisions use only what exists before the frame's inference: D (B, M from the current image; C, V, O, S from the
last processed frame), the global slot risk and the slot ages. Hysteresis: once escalated, D / risk must fall
`margin` below the threshold to leave; once reduced, they must rise `margin` above the threshold to leave.
Minimum dwell: a mode is kept for at least `min_dwell` frames before DE-escalating (escalation is never delayed).
Skipped frames never produce a vacant observation: SlotTracker.serve returns the last reliable state.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from src.slot_risk import risk as rk

MODES = ("reduced", "standard", "enhanced", "forced_refresh")
LEVEL = {"reduced": 0, "standard": 1, "enhanced": 2}


class SlotTracker(object):
    """Per-slot memory for one camera: last score/state, the score before it, when it was observed, occlusion."""

    def __init__(self, n_slots: int):
        self.score = np.full(n_slots, np.nan)
        self.prev = np.full(n_slots, np.nan)
        self.thr = np.full(n_slots, np.nan)
        self.state = np.full(n_slots, -1, np.int8)
        self.last_t = np.full(n_slots, np.nan)
        self.occ = np.full(n_slots, np.nan)

    def observe(self, cols, scores, thr: float, occ, t: float) -> None:
        cols = np.asarray(cols, int)
        self.prev[cols] = self.score[cols]
        self.score[cols] = scores
        self.thr[cols] = thr
        self.state[cols] = (np.asarray(scores) >= thr).astype(np.int8)
        self.last_t[cols] = t
        self.occ[cols] = occ

    def ages(self, cols, t: float) -> np.ndarray:
        return t - self.last_t[np.asarray(cols, int)]          # NaN = never observed

    def components(self, cols, t: float, age_ref_s: float) -> np.ndarray:
        """(len(cols), 4) risk components U, Delta, A, O; NaN where not available."""
        cols = np.asarray(cols, int)
        s, p, th = self.score[cols], self.prev[cols], self.thr[cols]
        sf_, tf_ = np.nan_to_num(s), np.nan_to_num(th, nan=0.5)
        span = np.where(sf_ < tf_, tf_, 1.0 - tf_)
        U = np.clip(1.0 - np.abs(sf_ - tf_) / np.where(span > 0, span, 1.0), 0.0, 1.0)
        U = np.where(np.isnan(s), np.nan, U)
        D = np.where(np.isnan(s) | np.isnan(p), np.nan, np.abs(s - p))
        age = t - self.last_t[cols]
        A = np.where(np.isnan(age), np.nan, rk.age_component(np.nan_to_num(age), age_ref_s))
        return np.stack([U, D, A, self.occ[cols]], 1)

    def serve(self, cols) -> np.ndarray:
        return self.state[np.asarray(cols, int)].copy()


class ModeController(object):
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.mode = "standard"
        self.dwell = 0
        self.since_run = 10 ** 9

    def _target(self, D: float, Rg: float) -> str:
        c, m = self.cfg, float(self.cfg.get("margin", 0.0))
        use_r = c.get("use_risk", True)
        esc = self.mode == "enhanced"
        hi = (not np.isnan(D) and D >= c["d_hi"] - (m if esc else 0.0)) or \
             (use_r and not np.isnan(Rg) and Rg >= c["r_hi"] - (m if esc else 0.0))
        red = self.mode == "reduced"
        d_lo, r_lo = c.get("d_lo"), c.get("r_lo")
        lo = d_lo is not None and not np.isnan(D) and D < d_lo + (m if red else 0.0)
        if use_r:
            lo = lo and r_lo is not None and not np.isnan(Rg) and Rg < r_lo + (m if red else 0.0)
        return "enhanced" if hi else ("reduced" if lo else "standard")

    def decide(self, D: float, Rg: float, Rmax: float, age_max: float, first: bool) -> Dict[str, Any]:
        c = self.cfg
        target = self._target(D, Rg)
        if LEVEL[target] < LEVEL[self.mode] and self.dwell < int(c.get("min_dwell", 1)):
            target = self.mode                                   # dwell: no de-escalation yet
        if target != self.mode:
            self.mode, self.dwell = target, 0
        self.dwell += 1
        reasons = []
        if target == "enhanced":
            if not np.isnan(D) and D >= c["d_hi"] - float(c.get("margin", 0.0)):
                reasons.append("difficulty")
            if c.get("use_risk", True) and not np.isnan(Rg) and Rg >= c["r_hi"] - float(c.get("margin", 0.0)):
                reasons.append("global_slot_risk")
            reasons = reasons or ["hold (hysteresis/dwell)"]
        run = target != "reduced" or self.since_run + 1 >= int(c.get("cadence", 1))
        safety = []
        if first:
            safety.append("sequence_start")
        if c.get("max_age_s") is not None and not np.isnan(age_max) and age_max >= c["max_age_s"]:
            safety.append("max_age")
        if c.get("use_risk", True) and c.get("r_crit") is not None and not np.isnan(Rmax) and Rmax >= c["r_crit"]:
            safety.append("critical_slot_risk")
        mode = target
        if not run and safety:
            run, mode = True, "forced_refresh"
        model = None
        if run:
            model = c["heavy"] if target == "enhanced" else c["light"]
        self.since_run = 0 if run else self.since_run + 1
        return {"mode": mode, "level_mode": target, "run": run, "model": model, "reasons": reasons + safety}
