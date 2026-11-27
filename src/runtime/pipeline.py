"""Per-frame AdaptivePark pipeline for one camera: image in, slot states out (notebook 14 on Kaggle; the Orin replay).

Same logic, step for step, as the batch replay `src/controller/replay.py` (a unit test checks they agree exactly),
but driven by real images and a detector callable, with a wall-clock timing of every stage:

  scene      gray crop of the slot region, darkness B, motion M against the previous frame
  decide     detector features of the last processed frame, slot risk, difficulty D, controller decision
  detect     the chosen model on this frame (skipped frames: none)
  associate  vehicle-to-slot scores, slot occlusion, detection features, slot misalignment
  update     slot tracker and temporal filter
  drift      drift monitor update (optional)

Skipped frames serve the last reliable state with its age; they never produce a fresh vacant.
Python 3.8 + numpy (+ PIL for the image); the detector is any callable (model_name, image) -> (N, 5) array
[x0, y0, x1, y1, conf] in image pixels.
"""
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.controller.adaptive import ModeController, SlotTracker
from src.controller.replay import _controller_cfg
from src.drift.monitor import slot_misalignment
from src.occupancy import association as asc
from src.scene_analysis import features as sf
from src.slot_risk import risk as rk
from src.temporal.filters import TemporalFilter

STAGES = ("scene", "decide", "detect", "associate", "update", "drift")
STRATEGIES = ("fixed_n", "fixed_s", "periodic_n", "scene_only", "adaptive")


def slot_region(polygons: Sequence[Sequence[float]]) -> List[float]:
    """Bounding box of all slot polygons of a camera (where B and M are measured)."""
    xs = [x for p in polygons for x in p[0::2]]
    ys = [y for p in polygons for y in p[1::2]]
    return [min(xs), min(ys), max(xs), max(ys)]


class FramePipeline(object):
    """One camera stream.

    slots: [{"slot_id", "polygon"}] of the camera; list order = slot column.
    ctx: as replay (thr = decision threshold per model, wD, wR, agg, age_ref_s, inst_frames, occ_scale, light,
         heavy, optional filter).
    assoc: per model {"method", "overlap_thr", "exclusive"}; feat_thr: per model score threshold for the detection
    features (C, V, O), slot occlusion and misalignment (notebook 07 thresholds, as in notebooks 09-12).
    """

    def __init__(self, slots: List[Dict[str, Any]], ctx: Dict[str, Any], strategy: str, controller_cfg: Dict[str, Any],
                 assoc: Dict[str, Dict[str, Any]], feat_thr: Dict[str, float], detector: Callable,
                 motion_scale: float, region: Optional[Sequence[float]] = None, drift=None, clock: Callable = time.perf_counter):
        if strategy not in STRATEGIES:
            raise ValueError("unknown strategy {}".format(strategy))
        self.ids = [s["slot_id"] for s in slots]
        self.col = {s: i for i, s in enumerate(self.ids)}
        self.poly = {s["slot_id"]: s["polygon"] for s in slots}
        self.S = len(slots)
        self.ctx, self.strategy, self.cfg = ctx, strategy, dict(controller_cfg or {})
        self.assoc, self.feat_thr, self.detector = assoc, feat_thr, detector
        self.motion_scale = float(motion_scale)
        self.region = list(region) if region is not None else slot_region([s["polygon"] for s in slots])
        self.drift, self.clock = drift, clock
        self.tr = SlotTracker(self.S)
        self.ctl = ModeController(_controller_cfg(strategy, self.cfg, ctx)) if strategy in ("scene_only", "adaptive") else None
        self.filt = TemporalFilter(self.S, ctx["filter"], ctx["thr"][ctx["light"]]) if ctx.get("filter") else None
        self.last_detf, self.hist, self.prev_gray, self.t_index = None, [], None, 0
        self._prep = {}

    def _prepared(self, sids):
        if sids not in self._prep:
            self._prep[sids] = asc.prepare_slots([self.poly[s] for s in sids])
        return self._prep[sids]

    def step(self, image, t: float, sids: Optional[Sequence[str]] = None, valid: Optional[np.ndarray] = None,
             B: Optional[float] = None, M: Optional[float] = None, frame: Any = None) -> Dict[str, Any]:
        """Process one frame taken at time t (seconds). sids: the slots annotated in this frame (default: all);
        valid: which of them count. B / M override the image features (for parity tests with cached features)."""
        ck, ms = self.clock, {k: 0.0 for k in STAGES}
        ctx = self.ctx
        sids = tuple(self.ids) if sids is None else tuple(sids)
        cols = np.array([self.col[s] for s in sids], int)
        v = np.ones(len(sids), bool) if valid is None else np.asarray(valid, bool)
        vcols = cols[v]
        t0 = ck()
        if B is None or M is None:
            gray = sf.to_gray_small(image, self.region)
            Bv = sf.brightness(gray)
            m = sf.motion_raw(gray, self.prev_gray)
            Mv = np.nan if m is None else float(sf.scale(m, self.motion_scale))
            self.prev_gray = gray
        else:
            Bv, Mv = float(B), float(M)
        t1 = ck(); ms["scene"] = t1 - t0
        if self.last_detf is None:
            C = V = O = np.nan
        else:
            d = self.last_detf
            C = np.nan if d["confidence"] is None else d["confidence"]
            V = np.nan if d["density"] is None else d["density"]
            O = np.nan if d["occlusion_raw"] is None else float(sf.scale(d["occlusion_raw"], ctx["occ_scale"]))
        Sv = sf.instability(self.hist[-(ctx["inst_frames"] + 1):])
        Fv = np.array([[Bv, Mv, C, V, O, np.nan if Sv is None else Sv]])
        D = float(sf.weighted_score(Fv, ctx["wD"])[0][0])
        comps = self.tr.components(vcols, t, ctx["age_ref_s"])
        R = sf.weighted_score(comps, ctx["wR"])[0] if len(vcols) else np.zeros(0)
        Rg = rk.aggregate(R, ctx["agg"]["method"], ctx["agg"].get("k")) if len(R) else np.nan
        Rmax = float(np.nanmax(R)) if len(R) and (~np.isnan(R)).any() else np.nan
        ages = self.tr.ages(vcols, t)
        never = np.isnan(ages)
        first = bool(never.all()) if len(ages) else True
        age_max = float(np.inf) if never.any() else (float(ages.max()) if len(ages) else np.nan)
        if self.strategy == "fixed_n":
            dec = {"mode": "standard", "run": True, "model": ctx["light"], "reasons": []}
        elif self.strategy == "fixed_s":
            dec = {"mode": "enhanced", "run": True, "model": ctx["heavy"], "reasons": []}
        elif self.strategy == "periodic_n":
            run = self.t_index % int(self.cfg.get("cadence", 2)) == 0
            dec = {"mode": "reduced", "run": run, "model": ctx["light"] if run else None, "reasons": []}
        else:
            dec = self.ctl.decide(D, Rg, Rmax, age_max, first)
        t2 = ck(); ms["decide"] = t2 - t1
        mis = np.nan
        if dec["run"]:
            mname = dec["model"]
            dets = np.asarray(self.detector(mname, image), float).reshape(-1, 5)
            t3 = ck(); ms["detect"] = t3 - t2
            a = self.assoc[mname]
            pairs = asc.image_pairs(self._prepared(sids), dets)
            score = asc.slot_scores(len(sids), pairs, a["method"], a.get("overlap_thr"), a["exclusive"])
            occ = rk.slot_occlusion(len(sids), pairs, self.feat_thr[mname], float(a["overlap_thr"]))
            detf = sf.detection_features(dets, self.feat_thr[mname], len(sids))
            mis = slot_misalignment(pairs, self.feat_thr[mname])
            t4 = ck(); ms["associate"] = t4 - t3
            self.tr.observe(cols[v], np.asarray(score)[v], ctx["thr"][mname], np.asarray(occ)[v], t)
            self.last_detf = detf
            if self.filt is not None:
                sc = np.zeros(self.S); ob = np.zeros(self.S, bool)
                sc[cols[v]] = np.asarray(score)[v]; ob[cols[v]] = True
                self.filt.step(sc, ob, thr=ctx["thr"][mname])
        else:
            t4 = ck()
        served = self.filt.state[vcols].copy() if self.filt is not None else self.tr.serve(vcols)
        age = self.tr.ages(vcols, t)
        if dec["run"]:
            full = np.full(self.S, -1, np.int8); full[vcols] = served; self.hist.append(full)
        t5 = ck(); ms["update"] = t5 - t4
        drift = None
        if self.drift is not None:
            drift = self.drift.update({"brightness": Bv, "confidence": C, "density": V, "occlusion": O,
                                       "instability": np.nan if Sv is None else Sv, "difficulty": D, "slot_risk": Rg,
                                       "misalignment": mis}, frame=frame, timestamp=t)
        ms["drift"] = ck() - t5
        self.t_index += 1
        return {"slot_ids": [sids[i] for i in np.nonzero(v)[0]], "state": served, "age_s": age,
                "stale": ~np.isnan(age) & (age > 0), "mode": dec["mode"], "model": dec["model"] if dec["run"] else None,
                "reasons": dec["reasons"], "D": D, "Rg": Rg, "Rmax": Rmax, "drift": drift,
                "ms": {k: 1000.0 * x for k, x in ms.items()}}
