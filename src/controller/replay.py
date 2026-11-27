"""Replay a camera sequence through a strategy using cached per-frame model outputs (notebook 10).

Every strategy goes through the same loop, so the comparison is like for like:
  fixed_n / fixed_s   the light / heavy model on every frame
  periodic_n          the light model every `cadence` frames, nothing in between (no safety refresh)
  scene_only          ModeController driven by D only (no slot risk, no critical-slot refresh)
  adaptive            ModeController driven by D and global slot risk, with critical-slot and max-age refresh
  oracle              the heavy model exactly on frames where it fixes a light-model slot error (uses labels:
                      an upper bound, not a deployable strategy)

seq (one camera sequence, frames in time order):
  sec (T,), n_cols, cols[t] (slot column indices), valid[t], gt[t], B (T,), M (T,) scaled with NaN = missing,
  fixable (T,), out[model] = {"score": [arrays], "detf": [dicts], "occ": [arrays]}
ctx: thr {model}, wD (6,), wR (4,), agg {"method", "k"}, age_ref_s, inst_frames, occ_scale, light, heavy
  optional: filter (temporal filter cfg, src/temporal/filters.py) applied to every executed frame's slot scores at
  that model's threshold; the served state is the filtered state. fixable_key (default "fixable") for the oracle.
"""
from typing import Any, Dict, List

import numpy as np

from src.controller.adaptive import ModeController, SlotTracker
from src.scene_analysis import features as sf
from src.temporal.filters import TemporalFilter
from src.slot_risk import risk as rk

STRATEGIES = ("fixed_n", "fixed_s", "periodic_n", "scene_only", "adaptive", "oracle")


def _controller_cfg(strategy: str, cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    c = dict(cfg)
    c.update(light=ctx["light"], heavy=ctx["heavy"])
    if strategy == "scene_only":
        c.update(use_risk=False, r_crit=None)
    return c


def run_sequence(seq: Dict[str, Any], strategy: str, cfg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError("unknown strategy {}".format(strategy))
    T, S = len(seq["sec"]), seq["n_cols"]
    tr = SlotTracker(S)
    ctl = ModeController(_controller_cfg(strategy, cfg, ctx)) if strategy in ("scene_only", "adaptive") else None
    last_detf, hist = None, []
    filt = TemporalFilter(S, ctx["filter"], ctx["thr"][ctx["light"]]) if ctx.get("filter") else None
    out = {"pred": [], "age_s": [], "stale": [], "model": [], "mode": [], "reasons": [], "D": np.full(T, np.nan),
           "Rg": np.full(T, np.nan), "Rmax": np.full(T, np.nan), "F": np.full((T, 6), np.nan)}   # F: B, M, C, V, O, S
    for t in range(T):
        cols = np.asarray(seq["cols"][t], int)
        vcols = cols[seq["valid"][t]]
        # signals available BEFORE this frame's inference
        if last_detf is None:
            C = V = O = np.nan
        else:
            C = np.nan if last_detf["confidence"] is None else last_detf["confidence"]
            V = np.nan if last_detf["density"] is None else last_detf["density"]
            O = np.nan if last_detf["occlusion_raw"] is None else float(sf.scale(last_detf["occlusion_raw"], ctx["occ_scale"]))
        Sv = sf.instability(hist[-(ctx["inst_frames"] + 1):])
        Fv = np.array([[seq["B"][t], seq["M"][t], C, V, O, np.nan if Sv is None else Sv]])
        D = float(sf.weighted_score(Fv, ctx["wD"])[0][0])
        comps = tr.components(vcols, seq["sec"][t], ctx["age_ref_s"])
        R = sf.weighted_score(comps, ctx["wR"])[0] if len(vcols) else np.zeros(0)
        Rg = rk.aggregate(R, ctx["agg"]["method"], ctx["agg"].get("k")) if len(R) else np.nan
        Rmax = float(np.nanmax(R)) if len(R) and (~np.isnan(R)).any() else np.nan
        ages = tr.ages(vcols, seq["sec"][t])
        never = np.isnan(ages)
        first = bool(never.all()) if len(ages) else True
        age_max = float(np.inf) if never.any() else (float(ages.max()) if len(ages) else np.nan)
        out["D"][t], out["Rg"][t], out["Rmax"][t] = D, Rg, Rmax
        out["F"][t] = Fv[0]
        if strategy == "fixed_n":
            dec = {"mode": "standard", "run": True, "model": ctx["light"], "reasons": []}
        elif strategy == "fixed_s":
            dec = {"mode": "enhanced", "run": True, "model": ctx["heavy"], "reasons": []}
        elif strategy == "periodic_n":
            run = t % int(cfg.get("cadence", 2)) == 0
            dec = {"mode": "reduced", "run": run, "model": ctx["light"] if run else None, "reasons": []}
        elif strategy == "oracle":
            heavy = seq[ctx.get("fixable_key", "fixable")][t] > 0
            dec = {"mode": "enhanced" if heavy else "standard", "run": True, "model": ctx["heavy"] if heavy else ctx["light"],
                   "reasons": ["oracle"] if heavy else []}
        else:
            dec = ctl.decide(D, Rg, Rmax, age_max, first)
        if dec["run"]:
            m = dec["model"]
            o = seq["out"][m]
            v = np.asarray(seq["valid"][t], bool)
            tr.observe(cols[v], np.asarray(o["score"][t])[v], ctx["thr"][m], np.asarray(o["occ"][t])[v], seq["sec"][t])
            last_detf = o["detf"][t]
            if filt is not None:
                sc = np.zeros(S); ob = np.zeros(S, bool)
                sc[cols[v]] = np.asarray(o["score"][t])[v]; ob[cols[v]] = True
                filt.step(sc, ob, thr=ctx["thr"][m])
        # skipped frame: the last reliable state, never a fresh 'vacant'
        served = filt.state[np.asarray(vcols, int)].copy() if filt is not None else tr.serve(vcols)
        age = tr.ages(vcols, seq["sec"][t])
        if dec["run"]:
            full = np.full(S, -1, np.int8); full[vcols] = served; hist.append(full)
        out["pred"].append(served); out["age_s"].append(age); out["stale"].append(~np.isnan(age) & (age > 0))
        out["model"].append(dec["model"] if dec["run"] else None); out["mode"].append(dec["mode"]); out["reasons"].append(dec["reasons"])
    return out


def evaluate(seqs: List[Dict[str, Any]], outs: List[Dict[str, Any]], cost: Dict[str, float], max_age_s=None) -> Dict[str, Any]:
    """Slot accuracy over EVERY frame (skipped frames are judged on the state they show), cost and freshness."""
    tp = tn = fo = fv = unknown = stale = n = 0
    frames = skipped = switches = 0
    calls, modes, reasons = {}, {}, {}
    age_sum, age_viol, hours = 0.0, 0, 0.0
    for seq, o in zip(seqs, outs):
        prev_mode = None
        for t in range(len(seq["sec"])):
            g = np.asarray(seq["gt"][t])[np.asarray(seq["valid"][t], bool)].astype(bool)
            p = o["pred"][t]
            known = p >= 0
            pp = p[known].astype(bool); gg = g[known]
            tp += int((pp & gg).sum()); tn += int((~pp & ~gg).sum()); fo += int((pp & ~gg).sum()); fv += int((~pp & gg).sum())
            unknown += int((~known).sum()); stale += int(o["stale"][t].sum()); n += len(p)
            a = o["age_s"][t][known]
            age_sum += float(np.nansum(a))
            if max_age_s is not None:
                age_viol += int((a > max_age_s).sum())
            frames += 1
            m = o["model"][t]
            if m is None:
                skipped += 1
            else:
                calls[m] = calls.get(m, 0) + 1
            modes[o["mode"][t]] = modes.get(o["mode"][t], 0) + 1
            for r in o["reasons"][t]:
                reasons[r] = reasons.get(r, 0) + 1
            if prev_mode is not None and o["mode"][t] != prev_mode:
                switches += 1
            prev_mode = o["mode"][t]
        if len(seq["sec"]) > 1:
            hours += (seq["sec"][-1] - seq["sec"][0]) / 3600.0
    from src.occupancy.association import summarize
    s = summarize(tp, tn, fo, fv)
    total_cost = sum(cost[m] * k for m, k in calls.items())
    s.update({"frames": frames, "calls": calls, "skipped_frames": skipped, "skipped_share": round(skipped / frames, 4) if frames else None,
              "cost_per_frame": round(total_cost / frames, 4) if frames else None, "unknown_slot_frames": unknown,
              "stale_share": round(stale / n, 4) if n else None, "mean_served_age_min": round(age_sum / max(1, n - unknown) / 60.0, 2),
              "max_age_violations": age_viol if max_age_s is not None else None,
              "mode_share": {k: round(v / frames, 4) for k, v in sorted(modes.items())}, "reasons": reasons,
              "mode_switches_per_hour": round(switches / hours, 3) if hours else None})
    return s
