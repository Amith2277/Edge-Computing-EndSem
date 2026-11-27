import numpy as np
import pytest

from src.controller import replay as rp
from src.drift import monitor as dm
from src.occupancy import association as asc
from src.runtime.pipeline import FramePipeline, slot_region
from src.scene_analysis import features as sf
from src.slot_risk import risk as rk

ASSOC = {"method": "box_iou", "overlap_thr": 0.2, "exclusive": True}
SLOTS = [{"slot_id": "cam#{:03d}".format(j), "polygon": [30 * j, 0, 30 * j + 24, 0, 30 * j + 24, 40, 30 * j, 40]} for j in range(5)]


def _synthetic(T=40, seed=0):
    rng = np.random.RandomState(seed)
    occ = rng.rand(5) < 0.5
    dets = {"n": [], "s": []}
    for t in range(T):
        occ = occ ^ (rng.rand(5) < 0.1)
        for m, noise in (("n", 0.3), ("s", 0.15)):
            rows = []
            for j in range(5):
                c = float(np.clip((0.8 if occ[j] else 0.1) + rng.normal(0, noise), 0, 1))
                if c >= 0.05:
                    dx = rng.randint(-3, 4)
                    rows.append([30 * j + dx, 0, 30 * j + 24 + dx, 40, c])
            dets[m].append(np.array(rows, float).reshape(-1, 5))
    return dets, rng.rand(T), rng.rand(T) * 0.3


def _seq(dets, B, M, thr_feat):
    T = len(B); sids = tuple(s["slot_id"] for s in SLOTS); prep = asc.prepare_slots([s["polygon"] for s in SLOTS])
    seq = {"sec": np.arange(T) * 300.0, "n_cols": 5, "cols": [np.arange(5)] * T, "valid": [np.ones(5, bool)] * T,
           "gt": [np.ones(5, int)] * T, "B": B, "M": M, "fixable": np.zeros(T), "out": {}}
    for m in ("n", "s"):
        o = {"score": [], "detf": [], "occ": []}
        for t in range(T):
            pairs = asc.image_pairs(prep, dets[m][t])
            o["score"].append(asc.slot_scores(5, pairs, ASSOC["method"], ASSOC["overlap_thr"], ASSOC["exclusive"]))
            o["detf"].append(sf.detection_features(dets[m][t], thr_feat[m], 5))
            o["occ"].append(rk.slot_occlusion(5, pairs, thr_feat[m], ASSOC["overlap_thr"]))
        seq["out"][m] = o
    return seq


@pytest.mark.parametrize("strategy,filt", [("adaptive", None), ("adaptive", {"method": "asymmetric", "k_occ": 1, "k_vac": 3, "gate": 0.75}),
                                           ("periodic_n", None), ("fixed_s", {"method": "asymmetric", "k_occ": 1, "k_vac": 2})])
def test_pipeline_matches_replay(strategy, filt):
    dets, B, M = _synthetic()
    feat = {"n": 0.25, "s": 0.2}
    ctx = {"thr": {"n": 0.15, "s": 0.1}, "wD": np.array([0.3, 0.1, 0.3, 0.0, 0.3, 0.0]), "wR": np.array([0.0, 0.8, 0.1, 0.1]),
           "agg": {"method": "topk_mean", "k": 2}, "age_ref_s": 1800.0, "inst_frames": 3, "occ_scale": 0.3, "light": "n", "heavy": "s",
           "filter": filt}
    cfg = {"d_hi": 0.45, "d_lo": 0.3, "r_hi": 0.3, "r_lo": 0.05, "r_crit": 0.6, "use_risk": True, "margin": 0.02, "min_dwell": 2,
           "cadence": 2, "max_age_s": 900}
    seq = _seq(dets, B, M, feat)
    ref = rp.run_sequence(seq, strategy, cfg, ctx)
    pipe = FramePipeline(SLOTS, ctx, strategy, cfg, {"n": ASSOC, "s": ASSOC}, feat, lambda m, t: dets[m][t], motion_scale=1.0)
    out = [pipe.step(t, seq["sec"][t], B=B[t], M=M[t]) for t in range(len(B))]
    assert [o["model"] for o in out] == ref["model"] and [o["mode"] for o in out] == ref["mode"]
    assert all((o["state"] == p).all() for o, p in zip(out, ref["pred"]))
    assert np.allclose([o["D"] for o in out], ref["D"]) and np.allclose([o["Rg"] for o in out], ref["Rg"], equal_nan=True)
    assert len({m for m in ref["model"]}) > 1 or strategy == "fixed_s"            # the test exercises switching
    assert set(out[0]["ms"]) == {"scene", "decide", "detect", "associate", "update", "drift"}


def test_pipeline_images_and_drift():
    from PIL import Image
    dets, _, _ = _synthetic(T=12)
    base = {s: dm.baseline_quantiles(np.random.RandomState(1).rand(200)) for s in dm.SIGNALS}
    thr = dm.calibrate([{s: float(x) for s in dm.SIGNALS} for x in np.random.RandomState(2).rand(200)])
    mon = dm.DriftMonitor("cam", base, thr, "median_shift", window=4)
    ctx = {"thr": {"n": 0.15, "s": 0.1}, "wD": np.full(6, 1 / 6), "wR": np.array([0.0, 0.8, 0.1, 0.1]), "agg": {"method": "max"},
           "age_ref_s": 1800.0, "inst_frames": 3, "occ_scale": 0.3, "light": "n", "heavy": "s"}
    frames = [Image.fromarray(np.full((60, 160, 3), 40 + 15 * t, np.uint8)) for t in range(12)]
    idx = {id(f): t for t, f in enumerate(frames)}
    pipe = FramePipeline(SLOTS, ctx, "fixed_n", {}, {"n": ASSOC, "s": ASSOC}, {"n": 0.25, "s": 0.2},
                         lambda m, im: dets[m][idx[id(im)]], motion_scale=0.2, drift=mon)
    out = [pipe.step(f, 300.0 * t, frame=t) for t, f in enumerate(frames)]
    assert out[0]["D"] == out[0]["D"] and out[1]["drift"]["window_full"] is False and out[-1]["drift"]["window_full"] is True
    assert slot_region([s["polygon"] for s in SLOTS]) == [0, 0, 144, 40]
    assert all(o["model"] == "n" for o in out) and all(o["ms"]["detect"] >= 0 for o in out)
