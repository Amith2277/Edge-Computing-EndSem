import numpy as np
import pytest

from src.controller import replay as rp
from src.evaluation import bootstrap as bs
from src.occupancy import cost_sensitive as cs
from src.temporal import filters as tf


def test_score_threshold_for_matches_probability_rule():
    rng = np.random.RandomState(0)
    s = rng.rand(4000); y = (rng.rand(4000) < s).astype(int)
    for meth in ("isotonic", "platt"):
        cal = cs.make(meth).fit(s, y)
        for p in (1 / 6, 0.5, 0.8):
            t = cs.score_threshold_for(cal, p)
            probe = np.r_[s, np.linspace(0, 1, 501)]
            assert ((cal.predict(probe) > p) == (probe >= t)).all(), (meth, p)
    iso = cs.Isotonic(); iso.x, iso.y = np.array([0.0, 0.5]), np.array([0.3, 0.9])
    assert cs.score_threshold_for(iso, 0.2) == -np.inf and cs.score_threshold_for(iso, 0.95) == np.inf


def test_filter_threshold_override_per_frame():
    f = tf.TemporalFilter(1, {"method": "none"}, 0.5)
    assert f.step(np.array([0.3]))[0] == 0
    assert f.step(np.array([0.3]), thr=0.2)[0] == 1          # other model, lower threshold
    assert f.thr == 0.5


def _seq(scores_n, scores_s, gt):
    T = len(gt)
    det = {"confidence": 0.2, "density": 0.5, "occlusion_raw": 0.0}
    return {"sec": np.arange(T) * 300.0, "n_cols": 1, "cols": [np.array([0])] * T, "valid": [np.array([True])] * T,
            "gt": [np.array([g]) for g in gt], "B": np.zeros(T), "M": np.zeros(T), "fixable": np.zeros(T),
            "out": {"n": {"score": [np.array([x]) for x in scores_n], "detf": [det] * T, "occ": [np.zeros(1)] * T},
                    "s": {"score": [np.array([x]) for x in scores_s], "detf": [det] * T, "occ": [np.zeros(1)] * T}}}


def test_replay_with_asymmetric_filter_holds_back_vacant():
    ctx = {"thr": {"n": 0.5, "s": 0.5}, "wD": np.full(6, 1 / 6), "wR": np.full(4, 0.25), "agg": {"method": "max"},
           "age_ref_s": 1800.0, "inst_frames": 3, "occ_scale": 1.0, "light": "n", "heavy": "s"}
    seq = _seq([0.9, 0.4, 0.9, 0.4, 0.4, 0.4], [0.9] * 6, [1, 1, 1, 1, 1, 1])
    raw = rp.run_sequence(seq, "fixed_n", {}, ctx)
    assert [int(p[0]) for p in raw["pred"]] == [1, 0, 1, 0, 0, 0]
    filt = rp.run_sequence(seq, "fixed_n", {}, dict(ctx, filter={"method": "asymmetric", "k_occ": 1, "k_vac": 3}))
    assert [int(p[0]) for p in filt["pred"]] == [1, 1, 1, 1, 1, 0]   # one-frame dip ignored, 3rd vacant frame flips
    m = rp.evaluate([seq], [filt], {"n": 1.0, "s": 2.0})
    assert m["FV"] == 1 and rp.evaluate([seq], [raw], {"n": 1.0, "s": 2.0})["FV"] == 4


def test_paired_bootstrap():
    rng = np.random.RandomState(1)
    rows = lambda fv: [{"TP": 50, "TN": 50, "FO": 1, "FV": fv + rng.randint(0, 2), "frames": 10, "cost": 10} for _ in range(30)]
    a, b = bs.counts(rows(5)), bs.counts(rows(1))
    r = bs.paired(a, b, bs.cost_per_1000(5, 1), n_boot=500)
    assert r["difference"] > 0 and r["ci95"][0] > 0 and r["share_above_0"] == 1.0
    same = bs.paired(a, a, bs.macro_f1, n_boot=200)
    assert same["difference"] == 0 and same["ci95"] == [0.0, 0.0]
    assert bs.compute_per_frame(a.sum(0)) == pytest.approx(1.0)
    assert 0 < bs.macro_f1(a.sum(0)) < 1 and 0 < bs.accuracy(a.sum(0)) < 1
