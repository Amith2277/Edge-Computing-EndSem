import numpy as np
import pytest

from src.controller.adaptive import ModeController, SlotTracker
from src.controller import replay as rp

BASE = {"d_hi": 0.6, "d_lo": 0.2, "r_hi": 0.7, "r_lo": 0.3, "r_crit": 0.95, "margin": 0.05, "min_dwell": 2,
        "cadence": 3, "max_age_s": 1000, "light": "n", "heavy": "s", "use_risk": True}
nan = float("nan")


def test_modes_hysteresis_dwell_and_cadence():
    c = ModeController(BASE)
    d = c.decide(0.7, 0.1, 0.1, 0.0, False)
    assert d["mode"] == "enhanced" and d["model"] == "s" and "difficulty" in d["reasons"]
    d = c.decide(0.57, 0.1, 0.1, 0.0, False)                  # below d_hi but within the margin: stay
    assert d["mode"] == "enhanced"
    d = c.decide(0.4, 0.1, 0.1, 0.0, False)                   # clearly below: de-escalate (dwell 3 >= 2)
    assert d["mode"] == "standard" and d["model"] == "n"
    d = c.decide(0.1, 0.1, 0.1, 0.0, False)                   # easy, but dwell of standard is only 1
    assert d["mode"] == "standard"
    runs = [c.decide(0.1, 0.1, 0.1, 0.0, False)["run"] for _ in range(6)]
    assert c.mode == "reduced" and runs == [False, False, True, False, False, True]   # cadence 3
    d = c.decide(0.1, 0.1, 0.1, 2000.0, False)                # max age reached on a frame that would be skipped
    assert d["mode"] == "forced_refresh" and d["run"] and d["model"] == "n" and "max_age" in d["reasons"]
    d = c.decide(0.1, 0.1, 0.99, 0.0, False)                  # one critical slot while the scene is easy
    assert d["run"] and "critical_slot_risk" in d["reasons"]
    d = c.decide(0.1, 0.8, 0.1, 0.0, False)                   # global slot risk escalates even with low D
    assert d["mode"] == "enhanced" and "global_slot_risk" in d["reasons"]
    assert ModeController(BASE).decide(nan, nan, nan, nan, True)["mode"] == "standard"   # missing D: no reduced


def test_tracker_components():
    tr = SlotTracker(3)
    tr.observe([0, 1], np.array([0.9, 0.2]), 0.2, np.array([0.0, 0.5]), 0.0)
    tr.observe([0], np.array([0.3]), 0.2, np.array([0.1]), 300.0)
    c = tr.components([0, 1, 2], 600.0, 1800.0)
    assert c[0].tolist() == pytest.approx([1 - 0.1 / 0.8, 0.6, 300 / 1800, 0.1])
    assert np.isnan(c[1, 1]) and c[1, 0] == 1.0 and np.isnan(c[2]).all()
    assert tr.serve([0, 1, 2]).tolist() == [1, 1, -1]


def seq_one_slot(gt, sn, ss, dt=300.0):
    T = len(gt)
    det = {"confidence": 0.2, "density": 0.5, "occlusion_raw": 0.0}
    return {"sec": np.arange(T) * dt, "n_cols": 1, "cols": [np.array([0])] * T, "valid": [np.array([True])] * T,
            "gt": [np.array([g]) for g in gt], "B": np.zeros(T), "M": np.zeros(T), "fixable": np.array(
                [int((a >= 0.5) != g and (b >= 0.5) == g) for g, a, b in zip(gt, sn, ss)]),
            "out": {"n": {"score": [np.array([x]) for x in sn], "detf": [det] * T, "occ": [np.zeros(1)] * T},
                    "s": {"score": [np.array([x]) for x in ss], "detf": [det] * T, "occ": [np.zeros(1)] * T}}}


CTX = {"thr": {"n": 0.5, "s": 0.5}, "wD": np.full(6, 1 / 6), "wR": np.full(4, 0.25), "agg": {"method": "max"},
       "age_ref_s": 1800.0, "inst_frames": 3, "occ_scale": 1.0, "light": "n", "heavy": "s"}


def test_skipped_frames_show_last_state_stale_never_vacant():
    seq = seq_one_slot([1, 0, 0, 0], [0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1])
    o = rp.run_sequence(seq, "periodic_n", {"cadence": 3}, CTX)
    assert o["model"] == ["n", None, None, "n"]
    assert [int(p[0]) for p in o["pred"]] == [1, 1, 1, 0]        # the car left at t=1, skipped frames still show it
    assert [bool(s[0]) for s in o["stale"]] == [False, True, True, False]
    assert [float(a[0]) for a in o["age_s"]] == [0.0, 300.0, 600.0, 0.0]
    m = rp.evaluate([seq], [o], {"n": 1.0, "s": 2.5}, max_age_s=450)
    assert (m["TP"], m["FO"], m["TN"]) == (1, 2, 1) and m["cost_per_frame"] == 0.5 and m["max_age_violations"] == 1
    assert m["stale_share"] == 0.5 and m["skipped_share"] == 0.5


def test_fixed_and_oracle_strategies():
    seq = seq_one_slot([1, 1, 0], [0.9, 0.3, 0.1], [0.9, 0.8, 0.1])   # n misses frame 1, s gets it
    cost = {"n": 1.0, "s": 2.5}
    mn = rp.evaluate([seq], [rp.run_sequence(seq, "fixed_n", {}, CTX)], cost)
    ms = rp.evaluate([seq], [rp.run_sequence(seq, "fixed_s", {}, CTX)], cost)
    mo = rp.evaluate([seq], [rp.run_sequence(seq, "oracle", {}, CTX)], cost)
    assert mn["FV"] == 1 and ms["FV"] == 0 and mo["FV"] == 0
    assert mn["cost_per_frame"] == 1.0 and ms["cost_per_frame"] == 2.5 and mo["cost_per_frame"] == pytest.approx(4.5 / 3, abs=1e-4)
    cfg = dict(BASE, d_hi=2.0, r_hi=2.0, d_lo=None, r_lo=None, r_crit=None)   # never escalates: behaves like fixed n
    ma = rp.evaluate([seq], [rp.run_sequence(seq, "adaptive", cfg, CTX)], cost)
    assert ma["FV"] == 1 and ma["cost_per_frame"] == 1.0
    with pytest.raises(ValueError):
        rp.run_sequence(seq, "random", {}, CTX)
