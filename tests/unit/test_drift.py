import numpy as np
import pytest

from src.drift import monitor as dm
from src.controller import replay as rp


def base_of(rng, n=2000, loc=0.5):
    return {s: dm.baseline_quantiles(rng.normal(loc, 0.1, n)) for s in dm.SIGNALS}


def test_psi_and_ks_zero_on_same_large_on_shift():
    rng = np.random.RandomState(0)
    q = dm.baseline_quantiles(rng.normal(0, 1, 5000))
    same, shifted = rng.normal(0, 1, 400), rng.normal(2, 1, 400)
    assert dm.psi(q, same) < 0.05 and dm.psi(q, shifted) > 1.0
    assert dm.ks(q, same) < 0.1 and dm.ks(q, shifted) > 0.6
    assert np.isnan(dm.psi(q, np.array([np.nan])))
    tied = dm.baseline_quantiles(np.r_[np.zeros(900), rng.rand(100)])   # 90% zeros: tied deciles merge
    assert dm.psi(tied, np.r_[np.zeros(45), rng.rand(5)]) < 0.05 and dm.psi(tied, rng.rand(50)) > 1.0


def test_calibrate_and_monitor_raise_explained_alert():
    rng = np.random.RandomState(1)
    base = base_of(rng)
    clean = [dm.window_scores("ks", base, {s: rng.normal(0.5, 0.1, 12) for s in dm.SIGNALS}) for _ in range(600)]
    thr = dm.calibrate(clean)
    fa = np.mean([dm.severity(w, thr) != "ok" for w in clean])
    assert 0.02 <= fa <= 0.08                                         # ~5% by construction
    mon = dm.DriftMonitor("cam", base, thr, "ks", window=12)
    for t in range(12):
        r = mon.update({s: rng.normal(0.5, 0.1) for s in dm.SIGNALS}, frame=t)
    for t in range(12, 30):
        vals = {s: rng.normal(0.5, 0.1) for s in dm.SIGNALS}; vals["brightness"] = 0.95
        r = mon.update(vals, frame=t)
    assert r["severity"] in ("warning", "critical") and mon.history
    ev = mon.history[0]
    assert "darkness" in ev["explanation"] and "higher" in ev["explanation"] and ev["acknowledged"] is False
    assert "No model is retrained" in ev["explanation"]
    assert dm.DriftMonitor("c", base, thr, "ks", window=12).update({})["severity"] == "ok"   # window not full


def test_replay_records_features():
    nan = float("nan")
    det = {"confidence": 0.2, "density": 0.5, "occlusion_raw": 0.0}
    seq = {"sec": np.arange(3) * 300.0, "n_cols": 1, "cols": [np.array([0])] * 3, "valid": [np.array([True])] * 3,
           "gt": [np.array([1])] * 3, "B": np.array([0.1, 0.2, 0.3]), "M": np.array([nan, 0.0, 0.0]), "fixable": np.zeros(3),
           "out": {"n": {"score": [np.array([0.9])] * 3, "detf": [det] * 3, "occ": [np.zeros(1)] * 3}}}
    ctx = {"thr": {"n": 0.5}, "wD": np.full(6, 1 / 6), "wR": np.full(4, 0.25), "agg": {"method": "max"},
           "age_ref_s": 1800.0, "inst_frames": 3, "occ_scale": 1.0, "light": "n", "heavy": "s"}
    o = rp.run_sequence(seq, "fixed_n", {}, ctx)
    assert o["F"].shape == (3, 6) and o["F"][:, 0].tolist() == [0.1, 0.2, 0.3]
    assert np.isnan(o["F"][0, 2]) and o["F"][1, 2] == pytest.approx(0.2)    # C from the previous frame only


def test_perturbations_change_the_image_as_intended():
    from PIL import Image
    from src.drift import perturb as pt
    rng = np.random.RandomState(0)
    im = Image.fromarray((rng.rand(40, 60, 3) * 200 + 20).astype(np.uint8))
    a = np.asarray(im, float)
    names = [v[0] for v in pt.variants()]
    assert names[0] == "none" and len(names) == 13
    out = {n: np.asarray(pt.apply(im, k, x, seed=1), float) for n, k, x in pt.variants()}
    assert (out["none"] == a).all()
    assert out["dark_strong"].mean() < out["dark_mild"].mean() < a.mean()
    assert out["blur_strong"].std() < a.std() and out["haze_strong"].std() < a.std()
    assert (out["shift_strong"][:3, :3] == 0).all() and (out["shift_strong"][10, 10] == a[7, 7]).all()   # 0.05*60 = 3 px
    assert all(out[n].shape == a.shape for n in names)


def test_tie_safe_thresholds_hold_the_clean_rate():
    # 30% of clean windows sit at the maximum score (saturated KS / PSI): a plain quantile put the threshold there
    # and every one of them warned. The tie-safe rule must disable the signal instead of exceeding the rate.
    x = np.r_[np.full(300, 1.0), np.linspace(0, 0.9, 700)]
    assert dm.upper_threshold(x, 0.05) > 1.0 and np.mean(x >= dm.upper_threshold(x, 0.05)) == 0
    y = np.r_[np.full(30, 1.0), np.linspace(0, 0.9, 970)]          # 3% at the maximum: threshold can be met
    t = dm.upper_threshold(y, 0.05)
    assert np.mean(y >= t) <= 0.05 and t < 1.0
    rng = np.random.RandomState(2)
    clean = []
    for _ in range(800):
        w = {s: float(rng.rand()) for s in dm.SIGNALS}
        w["brightness"] = 1.0 if rng.rand() < 0.2 else float(rng.rand() * 0.5)
        clean.append(w)
    thr = dm.calibrate(clean, 0.05, 0.01)
    assert thr["saturated"] == ["brightness"] and thr["clean_rate"]["ok"] >= 0.94
    assert thr["clean_rate"]["critical"] <= 0.011 and thr["critical_overall"] > 1.0
    assert dm.severity(dict(clean[0], brightness=1.0), thr) == "ok"       # saturated signal cannot alert


def test_median_shift_and_misalignment():
    rng = np.random.RandomState(3)
    q = dm.baseline_quantiles(rng.normal(0, 1, 5000))
    assert dm.median_shift(q, rng.normal(0, 1, 8)) < 1.0 < dm.median_shift(q, rng.normal(3, 1, 8))
    assert dm.median_shift(q, rng.normal(6, 1, 8)) > dm.median_shift(q, rng.normal(3, 1, 8))   # does not saturate
    from src.occupancy import association as asc
    slots = asc.prepare_slots([[0, 0, 10, 0, 10, 10, 0, 10], [20, 0, 30, 0, 30, 10, 20, 10]])
    on = np.array([[0, 0, 10, 10, 0.9], [20, 0, 30, 10, 0.9]], float)
    off = on + np.array([4, 4, 4, 4, 0.0])
    assert dm.slot_misalignment(asc.image_pairs(slots, on), 0.5) == pytest.approx(0.0)
    assert dm.slot_misalignment(asc.image_pairs(slots, off), 0.5) > 0.5
    assert np.isnan(dm.slot_misalignment(asc.image_pairs(slots, on), 0.95))       # nothing accepted
