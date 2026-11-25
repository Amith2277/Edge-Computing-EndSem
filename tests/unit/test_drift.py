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
