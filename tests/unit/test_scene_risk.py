import numpy as np
import pytest
from PIL import Image

from src.occupancy import association as asc
from src.scene_analysis import features as sf
from src.slot_risk import risk as rk


def test_image_features():
    dark = Image.new("RGB", (200, 100), (0, 0, 0)); light = Image.new("RGB", (200, 100), (255, 255, 255))
    gd, gl = sf.to_gray_small(dark, (0, 0, 200, 100), 50), sf.to_gray_small(light, (0, 0, 200, 100), 50)
    assert gd.shape == (25, 50)
    assert sf.brightness(gd) == 1.0 and sf.brightness(gl) == 0.0
    assert sf.motion_raw(gl, gd) == 1.0 and sf.motion_raw(gl, None) is None


def test_detection_features_and_instability():
    d = np.array([[0, 0, 10, 10, 0.9], [5, 0, 15, 10, 0.7], [50, 50, 60, 60, 0.1]])
    f = sf.detection_features(d, 0.5, n_slots=4)
    assert f["confidence"] == pytest.approx(0.2) and f["density"] == 0.5
    assert f["occlusion_raw"] == pytest.approx(50 / 150)               # two boxes overlap with IoU 1/3
    e = sf.detection_features(np.zeros((0, 5)), 0.5, 4)
    assert e["confidence"] is None and e["density"] == 0.0              # empty lot: confidence missing, not 0
    assert sf.detection_features(None, 0.5, 4)["density"] is None
    st = [np.array([1, 0, -1, 1]), np.array([1, 1, 0, 1]), np.array([1, 1, 1, 0])]
    assert sf.instability(st) == pytest.approx(3 / 4)                   # slots 1, 2, 3 changed; slot 2 seen twice
    assert sf.instability(st[:1]) is None


def test_weighted_score_renormalises_missing():
    F = np.array([[0.2, 0.8], [0.4, np.nan], [np.nan, np.nan]])
    s, den = sf.weighted_score(F, np.array([0.5, 0.5]))
    assert s[0] == pytest.approx(0.5) and s[1] == pytest.approx(0.4) and np.isnan(s[2])
    S, _ = sf.weighted_score(F, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert S.shape == (3, 2) and S[0, 1] == pytest.approx(0.8) and np.isnan(S[1, 1])


def test_simplex_grid_and_capture():
    g = sf.simplex_grid(3, 0.5)
    assert len(g) == 6 and np.allclose(g.sum(1), 1) and (g >= 0).all()
    assert len(sf.simplex_grid(6, 0.1)) == 3003
    t = np.array([0, 0, 0, 5])
    assert sf.capture_auc(np.array([0.1, 0.2, 0.3, 0.9]), t) == pytest.approx(1 - 0.5 / 4)
    assert sf.capture_auc(np.array([1, 1, 1, 1.0]), t) == pytest.approx(0.5)   # all tied = random
    assert sf.capture_at(np.array([0.1, 0.2, 0.3, 0.9]), t, (0.25,)) == {"top25%": 1.0}
    assert sf.spearman(np.arange(5.0), np.arange(5.0) ** 2) == pytest.approx(1.0)


def test_slot_risk_components_and_aggregation():
    assert rk.uncertainty(np.array([0.2, 0.0, 1.0, 0.6]), 0.2).tolist() == pytest.approx([1.0, 0.0, 0.0, 0.5])
    assert rk.age_component(np.array([300, 3600]), 1800).tolist() == pytest.approx([1 / 6, 1.0])
    sl = asc.prepare_slots([[0, 0, 10, 0, 10, 10, 0, 10], [10, 0, 20, 0, 20, 10, 10, 10]])
    p = asc.image_pairs(sl, np.array([[0, 0, 14, 10, 0.9]]))         # one car in slot 0, spilling 40% into slot 1
    o = rk.slot_occlusion(2, p, 0.5, 0.2)
    assert o.tolist() == pytest.approx([0.0, 0.4])
    R = np.array([0.1, 0.9, 0.5, np.nan])
    assert rk.aggregate(R, "max") == 0.9 and rk.aggregate(R, "topk_mean", 2) == pytest.approx(0.7)
    assert np.isnan(rk.aggregate(np.array([np.nan]), "max"))
    assert rk.roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert rk.roc_auc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
    with pytest.raises(ValueError):
        rk.aggregate(R, "median")
