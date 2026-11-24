import numpy as np
import pytest

from src.occupancy import cost_sensitive as cs
from src.temporal import filters as tf


def test_isotonic_is_monotone_and_matches_block_means():
    s = np.array([0.0, 0.0, 0.1, 0.2, 0.3, 0.9])
    y = np.array([0, 0, 1, 0, 1, 1])
    c = cs.Isotonic().fit(s, y)
    p = c.predict(np.array([0.0, 0.1, 0.2, 0.3, 0.9, 0.5, -1.0]))
    assert p.tolist() == pytest.approx([0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0])   # 0.1 and 0.2 pooled (1, 0)
    assert np.all(np.diff(c.predict(np.linspace(0, 1, 50))) >= 0)
    c2 = cs.from_params(c.params())
    assert c2.predict(s).tolist() == c.predict(s).tolist()


def test_platt_recovers_logistic_and_roundtrips():
    rng = np.random.RandomState(0)
    x = rng.rand(20000)
    y = (rng.rand(20000) < 1 / (1 + np.exp(-(8 * x - 4)))).astype(int)
    c = cs.Platt().fit(x, y)
    assert c.a == pytest.approx(8, abs=0.5) and c.b == pytest.approx(-4, abs=0.3)
    assert cs.ece(c.predict(x), y) < 0.02
    assert cs.from_params(c.params()).predict(np.array([0.5]))[0] == pytest.approx(c.predict(np.array([0.5]))[0], abs=1e-5)


def test_metrics_and_bayes_threshold():
    assert cs.bayes_threshold(c_fv=4, c_fo=1) == 0.2
    y = np.array([1, 0, 1, 0]); p = np.array([1.0, 0.0, 0.5, 0.5])
    assert cs.brier(p, y) == pytest.approx(0.125)
    assert cs.ece(p, y) == pytest.approx(0.0)              # 0.5 bin holds one of each
    assert cs.ece(np.full(4, 0.9), y) == pytest.approx(0.4)
    folds = cs.group_folds(["a", "a", "b", "c", "c"], k=2)
    assert folds[0] == folds[1] and folds[3] == folds[4]


def test_best_score_threshold_moves_down_when_false_vacant_costs_more():
    s = np.array([0.1, 0.3, 0.3, 0.6, 0.8, 0.2])
    gt = np.array([1, 1, 0, 1, 1, 0])
    even = cs.best_score_threshold(s, gt, 1, 1, grid=np.array([0.15, 0.25, 0.5, 0.7]))
    fv_heavy = cs.best_score_threshold(s, gt, 10, 1, grid=np.array([0.05, 0.15, 0.25, 0.5, 0.7]))
    assert (even["thr"], even["FV"], even["FO"]) == (0.25, 1, 1)
    assert fv_heavy["thr"] == 0.05 and fv_heavy["FV"] == 0 and fv_heavy["FO"] == 2
    sw = cs.score_threshold_sweep(s, gt, np.array([0.3]))
    assert (sw["FV"][0], sw["FO"][0]) == (1, 1)            # 0.3 counts as occupied (score >= thr)


def test_asymmetric_filter_holds_departures_not_arrivals():
    cfg = {"method": "asymmetric", "k_occ": 1, "k_vac": 2}
    sc = np.array([[0.1], [0.9], [0.1], [0.9], [0.1], [0.1]])
    out = tf.run_sequence(sc, np.ones_like(sc, bool), cfg, 0.5)[:, 0].tolist()
    assert out == [0, 1, 1, 1, 1, 0]        # arrival at once; one-frame dip held; two vacant frames flip
    gated = dict(cfg, gate=0.9)
    out = tf.run_sequence(np.array([[0.9], [0.0]]), np.ones((2, 1), bool), gated, 0.5)[:, 0].tolist()
    assert out == [1, 0]                    # a confident vacant flips at once
    assert tf.label(cfg) == "asymmetric k_occ=1 k_vac=2"
