import numpy as np
import pytest

from src.temporal import filters as tf

THR = 0.5


def one_slot(seq, cfg, observed=None):
    s = np.array(seq, float)[:, None]
    o = np.ones_like(s, bool) if observed is None else np.array(observed, bool)[:, None]
    return list(tf.run_sequence(s, o, cfg, THR)[:, 0])


def test_methods_on_a_flicker():
    seq = [0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]       # one-frame dip, then a real departure
    assert one_slot(seq, {"method": "none"}) == [1, 1, 0, 1, 1, 0, 0, 0, 0]
    assert one_slot(seq, {"method": "consecutive", "k": 2}) == [1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert one_slot(seq, {"method": "window", "w": 3}) == [1, 1, 1, 1, 1, 1, 0, 0, 0]
    # EMA alpha 0.5: 0.9, 0.9, 0.5 (>= thr, stays occupied), 0.7, 0.8, 0.45 -> vacant
    assert one_slot(seq, {"method": "weighted_vote", "alpha": 0.5}) == [1, 1, 1, 1, 1, 0, 0, 0, 0]


def test_unobserved_frames_are_not_vacant_and_do_not_count():
    seq = [0.9, 0.1, 0.1, 0.1]
    obs = [1, 0, 1, 1]
    out = one_slot(seq, {"method": "consecutive", "k": 2}, obs)
    assert out == [1, -1, 1, 0]                       # the missing frame neither votes nor reads as vacant


def test_no_future_leakage():
    rng = np.random.default_rng(0)
    s = rng.random((40, 6))
    o = rng.random((40, 6)) > 0.1
    for cfg in tf.param_grid():
        full = tf.run_sequence(s, o, cfg, THR)
        s2 = s.copy(); s2[25:] = rng.random((15, 6))  # change the future only
        assert (tf.run_sequence(s2, o, cfg, THR)[:25] == full[:25]).all(), tf.label(cfg)
        # streaming one frame at a time equals the batch replay
        f = tf.TemporalFilter(6, cfg, THR)
        assert (np.stack([f.step(s[t], o[t]) for t in range(40)]) == full).all()


def test_sequences_split_on_camera_day_and_gaps():
    cams = ["A"] * 5 + ["B"] * 2
    days = ["d1", "d1", "d1", "d1", "d2", "d1", "d1"]
    sec = np.array([0, 300, 600, 3000, 90000, 0, 1800], float)   # A: 5-min spacing, a 40-min gap, next day
    r = tf.split_sequences(cams, sec, days, max_gap_factor=3)
    assert r["spacing_s"]["A"] == 300
    assert len(set(r["seq"][:3])) == 1 and r["seq"][3] != r["seq"][2] and r["seq"][4] != r["seq"][3]
    assert r["seq"][5] == r["seq"][6] and r["seq"][5] not in r["seq"][:5]


def test_transition_metrics():
    gt = np.array([[0], [0], [1], [1], [1], [1], [0], [0]], np.int8)
    pred = np.array([[0], [0], [0], [1], [1], [0], [1], [0]], np.int8)
    sec = np.arange(8) * 300.0
    m = tf.transition_metrics(gt, pred, sec, tol_frames=1)
    assert m["true_transitions"] == 2 and m["pred_transitions"] == 4
    assert m["detected"] == 2 and m["delay_frames"] == [1, 1] and m["delay_seconds"] == [300.0, 300.0]
    # false: the drop at frame 5 (truth still occupied) and the flip to occupied at frame 6 (truth just became vacant)
    assert m["false_transitions"] == 2
    tot = tf.merge_transition_metrics([m], slot_frames=8, slot_hours=8 * 300 / 3600)
    assert tot["detection_rate"] == 1.0 and tot["mean_delay_frames"] == 1.0 and tot["mean_delay_minutes"] == 5.0


def test_choose_prefers_fewer_false_transitions_within_tolerance():
    rows = [{"grid_index": 0, "macro_f1": 0.9960, "false_per_1000_slot_frames": 5.0, "mean_delay_frames": 0.0},
            {"grid_index": 1, "macro_f1": 0.9955, "false_per_1000_slot_frames": 1.0, "mean_delay_frames": 0.4},
            {"grid_index": 2, "macro_f1": 0.9900, "false_per_1000_slot_frames": 0.1, "mean_delay_frames": 1.0}]
    assert tf.choose(rows, 0.001)["grid_index"] == 1
    assert tf.choose(rows, 0.0)["grid_index"] == 0
    with pytest.raises(ValueError):
        tf.TemporalFilter(1, {"method": "median"}, THR)
