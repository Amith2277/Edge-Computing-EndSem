import numpy as np
import pytest

from src.occupancy import association as asc
from src.occupancy import geometry as geo
from src.occupancy import schema

SQ = [0, 0, 10, 0, 10, 10, 0, 10]          # 10x10 square, area 100


def test_geometry_basics():
    pts = geo.to_points(SQ)
    assert geo.area(pts) == 100
    assert geo.rect_intersection_area(pts, 5, 5, 20, 20) == pytest.approx(25)
    assert geo.rect_intersection_area(pts, 20, 20, 30, 30) == 0
    assert geo.point_in_polygon(5, 5, pts) and not geo.point_in_polygon(15, 5, pts)
    assert geo.is_simple(pts) and not geo.is_simple(geo.to_points([0, 0, 10, 10, 10, 0, 0, 10]))  # bow-tie
    tri = geo.to_points([0, 0, 10, 0, 0, 10])
    assert geo.rect_intersection_area(tri, 0, 0, 5, 5) == pytest.approx(25)
    assert geo.roi_fraction(SQ, [0, 0, 5, 10]) == pytest.approx(0.5) and geo.roi_fraction(SQ, None) == 1.0


def test_registry_and_validation():
    images = [{"image_id": 1, "camera": "A/cam", "width": 100, "height": 50},
              {"image_id": 2, "camera": "A/cam", "width": 100, "height": 50}]
    upper = [0, 0, 10, 0, 10, 10, 0, 10]
    lower = [0, 30, 10, 30, 10, 40, 0, 40]
    spots = [{"ann_id": 1, "image_id": 1, "polygon": lower}, {"ann_id": 2, "image_id": 1, "polygon": upper},
             {"ann_id": 3, "image_id": 2, "polygon": [v + 0.2 for v in upper]}]   # same slot, rounding noise
    slots, sid = geo.build_registry(images, spots)
    assert sorted(slots) == ["A/cam#000", "A/cam#001"]
    assert sid[2] == sid[3] == "A/cam#000" and sid[1] == "A/cam#001"     # numbered top to bottom
    assert slots["A/cam#000"]["images_seen"] == 2
    v = geo.validate_slots(slots, {a: (img, s) for a, s, img in ((1, sid[1], 1), (2, sid[2], 1), (3, sid[3], 2))})
    assert v["ok"] and v["slots"] == 2
    slots["A/cam#001"]["polygon"] = [0, 30, 10, 30, 10, 400, 0, 40]   # outside the 100x50 image
    assert not geo.validate_slots(slots)["ok"]


def test_association_methods_and_batch_matches_runtime():
    # two slots side by side; one big box straddles both (mostly the left), one small box in the right slot
    sl = asc.prepare_slots([[0, 0, 10, 0, 10, 10, 0, 10], [10, 0, 20, 0, 20, 10, 10, 10]])
    dets = np.array([[0, 0, 14, 10, 0.9], [16, 2, 19, 5, 0.4]])
    p = asc.image_pairs(sl, dets)
    assert len(p) == 3
    left = p[(p["slot"] == 0) & (p["det"] == 0)][0]
    assert left["ios"] == pytest.approx(1.0) and left["iou"] == pytest.approx(100 / 140) and left["center"] == 1
    s = asc.slot_scores(2, p, "intersection_over_slot", 0.3, False)
    assert list(s) == pytest.approx([0.9, 0.9])          # big box covers 40% of the right slot
    s = asc.slot_scores(2, p, "intersection_over_slot", 0.3, True)
    assert list(s) == pytest.approx([0.9, 0.0])          # exclusive: big box only supports the left slot; small box too small
    s = asc.slot_scores(2, p, "center_in_polygon", None, False)
    assert list(s) == pytest.approx([0.9, 0.4])
    for method, t in (("box_iou", 0.1), ("intersection_over_slot", 0.05), ("center_in_polygon", None)):
        for ex in (False, True):
            cfg = {"method": method, "overlap_thr": t, "exclusive": ex, "conf_thr": 0.5}
            rt = asc.assign_image(sl, dets, cfg)
            bt = asc.slot_scores(2, p, method, t, ex)
            assert [r["score"] for r in rt] == pytest.approx(list(bt))
            assert [r["occupied"] for r in rt] == list(bt >= 0.5)
    assert asc.assign_image(sl, np.zeros((0, 5)), cfg)[0] == {"score": 0.0, "occupied": False, "support": []}


def test_metrics_sweep_and_best():
    m = asc.confusion([1, 1, 0, 0], [1, 0, 1, 0])
    assert (m["TP"], m["FV"], m["FO"], m["TN"]) == (1, 1, 1, 1) and m["macro_f1"] == 0.5
    sl = asc.prepare_slots([[0, 0, 10, 0, 10, 10, 0, 10], [10, 0, 20, 0, 20, 10, 10, 10]])
    p = asc.image_pairs(sl, np.array([[0, 0, 10, 10, 0.8], [10, 0, 20, 10, 0.3]]))
    rows = asc.sweep(2, p, np.array([1, 0]), np.array([True, True]), asc.conf_grid())
    b = asc.best(rows)
    assert b["accuracy"] == 1.0 and 0.3 < b["conf_thr"] <= 0.8 and not b["exclusive"]
    assert asc.best(rows, method="center_in_polygon")["method"] == "center_in_polygon"
    assert asc.uncertainty(0.5, 0.5) == 1.0 and asc.uncertainty(1.0, 0.5) == 0.0
    assert asc.uncertainty(0.0, 0.25) == 0.0 and asc.uncertainty(1.0, 0.25) == 0.0   # both ends certain
    assert asc.uncertainty(0.125, 0.25) == 0.5 and asc.uncertainty(0.625, 0.25) == 0.5


def test_slot_record():
    r = schema.slot_record("A#000", "A", 3, None, "occupied", 0.9, 0.2, [{"det": 0}], 3, None, 0, None, "x")
    assert tuple(r) == schema.FIELDS and r["transition_risk"] == schema.NOT_MEASURED
    with pytest.raises(ValueError):
        schema.slot_record("A#000", "A", 3, None, "vacant", 0.0, 1.0, [], 3, None, 0, None, "x", inference="skipped")
