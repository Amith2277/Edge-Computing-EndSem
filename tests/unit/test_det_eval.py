import numpy as np
import pytest

from src.evaluation import det_eval as de

ROWS = [
    {"file_name": "PKLot/PUCPR/Sunny/d1/a.jpg", "camera": "PKLot/PUCPR", "site": "PUCPR", "day": "d1",
     "climate": "sunny", "split": "test", "day_flag": ""},
    {"file_name": "PKLot/PUCPR/Rainy/d2/b.jpg", "camera": "PKLot/PUCPR", "site": "PUCPR", "day": "d2",
     "climate": "rainy", "split": "test", "day_flag": "non-working"},
    {"file_name": "CNR/camera1/d3/c.jpg", "camera": "CNRPark-EXT/camera1", "site": "CNRPark-EXT", "day": "d3",
     "climate": "overcast", "split": "test", "day_flag": ""},
    {"file_name": "CNR/camera1/d3/missing.jpg", "camera": "CNRPark-EXT/camera1", "site": "CNRPark-EXT", "day": "d3",
     "climate": "overcast", "split": "test", "day_flag": ""},
    {"file_name": "PKLot/PUCPR/Sunny/d0/t.jpg", "camera": "PKLot/PUCPR", "site": "PUCPR", "day": "d0",
     "climate": "sunny", "split": "train", "day_flag": ""},
]
AVAIL = ["PKLot__PUCPR__Sunny__d1__a.jpg", "PKLot__PUCPR__Rainy__d2__b.jpg", "CNR__camera1__d3__c.jpg",
         "PKLot__PUCPR__Sunny__d0__t.jpg"]


def test_groups_from_manifest():
    g = de.test_groups(ROWS, AVAIL)
    assert len(g["all"]["images"]) == 3 and g["all"]["days"] == 3   # train row and unexported image excluded
    assert g["site:PUCPR"]["images"] == ["PKLot__PUCPR__Rainy__d2__b.jpg", "PKLot__PUCPR__Sunny__d1__a.jpg"]
    assert g["day_type:flagged"]["images"] == ["PKLot__PUCPR__Rainy__d2__b.jpg"]
    assert g["camera:CNRPark-EXT/camera1"]["days"] == 1 and "climate:overcast" in g


def test_build_group_dataset_and_count(tmp_path):
    src = tmp_path / "local"
    (src / "images" / "test").mkdir(parents=True)
    (src / "labels" / "test").mkdir(parents=True)
    for n in AVAIL[:3]:
        (src / "images" / "test" / n).write_bytes(b"x")
    (src / "labels" / "test" / "CNR__camera1__d3__c.txt").write_text("0 .5 .5 .1 .1\n0 .2 .2 .1 .1\n")
    y = de.build_group_dataset(src, tmp_path / "g" / "x", AVAIL[1:3])
    assert "test: images/test" in y.read_text()
    assert sorted(p.name for p in (tmp_path / "g" / "x" / "images" / "test").iterdir()) == sorted(AVAIL[1:3])
    de.build_group_dataset(src, tmp_path / "g" / "x", AVAIL[1:3])   # second call is a no-op
    assert de.count_boxes(src / "labels" / "test", AVAIL[:3]) == 2


def test_conf_helpers():
    px = np.linspace(0, 1, 11)
    p = np.array([px])               # precision rises with conf
    r = np.array([1 - px])           # recall falls
    f1 = 2 * p * r / (p + r + 1e-16)
    assert de.best_f1_conf(px, f1) == {"conf": 0.5, "f1": 0.5}
    m = de.metrics_at_conf(px, p, r, 0.25)
    assert m["precision"] == 0.25 and m["recall"] == 0.75 and m["f1"] == pytest.approx(0.375)
    assert de.best_f1_conf(px, np.zeros((1, 0)))["conf"] == de.NOT_MEASURED


def test_box_summary_and_sampling():
    class Box:
        mp, mr, map50, map = 0.9, 0.8, 0.95, 0.7
        px = np.linspace(0, 1, 11)
        p_curve = np.array([px]); r_curve = np.array([1 - px])
        f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-16)
    s = de.box_summary(Box(), conf=0.5)
    assert s["mAP50-95"] == 0.7 and s["best_f1_on_this_set"]["conf"] == 0.5
    assert s["at_operating_conf"]["recall"] == 0.5
    assert de.stride_sample(list("abcdefghij"), 5) == list("acegi") and de.stride_sample(["b", "a"], None) == ["a", "b"]
    assert de.small_group_note(50, 2) == "fewer than 100 images; fewer than 3 days" and de.small_group_note(500, 5) == ""
