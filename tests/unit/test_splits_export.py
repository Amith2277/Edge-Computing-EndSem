import pytest
from PIL import Image

from src.data import splits as sp
from src.data import yolo_export as ye


def fake_images():
    """2 sites: CNR (2 cameras, 10 shared days) and PUCPR (10 days); 4 frames/day; climate alternates by day."""
    imgs, i = [], 0
    for cam, site_days in (("CNRPark-EXT/camera1", "2015-11-{:02d}"), ("CNRPark-EXT/camera2", "2015-11-{:02d}"),
                           ("PKLot/PUCPR", "2012-10-{:02d}")):
        for d in range(1, 11):
            day = site_days.format(d)
            for f in range(4):
                imgs.append({"image_id": i, "file_name": "{}/{}/{}.jpg".format(cam, day, f), "camera": cam,
                             "day": day, "timestamp": "{}T10:{:02d}:00".format(day, f * 5),
                             "climate": "sunny" if d % 2 else "rainy", "width": 100, "height": 80, "rect": None})
                i += 1
    return imgs


def test_site_of_and_normalize():
    assert sp.site_of("CNRPark-EXT/camera7") == "CNRPark-EXT"
    assert sp.site_of("PKLot/UFPR04") == "UFPR04"
    assert sp.normalize_day("2012-9-5 / non-working day") == "2012-09-05"
    assert sp.normalize_day("garbage") is None


def test_parse_day_lists():
    flags = sp.parse_day_lists({
        "non_working_days/PKLot/PUCPR/days_ignore": ["2012-09-15 / Saturday"],
        "non_representative_days/PKLot/PUCPR/non_representative_days": ["2012-9-15 / non-working day"],
        "non_representative_days/CNRPark-EXT/camera1/non_representative_days": ["2015-11-21 / non-working day"],
        "non_representative_days/PLds/ISSHK/non_representative_days": ["2016-02-18 / day with low occupance"]})
    assert flags[("PUCPR", "2012-09-15")] == "non-working day; saturday"
    assert ("CNRPark-EXT", "2015-11-21") in flags
    assert not any(k[0] == "ISSHK" for k in flags)


def test_assign_splits_no_leakage_and_site_level_days():
    imgs = fake_images()
    a, info = sp.assign_splits(imgs, seed=1)
    rows = sp.manifest_rows(imgs, a, set(a))
    assert sp.check_no_leakage(rows) == []
    # the two CNR cameras share days, so a CNR day must land in the same split for both cameras
    by_day = {}
    for r in rows:
        if r["site"] == "CNRPark-EXT":
            by_day.setdefault(r["day"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_day.values())
    assert info["climate_coverage_ok"]
    assert set(a.values()) == {"train", "val", "test"}


def test_assign_splits_is_deterministic_and_validates():
    imgs = fake_images()
    assert sp.assign_splits(imgs, seed=7)[0] == sp.assign_splits(imgs, seed=7)[0]
    with pytest.raises(ValueError):
        sp.assign_splits(imgs, ratios=(0.5, 0.5, 0.5))


def test_holdout_camera():
    imgs = fake_images()
    a, info = sp.assign_splits(imgs, holdout_cameras=["CNRPark-EXT/camera2"])
    assert {a[i["image_id"]] for i in imgs if i["camera"] == "CNRPark-EXT/camera2"} == {"holdout_test"}
    assert info["holdout_cameras"] == ["CNRPark-EXT/camera2"]


def test_subsample_only_thins_train():
    imgs = fake_images()
    a, _ = sp.assign_splits(imgs, seed=3)
    keep = sp.subsample_train(imgs, a, 0.5)
    train = [i for i in imgs if a[i["image_id"]] == "train"]
    assert sum(1 for i in train if i["image_id"] in keep) == len(train) // 2
    assert all(i["image_id"] in keep for i in imgs if a[i["image_id"]] != "train")
    rows = sp.manifest_rows(imgs, a, keep)
    assert sp.manifest_hash(rows) == sp.manifest_hash(list(reversed(rows)))
    assert sp.split_summary(rows)["train"]["in_subset"] == len(train) // 2
    with pytest.raises(ValueError):
        sp.subsample_train(imgs, a, 0)


def test_yolo_label_lines_normalize_and_clip():
    lines = ye.yolo_label_lines([[10, 20, 30, 40]], 100, 80)
    assert lines == ["0 0.250000 0.500000 0.300000 0.500000"]
    # box half outside the ROI is clipped; box fully outside is dropped
    clipped = ye.yolo_label_lines([[40, 10, 40, 20], [90, 70, 5, 5]], 100, 80, rect=[0, 0, 60, 80])
    assert clipped == ["0 0.500000 0.250000 0.200000 0.250000"]


def test_export_masks_outside_roi_and_writes_yaml(tmp_path):
    root = tmp_path / "ds"
    (root / "cam").mkdir(parents=True)
    Image.new("RGB", (100, 80), (255, 0, 0)).save(root / "cam" / "a.jpg")
    Image.new("RGB", (100, 80), (0, 0, 255)).save(root / "cam" / "b.jpg")
    images = {1: {"width": 100, "height": 80, "rect": [0, 20, 100, 60]},
              2: {"width": 100, "height": 80, "rect": [0, 0, 100, 80]}}
    rows = [{"image_id": 1, "file_name": "cam/a.jpg", "split": "train", "in_subset": True},
            {"image_id": 2, "file_name": "cam/b.jpg", "split": "val", "in_subset": True},
            {"image_id": 3, "file_name": "cam/c.jpg", "split": "train", "in_subset": False}]
    out = tmp_path / "yolo"
    c = ye.export_dataset(root, out, rows, images, {1: [[10, 30, 20, 20]]}, workers=2)
    assert c["masked"] == 1 and c["copied"] == 1 and c["boxes"] == 1 and c["empty_label_images"] == 1
    with Image.open(out / "images" / "train" / "cam__a.jpg") as im:
        assert im.getpixel((50, 5)) == (114, 114, 114)          # outside ROI -> gray
        assert im.getpixel((50, 50))[0] > 200                   # inside ROI -> original red
    assert (out / "labels" / "val" / "cam__b.txt").read_text() == ""
    again = ye.export_dataset(root, out, rows, images, {1: [[10, 30, 20, 20]]}, workers=2)
    assert again["skipped_existing"] == 2
    y = ye.write_data_yaml(out).read_text()
    assert "train: images/train" in y and "0: car" in y
