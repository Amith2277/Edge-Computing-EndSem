import io
import json
import tarfile

import pytest
from PIL import Image

from src.data import metapklot as mp

CLIMATES = [{"id": 2, "name": "Sunny"}, {"id": 3, "name": "Rainy"}]


def _img(i, name, t, w=100, h=80):
    return {"id": i, "file_name": name, "width": w, "height": h, "annotationsRectangle": [0, 0, 60, 80],
            "date": [2012, 9, 11], "time": t, "climate": 2}


def make_root(tmp_path):
    root = tmp_path / "ds"
    names = ["PKLot/PUCPR/Sunny/2012-09-11/a.jpg", "PKLot/PUCPR/Sunny/2012-09-11/b.jpg"]
    for n in names:
        (root / n).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 80)).save(root / n)
    images = [_img(1, names[0], [10, 0, 0]), _img(2, names[1], [10, 5, 0])]
    sq = [10, 10, 30, 10, 30, 30, 10, 30]
    sq2 = [40, 10, 55, 10, 55, 30, 40, 30]
    cars = {"categories": [{"id": 1, "name": "car", "supercategory": "vehicle"}], "climates": CLIMATES,
            "images": images,
            "annotations": [{"id": 0, "image_id": 1, "car_id": 7, "category_id": 1, "bbox": [10, 10, 20, 20],
                             "segmentation": [sq]},
                            {"id": 1, "image_id": 2, "car_id": 7, "category_id": 1, "bbox": [10, 10, 20, 20],
                             "segmentation": [sq]},
                            {"id": 2, "image_id": 2, "car_id": 8, "category_id": 1, "bbox": [70, 10, 20, 20],
                             "segmentation": [[70, 10, 90, 10, 90, 30]]}]}
    spots = {"categories": [{"id": 0, "name": "empty"}, {"id": 1, "name": "occupied"}], "climates": CLIMATES,
             "images": images,
             "annotations": [
                 {"id": 0, "image_id": 1, "car_id": 7, "category_id": 1, "segmentation": [sq], "bbox": [10, 10, 20, 20],
                  "best_rotated_rect": [[20, 20], [20, 20], 0]},
                 {"id": 1, "image_id": 1, "car_id": -1, "category_id": 0, "segmentation": [sq2], "bbox": [40, 10, 15, 20],
                  "best_rotated_rect": [[47, 20], [15, 20], 0]},
                 {"id": 2, "image_id": 2, "car_id": 7, "category_id": 1, "segmentation": [sq], "bbox": [10, 10, 20, 20],
                  "best_rotated_rect": [[20, 20], [20, 20], 0]},
                 {"id": 3, "image_id": 2, "car_id": 99, "category_id": 1, "segmentation": [sq2], "bbox": [40, 10, 15, 20],
                  "best_rotated_rect": [[47, 20], [15, 20], 0]}]}
    for kind, data in (("cars", cars), ("spots", spots)):
        d = root / "annotations" / "original" / kind / "PKLot"
        d.mkdir(parents=True)
        raw = json.dumps(data).encode()
        with tarfile.open(str(d / "pucpr_{}.tar.xz".format(kind)), "w:xz") as tf:
            info = tarfile.TarInfo("pucpr_{}.json".format(kind))
            info.size = len(raw)
            tf.addfile(info, io.BytesIO(raw))
    (root / "non_working_days" / "PKLot" / "PUCPR").mkdir(parents=True)
    (root / "non_working_days" / "PKLot" / "PUCPR" / "days_ignore").write_text("2012-09-12\n2012-09-13\n")
    return root


def test_load_and_parse(tmp_path):
    root = make_root(tmp_path)
    s = mp.load_set(root, "original", "spots")
    assert len(s["images"]) == 2 and len(s["annotations"]) == 4
    img = s["images"][0]
    assert img["camera"] == "PKLot/PUCPR" and img["climate"] == "sunny" and img["day"] == "2012-09-11"
    assert s["annotations"][0]["category"] == "occupied"
    assert "best_rotated_rect" in s["schema"][0]["annotation_keys"]


def test_bad_variant_rejected(tmp_path):
    with pytest.raises(ValueError):
        mp.annotation_files(tmp_path, "weird", "spots")


def test_validators(tmp_path):
    root = make_root(tmp_path)
    cars = mp.load_set(root, "original", "cars")
    spots = mp.load_set(root, "original", "spots")
    vc = mp.validate_cars(cars["images"], cars["annotations"])
    assert vc["car_annotations"] == 3 and vc["distinct_car_ids"] == 2
    assert vc["center_outside_annotation_rectangle"] == 1      # car 8 at x=80 is outside rect width 60
    vs = mp.validate_spots(spots["images"], spots["annotations"], cars["annotations"])
    assert vs["category_counts"] == {"occupied": 3, "empty": 1}
    assert vs["occupied_vs_car_id_inconsistent"] == 0
    assert vs["occupied_spots_linked_to_car_annotation"] == 2
    assert vs["occupied_spots_without_car_annotation"] == 1    # car_id 99 has no car annotation


def test_disk_check_slot_identity_and_time(tmp_path):
    root = make_root(tmp_path)
    spots = mp.load_set(root, "original", "spots")
    d = mp.check_images_on_disk(root, spots["images"])
    assert d["missing_on_disk"] == 0 and d["size_mismatch"] == 0
    ident = mp.slot_identity(spots["images"], spots["annotations"])
    assert ident["PKLot/PUCPR"]["distinct_polygons"] == 2 and ident["PKLot/PUCPR"]["polygons_seen_once"] == 0
    t = mp.temporal_structure(spots["images"])
    assert t["cameras"]["PKLot/PUCPR"]["gap_min_p10_p50_p90"][1] == 5.0
    assert mp.read_day_lists(root)["non_working_days/PKLot/PUCPR/days_ignore"] == ["2012-09-12", "2012-09-13"]


def test_polygon_area():
    assert mp.polygon_area([0, 0, 2, 0, 2, 2, 0, 2]) == 4.0
    assert mp.polygon_area([0, 0, 1, 1]) == 0.0
