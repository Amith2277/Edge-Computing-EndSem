import json

import pytest
from PIL import Image

from src.data.audit import (find_near_duplicates, hamming, run_audit, sample_evenly,
                            summarize_annotation, tree_text)


def make_dataset(root):
    (root / "cam1" / "2024-05-01").mkdir(parents=True)
    (root / "cam2").mkdir()
    Image.new("RGB", (64, 48), (10, 20, 30)).save(root / "cam1" / "2024-05-01" / "a_101500.jpg")
    Image.new("RGB", (64, 48), (10, 20, 30)).save(root / "cam1" / "2024-05-01" / "a_101500_copy.jpg")
    Image.new("RGB", (32, 32), (200, 100, 0)).save(root / "cam2" / "b.png")
    (root / "cam2" / "broken.jpg").write_bytes(b"not an image")
    (root / "cam1" / "labels.json").write_text(json.dumps({"images": [1, 2], "annotations": [1]}))
    (root / "cam2" / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def test_sample_evenly_is_deterministic_and_spread():
    items = list(range(100))
    s = sample_evenly(items, 10)
    assert s == sample_evenly(items, 10)
    assert len(s) == 10 and s[0] == 0 and s[-1] >= 90
    assert sample_evenly(items, None) == items


def test_hamming_and_near_duplicates():
    assert hamming(0b1010, 0b1000) == 1
    pairs = find_near_duplicates([("a", 5), ("b", 5), ("c", 2 ** 60)], max_hamming=0)
    assert [(p["a"], p["b"]) for p in pairs] == [("a", "b")]


def test_summarize_annotation_json_and_txt(tmp_path):
    j = tmp_path / "x.json"
    j.write_text(json.dumps({"images": [1, 2, 3], "categories": []}))
    s = summarize_annotation(j)
    assert s["kind"] == "json:dict" and "images=list[3]" in s["detail"]
    t = tmp_path / "y.txt"
    t.write_text("0 0.1 0.2 0.3 0.4\n")
    assert "first_line_fields=5" in summarize_annotation(t)["detail"]
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    assert summarize_annotation(bad)["error"]


def test_full_audit_reports_facts(tmp_path):
    root = tmp_path / "ds"
    make_dataset(root)
    out = tmp_path / "out"
    s = run_audit(root, out)
    assert s["image_files"] == 4 and s["corrupt_or_unreadable_images"] == 1
    assert s["exact_duplicate_groups"] == 1
    assert s["near_duplicate_pairs_found"] >= 1
    assert s["annotation_like_files"] == 2
    assert s["metadata_hints"]["paths_with_date_like_token"] >= 2
    assert (out / "manifests" / "image_inventory.csv").exists()
    assert (out / "manifests" / "annotation_inventory.csv").exists()
    assert json.loads((out / "corrupt_report.json").read_text())[0]["path"].endswith("broken.jpg")
    assert "cam1" in (out / "tree.txt").read_text()


def test_smoke_mode_samples_and_warns(tmp_path):
    root = tmp_path / "ds"
    make_dataset(root)
    s = run_audit(root, tmp_path / "out", smoke=True, smoke_max_files=2)
    assert s["mode"] == "smoke" and s["images_inspected"] == 2 and s["image_files"] == 4
    assert s["smoke_warning"]


def test_missing_root_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        run_audit(tmp_path / "nope", tmp_path / "out")
    assert "ADAPTIVEPARK_DATA_ROOT" in str(e.value)


def test_tree_text(tmp_path):
    make_dataset(tmp_path / "ds")
    assert "cam2/" in tree_text(tmp_path / "ds")
