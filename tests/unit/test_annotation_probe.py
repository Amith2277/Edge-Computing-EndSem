import json
import lzma
import tarfile

from src.data.annotation_probe import (basename_index, check_listed_paths, dir_summary,
                                       first_file_per_dir, probe_file)


def test_probe_xz_json(tmp_path):
    p = tmp_path / "a.json.xz"
    with lzma.open(str(p), "wb") as f:
        f.write(json.dumps({"images": [{"id": 1, "file": "x.jpg"}], "annotations": []}).encode())
    r = probe_file(p)
    assert r["container"] == "xz" and "images=list[1]" in r["json_shape"] and not r["error"]


def test_probe_tar_xz(tmp_path):
    src = tmp_path / "x.txt"
    src.write_text("1 2 3 4\n")
    p = tmp_path / "t.tar.xz"
    with tarfile.open(str(p), "w:xz") as tf:
        tf.add(str(src), arcname="d/x.txt")
    r = probe_file(p)
    assert r["container"] == "tar" and r["members_total"] == 1 and "1 2 3 4" in r["head"]


def test_probe_bad_file_reports_error(tmp_path):
    p = tmp_path / "bad.xz"
    p.write_bytes(b"not xz")
    assert probe_file(p)["error"]


def test_dir_summary_and_picks(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "1.xz").write_bytes(b"x")
    (tmp_path / "a" / "2.xz").write_bytes(b"x")
    rows = dir_summary(tmp_path)
    assert rows[0]["files"] == 2 and rows[0]["extensions"] == {".xz": 2}
    assert first_file_per_dir(tmp_path)[0].name == "1.xz"


def test_check_listed_paths_uses_basename(tmp_path):
    (tmp_path / "cam1").mkdir()
    (tmp_path / "cam1" / "f1.jpg").write_bytes(b"x")
    lst = tmp_path / "removed.txt"
    lst.write_text("RAINY/2016/cam1/f1.jpg\nRAINY/2016/cam1/gone.jpg\n")
    r = check_listed_paths(lst, basename_index(tmp_path))
    assert r["entries"] == 2 and r["basename_found_on_disk"] == 1
    assert r["not_found_sample"] == ["RAINY/2016/cam1/gone.jpg"]
