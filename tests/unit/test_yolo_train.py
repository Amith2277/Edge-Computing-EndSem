import pytest

from src.training import yolo_train as yt

CFG = {"model": "yolo11n.pt", "imgsz": 640, "epochs": 100, "batch": 32, "optimizer": "auto", "patience": 20,
       "seed": 42, "deterministic": True, "workers": 4, "cache": "ram", "save_period": 10, "amp": True,
       "subset_fraction": 0.1, "smoke": {"epochs": 1, "fraction": 0.1, "cache": False}}


def make_export(root, name):
    d = root / name
    for s in ("train", "val", "test"):
        (d / "images" / s).mkdir(parents=True)
    (d / "data.yaml").write_text("path: /kaggle/working/{}\n".format(name))
    return d


def test_find_export_dir_and_rewrite(tmp_path):
    make_export(tmp_path / "a", "yolo_f1.0")
    want = make_export(tmp_path / "b" / "c", "yolo_f0.1")
    assert yt.find_export_dir(tmp_path, 0.1) == want
    with pytest.raises(FileNotFoundError):
        yt.find_export_dir(tmp_path, 0.5)
    y = yt.write_local_data_yaml(want, tmp_path / "w" / "data.yaml").read_text()
    assert "path: {}".format(want.as_posix()) in y and "0: car" in y and "val: images/val" in y


def test_resolve_train_args_smoke_and_nulls():
    a = yt.resolve_train_args(CFG, "d.yaml", "proj", "yolo11n_f0.1")
    assert a["epochs"] == 100 and a["cache"] == "ram" and a["single_cls"] and "model" not in a
    s = yt.resolve_train_args(CFG, "d.yaml", "proj", "x", smoke=True)
    assert s["epochs"] == 1 and s["fraction"] == 0.1 and s["cache"] is False
    with pytest.raises(ValueError):
        yt.resolve_train_args(dict(CFG, epochs=None), "d.yaml", "p", "n")


def test_results_best_epoch_and_record(tmp_path):
    csvp = tmp_path / "results.csv"
    csvp.write_text("                  epoch,  metrics/mAP50(B),  metrics/mAP50-95(B)\n"
                    "1, 0.5, 0.3\n2, 0.9, 0.6\n3, 0.95, 0.55\n")
    rows = yt.read_results(csvp)
    assert rows[1]["metrics/mAP50(B)"] == 0.9
    assert yt.best_epoch(rows)["epoch"] == 2
    w = tmp_path / "best.pt"
    w.write_bytes(b"1234")
    rec = yt.build_record("yolo11n", CFG, {"epochs": 100, "patience": 20, "lr0": 0.01}, rows, w, "abc",
                          {"manifest_hash": "fbcd", "subset_fraction": 0.1}, 12.34, "AdamW",
                          {"parameters": 2_582_347}, {"status": "x"}, "r1", "h", False)
    assert rec["checkpoint_size_bytes"] == 4 and rec["dataset_split_hash"] == "fbcd"
    assert rec["early_stopping"] == {"patience": 20, "stopped_early": True}
    assert rec["lr_schedule"]["lr0"] == 0.01 and rec["lr_schedule"]["lrf"] == yt.NOT_MEASURED
    assert rec["gflops_at_imgsz"] == yt.NOT_MEASURED and rec["optimizer_used"] == "AdamW"


def test_latency_summary():
    s = yt.latency_summary([float(i) for i in range(1, 101)])
    assert s["n"] == 100 and s["p50_ms"] == 50.5 and s["max_ms"] == 100.0
    assert yt.latency_summary([])["status"] == yt.NOT_MEASURED


def test_find_export_follows_symlinks_and_reports_tree(tmp_path):
    real = make_export(tmp_path / "real", "yolo_f0.1")
    inp = tmp_path / "input" / "notebooks" / "me"
    inp.mkdir(parents=True)
    try:
        (inp / "nb03").symlink_to(real.parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted here")
    assert yt.find_export_dir(tmp_path / "input", 0.1).resolve() == real.resolve()
    with pytest.raises(FileNotFoundError) as e:
        yt.find_export_dir(tmp_path / "input", 0.5)
    assert "nb03" in str(e.value)
