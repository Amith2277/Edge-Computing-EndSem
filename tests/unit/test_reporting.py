import json

from src.reporting import final as fr


def test_get_paths_and_selectors():
    r = {"a": {"b": 1}, "rows": [{"model": "n", "x": 2}, {"model": "s", "ratio": 5, "x": 3}], "k-1": {"m": None}}
    assert fr.get(r, "a/b") == 1 and fr.get(r, "a/c") == fr.MISSING and fr.get(None, "a") == fr.MISSING
    assert fr.get(r, "rows[model=s,ratio=5]/x") == 3 and fr.get(r, "rows[model=q]/x") == fr.MISSING
    assert fr.get(r, "k-1/m") is None


def _records(tmp_path):
    def w(rel, obj):
        p = tmp_path / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj))
    w("nb03/artifacts/splits/split_info.json", {"manifest_hash": "H"})
    for v in fr.MODELS:
        w("nb04/train/{}_f0.1/training_record.json".format(v), {"run_id": "t" + v, "smoke_test": False, "dataset_split_hash": "H",
                                                                 "checkpoint_sha256": "sha" + v, "parameters": 10})
        w("nb06/eval_det/{}/eval_record.json".format(v), {"smoke_test": False, "dataset_split_hash": "H", "checkpoint_sha256": "sha" + v,
                                                          "test": {"mAP50": 0.9}})
        w("nb07/occupancy/occupancy_record_{}.json".format(v), {"run_id": "occ1", "smoke_test": False, "dataset_split_hash": "H",
                                                                "checkpoint_sha256": "sha" + v, "test": {"accuracy": 0.99}})
    w("nb07/occupancy/status.json", {"state": "COMPLETE", "run_id": "occ1"})
    w("nb10/controller/controller_record.json", {"smoke_test": False, "dataset_split_hash": "X", "notebook07_run_id": "occ1"})
    w("nb03/artifacts/splits/split_info_smoke.json", {})


def test_discover_checks_catalogue_and_status(tmp_path):
    _records(tmp_path)
    found = fr.discover(tmp_path, 0.1)
    assert found["occ:yolo11s"]["record"]["test"]["accuracy"] == 0.99 and found["occ:yolo11s"]["status"]["state"] == "COMPLETE"
    assert found["eval"]["record"] is None and found["jetson"]["record"] is None
    rows = fr.checks(found)
    res = {r["check"]: r["result"] for r in rows}
    assert res["ctl split hash = notebook 03"] == "FAIL" and res["occ:yolo11n weights = notebook 04 checkpoint"] == "PASS"
    assert res["eval present (notebook 13)"] == "FAIL" and res["jetson present (notebook Orin (plan G8))"] == "SKIP"
    R = {k: v["record"] for k, v in found.items()}
    cat = fr.catalogue_rows(R)
    acc = [r for r in cat if r["metric"] == "accuracy" and r["section"] == "18 occupancy"][0]
    assert acc["values"] == {"yolo11n": 0.99, "yolo11s": 0.99}
    assert all(r["values"]["value"] == "NOT MEASURED" for r in cat if r["section"] == "22 Jetson AGX Orin")
    assert [r for r in cat if r["metric"] == "parameters"][0]["values"]["yolo11n"] == 10
    st = fr.overall_status(found, {"FastAPI backend": False}, rows)
    assert st["state"] == "BLOCKED" and "eval" in st["missing_records"]
    md = fr.md_table([{"a": 0.123456, "b": {"x": 1}}, {"a": None, "b": "p|q"}], ["a", "b"])
    assert "0.1235" in md and "x 1" in md and "p/q" in md and md.count("\n") == 4
