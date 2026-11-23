import json

import pytest

from src.common.config import config_hash, find_nulls, load_yaml, resolve_data_root
from src.common.env_record import NA, collect_environment, save_environment
from src.common.hashing import sha256_file
from src.common.status import require_inputs, write_status


def test_config_hash_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_find_nulls_reports_paths():
    cfg = {"x": None, "y": {"z": None, "w": 1}, "l": [1, None]}
    assert sorted(find_nulls(cfg)) == ["l[1]", "x", "y.z"]


def test_controller_config_is_untuned_and_safe():
    from pathlib import Path
    cfg = load_yaml(Path("configs/adaptive_controller.yaml"))
    assert cfg["allow_untuned"] is False
    assert cfg["freshness"]["skipped_frame_is_vacant"] is False
    assert len(find_nulls(cfg)) > 0  # nothing is silently pre-tuned


def test_resolve_data_root_priority(monkeypatch, tmp_path):
    cfg = {"dataset": {"root": str(tmp_path / "cfg")}}
    monkeypatch.delenv("ADAPTIVEPARK_DATA_ROOT", raising=False)
    assert resolve_data_root(cfg).name == "cfg"
    monkeypatch.setenv("ADAPTIVEPARK_DATA_ROOT", str(tmp_path / "env"))
    assert resolve_data_root(cfg).name == "env"
    assert resolve_data_root(cfg, str(tmp_path / "arg")).name == "arg"
    assert resolve_data_root({"dataset": {"root": None}}, None) is not None  # env still set
    monkeypatch.delenv("ADAPTIVEPARK_DATA_ROOT")
    assert resolve_data_root({"dataset": {"root": None}}) is None


def test_sha256(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_status_roundtrip_and_validation(tmp_path):
    p = tmp_path / "s" / "status.json"
    write_status(p, "nb", "run1", "RUNNING", "abc")
    assert json.loads(p.read_text())["state"] == "RUNNING"
    with pytest.raises(ValueError):
        write_status(p, "nb", "run1", "DONE")


def test_require_inputs_message(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        require_inputs([tmp_path / "missing.json"])
    assert "missing.json" in str(e.value)


def test_environment_never_guesses(tmp_path):
    env = collect_environment()
    assert "python" in env and "packages" in env
    # Not on a Jetson here: Jetson fields must say NOT AVAILABLE, not a made-up value.
    if not env["is_jetson"]:
        assert env["jetson_model"] == NA
    save_environment(tmp_path / "env.json")
    assert (tmp_path / "env.json").exists()
