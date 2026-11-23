import pytest

from src.data.discover import candidates, common_data_root, find_dataset_root


def touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_legacy_layout_strips_wrapper_folders(tmp_path):
    touch(tmp_path / "slug" / "inner" / "images" / "a.jpg")
    touch(tmp_path / "slug" / "inner" / "labels" / "a.txt")
    assert find_dataset_root(tmp_path) == tmp_path / "slug" / "inner"


def test_datasets_owner_slug_layout(tmp_path):
    touch(tmp_path / "datasets" / "owner" / "myset" / "train" / "a.jpg")
    touch(tmp_path / "datasets" / "owner" / "myset" / "val" / "b.jpg")
    assert find_dataset_root(tmp_path) == tmp_path / "datasets" / "owner" / "myset"


def test_ambiguity_requires_hint(tmp_path):
    touch(tmp_path / "datasets" / "o" / "parking-set" / "a.jpg")
    touch(tmp_path / "datasets" / "o" / "other" / "b.jpg")
    # Both live under the same owner folder -> one merged candidate; two top-level slugs -> ambiguous.
    touch(tmp_path / "second" / "c.jpg")
    with pytest.raises(RuntimeError):
        find_dataset_root(tmp_path)
    assert find_dataset_root(tmp_path, hint="second") == tmp_path / "second"


def test_missing_and_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_dataset_root(tmp_path)
    (tmp_path / "empty_dir").mkdir()
    assert candidates(tmp_path) == []
    assert common_data_root(tmp_path / "empty_dir") is None
