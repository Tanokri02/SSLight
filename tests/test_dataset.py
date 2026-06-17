import json
from pathlib import Path

import pytest

from sslight.data.dataset import load_jsonl


def test_load_project_dataset():
    path = Path("data/dataset.jsonl")
    if not path.exists():
        pytest.skip("dataset not present")

    records = load_jsonl(path)
    assert len(records) > 0
    splits = {record.split for record in records}
    assert splits <= {"train", "val", "test"}


def test_sequence_label_length_match(tmp_path: Path):
    record = {
        "id": "demo",
        "sequence": "ACDE",
        "q8": "CCCC",
        "split": "train",
    }
    path = tmp_path / "demo.jsonl"
    path.write_text(json.dumps(record) + "\n")
    loaded = load_jsonl(path)[0]
    assert loaded.length == len(loaded.q8) == 4


def test_reject_ambiguous_sequence(tmp_path: Path):
    record = {
        "id": "bad",
        "sequence": "ACX",
        "q8": "CCC",
        "split": "train",
    }
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        load_jsonl(path)
