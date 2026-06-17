"""Normalized JSONL dataset loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from sslight.constants import (
    VALID_SPLITS,
    normalize_split,
    q8_to_q3,
    validate_q8_labels,
    validate_sequence,
)


@dataclass(frozen=True)
class ProteinRecord:
    id: str
    sequence: str
    q8: str
    split: str

    @property
    def q3(self) -> str:
        return q8_to_q3(self.q8)

    @property
    def length(self) -> int:
        return len(self.sequence)


def _validate_record(raw: dict) -> ProteinRecord:
    required = {"id", "sequence", "q8", "split"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Record {raw.get('id', '<unknown>')} missing fields: {sorted(missing)}")

    record = ProteinRecord(
        id=str(raw["id"]),
        sequence=str(raw["sequence"]),
        q8=str(raw["q8"]),
        split=str(raw["split"]),
    )

    if record.length == 0:
        raise ValueError(f"Record {record.id} has empty sequence")

    if len(record.q8) != record.length:
        raise ValueError(
            f"Record {record.id}: sequence length ({record.length}) "
            f"!= q8 length ({len(record.q8)})"
        )

    validate_sequence(record.sequence)
    validate_q8_labels(record.q8)

    if record.split not in VALID_SPLITS:
        raise ValueError(f"Record {record.id} has invalid split: {record.split!r}")

    return record


def load_jsonl(path: str | Path) -> list[ProteinRecord]:
    """Load and validate all records from a JSONL file."""
    records: list[ProteinRecord] = []
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                records.append(_validate_record(raw))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return records


def load_jsonl_by_split(
    path: str | Path,
    split: str | None = None,
) -> list[ProteinRecord]:
    """Load records, optionally filtered by split."""
    records = load_jsonl(path)
    if split is None:
        return records
    canonical = normalize_split(split)
    return [record for record in records if record.split == canonical]


def iter_jsonl(path: str | Path) -> Iterator[ProteinRecord]:
    """Stream records from a JSONL file."""
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield _validate_record(json.loads(line))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
