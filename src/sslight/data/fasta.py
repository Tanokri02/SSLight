"""FASTA parsing and sequence validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sslight.constants import validate_sequence


@dataclass(frozen=True)
class FastaRecord:
    id: str
    sequence: str


def parse_fasta(text: str) -> list[FastaRecord]:
    """Parse FASTA text into validated records."""
    records: list[FastaRecord] = []
    current_id: str | None = None
    chunks: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                sequence = "".join(chunks)
                validate_sequence(sequence)
                records.append(FastaRecord(id=current_id, sequence=sequence))
            current_id = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line.upper())

    if current_id is not None:
        sequence = "".join(chunks)
        validate_sequence(sequence)
        records.append(FastaRecord(id=current_id, sequence=sequence))

    return records


def load_fasta(path: str | Path) -> list[FastaRecord]:
    """Load and validate sequences from a FASTA file."""
    return parse_fasta(Path(path).read_text())
