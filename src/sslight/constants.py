"""Shared constants for sequences, labels, and ESM defaults."""

from __future__ import annotations

STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Residues rejected during dataset filtering and sequence validation.
AMBIGUOUS_AMINO_ACIDS = frozenset("XBZUJOU*-")

Q8_LABELS = frozenset("HGBTIESCP")

Q3_LABELS = frozenset("HEC")

Q8_TO_Q3: dict[str, str] = {
    "H": "H",
    "G": "H",
    "I": "H",
    "E": "E",
    "B": "E",
    "T": "C",
    "S": "C",
    "C": "C",
    "P": "C",
}

DEFAULT_ESM_MODEL = "esm2_t6_8M_UR50D"

ESM_MODEL_REGISTRY: dict[str, str] = {
    "esm2_t6_8M_UR50D": "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D": "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D": "esm2_t30_150M_UR50D",
}

PADDING_LABEL_INDEX = -100

VALID_SPLITS = frozenset({"train", "val", "validation", "test"})

# CLI alias -> canonical split name used in JSONL and embedding directories.
SPLIT_ALIASES: dict[str, str] = {
    "val": "validation",
}


def normalize_split(split: str) -> str:
    """Return the canonical split name."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}. Expected one of {sorted(VALID_SPLITS)}")
    return SPLIT_ALIASES.get(split, split)


def q8_to_q3(q8_string: str) -> str:
    """Convert a Q8 label string to Q3."""
    return "".join(Q8_TO_Q3.get(char, "C") for char in q8_string)


def validate_sequence(sequence: str) -> None:
    """Raise ValueError if the sequence contains invalid residues."""
    invalid = {char for char in sequence if char not in STANDARD_AMINO_ACIDS}
    if invalid:
        raise ValueError(
            f"Sequence contains non-standard or ambiguous residues: {sorted(invalid)}"
        )


def validate_q8_labels(q8_string: str) -> None:
    """Raise ValueError if Q8 labels are invalid."""
    invalid = {char for char in q8_string if char not in Q8_LABELS}
    if invalid:
        raise ValueError(f"Invalid Q8 labels: {sorted(invalid)}")
