"""Label encoding and decoding for secondary structure prediction."""

from __future__ import annotations

from sslight.constants import PADDING_LABEL_INDEX, Q3_LABELS

LABEL_MODES = frozenset({"q3", "q8"})

# Stable Q3 class order used by the model and metrics.
Q3_LABEL_TO_INDEX: dict[str, int] = {"H": 0, "E": 1, "C": 2}
Q3_INDEX_TO_LABEL: dict[int, str] = {index: label for label, index in Q3_LABEL_TO_INDEX.items()}


def encode_q3_label_string(label_string: str) -> list[int]:
    """Encode a per-residue Q3 label string to class indices."""
    unknown = {char for char in label_string if char not in Q3_LABELS}
    if unknown:
        raise ValueError(f"Invalid Q3 labels: {sorted(unknown)}")
    return [Q3_LABEL_TO_INDEX[char] for char in label_string]


def decode_q3_indices(indices: list[int] | "torch.Tensor") -> str:
    """Decode class indices to a Q3 label string."""
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    return "".join(Q3_INDEX_TO_LABEL[int(index)] for index in indices)


def encode_labels(label_string: str, label_mode: str = "q3") -> list[int]:
    """Encode labels for the requested label mode."""
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unsupported label mode {label_mode!r}. Expected one of {sorted(LABEL_MODES)}")
    if label_mode == "q3":
        return encode_q3_label_string(label_string)
    raise NotImplementedError("Q8 training mode is reserved for a future version.")


def labels_from_record(record, label_mode: str = "q3") -> list[int]:
    """Return encoded residue labels for one dataset record."""
    if label_mode == "q3":
        return encode_q3_label_string(record.q3)
    raise NotImplementedError("Q8 training mode is reserved for a future version.")


def num_classes(label_mode: str = "q3") -> int:
    """Return the number of output classes for a label mode."""
    if label_mode == "q3":
        return len(Q3_LABEL_TO_INDEX)
    raise NotImplementedError("Q8 training mode is reserved for a future version.")
