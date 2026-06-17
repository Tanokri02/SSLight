"""Masked training and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sslight.constants import PADDING_LABEL_INDEX
from sslight.labels import Q3_INDEX_TO_LABEL


@dataclass
class MetricResult:
    accuracy: float
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    num_residues: int

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "confusion_matrix": self.confusion_matrix,
            "num_residues": self.num_residues,
        }


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy over real residues only."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [batch, length, classes], got {tuple(logits.shape)}")

    batch_size, length, num_classes = logits.shape
    flat_logits = logits.reshape(batch_size * length, num_classes)
    flat_labels = labels.reshape(batch_size * length)
    flat_mask = mask.reshape(batch_size * length)

    if flat_mask.any():
        valid_logits = flat_logits[flat_mask]
        valid_labels = flat_labels[flat_mask]
    else:
        valid_logits = flat_logits
        valid_labels = flat_labels

    return torch.nn.functional.cross_entropy(
        valid_logits,
        valid_labels,
        ignore_index=PADDING_LABEL_INDEX,
    )


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    class_names: list[str] | None = None,
) -> MetricResult:
    """Compute residue-level metrics ignoring padded positions."""
    if class_names is None:
        class_names = [Q3_INDEX_TO_LABEL[index] for index in range(num_classes)]

    predictions = logits.argmax(dim=-1)
    valid = mask & (labels != PADDING_LABEL_INDEX)
    if not valid.any():
        empty = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        zeros = {name: 0.0 for name in class_names}
        return MetricResult(
            accuracy=0.0,
            macro_f1=0.0,
            per_class_precision=zeros.copy(),
            per_class_recall=zeros.copy(),
            per_class_f1=zeros.copy(),
            confusion_matrix=empty,
            num_residues=0,
        )

    y_true = labels[valid].detach().cpu()
    y_pred = predictions[valid].detach().cpu()

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion[int(true_label), int(pred_label)] += 1

    accuracy = float((y_true == y_pred).float().mean())
    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    f1_values: list[float] = []

    for class_index, class_name in enumerate(class_names):
        tp = int(confusion[class_index, class_index])
        fp = int(confusion[:, class_index].sum() - tp)
        fn = int(confusion[class_index, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_precision[class_name] = precision
        per_class_recall[class_name] = recall
        per_class_f1[class_name] = f1
        f1_values.append(f1)

    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    return MetricResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        confusion_matrix=confusion.tolist(),
        num_residues=int(valid.sum()),
    )
