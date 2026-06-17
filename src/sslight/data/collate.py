"""Batch collation with padding and masking for variable-length proteins."""

from __future__ import annotations

from typing import Any

import torch

from sslight.constants import PADDING_LABEL_INDEX


def collate_secondary_structure_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad embeddings and labels to the longest sequence in the batch."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    lengths = [int(item["length"]) for item in batch]
    max_length = max(lengths)
    embedding_dim = batch[0]["embeddings"].shape[-1]
    batch_size = len(batch)

    embeddings = torch.zeros(batch_size, max_length, embedding_dim, dtype=torch.float32)
    labels = torch.full(
        (batch_size, max_length),
        PADDING_LABEL_INDEX,
        dtype=torch.long,
    )
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for index, item in enumerate(batch):
        length = int(item["length"])
        embeddings[index, :length] = item["embeddings"]
        labels[index, :length] = item["labels"]
        mask[index, :length] = True

    return {
        "protein_ids": [item["protein_id"] for item in batch],
        "sequences": [item["sequence"] for item in batch],
        "embeddings": embeddings,
        "labels": labels,
        "mask": mask,
        "lengths": lengths,
    }
