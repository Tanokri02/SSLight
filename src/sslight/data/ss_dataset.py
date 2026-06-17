"""Dataset combining JSONL records with precomputed ESM embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from sslight.constants import normalize_split
from sslight.data.dataset import ProteinRecord, load_jsonl_by_split
from sslight.embeddings.storage import find_embedding_path, load_embedding
from sslight.labels import labels_from_record


@dataclass(frozen=True)
class SecondaryStructureExample:
    protein_id: str
    sequence: str
    embeddings: torch.Tensor
    labels: torch.Tensor
    length: int


class SecondaryStructureDataset(Dataset):
    """PyTorch dataset over JSONL records and saved ESM embeddings."""

    def __init__(
        self,
        data_path: str | Path,
        embeddings_dir: str | Path,
        split: str,
        label_mode: str = "q3",
    ) -> None:
        self.data_path = Path(data_path)
        self.embeddings_dir = Path(embeddings_dir)
        self.split = normalize_split(split)
        self.label_mode = label_mode
        self.records = load_jsonl_by_split(self.data_path, self.split)

        if not self.records:
            raise ValueError(f"No records found for split {split!r} in {self.data_path}")

        self._missing: list[str] = []
        self._examples: list[SecondaryStructureExample] = []
        for record in self.records:
            example = self._load_example(record)
            if example is not None:
                self._examples.append(example)

        if not self._examples:
            raise RuntimeError(
                f"No usable examples for split {split!r}. Missing embeddings: {len(self._missing)}"
            )

        if self._missing:
            raise RuntimeError(
                f"Missing or mismatched embeddings for {len(self._missing)} record(s). "
                f"Examples: {self._missing[:5]}"
            )

    def _load_example(self, record: ProteinRecord) -> SecondaryStructureExample | None:
        embedding_path = find_embedding_path(
            self.embeddings_dir,
            record.split,
            record.id,
            record.sequence,
        )
        if embedding_path is None:
            self._missing.append(record.id)
            return None

        embedding_record = load_embedding(embedding_path)
        if embedding_record.sequence != record.sequence:
            raise ValueError(
                f"{record.id}: embedding sequence does not match dataset sequence"
            )
        if embedding_record.embeddings.shape[0] != record.length:
            raise ValueError(
                f"{record.id}: embedding length ({embedding_record.embeddings.shape[0]}) "
                f"!= label length ({record.length})"
            )

        labels = labels_from_record(record, self.label_mode)
        if len(labels) != record.length:
            raise ValueError(
                f"{record.id}: encoded label length ({len(labels)}) "
                f"!= sequence length ({record.length})"
            )

        return SecondaryStructureExample(
            protein_id=record.id,
            sequence=record.sequence,
            embeddings=embedding_record.embeddings.float(),
            labels=torch.tensor(labels, dtype=torch.long),
            length=record.length,
        )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict:
        example = self._examples[index]
        return {
            "protein_id": example.protein_id,
            "sequence": example.sequence,
            "embeddings": example.embeddings,
            "labels": example.labels,
            "length": example.length,
        }

    @property
    def embedding_dim(self) -> int:
        return int(self._examples[0].embeddings.shape[-1])

    @property
    def esm_model_name(self) -> str:
        embedding_path = find_embedding_path(
            self.embeddings_dir,
            self.records[0].split,
            self.records[0].id,
            self.records[0].sequence,
        )
        if embedding_path is None:
            raise RuntimeError("Unable to resolve ESM model name from embeddings.")
        return load_embedding(embedding_path).esm_model_name
