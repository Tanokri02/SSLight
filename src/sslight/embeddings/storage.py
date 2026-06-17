"""Save and load per-protein ESM embedding files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class EmbeddingRecord:
    protein_id: str
    sequence: str
    esm_model_name: str
    embeddings: torch.Tensor
    embedding_dim: int
    sequence_length: int


def verify_embedding_record(record: EmbeddingRecord) -> None:
    """Ensure embedding tensor shape matches sequence metadata."""
    if record.embeddings.ndim != 2:
        raise ValueError(
            f"{record.protein_id}: expected 2D embeddings, got shape {tuple(record.embeddings.shape)}"
        )

    num_residues, dim = record.embeddings.shape
    if num_residues != record.sequence_length:
        raise ValueError(
            f"{record.protein_id}: embedding length ({num_residues}) "
            f"!= sequence length ({record.sequence_length})"
        )
    if num_residues != len(record.sequence):
        raise ValueError(
            f"{record.protein_id}: embedding length ({num_residues}) "
            f"!= sequence string length ({len(record.sequence)})"
        )
    if dim != record.embedding_dim:
        raise ValueError(
            f"{record.protein_id}: embedding dim ({dim}) "
            f"!= recorded embedding_dim ({record.embedding_dim})"
        )


def save_embedding(path: str | Path, record: EmbeddingRecord) -> None:
    """Save one protein embedding to disk."""
    verify_embedding_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protein_id": record.protein_id,
        "sequence": record.sequence,
        "esm_model_name": record.esm_model_name,
        "embeddings": record.embeddings.cpu(),
        "embedding_dim": record.embedding_dim,
        "sequence_length": record.sequence_length,
    }
    torch.save(payload, path)


def load_embedding(path: str | Path) -> EmbeddingRecord:
    """Load one protein embedding from disk."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    record = EmbeddingRecord(
        protein_id=str(payload["protein_id"]),
        sequence=str(payload["sequence"]),
        esm_model_name=str(payload["esm_model_name"]),
        embeddings=payload["embeddings"],
        embedding_dim=int(payload["embedding_dim"]),
        sequence_length=int(payload["sequence_length"]),
    )
    verify_embedding_record(record)
    return record


def embedding_path(root: str | Path, split: str, protein_id: str) -> Path:
    """Return the canonical path for one saved embedding file."""
    safe_id = protein_id.replace("/", "_")
    return Path(root) / split / f"{safe_id}.pt"


def embedding_path_hashed(root: str | Path, split: str, protein_id: str) -> Path:
    """Return a case-safe path for IDs that collide on case-insensitive filesystems."""
    safe_id = protein_id.replace("/", "_")
    digest = hashlib.sha256(protein_id.encode()).hexdigest()[:12]
    return Path(root) / split / f"{safe_id}__{digest}.pt"


def find_embedding_path(
    root: str | Path,
    split: str,
    protein_id: str,
    sequence: str | None = None,
) -> Path | None:
    """Locate a saved embedding, validating sequence when provided."""
    root = Path(root)
    exact = embedding_path(root, split, protein_id)
    if exact.exists():
        if sequence is None:
            return exact
        record = load_embedding(exact)
        if record.sequence == sequence:
            return exact

    hashed = embedding_path_hashed(root, split, protein_id)
    if hashed.exists():
        if sequence is None:
            return hashed
        record = load_embedding(hashed)
        if record.sequence == sequence:
            return hashed

    split_dir = root / split
    if not split_dir.exists():
        return None

    safe_id = protein_id.replace("/", "_").lower()
    for candidate in split_dir.glob("*.pt"):
        if candidate.stem.lower() != safe_id and not candidate.stem.lower().startswith(f"{safe_id}__"):
            continue
        if sequence is None:
            return candidate
        record = load_embedding(candidate)
        if record.sequence == sequence:
            return candidate

    return None
