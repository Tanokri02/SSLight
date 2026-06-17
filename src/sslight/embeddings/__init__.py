"""Embedding storage and verification."""

from sslight.embeddings.generator import EmbeddingGenerator
from sslight.embeddings.storage import (
    EmbeddingRecord,
    load_embedding,
    save_embedding,
    verify_embedding_record,
)

__all__ = [
    "EmbeddingGenerator",
    "EmbeddingRecord",
    "load_embedding",
    "save_embedding",
    "verify_embedding_record",
]
