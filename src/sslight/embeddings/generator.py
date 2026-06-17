"""Generate frozen ESM residue embeddings for protein sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import esm
import torch

from sslight.constants import (
    DEFAULT_ESM_MODEL,
    ESM_MODEL_REGISTRY,
    normalize_split,
    validate_sequence,
)
from sslight.data.dataset import ProteinRecord, iter_jsonl
from sslight.data.fasta import FastaRecord, load_fasta
from sslight.embeddings.storage import (
    EmbeddingRecord,
    embedding_path,
    embedding_path_hashed,
    find_embedding_path,
    load_embedding,
    save_embedding,
)


class EmbeddingGenerator:
    """Compute and save per-residue ESM embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_ESM_MODEL,
        device: str | torch.device | None = None,
        batch_size: int = 8,
    ) -> None:
        if model_name not in ESM_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown ESM model {model_name!r}. "
                f"Supported: {sorted(ESM_MODEL_REGISTRY)}"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        loader = getattr(esm.pretrained, model_name)
        self.model, self.alphabet = loader()
        self.model.eval()
        self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    @torch.inference_mode()
    def embed_sequence(self, protein_id: str, sequence: str) -> EmbeddingRecord:
        """Generate residue embeddings for one sequence."""
        validate_sequence(sequence)
        _, _, tokens = self.batch_converter([(protein_id, sequence)])
        tokens = tokens.to(self.device)

        results = self.model(tokens, repr_layers=[self.model.num_layers], return_contacts=False)
        token_embeddings = results["representations"][self.model.num_layers][0]

        # Remove BOS/EOS special tokens so length matches the amino acid sequence.
        residue_embeddings = token_embeddings[1 : len(sequence) + 1].detach().cpu()

        if residue_embeddings.shape[0] != len(sequence):
            raise ValueError(
                f"{protein_id}: embedding length ({residue_embeddings.shape[0]}) "
                f"!= sequence length ({len(sequence)})"
            )

        return EmbeddingRecord(
            protein_id=protein_id,
            sequence=sequence,
            esm_model_name=self.model_name,
            embeddings=residue_embeddings,
            embedding_dim=residue_embeddings.shape[1],
            sequence_length=len(sequence),
        )

    @torch.inference_mode()
    def embed_batch(self, items: Iterable[tuple[str, str]]) -> list[EmbeddingRecord]:
        """Embed a batch of (protein_id, sequence) pairs."""
        batch = list(items)
        if not batch:
            return []

        for protein_id, sequence in batch:
            validate_sequence(sequence)

        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)
        results = self.model(tokens, repr_layers=[self.model.num_layers], return_contacts=False)
        layer = results["representations"][self.model.num_layers]

        records: list[EmbeddingRecord] = []
        for index, (protein_id, sequence) in enumerate(batch):
            token_embeddings = layer[index]
            residue_embeddings = token_embeddings[1 : len(sequence) + 1].detach().cpu()

            if residue_embeddings.shape[0] != len(sequence):
                raise ValueError(
                    f"{protein_id}: embedding length ({residue_embeddings.shape[0]}) "
                    f"!= sequence length ({len(sequence)})"
                )

            records.append(
                EmbeddingRecord(
                    protein_id=protein_id,
                    sequence=sequence,
                    esm_model_name=self.model_name,
                    embeddings=residue_embeddings,
                    embedding_dim=residue_embeddings.shape[1],
                    sequence_length=len(sequence),
                )
            )
        return records

    def embed_jsonl(
        self,
        data_path: str | Path,
        output_dir: str | Path,
        split: str | None = None,
        overwrite: bool = False,
    ) -> int:
        """Embed all records from a JSONL dataset."""
        output_dir = Path(output_dir)
        saved = 0
        skipped = 0
        target_split = normalize_split(split) if split is not None else None

        buffer: list[tuple[str, str, str]] = []
        for record in iter_jsonl(data_path):
            if target_split is not None and record.split != target_split:
                continue

            out_path = find_embedding_path(output_dir, record.split, record.id, record.sequence)
            if out_path is not None and not overwrite:
                skipped += 1
                continue

            buffer.append((record.id, record.sequence, record.split))
            if len(buffer) >= self.batch_size:
                saved += self._flush_buffer(buffer, output_dir)
                buffer.clear()
                if saved % 100 == 0:
                    print(f"  embedded {saved} new file(s)...", flush=True)

        if buffer:
            saved += self._flush_buffer(buffer, output_dir)

        if skipped:
            print(f"  skipped {skipped} existing file(s)", flush=True)

        return saved

    def embed_fasta(
        self,
        fasta_path: str | Path,
        output_dir: str | Path,
        split: str = "predict",
        overwrite: bool = False,
    ) -> int:
        """Embed all sequences from a FASTA file."""
        records = load_fasta(fasta_path)
        output_dir = Path(output_dir)
        saved = 0
        buffer: list[tuple[str, str, str]] = []

        for record in records:
            out_path = embedding_path(output_dir, split, record.id)
            if out_path.exists() and not overwrite:
                continue
            buffer.append((record.id, record.sequence, split))
            if len(buffer) >= self.batch_size:
                saved += self._flush_buffer(buffer, output_dir)
                buffer.clear()

        if buffer:
            saved += self._flush_buffer(buffer, output_dir)

        return saved

    def _flush_buffer(
        self,
        buffer: list[tuple[str, str, str]],
        output_dir: Path,
    ) -> int:
        pairs = [(protein_id, sequence) for protein_id, sequence, _split in buffer]
        embedding_records = self.embed_batch(pairs)
        saved = 0

        for (_, _, split), embedding_record in zip(buffer, embedding_records):
            out_path = find_embedding_path(
                output_dir, split, embedding_record.protein_id, embedding_record.sequence
            )
            if out_path is None:
                exact = embedding_path(output_dir, split, embedding_record.protein_id)
                if exact.exists():
                    existing = load_embedding(exact)
                    if existing.sequence != embedding_record.sequence:
                        out_path = embedding_path_hashed(
                            output_dir, split, embedding_record.protein_id
                        )
                    else:
                        out_path = exact
                else:
                    out_path = exact
            save_embedding(out_path, embedding_record)
            saved += 1

        return saved


def embed_dataset_records(
    records: Iterable[ProteinRecord | FastaRecord],
    output_dir: str | Path,
    model_name: str = DEFAULT_ESM_MODEL,
    split: str | None = None,
    batch_size: int = 8,
    device: str | torch.device | None = None,
    overwrite: bool = False,
    ):
    """Embed an in-memory iterable of records."""
    generator = EmbeddingGenerator(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    output_dir = Path(output_dir)
    saved = 0
    buffer: list[tuple[str, str, str]] = []

    for record in records:
        record_split = split or getattr(record, "split", "predict")
        out_path = embedding_path(output_dir, record_split, record.id)
        if out_path.exists() and not overwrite:
            continue
        buffer.append((record.id, record.sequence, record_split))
        if len(buffer) >= batch_size:
            saved += generator._flush_buffer(buffer, output_dir)
            buffer.clear()

    if buffer:
        saved += generator._flush_buffer(buffer, output_dir)

    return saved
