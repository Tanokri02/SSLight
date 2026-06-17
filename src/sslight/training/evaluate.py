"""Evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sslight.constants import normalize_split
from sslight.data.collate import collate_secondary_structure_batch
from sslight.data.ss_dataset import SecondaryStructureDataset
from sslight.labels import num_classes
from sslight.training.checkpoint import load_checkpoint
from sslight.training.metrics import compute_metrics, masked_cross_entropy
from sslight.training.trainer import _aggregate_metrics, _resolve_device


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_path: str | Path,
    embeddings_dir: str | Path,
    split: str = "test",
    batch_size: int = 8,
    num_workers: int = 0,
    device: str | None = None,
    output_json: str | Path | None = None,
) -> dict:
    """Evaluate a saved checkpoint on one dataset split."""
    split = normalize_split(split)
    device_obj = _resolve_device(device)
    model, payload = load_checkpoint(checkpoint_path, device=device_obj)
    label_mode = payload["label_mode"]

    dataset = SecondaryStructureDataset(
        data_path=data_path,
        embeddings_dir=embeddings_dir,
        split=split,
        label_mode=label_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_secondary_structure_batch,
    )

    model.eval()
    total_loss = 0.0
    num_batches = 0
    metric_batches = []

    with torch.inference_mode():
        for batch in loader:
            embeddings = batch["embeddings"].to(device_obj)
            labels = batch["labels"].to(device_obj)
            mask = batch["mask"].to(device_obj)
            logits = model(embeddings, mask=mask)
            loss = masked_cross_entropy(logits, labels, mask)
            total_loss += float(loss.item())
            num_batches += 1
            metric_batches.append(
                compute_metrics(
                    logits,
                    labels,
                    mask,
                    num_classes=num_classes(label_mode),
                )
            )

    metrics = _aggregate_metrics(metric_batches)
    results = {
        "split": split,
        "loss": total_loss / max(num_batches, 1),
        "checkpoint": str(checkpoint_path),
        **metrics,
    }

    _print_results(results)
    if output_json is not None:
        Path(output_json).write_text(json.dumps(results, indent=2) + "\n")
    return results


def _print_results(results: dict) -> None:
    print(f"Split: {results['split']}")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print("Per-class precision:", results["per_class_precision"])
    print("Per-class recall:", results["per_class_recall"])
    print("Per-class F1:", results["per_class_f1"])
    print("Confusion matrix [true x pred] (H, E, C):")
    for row in results["confusion_matrix"]:
        print(" ", row)
