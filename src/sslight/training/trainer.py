"""Training loop for the secondary structure prediction head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sslight.data.collate import collate_secondary_structure_batch
from sslight.data.ss_dataset import SecondaryStructureDataset
from sslight.labels import Q3_LABEL_TO_INDEX, num_classes
from sslight.models.cnn_head import CNNHeadConfig, SecondaryStructureCNN
from sslight.training.checkpoint import save_checkpoint
from sslight.training.metrics import compute_metrics, masked_cross_entropy


@dataclass
class TrainConfig:
    data_path: Path
    embeddings_dir: Path
    output_dir: Path
    label_mode: str = "q3"
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    num_layers: int = 3
    kernel_size: int = 5
    dropout: float = 0.2
    num_workers: int = 0
    device: str | None = None
    selection_metric: str = "macro_f1"
    seed: int = 42


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _aggregate_metrics(metric_batches: list) -> dict:
    total_residues = sum(batch.num_residues for batch in metric_batches)
    if total_residues == 0:
        return compute_metrics(
            logits=torch.zeros(1, 1, 3),
            labels=torch.zeros(1, 1, dtype=torch.long),
            mask=torch.zeros(1, 1, dtype=torch.bool),
            num_classes=3,
        ).to_dict()

    weighted = {
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "per_class_precision": {label: 0.0 for label in "HEC"},
        "per_class_recall": {label: 0.0 for label in "HEC"},
        "per_class_f1": {label: 0.0 for label in "HEC"},
        "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "num_residues": total_residues,
    }

    for metrics in metric_batches:
        weight = metrics.num_residues / total_residues
        weighted["accuracy"] += metrics.accuracy * weight
        weighted["macro_f1"] += metrics.macro_f1 * weight
        for label in "HEC":
            weighted["per_class_precision"][label] += metrics.per_class_precision[label] * weight
            weighted["per_class_recall"][label] += metrics.per_class_recall[label] * weight
            weighted["per_class_f1"][label] += metrics.per_class_f1[label] * weight
        for row in range(3):
            for col in range(3):
                weighted["confusion_matrix"][row][col] += metrics.confusion_matrix[row][col]

    return weighted


def _run_epoch(
    model: SecondaryStructureCNN,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    num_output_classes: int,
) -> tuple[float, dict]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    num_batches = 0
    metric_batches = []

    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        logits = model(embeddings, mask=mask)
        loss = masked_cross_entropy(logits, labels, mask)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1
        metric_batches.append(
            compute_metrics(logits, labels, mask, num_classes=num_output_classes)
        )

    avg_loss = total_loss / max(num_batches, 1)
    metrics = _aggregate_metrics(metric_batches)
    return avg_loss, metrics


def train_model(config: TrainConfig) -> dict:
    """Train the CNN head and save the best checkpoint."""
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SecondaryStructureDataset(
        data_path=config.data_path,
        embeddings_dir=config.embeddings_dir,
        split="train",
        label_mode=config.label_mode,
    )
    val_dataset = SecondaryStructureDataset(
        data_path=config.data_path,
        embeddings_dir=config.embeddings_dir,
        split="validation",
        label_mode=config.label_mode,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_secondary_structure_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_secondary_structure_batch,
    )

    model_config = CNNHeadConfig(
        embedding_dim=train_dataset.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        num_classes=num_classes(config.label_mode),
    )
    model = SecondaryStructureCNN(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_metric_value = float("-inf")
    best_epoch = 0
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            model_config.num_classes,
        )
        val_loss, val_metrics = _run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            num_output_classes=model_config.num_classes,
        )

        selection_value = val_metrics[config.selection_metric]
        improved = selection_value > best_metric_value
        if improved:
            best_metric_value = selection_value
            best_epoch = epoch
            save_checkpoint(
                config.output_dir / "best.pt",
                model,
                label_mode=config.label_mode,
                esm_model_name=train_dataset.esm_model_name,
                label_vocab=Q3_LABEL_TO_INDEX,
                epoch=epoch,
                best_metric_name=config.selection_metric,
                best_metric_value=best_metric_value,
                optimizer_state=optimizer.state_dict(),
                extra={"train_config": config.__dict__},
            )

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "best": improved,
        }
        history.append(epoch_summary)
        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
            + (" *" if improved else "")
        )

    save_checkpoint(
        config.output_dir / "last.pt",
        model,
        label_mode=config.label_mode,
        esm_model_name=train_dataset.esm_model_name,
        label_vocab=Q3_LABEL_TO_INDEX,
        epoch=config.epochs,
        best_metric_name=config.selection_metric,
        best_metric_value=best_metric_value,
        optimizer_state=optimizer.state_dict(),
        extra={"train_config": config.__dict__, "history": history},
    )

    return {
        "best_epoch": best_epoch,
        "best_metric_name": config.selection_metric,
        "best_metric_value": best_metric_value,
        "history": history,
        "checkpoint_dir": str(config.output_dir),
    }
