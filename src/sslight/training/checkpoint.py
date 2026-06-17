"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sslight.models.cnn_head import CNNHeadConfig, SecondaryStructureCNN


def save_checkpoint(
    path: str | Path,
    model: SecondaryStructureCNN,
    *,
    label_mode: str,
    esm_model_name: str,
    label_vocab: dict[str, int],
    epoch: int,
    best_metric_name: str,
    best_metric_value: float,
    optimizer_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "label_mode": label_mode,
        "label_vocab": label_vocab,
        "esm_model_name": esm_model_name,
        "epoch": epoch,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "optimizer_state": optimizer_state,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[SecondaryStructureCNN, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = CNNHeadConfig(**payload["model_config"])
    model = SecondaryStructureCNN(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, payload
