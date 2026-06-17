"""Lightweight CNN prediction head for per-residue secondary structure."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class CNNHeadConfig:
    embedding_dim: int = 320
    hidden_dim: int = 128
    num_layers: int = 3
    kernel_size: int = 5
    dropout: float = 0.2
    num_classes: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


class SecondaryStructureCNN(nn.Module):
    """1D CNN head over frozen ESM residue embeddings."""

    def __init__(self, config: CNNHeadConfig) -> None:
        super().__init__()
        self.config = config

        if config.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if config.kernel_size < 1 or config.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        padding = config.kernel_size // 2
        self.input_projection = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        conv_layers: list[nn.Module] = []
        for _ in range(config.num_layers):
            conv_layers.extend(
                [
                    nn.Conv1d(
                        config.hidden_dim,
                        config.hidden_dim,
                        kernel_size=config.kernel_size,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
        self.conv_layers = nn.Sequential(*conv_layers)
        self.classifier = nn.Conv1d(config.hidden_dim, config.num_classes, kernel_size=1)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            embeddings: [batch, length, embedding_dim]
            mask: optional [batch, length] bool mask (unused in forward)

        Returns:
            logits: [batch, length, num_classes]
        """
        del mask
        hidden = self.input_projection(embeddings)
        hidden = hidden.transpose(1, 2)
        hidden = self.conv_layers(hidden)
        logits = self.classifier(hidden).transpose(1, 2)
        return logits

    @classmethod
    def from_config_dict(cls, config: dict) -> "SecondaryStructureCNN":
        return cls(CNNHeadConfig(**config))
