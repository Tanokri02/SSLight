"""Training package."""

from sslight.training.evaluate import evaluate_checkpoint
from sslight.training.trainer import TrainConfig, train_model

__all__ = ["TrainConfig", "evaluate_checkpoint", "train_model"]
