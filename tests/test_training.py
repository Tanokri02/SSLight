import pytest
import torch

from sslight.constants import PADDING_LABEL_INDEX
from sslight.models.cnn_head import CNNHeadConfig, SecondaryStructureCNN
from sslight.training.metrics import compute_metrics, masked_cross_entropy


def test_masked_cross_entropy_ignores_padding():
    logits = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    labels = torch.tensor([[0, 1, PADDING_LABEL_INDEX]])
    mask = torch.tensor([[True, True, False]])

    loss = masked_cross_entropy(logits, labels, mask)
    expected = torch.nn.functional.cross_entropy(
        logits[0, :2],
        labels[0, :2],
    )
    assert torch.isclose(loss, expected)


def test_compute_metrics_ignores_padding():
    logits = torch.tensor(
        [
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0]],
            [[0.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    labels = torch.tensor(
        [
            [0, 1, PADDING_LABEL_INDEX],
            [1, 0, PADDING_LABEL_INDEX],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
        ]
    )

    metrics = compute_metrics(logits, labels, mask, num_classes=3)
    assert metrics.num_residues == 4
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == pytest.approx(2 / 3)


def test_cnn_head_output_shape():
    model = SecondaryStructureCNN(CNNHeadConfig(embedding_dim=8, hidden_dim=4, num_layers=2, kernel_size=3))
    embeddings = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    logits = model(embeddings, mask=mask)
    assert logits.shape == (2, 5, 3)
