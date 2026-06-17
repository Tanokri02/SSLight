import torch

from sslight.constants import PADDING_LABEL_INDEX
from sslight.data.collate import collate_secondary_structure_batch
from sslight.labels import Q3_LABEL_TO_INDEX, decode_q3_indices, encode_q3_label_string


def test_encode_decode_q3_roundtrip():
    labels = "HECCHH"
    encoded = encode_q3_label_string(labels)
    assert encoded == [Q3_LABEL_TO_INDEX[char] for char in labels]
    assert decode_q3_indices(encoded) == labels


def test_collate_pads_embeddings_labels_and_mask():
    batch = [
        {
            "protein_id": "a",
            "sequence": "AC",
            "embeddings": torch.ones(2, 4),
            "labels": torch.tensor([0, 1], dtype=torch.long),
            "length": 2,
        },
        {
            "protein_id": "b",
            "sequence": "ACDE",
            "embeddings": torch.full((4, 4), 2.0),
            "labels": torch.tensor([1, 2, 2, 0], dtype=torch.long),
            "length": 4,
        },
    ]

    collated = collate_secondary_structure_batch(batch)
    assert collated["embeddings"].shape == (2, 4, 4)
    assert collated["labels"].shape == (2, 4)
    assert collated["mask"].shape == (2, 4)

    assert collated["mask"][0].tolist() == [True, True, False, False]
    assert collated["labels"][0, 2].item() == PADDING_LABEL_INDEX
    assert collated["labels"][0, 3].item() == PADDING_LABEL_INDEX
    assert torch.all(collated["embeddings"][0, 2:] == 0)
    assert collated["embeddings"][1].sum().item() == 4 * 4 * 2.0
