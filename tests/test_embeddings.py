import pytest
import torch

from sslight.embeddings.storage import EmbeddingRecord, save_embedding, load_embedding, verify_embedding_record


def test_embedding_roundtrip(tmp_path):
    record = EmbeddingRecord(
        protein_id="demo",
        sequence="ACDE",
        esm_model_name="esm2_t6_8M_UR50D",
        embeddings=torch.randn(4, 320),
        embedding_dim=320,
        sequence_length=4,
    )
    path = tmp_path / "demo.pt"
    save_embedding(path, record)
    loaded = load_embedding(path)
    assert loaded.protein_id == "demo"
    assert loaded.embeddings.shape == (4, 320)


def test_embedding_length_mismatch_raises():
    record = EmbeddingRecord(
        protein_id="bad",
        sequence="ACDE",
        esm_model_name="esm2_t6_8M_UR50D",
        embeddings=torch.randn(3, 320),
        embedding_dim=320,
        sequence_length=4,
    )
    with pytest.raises(ValueError, match="embedding length"):
        verify_embedding_record(record)
