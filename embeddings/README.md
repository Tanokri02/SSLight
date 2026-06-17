# Embeddings

Precomputed ESM embeddings are stored one protein per file:

```text
embeddings/
  train/
    protein_1.pt
  val/
    protein_2.pt
  test/
    protein_3.pt
```

Each `.pt` file contains:

- `protein_id`
- `sequence`
- `esm_model_name`
- `embeddings` tensor with shape `[sequence_length, embedding_dim]`
- `embedding_dim`
- `sequence_length`

The embedding length must exactly match the sequence length. Special ESM tokens are removed before saving.

## Generate embeddings

```bash
pip install -e .

sslight embed --data data/dataset_cleaned.jsonl --out embeddings/ --model esm2_t6_8M_UR50D
```

Optional flags:

- `--split train|val|validation|test` to embed only one split
- `--batch-size 8` for throughput tuning
- `--overwrite` to recompute existing files
- `--device cuda` or `--device cpu`

For ad-hoc sequences:

```bash
sslight embed --fasta query.fasta --out embeddings/predict/
```
