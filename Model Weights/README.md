# Model Weights

Trained secondary structure prediction checkpoints.

| File | Description |
|------|-------------|
| `best.pt` | Best checkpoint selected on the validation split (macro F1) |
| `last.pt` | Final checkpoint after the last training epoch |

Load for evaluation:

```bash
sslight evaluate --checkpoint "Model Weights/best.pt" --data data/dataset_cleaned.jsonl --embeddings embeddings/
```

Checkpoints include model weights, architecture config, label vocabulary, and ESM model metadata.
