# SSLight

Lightweight, reproducible **protein secondary structure prediction** using frozen ESM-2 residue embeddings and a small per-residue classifier.

This project is not intended to outperform mature tools such as JPred, PSIPRED, or NetSurfP. Its goal is to provide a clean local framework for ESM-based, sequence-only secondary structure experiments.

## Labels

- **Q3**: `H` (helix), `E` (beta strand), `C` (coil/other)
- **Q8**: DSSP fine-grained labels (`H`, `B`, `E`, `G`, `I`, `T`, `S`, `C`)

Q3 labels are derived from Q8 during dataset loading.

## Repository layout

```text
SSLight/
├── data/                  # JSONL dataset (dataset_cleaned.jsonl included)
├── Model Weights/         # trained checkpoints (best.pt, last.pt)
├── embeddings/            # precomputed ESM embeddings (generated locally)
├── scripts/               # dataset prep, download, embedding helpers
├── src/sslight/           # Python package (CLI, model, training)
└── tests/
```

**Not in the repo (generated locally):** `structures/` (PDB mmCIF files), `embeddings/*.pt`, `data/raw/`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,data]"
```

## Workflow

### 1. Generate ESM embeddings

```bash
sslight embed --data data/dataset_cleaned.jsonl --out embeddings/
```

### 2. Train

```bash
sslight train --data data/dataset_cleaned.jsonl --embeddings embeddings/ --output-dir "Model Weights"
```

### 3. Evaluate

```bash
sslight evaluate --checkpoint "Model Weights/best.pt" --data data/dataset_cleaned.jsonl --embeddings embeddings/ --split test
```

To rebuild the dataset from PDB structures locally, see [data/README.md](data/README.md).

## Data format

See [data/README.md](data/README.md) and [embeddings/README.md](embeddings/README.md).

## Limitations

- Uses frozen ESM embeddings only (no fine-tuning yet)
- Requires standard amino acid sequences without ambiguous residues
- Small CNN head; accuracy will depend on dataset size and quality

## Future work

- Q8 prediction mode
- Prediction CLI for new sequences
- Optional visualization of predicted secondary structure
- Support for larger ESM models and batched HDF5/Zarr storage
