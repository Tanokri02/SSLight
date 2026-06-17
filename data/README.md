# Dataset

SSLight expects a normalized JSONL file where each line is one protein chain:

```json
{"id": "protein_1", "sequence": "MKT...", "q8": "CCCHHH...", "split": "train"}
```

Required fields:

- `id`: unique protein/chain identifier
- `sequence`: one-letter amino acid sequence (20 standard amino acids only)
- `q8`: per-residue DSSP secondary structure string, same length as `sequence`
- `split`: one of `train`, `val`, or `test`

Q3 labels are derived at load time using:

```text
H, G, I -> H
E, B     -> E
T, S, C  -> C
```

## Current project dataset

The filtered dataset lives at `data/dataset_cleaned.jsonl`:

- ambiguous residues removed
- train / validation / test splits
- sequence lengths filtered to the project range

The unfiltered DSSP source (`data/raw/dataset.jsonl`) is generated locally and is **not** included in the GitHub repository.

## Building a dataset from PDB structures

1. Download structures:

```bash
python scripts/data/download_structures.py --pdb-list data/pdb_list.csv
```

2. Run DSSP labeling:

```bash
python scripts/data/generate_jsonl.py --input-dir structures --output data/raw/dataset.jsonl
```

3. Filter and create train/val/test splits:

```bash
python scripts/prepare_dataset.py --input data/raw/dataset.jsonl --output data/dataset_cleaned.jsonl
```

## External dataset sources

You can also prepare JSONL from sources such as CullPDB, CB513, TAPE secondary structure datasets, or other DSSP-derived PDB datasets. Convert them to the normalized format above before training.
