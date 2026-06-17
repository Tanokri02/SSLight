#!/usr/bin/env python3
"""Generate normalized JSONL labels from PDB mmCIF structures using DSSP. Acts as the training data ground truth"""

from __future__ import annotations
import argparse
import gzip
import json
import os
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from Bio.PDB import MMCIFParser
from Bio.PDB.DSSP import DSSP

def process_single_file(file_info: tuple[str, str, str]) -> list[dict]:
    
    """
    """
    cif_gz_path, pdb_id, split_label = file_info
    records: list[dict] = []

    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        temp_cif_path = tmp.name

    try:
        with gzip.open(cif_gz_path, "rb") as f_in, open(temp_cif_path, "wb") as f_out:
            f_out.write(f_in.read())

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, temp_cif_path)
        model = structure[0]
        dssp_obj = DSSP(model, temp_cif_path, dssp="mkdssp", file_type="")

        chain_sequences: dict[str, list[str]] = {}
        chain_q8: dict[str, list[str]] = {}

        for key in dssp_obj.keys():
            chain_id = key[0]
            residue_data = dssp_obj[key]
            aa = residue_data[1]
            ss = residue_data[2]

            if aa == "!":
                continue
            if aa.islower():
                aa = aa.upper()
            if ss in ("-", " "):
                ss = "C"

            chain_sequences.setdefault(chain_id, []).append(aa)
            chain_q8.setdefault(chain_id, []).append(ss)

        for chain_id, aa_list in chain_sequences.items():
            seq_str = "".join(aa_list)
            q8_str = "".join(chain_q8[chain_id])
            if not seq_str:
                continue

            records.append(
                {
                    "id": f"{pdb_id}_{chain_id}",
                    "sequence": seq_str,
                    "q8": q8_str,
                    "split": split_label,
                }
            )
    except Exception as exc:
        print(f"Skipping {pdb_id} due to parsing error: {exc}")
    finally:
        if os.path.exists(temp_cif_path):
            os.remove(temp_cif_path)

    return records


def build_dataset(
    input_dir: Path,
    output_file: Path,
    split_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    files = sorted(f for f in input_dir.iterdir() if f.name.lower().endswith(".cif.gz"))
    if not files:
        raise FileNotFoundError(f"No .cif.gz files found in {input_dir}")

    random.seed(seed)
    random.shuffle(files)
    split_idx = int(len(files) * split_ratio)

    file_infos = [
        (
            str(file_path),
            file_path.name.replace(".cif.gz", "").upper(),
            "train" if index < split_idx else "test",
        )
        for index, file_path in enumerate(files)
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_chains = 0

    with output_file.open("w") as out_f:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(process_single_file, info): info for info in file_infos
            }
            for index, future in enumerate(as_completed(futures), start=1):
                for record in future.result():
                    out_f.write(json.dumps(record) + "\n")
                    total_chains += 1
                if index % 500 == 0:
                    print(f"Progress: {index} / {len(files)} files processed")

    print(f"Saved {total_chains} chain records to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("structures"),
        help="Directory containing mmCIF.gz structure files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/dataset.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_dataset(
        input_dir=args.input_dir,
        output_file=args.output,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
