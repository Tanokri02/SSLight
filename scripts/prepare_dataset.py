from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path

AMBIGUOUS_AMINO_ACIDS = frozenset("XBZUJOU*-")

def has_ambiguous_residue(sequence):
    """
    Args: sequence - str - The input sequence of amino acids
    
    The function returns whether the input sequence has any unresolved residues in it.
    """
    return any(char in AMBIGUOUS_AMINO_ACIDS for char in sequence)


def prepare_dataset(input_path, output_path, val_fraction=0.5, seed=42):
    """Filter invalid sequences and split former test records into val/test."""
    train_records: list[dict] = []
    test_candidates: list[dict] = []
    skipped = 0

    with input_path.open() as handle:
        for line in handle:
            raw = json.loads(line)
            sequence = raw["sequence"]
            q8 = raw["q8"]

            if has_ambiguous_residue(sequence):
                skipped += 1
                continue
            if len(sequence) != len(q8):
                skipped += 1
                continue

            record = {
                "id": raw["id"],
                "sequence": sequence,
                "q8": q8,
                "split": raw["split"],
            }

            if record["split"] == "test":
                test_candidates.append(record)
            else:
                train_records.append(record)

    random.seed(seed)
    random.shuffle(test_candidates)
    val_count = int(len(test_candidates) * val_fraction)
    val_records = test_candidates[:val_count]
    test_records = test_candidates[val_count:]

    for record in train_records:
        record["split"] = "train"
    for record in val_records:
        record["split"] = "val"
    for record in test_records:
        record["split"] = "test"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_records = train_records + val_records + test_records
    with output_path.open("w") as handle:
        for record in all_records:
            handle.write(json.dumps(record) + "\n")

    counts = Counter(record["split"] for record in all_records)
    return {
        "skipped": skipped,
        "train": counts["train"],
        "val": counts["val"],
        "test": counts["test"],
        "total": len(all_records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/dataset.jsonl"),
        help="Source JSONL file (unfiltered)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset.jsonl"),
        help="Filtered output JSONL file",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.5,
        help="Fraction of former test records assigned to validation",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = prepare_dataset(
        input_path=args.input,
        output_path=args.output,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print(f"Skipped records: {stats['skipped']}")
    print(
        "Saved "
        f"{stats['total']} records -> {args.output} "
        f"(train={stats['train']}, val={stats['val']}, test={stats['test']})"
    )


if __name__ == "__main__":
    main()