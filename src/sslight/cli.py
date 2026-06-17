"""Command-line interface for SSLight."""

from __future__ import annotations

import argparse
from pathlib import Path

from sslight.constants import DEFAULT_ESM_MODEL, ESM_MODEL_REGISTRY
from sslight.embeddings.generator import EmbeddingGenerator
from sslight.training.evaluate import evaluate_checkpoint
from sslight.training.trainer import TrainConfig, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sslight",
        description="Lightweight ESM-based secondary structure prediction",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser(
        "embed",
        help="Generate and save ESM residue embeddings",
    )
    embed_parser.add_argument("--data", type=Path, help="Normalized JSONL dataset path")
    embed_parser.add_argument("--fasta", type=Path, help="FASTA file path (alternative to --data)")
    embed_parser.add_argument("--out", type=Path, required=True, help="Output directory for embedding files")
    embed_parser.add_argument(
        "--split",
        choices=["train", "val", "validation", "test"],
        help="Optional split filter when using --data",
    )
    embed_parser.add_argument(
        "--model",
        default=DEFAULT_ESM_MODEL,
        choices=sorted(ESM_MODEL_REGISTRY),
        help="ESM-2 model variant",
    )
    embed_parser.add_argument("--batch-size", type=int, default=8)
    embed_parser.add_argument("--device", default=None)
    embed_parser.add_argument("--overwrite", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train the secondary structure prediction head")
    train_parser.add_argument("--data", type=Path, default=Path("data/dataset_cleaned.jsonl"))
    train_parser.add_argument("--embeddings", type=Path, default=Path("embeddings"))
    train_parser.add_argument("--output-dir", type=Path, default=Path("Model Weights"))
    train_parser.add_argument("--label-mode", default="q3", choices=["q3"])
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--num-layers", type=int, default=3)
    train_parser.add_argument("--kernel-size", type=int, default=5)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument(
        "--selection-metric",
        default="macro_f1",
        choices=["macro_f1", "accuracy"],
    )
    train_parser.add_argument("--seed", type=int, default=42)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--data", type=Path, default=Path("data/dataset_cleaned.jsonl"))
    eval_parser.add_argument("--embeddings", type=Path, default=Path("embeddings"))
    eval_parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "validation", "test"],
    )
    eval_parser.add_argument("--batch-size", type=int, default=8)
    eval_parser.add_argument("--num-workers", type=int, default=0)
    eval_parser.add_argument("--device", default=None)
    eval_parser.add_argument("--output-json", type=Path, default=None)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "embed":
        if bool(args.data) == bool(args.fasta):
            parser.error("Provide exactly one of --data or --fasta")

        generator = EmbeddingGenerator(
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
        )

        if args.data:
            saved = generator.embed_jsonl(
                data_path=args.data,
                output_dir=args.out,
                split=args.split,
                overwrite=args.overwrite,
            )
        else:
            saved = generator.embed_fasta(
                fasta_path=args.fasta,
                output_dir=args.out,
                overwrite=args.overwrite,
            )

        print(f"Saved {saved} embedding file(s) to {args.out}")
        return

    if args.command == "train":
        config = TrainConfig(
            data_path=args.data,
            embeddings_dir=args.embeddings,
            output_dir=args.output_dir,
            label_mode=args.label_mode,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            num_workers=args.num_workers,
            device=args.device,
            selection_metric=args.selection_metric,
            seed=args.seed,
        )
        summary = train_model(config)
        print(
            f"Training complete. Best {summary['best_metric_name']}="
            f"{summary['best_metric_value']:.4f} at epoch {summary['best_epoch']}"
        )
        print(f"Checkpoint saved to {summary['checkpoint_dir']}/best.pt")
        return

    if args.command == "evaluate":
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            data_path=args.data,
            embeddings_dir=args.embeddings,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            output_json=args.output_json,
        )
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
