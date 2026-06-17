"""Run ESM embedding generation without requiring the sslight console script.

Usage:
    python scripts/embed_dataset.py --data data/dataset.jsonl --out embeddings/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports when the package has not been pip-installed yet.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sslight.cli import main  # noqa: E402


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("embed")
    elif sys.argv[1] != "embed":
        sys.argv.insert(1, "embed")
    main()
