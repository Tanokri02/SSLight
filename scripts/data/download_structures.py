#!/usr/bin/env python3
"""Download PDB mmCIF structure files from RCSB."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import aiohttp
import pandas as pd
from aiohttp import ClientSession


async def download_one(
    session: ClientSession,
    pdb_id: str,
    structures_dir: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    pdb_id = pdb_id.lower()
    file_path = structures_dir / f"{pdb_id}.cif.gz"
    if file_path.exists():
        return

    url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
    async with semaphore:
        try:
            async with session.get(url, timeout=15) as response:
                if response.status == 200:
                    file_path.write_bytes(await response.read())
                    print(f"Downloaded: {pdb_id}")
                else:
                    print(f"Failed {pdb_id}: status {response.status}")
        except Exception as exc:
            print(f"Error {pdb_id}: {exc}")


async def download_all(pdb_ids: list[str], structures_dir: Path, concurrency: int) -> None:
    structures_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    async with ClientSession() as session:
        tasks = [
            download_one(session, pdb_id, structures_dir, semaphore) for pdb_id in pdb_ids
        ]
        await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdb-list",
        type=Path,
        default=Path("data/pdb_list.csv"),
        help="CSV file with a 'pdb' column",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("structures"),
        help="Directory for downloaded mmCIF.gz files",
    )
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    df = pd.read_csv(args.pdb_list)
    pdb_ids = df["pdb"].dropna().astype(str).unique().tolist()
    asyncio.run(download_all(pdb_ids, args.out_dir, args.concurrency))


if __name__ == "__main__":
    main()
