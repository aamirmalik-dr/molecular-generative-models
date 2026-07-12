"""Carve a small, license-clean SMILES sample from the full public corpus.

Reads the full downloaded corpus (``data/smiles.txt``, produced by
``download_data.py`` from the public MoleculeNet Lipophilicity set), canonicalizes
with RDKit, removes duplicates, keeps the shorter molecules so a small character
level model has a fair chance of learning valid strings, and writes a fixed
random subset to ``data/sample_smiles.txt``.

The committed ``data/sample_smiles.txt`` was produced by this script. Only the
SMILES strings are used; the original lipophilicity labels are dropped. Run it
only if you want to regenerate the sample from a fresh download.

Usage:
    python scripts/make_sample.py --n 400 --max-len 45 --seed 0
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infile", default="data/smiles.txt")
    parser.add_argument("--outfile", default="data/sample_smiles.txt")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--min-len", type=int, default=15)
    parser.add_argument("--max-len", type=int, default=45)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    lines = [s.strip() for s in Path(args.infile).read_text(encoding="utf-8").splitlines()]
    seen: set[str] = set()
    pool: list[str] = []
    for s in lines:
        if not s:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen:
            continue
        if args.min_len <= len(canon) <= args.max_len:
            seen.add(canon)
            pool.append(canon)

    random.seed(args.seed)
    random.shuffle(pool)
    sample = sorted(pool[: args.n])

    out = Path(args.outfile)
    out.write_text("\n".join(sample) + "\n", encoding="utf-8")
    print(f"Wrote {len(sample)} canonical SMILES to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
