"""Download a public SMILES corpus and write one SMILES per line.

Fetches the MoleculeNet Lipophilicity dataset (about 4200 public molecules,
hosted by DeepChem) and extracts the ``smiles`` column. The molecules are only
used as a SMILES corpus here; the lipophilicity labels are ignored.

Usage:
    python scripts/download_data.py --outdir data
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="data")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--max-len", type=int, default=100)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import requests

        resp = requests.get(args.url, timeout=60)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {exc}")
        print("The tests and the built-in SMILES demo path work without this file.")
        return 1

    reader = csv.DictReader(io.StringIO(resp.text))
    smiles = []
    for row in reader:
        s = (row.get("smiles") or row.get("SMILES") or "").strip()
        if s and len(s) <= args.max_len:
            smiles.append(s)

    out = outdir / "smiles.txt"
    out.write_text("\n".join(smiles) + "\n", encoding="utf-8")
    print(f"Wrote {len(smiles)} SMILES to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
