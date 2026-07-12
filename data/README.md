# Data

This directory is gitignored. No datasets are committed.

`scripts/download_data.py` fetches the public MoleculeNet Lipophilicity dataset
(about 4200 molecules, hosted by DeepChem) and writes one SMILES per line to
`data/smiles.txt`. Only the SMILES are used; the lipophilicity labels are
ignored, the molecules serve purely as a public SMILES corpus.

```bash
python scripts/download_data.py --outdir data
```

The unit tests and the `--builtin` demo path use a small built-in set of valid
SMILES and need no download.
