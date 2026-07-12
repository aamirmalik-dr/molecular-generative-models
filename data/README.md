# Data

## Committed sample: `sample_smiles.txt`

`data/sample_smiles.txt` is a small, license-clean SMILES sample of 400 molecules.
It is a carved subset of a public source, not synthetic. The molecules were taken
from the public MoleculeNet Lipophilicity set (hosted by DeepChem), canonicalized
with RDKit, de-duplicated, and filtered to a token length between 15 and 45 so a
small character model has a fair chance of learning valid strings. Only the SMILES
strings are kept; the original lipophilicity labels are dropped.

This sample is what the tests, the offline quickstart, and the default training
run use. It needs no download.

Regenerate it from a fresh download of the full corpus:

```bash
python scripts/download_data.py --outdir data     # writes data/smiles.txt (gitignored)
python scripts/make_sample.py --n 400             # writes data/sample_smiles.txt
```

## Full corpus (not committed)

`scripts/download_data.py` fetches the full MoleculeNet Lipophilicity dataset
(about 4200 molecules) and writes one SMILES per line to `data/smiles.txt`, which
is gitignored. Train on it with:

```bash
python scripts/build_artifacts.py --data data/smiles.txt --epochs 80
```

The `--builtin` path in `load_smiles` uses a tiny built-in set of valid SMILES for
unit tests and needs no files at all.
