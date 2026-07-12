# Molecular generative models

SMILES-based generative models for molecules in PyTorch: a GRU sequence
autoencoder and a variational autoencoder with a Gaussian latent space, latent
sampling, and generation quality metrics (validity, uniqueness, novelty). The
encoder, decoder, reparameterization, ELBO, and autoregressive sampling are all
written from scratch.

## What it does

- A character-level SMILES tokenizer with start, end, and pad tokens
  (`tokenizer.py`).
- A shared GRU encoder/decoder backbone, with a plain autoencoder and a
  variational autoencoder that adds a Gaussian latent, the reparameterization
  trick, and a KL term (`models.py`).
- A training loop with gradient clipping and KL annealing to reduce posterior
  collapse (`train.py`).
- Generation by sampling the latent prior and decoding autoregressively, scored
  with RDKit-based validity, uniqueness, and novelty (`metrics.py`).

## What it does not do

- It is a compact character-level model trained briefly on a CPU, not a tuned
  generator; validity is modest (see results) and improves with longer training,
  a larger model, and stronger KL control.
- It does not enforce chemical validity during decoding (no grammar or masking).
- No property-conditioned or reinforcement-learning-based optimization.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,chem]"
```

RDKit (the `chem` extra) is needed for the generation metrics; the models and
training loop run without it.

## Run

```bash
python scripts/download_data.py --outdir data
python scripts/train.py --data data/smiles.txt --epochs 30 --n-generate 1000
```

Fully offline on a tiny built-in corpus:

```bash
python scripts/train.py --builtin --epochs 30
```

`notebooks/demo.ipynb` is a short executed walkthrough.

## Results

Trained on 4188 public SMILES from the MoleculeNet Lipophilicity set (labels
ignored; used only as a SMILES corpus), 30 epochs, KL annealing to beta 0.1,
single CPU, seed 0. One thousand molecules were then sampled from the latent
prior. Produced by `scripts/train.py` in this repository.

| Metric     | Value | Meaning |
|------------|------:|---------|
| Validity   | 0.326 | fraction of samples RDKit can parse |
| Uniqueness | 0.988 | fraction of valid samples that are distinct |
| Novelty    | 0.981 | fraction of unique valid samples not in the training set |

The model samples diverse and almost entirely novel strings (uniqueness and
novelty near 1.0), but only about a third parse as valid molecules after this
short CPU run. That is the honest trade-off for a small character-level VAE
trained briefly: it has learned a lot of SMILES structure but not enough to keep
every sample syntactically and chemically valid. Longer training, a larger
latent and hidden size, and stronger KL scheduling all raise validity. Example
valid generations are written to `results/generated_sample.txt`.

## Layout

```
src/molgen/     tokenizer, data, models, train, metrics
scripts/        download_data.py, train.py
notebooks/      demo.ipynb (executed)
tests/          pytest suite for tokenizer, model shapes, and metrics
data/           gitignored; see data/README.md
```

## Tests

```bash
pytest -q
ruff check src tests scripts
```

## License

MIT, see [LICENSE](LICENSE).

## Author

Aamir Malik. [GitHub](https://github.com/aamirmalik-dr) ·
[LinkedIn](https://linkedin.com/in/dr-aamirmalik)
