# Model card: SMILES GRU-VAE

A small character-level variational autoencoder for SMILES strings. This card
records what the committed checkpoint is, how it was trained, what it measures,
and where it fails. All numbers were produced by `scripts/build_artifacts.py` in
this repository and are stored in `results/metrics.json`.

## Overview

- **Task**: unconditional generation of SMILES strings, and a continuous latent
  representation of molecules.
- **Architecture**: GRU encoder, Gaussian latent, GRU decoder. Embedding 48,
  hidden 128, latent 32. About 0.15 M parameters. Checkpoint `models/vae.pt` is
  roughly 0.6 MB.
- **Framework**: PyTorch, CPU only.
- **Checkpoint**: `models/vae.pt` bundles the weights and the tokenizer vocabulary
  so sampling runs with no corpus and no network.

## Training data

- 400 canonical SMILES carved from the public MoleculeNet Lipophilicity set
  (hosted by DeepChem). Only the SMILES strings are used; the lipophilicity labels
  are dropped. The molecules are filtered to a token length between 15 and 45 so a
  small character model has a fair chance of learning valid strings.
- The carving is reproducible with `scripts/make_sample.py` and the resulting
  sample is committed at `data/sample_smiles.txt`.

## Training procedure

- 80 epochs, Adam at 1e-3, batch size 64, gradient clipping at norm 5.
- KL annealing: linear ramp of the KL weight from 0 to 0.008 over 20 epochs.
- Word dropout: off for the committed checkpoint (see the method note for why it
  is available but unused here).
- Seed 0.

## Metrics (this session)

One thousand molecules were sampled from the latent prior at temperature 0.9 and
scored with RDKit, which parses each string to decide validity.

| Metric | Value | Meaning |
|--------|------:|---------|
| Validity | 0.150 | fraction of samples RDKit can parse as a molecule |
| Uniqueness | 0.980 | fraction of valid samples that are distinct after canonicalization |
| Novelty | 1.000 | fraction of unique valid samples not in the training set |
| Final reconstruction loss | 0.79 | per-character cross entropy at the last epoch |
| Final KL | 0.065 | KL to the prior at the last epoch |

Validity was checked with RDKit (version 2026.03.3 in this session). If RDKit is
not installed, the code falls back to a syntactic grammar check in
`src/molgen/grammar.py`, which is more permissive and reports a different, looser
validity. The metrics above are the RDKit numbers.

## Honest reading of the results

The headline is validity, and it is low. About one in seven sampled strings is a
parseable molecule. That is the expected outcome for a 0.15 M parameter character
model trained for 80 epochs on 400 molecules on a CPU. The model has clearly
learned a lot of SMILES structure, since many samples are close to valid and the
valid ones are drug-like, but it has not learned to close every ring and balance
every branch.

The other two numbers are near their ceilings and should be read with that in
mind. Uniqueness near 1.0 and novelty at 1.0 mean the model is not memorizing or
repeating the training set, but on their own they are easy to achieve: a model
that emits random noise also scores high on uniqueness and novelty. They are only
meaningful in combination with validity, which is the hard part.

The final KL of 0.065 is small but non-zero, so the latent is lightly used rather
than fully collapsed. This is why the latent-space map shows partial organization:
the first principal component of the latent means tracks molecular size and,
more weakly, Crippen logP, rather than being pure noise.

## Limitations and intended use

- This is a teaching and portfolio model, not a production molecule generator. Do
  not use its samples for any real chemistry decision.
- Validity is not enforced during decoding. There is no grammar mask, valence
  check, or reinforcement signal.
- Generation is unconditional. There is no property targeting or optimization.
- The latent organization is real but modest. It should be described as partial,
  not as clean disentanglement.
- Ways to raise validity, all outside the scope of this small demo: more training
  data, a larger model, longer training, a grammar-constrained or SELFIES
  representation that is valid by construction, and stronger latent regularization
  schedules.
