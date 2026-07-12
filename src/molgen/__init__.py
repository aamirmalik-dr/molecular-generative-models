"""SMILES-based molecular generative models.

A character-level SMILES tokenizer, a GRU sequence autoencoder and a variational
autoencoder with reparameterization and KL annealing, a training loop, and
metrics (validity, uniqueness, novelty) for the generated molecules. The demo
trains on a public SMILES corpus fetched by a download script.
"""

from molgen.checkpoint import load_vae, save_vae
from molgen.data import BUILTIN_SMILES, SmilesDataset, load_smiles
from molgen.grammar import grammar_metrics, grammar_valid
from molgen.latent import (
    compute_properties,
    latent_means,
    latent_space_map,
    pca_project,
)
from molgen.metrics import generation_metrics
from molgen.models import SmilesAE, SmilesVAE
from molgen.tokenizer import SmilesTokenizer
from molgen.train import Trainer, set_seed

__all__ = [
    "SmilesTokenizer",
    "SmilesDataset",
    "load_smiles",
    "BUILTIN_SMILES",
    "SmilesAE",
    "SmilesVAE",
    "Trainer",
    "set_seed",
    "generation_metrics",
    "grammar_valid",
    "grammar_metrics",
    "latent_means",
    "pca_project",
    "compute_properties",
    "latent_space_map",
    "save_vae",
    "load_vae",
]

__version__ = "0.1.0"
