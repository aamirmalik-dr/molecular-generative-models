"""SMILES-based molecular generative models.

A character-level SMILES tokenizer, a GRU sequence autoencoder and a variational
autoencoder with reparameterization and KL annealing, a training loop, and
metrics (validity, uniqueness, novelty) for the generated molecules. The demo
trains on a public SMILES corpus fetched by a download script.
"""

from molgen.data import BUILTIN_SMILES, SmilesDataset, load_smiles
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
]

__version__ = "0.1.0"
