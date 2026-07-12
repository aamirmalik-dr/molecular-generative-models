"""Save and load a trained SmilesVAE together with its tokenizer.

A checkpoint bundles the model weights, the model hyperparameters, and the
tokenizer vocabulary so a saved VAE can be reloaded and sampled from with no
access to the training corpus. This is what makes the offline quickstart work.
"""

from __future__ import annotations

from pathlib import Path

import torch

from molgen.models import SmilesVAE
from molgen.tokenizer import SmilesTokenizer


def save_vae(path: str | Path, model: SmilesVAE, tokenizer: SmilesTokenizer) -> None:
    """Write the model weights, config, and tokenizer vocabulary to ``path``."""
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size": model.embedding.num_embeddings,
            "embed_dim": model.embedding.embedding_dim,
            "hidden_dim": model.hidden_dim,
            "latent_dim": model.latent_dim,
            "pad_id": model.pad_id,
        },
        "stoi": tokenizer.stoi,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_vae(path: str | Path, device: str = "cpu") -> tuple[SmilesVAE, SmilesTokenizer]:
    """Reload a SmilesVAE and its tokenizer from a checkpoint written by :func:`save_vae`."""
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = SmilesVAE(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        pad_id=cfg["pad_id"],
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    tokenizer = SmilesTokenizer()
    tokenizer.stoi = dict(payload["stoi"])
    tokenizer.itos = {i: t for t, i in tokenizer.stoi.items()}
    return model, tokenizer
