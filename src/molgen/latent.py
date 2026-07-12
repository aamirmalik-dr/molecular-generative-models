"""Latent-space analysis and the latent-space-map hero figure.

The variational autoencoder maps every SMILES string to a Gaussian latent
distribution. The mean of that distribution, ``mu``, is a fixed-length code for
the molecule. This module encodes a corpus to its latent means, projects them to
two dimensions with PCA, and draws a scatter colored by a per-molecule property.
The result, the latent-space map, shows whether the learned latent space is
organized by chemistry rather than being an unstructured blob.
"""

from __future__ import annotations

import numpy as np
import torch

from molgen.data import SmilesDataset, collate
from molgen.models import SmilesVAE
from molgen.tokenizer import SmilesTokenizer, atomize


@torch.no_grad()
def latent_means(
    model: SmilesVAE,
    smiles: list[str],
    tokenizer: SmilesTokenizer,
    max_len: int = 120,
    device: str = "cpu",
) -> np.ndarray:
    """Encode SMILES to their posterior latent means ``mu``.

    Args:
        model: A trained SmilesVAE.
        smiles: SMILES strings to encode.
        tokenizer: The tokenizer whose vocabulary the model was trained on.
        max_len: Maximum token length used when encoding.
        device: Torch device string.

    Returns:
        An array of shape ``(len(smiles), latent_dim)`` of latent means.
    """
    model.eval()
    model.to(device)
    dataset = SmilesDataset(smiles, tokenizer, max_len=max_len)
    ids, _ = collate([dataset[i] for i in range(len(dataset))], pad_id=tokenizer.pad_id)
    ids = ids.to(device)
    h = model.encode(ids)
    mu = model.to_mu(h)
    return mu.cpu().numpy()


def pca_project(codes: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project latent codes to ``n_components`` dimensions with plain PCA.

    Uses a centered SVD so there is no dependency beyond NumPy. Returns the
    projected coordinates of shape ``(n_samples, n_components)``.
    """
    centered = codes - codes.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:n_components].T


def smiles_property(smiles: str, name: str = "logp") -> float:
    """Return a scalar molecular property for coloring the latent map.

    ``logp`` uses RDKit's Crippen logP when RDKit is available and falls back to
    token length if it is not. ``length`` always returns the token count. The
    fallback is reported by :func:`property_label`.
    """
    if name == "length":
        return float(len(atomize(smiles)))
    if name == "logp":
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem import Crippen

            RDLogger.DisableLog("rdApp.*")
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return float(Crippen.MolLogP(mol))
        except ImportError:
            pass
        return float(len(atomize(smiles)))
    raise ValueError(f"unknown property: {name}")


def rdkit_available() -> bool:
    """True if RDKit can be imported."""
    try:
        import rdkit  # noqa: F401

        return True
    except ImportError:
        return False


def property_label(name: str) -> str:
    """Human-readable label for the color bar, honest about the fallback."""
    if name == "length":
        return "SMILES token length"
    if name == "logp":
        return "Crippen logP" if rdkit_available() else "SMILES token length (RDKit absent)"
    return name


def compute_properties(smiles: list[str], name: str = "logp") -> np.ndarray:
    """Vector of the chosen property over a list of SMILES."""
    return np.array([smiles_property(s, name) for s in smiles], dtype=float)


def latent_space_map(
    model: SmilesVAE,
    smiles: list[str],
    tokenizer: SmilesTokenizer,
    out_path: str,
    prop: str = "logp",
    device: str = "cpu",
) -> np.ndarray:
    """Draw the latent-space map and save it to ``out_path``.

    Encodes every SMILES to its latent mean, projects to two dimensions with PCA,
    and scatters the points colored by ``prop``. Returns the 2D coordinates.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    codes = latent_means(model, smiles, tokenizer, device=device)
    coords = pca_project(codes, 2)
    values = compute_properties(smiles, prop)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], c=values, cmap="viridis", s=26, alpha=0.85, edgecolors="none"
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(property_label(prop))
    ax.set_xlabel("latent PC1")
    ax.set_ylabel("latent PC2")
    ax.set_title(f"Latent-space map ({len(smiles)} molecules, colored by {property_label(prop)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return coords
