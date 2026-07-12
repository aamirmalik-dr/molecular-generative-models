"""SMILES corpus loading, a torch Dataset, and batching.

The download script fetches a public SMILES corpus. For offline tests and CI a
small built-in set of valid SMILES is provided.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from molgen.tokenizer import SmilesTokenizer, atomize

# A small set of valid drug-like SMILES for offline tests and demos.
BUILTIN_SMILES = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CCN(CC)CC",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(=O)Oc1ccccc1C(=O)O",
    "C1CCCCC1",
    "CCOC(=O)C",
    "OCC(O)CO",
    "Cc1ccccc1",
    "ClCCl",
    "BrCCBr",
    "CC(N)C(=O)O",
    "c1ccncc1",
    "O=C(O)c1ccccc1",
    "CCCCCCO",
    "CC(C)O",
    "NCCO",
    "c1ccc2ccccc2c1",
]


def load_smiles(path: str | None = None, max_len: int = 120) -> list[str]:
    """Load SMILES from a text file (one per line), or the built-in set.

    Args:
        path: Path to a newline-delimited SMILES file, or None for the built-in set.
        max_len: Drop SMILES whose token length exceeds this.

    Returns:
        A list of SMILES strings.
    """
    if path is None:
        smiles = list(BUILTIN_SMILES)
    else:
        text = Path(path).read_text(encoding="utf-8")
        smiles = [line.strip() for line in text.splitlines() if line.strip()]
    return [s for s in smiles if 0 < len(atomize(s)) <= max_len - 2]


class SmilesDataset(Dataset):
    """Encodes SMILES strings to padded id sequences for the autoencoder."""

    def __init__(self, smiles: list[str], tokenizer: SmilesTokenizer, max_len: int = 120) -> None:
        self.smiles = smiles
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encoded = [tokenizer.encode(s, max_len=max_len) for s in smiles]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> list[int]:
        return self.encoded[idx]


def collate(batch: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of id sequences; return ids and true lengths."""
    lengths = [len(seq) for seq in batch]
    max_len = max(lengths)
    ids = np.full((len(batch), max_len), pad_id, dtype=np.int64)
    for i, seq in enumerate(batch):
        ids[i, : len(seq)] = seq
    return torch.from_numpy(ids), torch.tensor(lengths, dtype=torch.long)
