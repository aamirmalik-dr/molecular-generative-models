"""Training loop for the SMILES autoencoder and variational autoencoder."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from molgen.data import SmilesDataset, collate
from molgen.models import SmilesAE, SmilesVAE


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Trainer:
    """Trains a SmilesAE or SmilesVAE with Adam and optional KL annealing."""

    model: SmilesAE | SmilesVAE
    pad_id: int
    lr: float = 1e-3
    device: str = "cpu"
    kl_anneal_epochs: int = 10
    max_beta: float = 0.1
    word_dropout: float = 0.0
    history: dict[str, list[float]] = field(
        default_factory=lambda: {"loss": [], "recon": [], "kl": []}
    )

    def _loader(self, dataset: SmilesDataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=partial(collate, pad_id=self.pad_id),
        )

    def _beta(self, epoch: int) -> float:
        if self.kl_anneal_epochs <= 0:
            return self.max_beta
        return self.max_beta * min(1.0, (epoch + 1) / self.kl_anneal_epochs)

    def fit(
        self, dataset: SmilesDataset, epochs: int = 30, batch_size: int = 128, verbose: bool = True
    ) -> Trainer:
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loader = self._loader(dataset, batch_size, shuffle=True)
        is_vae = isinstance(self.model, SmilesVAE)
        for epoch in range(epochs):
            self.model.train()
            beta = self._beta(epoch)
            tot, rec, kld, n = 0.0, 0.0, 0.0, 0
            for ids, _ in loader:
                ids = ids.to(self.device)
                opt.zero_grad()
                if is_vae:
                    loss, recon, kl = self.model.loss(
                        ids, beta=beta, word_dropout=self.word_dropout
                    )
                else:
                    loss = self.model.loss(ids)
                    recon, kl = loss, torch.zeros(())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                bs = ids.shape[0]
                tot += loss.item() * bs
                rec += recon.item() * bs
                kld += kl.item() * bs
                n += bs
            self.history["loss"].append(tot / n)
            self.history["recon"].append(rec / n)
            self.history["kl"].append(kld / n)
            if verbose:
                print(
                    f"epoch {epoch + 1:3d}  loss={tot / n:.4f}  recon={rec / n:.4f}  "
                    f"kl={kld / n:.4f}  beta={beta:.3f}"
                )
        return self
