"""GRU sequence autoencoder and variational autoencoder for SMILES.

Both models share an embedding, a GRU encoder, and a GRU decoder. The
autoencoder passes the encoder's final hidden state straight to the decoder; the
variational autoencoder maps it to a Gaussian latent, samples with the
reparameterization trick, and adds a KL term to the loss. Generation samples a
latent vector and decodes autoregressively.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Seq2Seq(nn.Module):
    """Shared embedding, encoder GRU, and decoder GRU."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_id: int) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        _, h = self.encoder(self.embedding(ids))
        return h[-1]  # (batch, hidden)

    def decode_teacher(
        self, ids: torch.Tensor, h0: torch.Tensor, word_dropout: float = 0.0
    ) -> torch.Tensor:
        """Teacher-forced decode; predicts each next token from the previous one.

        ``word_dropout`` randomly replaces decoder input tokens with the pad
        embedding (a blank input) during training. This weakens the teacher
        forcing signal and forces the decoder to lean on the latent code, which
        counteracts posterior collapse (Bowman et al., 2016). It does not change
        the prediction targets.
        """
        inp = ids[:, :-1]
        if word_dropout > 0.0 and self.training:
            mask = torch.rand_like(inp, dtype=torch.float) < word_dropout
            inp = inp.masked_fill(mask, self.pad_id)
        out, _ = self.decoder(self.embedding(inp), h0.unsqueeze(0))
        return self.output(out)  # (batch, seq-1, vocab)


class SmilesAE(_Seq2Seq):
    """A plain GRU sequence autoencoder (no latent regularization)."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.encode(ids)
        return self.decode_teacher(ids, h)

    def loss(self, ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(ids)
        target = ids[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=self.pad_id
        )


class SmilesVAE(_Seq2Seq):
    """A GRU variational autoencoder with a Gaussian latent space."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        pad_id: int = 0,
    ) -> None:
        super().__init__(vocab_size, embed_dim, hidden_dim, pad_id)
        self.latent_dim = latent_dim
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(
        self, ids: torch.Tensor, word_dropout: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(ids)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        z = self.reparameterize(mu, logvar)
        h0 = torch.tanh(self.from_latent(z))
        logits = self.decode_teacher(ids, h0, word_dropout=word_dropout)
        return logits, mu, logvar

    def loss(
        self, ids: torch.Tensor, beta: float = 1.0, word_dropout: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total loss and its reconstruction and KL parts."""
        logits, mu, logvar = self.forward(ids, word_dropout=word_dropout)
        target = ids[:, 1:]
        recon = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=self.pad_id
        )
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return recon + beta * kl, recon, kl

    @torch.no_grad()
    def generate(
        self,
        n: int,
        sos_id: int,
        eos_id: int,
        max_len: int = 120,
        z: torch.Tensor | None = None,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> list[list[int]]:
        """Sample ``n`` molecules from the prior and decode them autoregressively."""
        self.eval()
        if z is None:
            z = torch.randn(n, self.latent_dim, device=device)
        h = torch.tanh(self.from_latent(z)).unsqueeze(0)
        tokens = torch.full((n, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        seqs: list[list[int]] = [[] for _ in range(n)]
        cur = tokens
        for _ in range(max_len):
            out, h = self.decoder(self.embedding(cur), h)
            logits = self.output(out[:, -1]) / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)  # (n, 1)
            for i in range(n):
                if not finished[i]:
                    tok = int(nxt[i])
                    seqs[i].append(tok)
                    if tok == eos_id:
                        finished[i] = True
            cur = nxt
            if bool(finished.all()):
                break
        return seqs
