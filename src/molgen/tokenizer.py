"""A character-level SMILES tokenizer with start, end, and pad tokens.

SMILES two-character elements such as ``Cl`` and ``Br`` are treated as single
tokens so the model never splits them; everything else is tokenized per
character. The vocabulary is built from the training corpus.
"""

from __future__ import annotations

import re

PAD, SOS, EOS = "<pad>", "<sos>", "<eos>"
_SPECIAL = [PAD, SOS, EOS]

# Multi-character atoms kept as single tokens, longest first.
_MULTI = ["Cl", "Br"]
_PATTERN = re.compile("|".join(_MULTI) + r"|.")


def atomize(smiles: str) -> list[str]:
    """Split a SMILES string into tokens, keeping Cl and Br whole."""
    return _PATTERN.findall(smiles)


class SmilesTokenizer:
    """Maps SMILES characters to integer ids and back."""

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def sos_id(self) -> int:
        return self.stoi[SOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def __len__(self) -> int:
        return len(self.stoi)

    def build(self, smiles: list[str]) -> SmilesTokenizer:
        """Build the vocabulary from a list of SMILES strings."""
        chars: set[str] = set()
        for s in smiles:
            chars.update(atomize(s))
        tokens = _SPECIAL + sorted(chars)
        self.stoi = {t: i for i, t in enumerate(tokens)}
        self.itos = {i: t for t, i in self.stoi.items()}
        return self

    def encode(self, smiles: str, max_len: int = 120) -> list[int]:
        """Encode a SMILES string to ``[SOS, ..., EOS]`` ids, truncated to max_len."""
        body = [self.stoi[c] for c in atomize(smiles) if c in self.stoi][: max_len - 2]
        return [self.sos_id, *body, self.eos_id]

    def decode(self, ids: list[int]) -> str:
        """Decode ids to a SMILES string, stopping at EOS and dropping specials."""
        out: list[str] = []
        for i in ids:
            tok = self.itos.get(int(i), "")
            if tok == EOS:
                break
            if tok in (SOS, PAD):
                continue
            out.append(tok)
        return "".join(out)
