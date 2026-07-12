"""A lightweight SMILES-grammar validity check used when RDKit is absent.

This is a syntactic heuristic, not a chemistry engine. It checks the structural
rules a valid SMILES string must satisfy: balanced parentheses, balanced square
brackets, paired ring-bond digits, and only characters that appear in SMILES.
It cannot judge valence or aromaticity, so it is more permissive than RDKit and
will accept some strings RDKit would reject. When RDKit is installed, prefer
:mod:`molgen.metrics`, which parses each molecule for real. The functions here
exist so the metrics and the offline quickstart still run with no RDKit.
"""

from __future__ import annotations

import re

_ALLOWED = set("BCNOPSFIHcnops()[]=#+-@/\\.%0123456789lrBnaeigou")


def grammar_valid(smiles: str) -> bool:
    """Return True if ``smiles`` is syntactically well formed.

    This checks bracket balance and ring-closure pairing only; it does not verify
    chemistry. See the module docstring for the caveat.
    """
    if not smiles:
        return False
    if any(ch not in _ALLOWED for ch in smiles):
        return False

    depth = 0
    for ch in smiles:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    if depth != 0:
        return False

    if smiles.count("[") != smiles.count("]"):
        return False

    # Ring-bond digits must appear an even number of times (open then close).
    # Two-digit ring labels are written as %NN; strip those first.
    stripped = re.sub(r"%\d\d", "", smiles)
    digits = [ch for ch in stripped if ch.isdigit()]
    counts: dict[str, int] = {}
    for d in digits:
        counts[d] = counts.get(d, 0) + 1
    if any(v % 2 != 0 for v in counts.values()):
        return False

    return True


def grammar_metrics(generated: list[str], training: list[str]) -> dict[str, float]:
    """Validity, uniqueness, and novelty using the grammar check instead of RDKit.

    Validity is the fraction of generated strings that pass :func:`grammar_valid`.
    Uniqueness is over the valid strings, and novelty is the fraction of unique
    valid strings not present verbatim in the training set. Because there is no
    canonicalization, these are string-level, not molecule-level, comparisons.
    """
    valid = [s for s in generated if grammar_valid(s)]
    n_gen = len(generated)
    n_valid = len(valid)
    unique = set(valid)
    train_set = set(training)
    novel = [u for u in unique if u not in train_set]
    return {
        "n_generated": float(n_gen),
        "validity": n_valid / n_gen if n_gen else 0.0,
        "uniqueness": len(unique) / n_valid if n_valid else 0.0,
        "novelty": len(novel) / len(unique) if unique else 0.0,
    }
