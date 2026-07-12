"""Validity, uniqueness, and novelty metrics for generated SMILES.

Validity uses RDKit to parse each generated string. RDKit is an optional
dependency (the ``chem`` extra); the functions raise a clear error if it is not
installed.
"""

from __future__ import annotations


def _require_rdkit():
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        return Chem
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "RDKit is required for generation metrics. Install it with "
            "`pip install rdkit` or `pip install -e '.[chem]'`."
        ) from exc


def canonical(smiles: str) -> str | None:
    """Return the RDKit canonical SMILES, or None if the string is invalid."""
    chem = _require_rdkit()
    mol = chem.MolFromSmiles(smiles)
    return None if mol is None else chem.MolToSmiles(mol)


def generation_metrics(generated: list[str], training: list[str]) -> dict[str, float]:
    """Compute validity, uniqueness, and novelty for generated SMILES.

    Args:
        generated: Raw generated SMILES strings.
        training: The training SMILES, used as the reference for novelty.

    Returns:
        A dict with ``n_generated``, ``validity``, ``uniqueness``, and
        ``novelty``. Uniqueness is over valid molecules; novelty is the fraction
        of unique valid molecules not present (canonically) in the training set.
    """
    chem = _require_rdkit()

    valid_canon: list[str] = []
    for s in generated:
        mol = chem.MolFromSmiles(s)
        if mol is not None:
            valid_canon.append(chem.MolToSmiles(mol))

    n_gen = len(generated)
    n_valid = len(valid_canon)
    unique = set(valid_canon)

    train_canon = set()
    for s in training:
        mol = chem.MolFromSmiles(s)
        if mol is not None:
            train_canon.add(chem.MolToSmiles(mol))

    novel = [u for u in unique if u not in train_canon]

    return {
        "n_generated": float(n_gen),
        "validity": n_valid / n_gen if n_gen else 0.0,
        "uniqueness": len(unique) / n_valid if n_valid else 0.0,
        "novelty": len(novel) / len(unique) if unique else 0.0,
    }
