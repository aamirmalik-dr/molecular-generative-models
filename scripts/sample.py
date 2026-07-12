"""Sample molecules from the committed pretrained VAE. Runs fully offline.

Loads ``models/vae.pt`` (weights plus tokenizer vocabulary), samples the latent
prior, decodes autoregressively, and prints the generated SMILES together with
their validity. No network and no training corpus are needed.

Usage:
    python scripts/sample.py --n 20
    python scripts/sample.py --n 500 --only-valid
"""

from __future__ import annotations

import argparse

from molgen.checkpoint import load_vae


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/vae.pt")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-valid", action="store_true", help="print only valid molecules")
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)

    model, tok = load_vae(args.checkpoint)
    gen_ids = model.generate(args.n, tok.sos_id, tok.eos_id, temperature=args.temperature)
    gen = [tok.decode(ids) for ids in gen_ids]

    try:
        from molgen.metrics import canonical

        def check(s: str) -> str | None:
            return canonical(s)

        checker = "RDKit"
    except ImportError:
        from molgen.grammar import grammar_valid

        def check(s: str) -> str | None:
            return s if grammar_valid(s) else None

        checker = "grammar heuristic"

    n_valid = 0
    print(f"sampling {args.n} molecules from {args.checkpoint} (validity via {checker})\n")
    for s in gen:
        c = check(s)
        ok = c is not None
        n_valid += int(ok)
        if args.only_valid and not ok:
            continue
        flag = "valid  " if ok else "invalid"
        print(f"  [{flag}] {s}")

    print(f"\n{n_valid}/{args.n} valid ({n_valid / args.n:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
