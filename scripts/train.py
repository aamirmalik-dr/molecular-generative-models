"""Train a SMILES VAE, generate molecules, and report generation metrics.

Usage:
    python scripts/train.py --data data/smiles.txt --epochs 40
    python scripts/train.py --builtin --epochs 40      # tiny offline demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from molgen.data import SmilesDataset, load_smiles
from molgen.metrics import generation_metrics
from molgen.models import SmilesVAE
from molgen.tokenizer import SmilesTokenizer
from molgen.train import Trainer, set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/smiles.txt")
    parser.add_argument("--builtin", action="store_true", help="use the built-in tiny corpus")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--n-generate", type=int, default=1000)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(0)
    smiles = load_smiles(None if args.builtin else args.data)
    print(f"loaded {len(smiles)} SMILES")

    tok = SmilesTokenizer().build(smiles)
    dataset = SmilesDataset(smiles, tok)
    model = SmilesVAE(len(tok), pad_id=tok.pad_id)
    trainer = Trainer(model, pad_id=tok.pad_id, lr=1e-3, max_beta=0.1, kl_anneal_epochs=15)
    trainer.fit(dataset, epochs=args.epochs, batch_size=128)

    # Generate and score.
    gen_ids = model.generate(args.n_generate, tok.sos_id, tok.eos_id, temperature=0.9)
    gen_smiles = [tok.decode(ids) for ids in gen_ids]
    metrics = generation_metrics(gen_smiles, smiles)
    print("\nGeneration metrics on {} samples:".format(int(metrics["n_generated"])))
    for k in ("validity", "uniqueness", "novelty"):
        print(f"  {k:<12} {metrics[k]:.4f}")

    # Show a few valid generated molecules.
    from molgen.metrics import canonical

    valid = [c for c in (canonical(s) for s in gen_smiles) if c]
    print("\nExample valid generated SMILES:")
    for s in valid[:10]:
        print("  ", s)

    # Loss curves.
    ep = range(1, len(trainer.history["loss"]) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(ep, trainer.history["recon"], label="reconstruction")
    plt.plot(ep, trainer.history["kl"], label="KL")
    plt.plot(ep, trainer.history["loss"], label="total (recon + beta*KL)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("SMILES VAE training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "vae_losses.png", dpi=120)
    plt.close()

    (out_dir / "generated_sample.txt").write_text("\n".join(valid[:50]) + "\n", encoding="utf-8")
    print(f"\nWrote figures and samples to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
