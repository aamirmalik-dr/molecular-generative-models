"""Train the SMILES VAE on the committed sample and build all repo artifacts.

Produces, under ``results/`` and ``models/``:
  - ``models/vae.pt``            the pretrained VAE checkpoint (weights + vocab)
  - ``results/metrics.json``     validity, uniqueness, novelty, and final losses
  - ``results/latent_space_map.png``  the hero figure
  - ``results/vae_losses.png``   reconstruction and KL curves
  - ``results/generated_sample.txt``  a handful of valid generated molecules

This is a small model trained briefly on a CPU. The validity number it reports
is modest and is meant to be read honestly, not as a benchmark.

Usage:
    python scripts/build_artifacts.py --data data/sample_smiles.txt --epochs 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from molgen.checkpoint import save_vae
from molgen.data import SmilesDataset, load_smiles
from molgen.latent import latent_space_map, rdkit_available
from molgen.models import SmilesVAE
from molgen.tokenizer import SmilesTokenizer
from molgen.train import Trainer, set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sample_smiles.txt")
    parser.add_argument("--builtin", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n-generate", type=int, default=1000)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--max-beta", type=float, default=0.008)
    parser.add_argument("--kl-anneal-epochs", type=int, default=20)
    parser.add_argument("--word-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", default="results")
    parser.add_argument("--models", default="models")
    parser.add_argument("--prop", default="logp", choices=["logp", "length"])
    args = parser.parse_args()

    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    models = Path(args.models)
    models.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    smiles = load_smiles(None if args.builtin else args.data)
    print(f"loaded {len(smiles)} SMILES")

    tok = SmilesTokenizer().build(smiles)
    dataset = SmilesDataset(smiles, tok)
    model = SmilesVAE(
        len(tok),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        pad_id=tok.pad_id,
    )
    trainer = Trainer(
        model,
        pad_id=tok.pad_id,
        lr=1e-3,
        max_beta=args.max_beta,
        kl_anneal_epochs=args.kl_anneal_epochs,
        word_dropout=args.word_dropout,
    )
    trainer.fit(dataset, epochs=args.epochs, batch_size=64)

    # Save the pretrained VAE so sampling runs offline.
    ckpt = models / "vae.pt"
    save_vae(ckpt, model, tok)
    print(f"saved checkpoint to {ckpt} ({ckpt.stat().st_size / 1024:.0f} KB)")

    # Generate and score.
    gen_ids = model.generate(args.n_generate, tok.sos_id, tok.eos_id, temperature=0.9)
    gen_smiles = [tok.decode(ids) for ids in gen_ids]

    checker = "rdkit" if rdkit_available() else "grammar"
    if rdkit_available():
        from molgen.metrics import canonical, generation_metrics

        metrics = generation_metrics(gen_smiles, smiles)
        valid = [c for c in (canonical(s) for s in gen_smiles) if c]
    else:
        from molgen.grammar import grammar_metrics, grammar_valid

        metrics = grammar_metrics(gen_smiles, smiles)
        valid = [s for s in gen_smiles if grammar_valid(s)]

    metrics["validity_check"] = checker
    metrics["final_recon"] = trainer.history["recon"][-1]
    metrics["final_kl"] = trainer.history["kl"][-1]
    metrics["epochs"] = args.epochs
    metrics["n_training"] = len(smiles)

    print(f"\nvalidity checked with: {checker}")
    for k in ("validity", "uniqueness", "novelty"):
        print(f"  {k:<12} {metrics[k]:.4f}")

    (results / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # Hero figure: latent-space map colored by a molecular property.
    latent_space_map(model, smiles, tok, str(results / "latent_space_map.png"), prop=args.prop)
    print(f"wrote {results / 'latent_space_map.png'}")

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
    plt.savefig(results / "vae_losses.png", dpi=120)
    plt.close()

    (results / "generated_sample.txt").write_text("\n".join(valid[:50]) + "\n", encoding="utf-8")
    print("\nExample valid generated SMILES:")
    for s in valid[:8]:
        print("  ", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
