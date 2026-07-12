from pathlib import Path

from molgen.data import BUILTIN_SMILES, SmilesDataset, collate, load_smiles
from molgen.grammar import grammar_metrics, grammar_valid
from molgen.latent import compute_properties, latent_means, pca_project
from molgen.models import SmilesAE, SmilesVAE
from molgen.tokenizer import SmilesTokenizer, atomize
from molgen.train import Trainer, set_seed


def test_atomize_keeps_multichar_atoms():
    assert atomize("ClCCBr") == ["Cl", "C", "C", "Br"]
    assert atomize("CCO") == ["C", "C", "O"]


def test_tokenizer_roundtrip():
    tok = SmilesTokenizer().build(BUILTIN_SMILES)
    ids = tok.encode("CCO")
    assert ids[0] == tok.sos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == "CCO"


def test_load_smiles_builtin_filtered():
    smiles = load_smiles(None)
    assert len(smiles) == len(BUILTIN_SMILES)
    assert all(isinstance(s, str) for s in smiles)


def test_collate_pads():
    tok = SmilesTokenizer().build(BUILTIN_SMILES)
    ds = SmilesDataset(["CCO", "c1ccccc1"], tok)
    ids, lengths = collate([ds[0], ds[1]], pad_id=tok.pad_id)
    assert ids.shape[0] == 2
    assert ids.shape[1] == max(lengths).item()


def test_ae_loss_scalar():
    tok = SmilesTokenizer().build(BUILTIN_SMILES)
    ds = SmilesDataset(BUILTIN_SMILES, tok)
    ids, _ = collate([ds[i] for i in range(4)], pad_id=tok.pad_id)
    model = SmilesAE(len(tok), embed_dim=16, hidden_dim=32, pad_id=tok.pad_id)
    assert model.loss(ids).ndim == 0


def test_vae_loss_and_generation():
    tok = SmilesTokenizer().build(BUILTIN_SMILES)
    ds = SmilesDataset(BUILTIN_SMILES, tok)
    ids, _ = collate([ds[i] for i in range(4)], pad_id=tok.pad_id)
    model = SmilesVAE(len(tok), embed_dim=16, hidden_dim=32, latent_dim=8, pad_id=tok.pad_id)
    total, recon, kl = model.loss(ids, beta=0.5)
    assert total.ndim == 0 and kl.item() >= 0
    gen = model.generate(5, tok.sos_id, tok.eos_id, max_len=40)
    assert len(gen) == 5
    assert all(isinstance(tok.decode(g), str) for g in gen)


def test_vae_trainer_reduces_loss():
    set_seed(0)
    tok = SmilesTokenizer().build(BUILTIN_SMILES)
    ds = SmilesDataset(BUILTIN_SMILES * 5, tok)
    model = SmilesVAE(len(tok), embed_dim=16, hidden_dim=32, latent_dim=8, pad_id=tok.pad_id)
    trainer = Trainer(model, pad_id=tok.pad_id, lr=1e-3, kl_anneal_epochs=5, max_beta=0.1)
    trainer.fit(ds, epochs=8, batch_size=16, verbose=False)
    assert trainer.history["recon"][-1] < trainer.history["recon"][0]


def test_metrics_optional_rdkit():
    import pytest

    rdkit = pytest.importorskip("rdkit")  # noqa: F841
    from molgen.metrics import generation_metrics

    m = generation_metrics(["CCO", "not_a_molecule", "c1ccccc1"], ["CCO"])
    assert m["validity"] == 2 / 3
    assert 0.0 <= m["novelty"] <= 1.0


def test_grammar_valid_rules():
    assert grammar_valid("CCO")
    assert grammar_valid("c1ccccc1")
    assert grammar_valid("C[C@H](N)O")
    assert not grammar_valid("")  # empty
    assert not grammar_valid("CC(C")  # unbalanced parenthesis
    assert not grammar_valid("c1ccccc")  # unpaired ring digit


def test_grammar_metrics_no_rdkit():
    m = grammar_metrics(["CCO", "CC(C", "c1ccccc1"], ["CCO"])
    assert m["validity"] == 2 / 3
    assert 0.0 <= m["novelty"] <= 1.0


def test_word_dropout_keeps_loss_scalar():
    tok = SmilesTokenizer().build(BUILTIN_SMILES)
    ds = SmilesDataset(BUILTIN_SMILES, tok)
    ids, _ = collate([ds[i] for i in range(4)], pad_id=tok.pad_id)
    model = SmilesVAE(len(tok), embed_dim=16, hidden_dim=32, latent_dim=8, pad_id=tok.pad_id)
    model.train()
    total, recon, kl = model.loss(ids, beta=0.1, word_dropout=0.5)
    assert total.ndim == 0 and recon.ndim == 0 and kl.item() >= 0


def test_latent_means_and_pca_shapes():
    tok = SmilesTokenizer().build(BUILTIN_SMILES)
    model = SmilesVAE(len(tok), embed_dim=16, hidden_dim=32, latent_dim=8, pad_id=tok.pad_id)
    mu = latent_means(model, BUILTIN_SMILES, tok)
    assert mu.shape == (len(BUILTIN_SMILES), 8)
    coords = pca_project(mu, 2)
    assert coords.shape == (len(BUILTIN_SMILES), 2)


def test_compute_properties_length():
    vals = compute_properties(["CCO", "c1ccccc1"], "length")
    assert vals.tolist() == [3.0, 8.0]


def test_checkpoint_roundtrip(tmp_path: Path):
    from molgen.checkpoint import load_vae, save_vae

    tok = SmilesTokenizer().build(BUILTIN_SMILES)
    model = SmilesVAE(len(tok), embed_dim=16, hidden_dim=32, latent_dim=8, pad_id=tok.pad_id)
    ckpt = tmp_path / "vae.pt"
    save_vae(ckpt, model, tok)
    model2, tok2 = load_vae(ckpt)
    assert len(tok2) == len(tok)
    assert tok2.decode(tok2.encode("CCO")) == "CCO"
    gen = model2.generate(3, tok2.sos_id, tok2.eos_id, max_len=30)
    assert len(gen) == 3
