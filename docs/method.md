# Method note: a character-level SMILES VAE

This note derives the objective the model optimizes and explains the two training
choices that matter most for a small SMILES VAE: KL annealing and word dropout.
It is written to be read alongside the code in `src/molgen/`.

## The model

A molecule is a SMILES string, a sequence of characters `x = (x_1, ..., x_T)`.
The generative story of the variational autoencoder is:

1. Draw a latent code `z` from a standard Gaussian prior `p(z) = N(0, I)`.
2. Decode `x` one character at a time with a GRU whose initial hidden state is a
   function of `z`, so `p_theta(x | z) = prod_t p_theta(x_t | x_{<t}, z)`.

The encoder is a separate GRU that reads `x` and outputs the parameters of an
approximate posterior `q_phi(z | x) = N(mu(x), diag(sigma^2(x)))`. In the code,
`SmilesVAE.encode` produces the GRU summary, `to_mu` and `to_logvar` produce the
posterior parameters, and `from_latent` maps a sampled `z` back to a decoder
hidden state.

## The evidence lower bound

We would like to maximize the marginal log likelihood of the data,
`log p_theta(x)`. It is intractable because it integrates over all `z`:

```
log p_theta(x) = log integral p_theta(x | z) p(z) dz
```

Introduce the approximate posterior `q_phi(z | x)` and multiply inside the log by
`q_phi(z | x) / q_phi(z | x)`:

```
log p_theta(x) = log E_{q_phi(z | x)} [ p_theta(x | z) p(z) / q_phi(z | x) ]
```

Jensen's inequality (the log of an expectation is at least the expectation of the
log) gives the evidence lower bound, the ELBO:

```
log p_theta(x) >= E_{q_phi(z | x)} [ log p_theta(x | z) ]
                  - KL( q_phi(z | x) || p(z) )
              =: ELBO(x)
```

The gap between the true log likelihood and the ELBO is exactly
`KL(q_phi(z | x) || p_theta(z | x))`, the divergence between the approximate and
the true posterior, which is non-negative. Maximizing the ELBO therefore both
raises a lower bound on the likelihood and tightens the posterior approximation.

The two ELBO terms have clear jobs:

- The **reconstruction term** `E_q [ log p_theta(x | z) ]` is the expected log
  likelihood of the characters given the code. With a categorical decoder this is
  the negative cross entropy of the next-character predictions, computed by
  `SmilesVAE.loss` as `F.cross_entropy` over the teacher-forced logits.
- The **KL term** `KL( q_phi(z | x) || p(z) )` pulls each posterior toward the
  prior. For two Gaussians with a unit-variance prior it has a closed form:

```
KL = -0.5 * sum_j ( 1 + log sigma_j^2 - mu_j^2 - sigma_j^2 )
```

which is the single line `kl = -0.5 * mean(sum(1 + logvar - mu^2 - exp(logvar)))`
in the code.

## The reparameterization trick

To backpropagate through the sampling of `z`, we do not sample `z` directly from
`q_phi`. We sample `eps` from a fixed `N(0, I)` and set `z = mu + sigma * eps`.
The randomness is now in `eps`, which carries no parameters, so gradients flow
through `mu` and `sigma`. This is `SmilesVAE.reparameterize`.

## KL annealing

The optimization has a failure mode called posterior collapse. Early in training
the decoder cannot yet use `z`, so the cheapest way to lower the loss is to drive
the KL term to zero by making `q_phi(z | x)` equal to the prior for every `x`.
Once that happens the latent carries no information and the encoder never
recovers.

KL annealing weights the KL term by a coefficient `beta` that starts near zero and
ramps up over the first several epochs:

```
loss = reconstruction + beta(epoch) * KL
```

Early on, `beta` is small, so the model is free to encode information in `z` and
learn a useful decoder. As `beta` rises to its target, the latent is gradually
regularized toward the prior. In this repository `Trainer._beta` implements a
linear ramp from 0 to `max_beta` over `kl_anneal_epochs`. The committed model uses
a deliberately small `max_beta` of 0.008, because on a 400 molecule corpus a
larger weight collapses the latent almost completely.

## Word dropout

Annealing alone is often not enough when the decoder is a strong autoregressive
model, because teacher forcing lets it predict each character from the previous
true character and ignore `z`. Word dropout (Bowman et al., 2016) randomly
replaces a fraction of the decoder input characters with a blank (the pad
embedding) during training. This weakens teacher forcing and forces the decoder to
rely on the latent code, which keeps the KL term positive and the latent
informative. It is implemented in `_Seq2Seq.decode_teacher` and exposed through
`Trainer(word_dropout=...)`.

There is a trade-off. Word dropout produces a more structured, more spread latent
space (visible in the latent-space map), but it lowers reconstruction quality and,
on this tiny corpus, lowers the validity of molecules sampled from the prior. The
committed checkpoint was trained with word dropout off because that gave the best
prior-sampling validity here; the feature is kept, tested, and documented because
it is the right tool when the latent collapses. Both behaviors were observed
directly during tuning and are reported honestly in the model card.

## Generation

To sample a new molecule we draw `z` from the prior, map it to the decoder hidden
state, and decode autoregressively, feeding each sampled character back in until
the end token or a length cap. `SmilesVAE.generate` does this with temperature
controlled multinomial sampling. Nothing constrains the output to be a valid
SMILES string, so validity is an emergent property of how well the decoder learned
the grammar, not something enforced.

## Reference

Bowman, Vilnis, Vinyals, Dai, Jozefowicz, Bengio. Generating Sentences from a
Continuous Space. CoNLL 2016.
