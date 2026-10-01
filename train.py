"""Deterministic demo of self-repulsive sampling. Not a training run.

The logit table is fixed. Nothing is learned. No benchmark is scored.
"""

import jax
import jax.numpy as jnp

from self_repulsion import self_repulsive_sample

# (L, V). Position 0 is a near tie so the tau = 0 cascade splits paths.
LOGITS = jnp.array(
    [
        [1.2, 1.0, 0.2, -0.4],
        [0.3, 0.3, 0.9, 0.1],
        [0.0, 0.5, 0.5, 0.1],
    ],
    dtype=jnp.float64,
)

N_PATHS = 4
SEQ_LEN = 3
VOCAB = 4
N_STEPS = 3


def model(canvases):
    return jnp.broadcast_to(LOGITS, (canvases.shape[0], SEQ_LEN, VOCAB))


def answer(canvas):
    return tuple(int(token) for token in canvas)


def counts_at(canvases, position):
    counts = [0] * VOCAB
    for token in canvases[:, position]:
        counts[int(token)] += 1
    return counts


def main():
    canvases, vote = self_repulsive_sample(
        model,
        n_paths=N_PATHS,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB,
        n_steps=N_STEPS,
        tau=0.0,
        lambda_pen=1.0,
        rho=1.0,
        answer_fn=answer,
    )
    print("tau=0 position=0 counts", counts_at(canvases, 0))
    print("tau=0 plurality", vote)

    canvases, vote = self_repulsive_sample(
        model,
        n_paths=N_PATHS,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB,
        n_steps=N_STEPS,
        tau=0.5,
        lambda_pen=1.0,
        rho=1.0,
        key=jax.random.PRNGKey(0),
        answer_fn=answer,
    )
    print("tau=0.5 position=0 counts", counts_at(canvases, 0))
    print("tau=0.5 plurality", vote)


if __name__ == "__main__":
    main()
