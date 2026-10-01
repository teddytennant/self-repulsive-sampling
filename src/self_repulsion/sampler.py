"""Self-repulsive sampling (Algorithm 1), training-free.

K canvases, vocabulary size V, sequence length L. The mask token id is V.
Logits at a position have shape (V,) and do not score the mask.

Gate g = lambda_pen / (K - 1). At a penalized step the paths share one batched
forward, then commit in order k = 0 .. K-1. Path k sees counts of tokens
already written at that position by paths that have committed (previous steps,
any path) and by paths 0 .. k-1 in this step. Paths that have not committed
yet are still mask and are not counted.

Penalty, tau > 0: logit'_v = logit_v - (g / tau) * c_v, then sample
softmax(logit'). Penalty, tau = 0: v = argmax_v (logit_v - g * c_v), smallest
index on ties. No softmax is formed for that choice. Confidence for the commit
schedule is the untilted softmax(logit) probability of the selected token.

Linear schedule: after step t (0-based) of T, round((t+1)/T * L) positions
are non-mask. Among still-masked positions, keep the ones with highest mean
untilted confidence. Writes at positions that are not kept stay mask and do
not count as committed. Positions already committed stay fixed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def gate(lambda_pen, n_paths):
    """g = lambda_pen / (K - 1). Requires K >= 2."""
    n_paths = int(n_paths)
    if n_paths < 2:
        raise ValueError("K must be >= 2")
    return float(lambda_pen) / (n_paths - 1)


def integer_objective(counts, logits, g):
    """sum_v c_v * logit_v - g * sum_v binom(c_v, 2).

    binom(c, 2) = c * (c - 1) / 2. The k-th copy of token v (0-based) has
    marginal score logit_v - k * g.
    """
    counts = jnp.asarray(counts, dtype=jnp.float64)
    logits = jnp.asarray(logits, dtype=jnp.float64)
    pairs = counts * (counts - 1.0) / 2.0
    return jnp.sum(counts * logits) - float(g) * jnp.sum(pairs)


def sparsemax(z):
    """Euclidean projection of a 1D vector onto the probability simplex.

    Martins and Astudillo, 2016. Support size k is the largest k with
    1 + k * z_(k) > sum_{j=1..k} z_(j), z sorted descending. Then
    p_i = max(z_i - tau, 0) with tau = (sum_{j=1..k} z_(j) - 1) / k.
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    if z.ndim != 1:
        raise ValueError("sparsemax expects a 1D vector")
    n = int(z.shape[0])
    if n == 0:
        return z
    z_sorted = jnp.flip(jnp.sort(z))
    cssv = jnp.cumsum(z_sorted)
    ks = jnp.arange(1, n + 1, dtype=jnp.float64)
    cond = (1.0 + ks * z_sorted) > cssv
    # index of the largest valid k; k = 1 is always valid for a finite vector
    best = jnp.max(jnp.where(cond, jnp.arange(n), 0))
    tau = (cssv[best] - 1.0) / (best + 1.0)
    return jnp.maximum(z - tau, 0.0)


def fractional_counts(logits, n_paths, g):
    """Continuous relaxation c* of the integer allocation.

    c*_v = max(0, (logit_v - mu) / g) with mu chosen so sum_v c*_v = K.
    This is K * sparsemax(logit / (g * K)). Requires g > 0.
    """
    logits = jnp.asarray(logits, dtype=jnp.float64)
    if logits.ndim != 1:
        raise ValueError("logits must be 1D")
    g = float(g)
    if g <= 0.0:
        raise ValueError("g must be > 0")
    n_paths = int(n_paths)
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    n = int(logits.shape[0])
    if n == 0:
        raise ValueError("logits must be non-empty")
    # Largest support of the top-r logits with logit_(r) > mu_r,
    # mu_r = (sum of those logits - g * K) / r.
    order = jnp.argsort(-logits)
    sorted_logits = logits[order]
    best_mu = None
    for r in range(1, n + 1):
        mu = (float(jnp.sum(sorted_logits[:r])) - g * n_paths) / r
        if float(sorted_logits[r - 1]) > mu:
            best_mu = mu
    if best_mu is None:
        best_mu = float(sorted_logits[0]) - g * n_paths
    return jnp.maximum(0.0, (logits - best_mu) / g)


def cascade_allocate(logits, n_paths, g):
    """Tau = 0 cascade at one position.

    Returns (tokens, counts). tokens[k] is the token path k commits.
    Path k sees counts from paths 0 .. k-1 only. Tie break: smallest index.
    counts sums to n_paths.
    """
    logits = jnp.asarray(logits, dtype=jnp.float64)
    if logits.ndim != 1:
        raise ValueError("logits must be 1D")
    n_paths = int(n_paths)
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    g = float(g)
    n_vocab = int(logits.shape[0])
    counts = jnp.zeros((n_vocab,), dtype=jnp.float64)
    tokens = []
    for _k in range(n_paths):
        scores = logits - g * counts
        choice = int(jnp.argmax(scores))
        tokens.append(choice)
        counts = counts.at[choice].add(1.0)
    token_arr = jnp.asarray(tokens, dtype=jnp.int32)
    count_arr = jnp.rint(counts).astype(jnp.int32)
    return token_arr, count_arr


def shared_allocate(logits, n_paths, g, counts=None):
    """Tau = 0 shared (non-cascade) penalty at one position.

    Counts are read once and not updated inside the path loop, so every path
    commits the same token. Tie break: smallest index.
    Returns (tokens, chosen_token_counts_after), where the returned counts
    are the shared read plus n_paths on the chosen token.
    """
    logits = jnp.asarray(logits, dtype=jnp.float64)
    if logits.ndim != 1:
        raise ValueError("logits must be 1D")
    n_paths = int(n_paths)
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    g = float(g)
    n_vocab = int(logits.shape[0])
    if counts is None:
        seen = jnp.zeros((n_vocab,), dtype=jnp.float64)
    else:
        seen = jnp.asarray(counts, dtype=jnp.float64)
        if seen.shape != (n_vocab,):
            raise ValueError("counts must have shape (V,)")
    choice = int(jnp.argmax(logits - g * seen))
    tokens = jnp.full((n_paths,), choice, dtype=jnp.int32)
    final = jnp.rint(seen).astype(jnp.int32)
    final = final.at[choice].add(n_paths)
    return tokens, final


def tilted_softmax(logits, counts, g, tau):
    """softmax(logit - (g / tau) * counts). Requires tau > 0."""
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("tau must be > 0")
    logits = jnp.asarray(logits, dtype=jnp.float64)
    counts = jnp.asarray(counts, dtype=jnp.float64)
    return jax.nn.softmax(logits - (float(g) / tau) * counts)


def plurality(answers):
    """Plurality vote. Ties: smallest answer in sorted order.

    Answers must be mutually sortable hashables. This is not the paper's
    uniform draw among ties.
    """
    answers = list(answers)
    if not answers:
        raise ValueError("no answers to vote on")
    tally = {}
    for answer in answers:
        tally[answer] = tally.get(answer, 0) + 1
    best = max(tally.values())
    winners = [answer for answer, count in tally.items() if count == best]
    return sorted(winners)[0]


def _as_key(key):
    if isinstance(key, int):
        return jax.random.PRNGKey(key)
    return key


def _token_counts(canvases, vocab_size):
    """Count non-mask tokens at each position. Returns (L, V) float64."""
    mask_id = vocab_size
    clipped = jnp.clip(canvases, 0, vocab_size - 1)
    one_hot = jax.nn.one_hot(clipped, vocab_size, dtype=jnp.float64)
    valid = (canvases != mask_id).astype(jnp.float64)[..., None]
    return jnp.sum(one_hot * valid, axis=0)


def _choose_tokens(path_logits, counts, g, tau, penalty_on, path_key):
    """Select a token at every position. path_logits and counts are (L, V)."""
    scores = path_logits
    if penalty_on:
        if tau == 0.0:
            scores = path_logits - g * counts
        else:
            scores = path_logits - (g / tau) * counts
    if tau == 0.0:
        return jnp.argmax(scores, axis=-1)
    return jax.random.categorical(path_key, scores, axis=-1)


def self_repulsive_sample(
    model,
    *,
    n_paths,
    seq_len,
    vocab_size,
    n_steps,
    tau=0.0,
    lambda_pen=1.0,
    rho=1.0,
    key=None,
    answer_fn=None,
    cascade=True,
):
    """Run Algorithm 1.

    model(canvases) -> logits with canvases shape (K, L) and logits shape
    (K, L, V). Called once per denoising step on the whole batch.

    Returns (canvases, vote). vote is None if answer_fn is None, otherwise
    plurality(answer_fn(canvas_k) for each path). answer_fn must return a
    sortable hashable. Plurality ties take the smallest answer in sorted order.

    cascade=False reads counts once per step (shared penalty). At tau = 0
    with identical logits that keeps every path on the same token.
    """
    n_paths = int(n_paths)
    seq_len = int(seq_len)
    vocab_size = int(vocab_size)
    n_steps = int(n_steps)
    tau = float(tau)
    lambda_pen = float(lambda_pen)
    rho = float(rho)
    if n_paths < 2:
        raise ValueError("K must be >= 2")
    if seq_len < 1:
        raise ValueError("L must be >= 1")
    if vocab_size < 1:
        raise ValueError("V must be >= 1")
    if n_steps < 1:
        raise ValueError("T must be >= 1")
    if tau < 0.0:
        raise ValueError("tau must be >= 0")
    g = gate(lambda_pen, n_paths)
    mask_id = vocab_size
    canvases = jnp.full((n_paths, seq_len), mask_id, dtype=jnp.int32)
    if tau > 0.0:
        if key is None:
            raise ValueError("key is required when tau > 0")
        key = _as_key(key)

    for t in range(n_steps):
        logits = jnp.asarray(model(canvases), dtype=jnp.float64)
        if logits.shape != (n_paths, seq_len, vocab_size):
            raise ValueError(
                "model must return logits of shape (K, L, V), "
                f"got {tuple(logits.shape)}"
            )
        penalty_on = ((t + 1) / n_steps) <= rho
        masked = canvases == mask_id
        prior = _token_counts(canvases, vocab_size)
        running = prior
        path_keys = None
        if tau > 0.0:
            key, step_key = jax.random.split(key)
            path_keys = jax.random.split(step_key, n_paths)
        chosen_rows = []
        for k in range(n_paths):
            path_key = None if path_keys is None else path_keys[k]
            choice = _choose_tokens(
                logits[k], running, g, tau, penalty_on, path_key
            )
            chosen_rows.append(choice)
            if cascade:
                update = jax.nn.one_hot(choice, vocab_size, dtype=jnp.float64)
                update = update * masked[k].astype(jnp.float64)[:, None]
                running = running + update
        selected = jnp.stack(chosen_rows, axis=0)

        # Untilted confidence. Not the penalized distribution.
        probs = jax.nn.softmax(logits, axis=-1)
        conf = jnp.take_along_axis(probs, selected[..., None], axis=-1)[..., 0]
        active = masked.astype(jnp.float64)
        mean_conf = (conf * active).sum(axis=0) / jnp.maximum(active.sum(axis=0), 1.0)

        already = jnp.all(~masked, axis=0)
        n_already = int(jnp.sum(already))
        target = int(round((t + 1) / n_steps * seq_len))
        n_new = max(0, target - n_already)
        candidates = [j for j in range(seq_len) if not bool(already[j])]
        candidates.sort(key=lambda j: (-float(mean_conf[j]), j))
        keep = candidates[:n_new]
        if keep:
            keep_mask = jnp.zeros((seq_len,), dtype=bool)
            keep_mask = keep_mask.at[jnp.asarray(keep, dtype=jnp.int32)].set(True)
            write = masked & keep_mask[None, :]
            canvases = jnp.where(write, selected.astype(jnp.int32), canvases)

    vote = None
    if answer_fn is not None:
        vote = plurality(answer_fn(canvases[k]) for k in range(n_paths))
    return canvases, vote
