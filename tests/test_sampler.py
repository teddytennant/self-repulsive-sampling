"""Deterministic tests for the self-repulsion kernel. No benchmark numbers."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from self_repulsion import (
    cascade_allocate,
    fractional_counts,
    gate,
    plurality,
    self_repulsive_sample,
    shared_allocate,
    sparsemax,
    tilted_softmax,
)


def _objective(counts, logits, g):
    total = 0.0
    for count, logit in zip(counts, logits):
        total += count * float(logit) - g * (count * (count - 1) / 2.0)
    return total


def _optimal_counts(logits, n_paths, g):
    """Brute force nonnegative integer vectors summing to K. V is small."""
    vocab = len(logits)
    best_val = None
    best = []

    def rec(index, remain, cur):
        nonlocal best_val
        if index == vocab:
            val = _objective(cur, logits, g)
            if best_val is None or val > best_val + 1e-9:
                best_val = val
                best.clear()
                best.append(tuple(cur))
            elif abs(val - best_val) <= 1e-9:
                best.append(tuple(cur))
            return
        if index == vocab - 1:
            cur[index] = remain
            rec(index + 1, 0, cur)
            return
        for taken in range(remain + 1):
            cur[index] = taken
            rec(index + 1, remain - taken, cur)

    rec(0, n_paths, [0] * vocab)
    return best_val, best


def _ref_sparsemax(z):
    z = np.asarray(z, dtype=np.float64)
    n = z.size
    z_sorted = np.sort(z)[::-1]
    cssv = np.cumsum(z_sorted)
    ks = np.arange(1, n + 1)
    cond = (1.0 + ks * z_sorted) > cssv
    best = int(np.max(np.where(cond, np.arange(n), 0)))
    tau = (cssv[best] - 1.0) / (best + 1.0)
    return np.maximum(z - tau, 0.0)


class TableModel:
    def __init__(self, table):
        self.table = jnp.asarray(table, dtype=jnp.float64)
        self.calls = 0
        self.batch_sizes = []
        self.seen = []

    def __call__(self, canvases):
        self.calls += 1
        self.batch_sizes.append(int(canvases.shape[0]))
        self.seen.append(np.asarray(canvases))
        return jnp.broadcast_to(
            self.table, (canvases.shape[0],) + self.table.shape
        )


def _position_counts(canvases, vocab):
    canvases = np.asarray(canvases)
    counts = np.zeros((canvases.shape[1], vocab), dtype=np.int32)
    for path in canvases:
        for j, token in enumerate(path):
            counts[j, int(token)] += 1
    return counts


def test_gate():
    assert gate(1.0, 2) == pytest.approx(1.0)
    assert gate(1.0, 5) == pytest.approx(1.0 / 4.0)
    assert gate(64.0, 10) == pytest.approx(64.0 / 9.0)
    assert gate(1.0, 4) == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError):
        gate(1.0, 1)
    with pytest.raises(ValueError):
        self_repulsive_sample(
            lambda c: c,
            n_paths=1,
            seq_len=1,
            vocab_size=1,
            n_steps=1,
        )


def test_tau0_cascade_matches_brute_force():
    hand = [
        ([1.0, 0.9], 2, 0.5, (1, 1)),
        ([3.0, 0.0, -1.0], 4, 0.2, (4, 0, 0)),
        ([1.0, 1.0], 4, 1.0, (2, 2)),
        ([0.0, 2.0, 2.0], 3, 1.0, (0, 2, 1)),
        ([1.0, 0.0], 3, 0.4, (3, 0)),
        ([1.0, 1.0, 0.5], 4, 1.0, (2, 1, 1)),
        ([2.0, 1.0, 0.0], 4, 1.0 / 3.0, (4, 0, 0)),
        ([1.0, 0.8, 0.0], 4, 1.0 / 3.0, (2, 2, 0)),
        ([5.0, 5.0, 1.0, -3.0], 6, 0.7, None),
        ([-1.5, -0.2, -0.2, 0.4], 5, 0.3, None),
    ]
    rng = np.random.default_rng(0)
    for _ in range(8):
        vocab = int(rng.integers(1, 5))
        n_paths = int(rng.integers(2, 7))
        g = float(rng.uniform(0.05, 3.0))
        logits = rng.normal(size=vocab)
        hand.append((logits.tolist(), n_paths, g, None))

    for logits, n_paths, g, expected in hand:
        assert len(logits) <= 4
        assert n_paths <= 6
        tokens, counts = cascade_allocate(jnp.asarray(logits), n_paths, g)
        counts_t = tuple(int(c) for c in counts)
        assert sum(counts_t) == n_paths
        assert all(c >= 0 for c in counts_t)
        best_val, optima = _optimal_counts(logits, n_paths, g)
        assert counts_t in optima
        assert _objective(counts_t, logits, g) == pytest.approx(best_val)
        if expected is not None:
            assert counts_t == expected
        # Marginal definition: each step takes argmax of logit - k * g.
        running = [0] * len(logits)
        for token in tokens:
            token = int(token)
            scores = [float(logits[i]) - running[i] * g for i in range(len(logits))]
            assert token == max(range(len(logits)), key=lambda i: (scores[i], -i))
            running[token] += 1
        assert tuple(running) == counts_t


def test_shared_penalty_identical_paths():
    logits = jnp.array([0.2, 1.5, 1.5, -0.3])
    tokens, _ = shared_allocate(logits, n_paths=6, g=0.8)
    assert len(set(int(t) for t in tokens)) == 1

    seen = jnp.array([1.0, 0.0, 2.0, 0.0])
    tokens, final = shared_allocate(logits, n_paths=5, g=1.2, counts=seen)
    assert len(set(int(t) for t in tokens)) == 1
    scores = logits - 1.2 * seen
    assert int(tokens[0]) == int(jnp.argmax(scores))
    assert int(final[int(tokens[0])]) == 5 + 0

    # Full sampler, tau 0, shared read. Logits would split under the cascade.
    table = jnp.array([[1.0, 0.8, 0.1], [0.2, 0.2, 1.4]])
    model = TableModel(table)
    canvases, _ = self_repulsive_sample(
        model,
        n_paths=4,
        seq_len=2,
        vocab_size=3,
        n_steps=2,
        tau=0.0,
        lambda_pen=1.0,
        rho=1.0,
        cascade=False,
    )
    canvases = np.asarray(canvases)
    for k in range(1, 4):
        assert np.array_equal(canvases[k], canvases[0])
    # Cascade on the same table does not keep paths identical.
    split, _ = self_repulsive_sample(
        TableModel(table),
        n_paths=4,
        seq_len=2,
        vocab_size=3,
        n_steps=2,
        tau=0.0,
        lambda_pen=1.0,
        rho=1.0,
        cascade=True,
    )
    assert not np.array_equal(np.asarray(split)[0], np.asarray(split)[1])


def test_tilted_softmax_matches_formula():
    logits = jnp.array([0.4, -1.2, 0.7, 0.1])
    counts = jnp.array([2.0, 0.0, 1.0, 3.0])
    g = 0.75
    tau = 0.4
    got = np.asarray(tilted_softmax(logits, counts, g, tau))
    expected = np.asarray(
        jax.nn.softmax(logits - (g / tau) * counts)
    )
    assert got == pytest.approx(expected)
    assert got.sum() == pytest.approx(1.0)
    # A second draw with zeros and a different tau.
    counts0 = jnp.zeros((4,))
    got0 = np.asarray(tilted_softmax(logits, counts0, g=1.5, tau=0.8))
    expected0 = np.asarray(jax.nn.softmax(logits))
    assert got0 == pytest.approx(expected0)
    with pytest.raises(ValueError):
        tilted_softmax(logits, counts, g, tau=0.0)


def test_fractional_matches_sparsemax_and_sums_to_k():
    hand = [
        (np.array([2.0, 1.0, 0.0]), 3, 1.0, np.array([2.0, 1.0, 0.0])),
        (np.array([3.0, 1.0, -10.0]), 2, 1.0, np.array([2.0, 0.0, 0.0])),
        (np.array([0.0, 0.0, 0.0]), 6, 0.5, np.full(3, 2.0)),
    ]
    for logits, n_paths, g, expected in hand:
        got = np.asarray(fractional_counts(logits, n_paths, g))
        via_lib = n_paths * np.asarray(sparsemax(logits / (g * n_paths)))
        via_ref = n_paths * _ref_sparsemax(logits / (g * n_paths))
        assert got == pytest.approx(expected, abs=1e-8)
        assert got == pytest.approx(via_lib, abs=1e-8)
        assert got == pytest.approx(via_ref, abs=1e-8)
        assert got.sum() == pytest.approx(n_paths)

    # Direct sparsemax checks.
    assert np.asarray(sparsemax(jnp.array([1.0, 0.0, 0.0]))) == pytest.approx(
        [1.0, 0.0, 0.0]
    )
    assert np.asarray(sparsemax(jnp.zeros(3))) == pytest.approx([1.0 / 3.0] * 3)

    rng = np.random.default_rng(0)
    for _ in range(12):
        vocab = int(rng.integers(1, 6))
        n_paths = int(rng.integers(2, 9))
        g = float(rng.uniform(0.05, 4.0))
        logits = rng.normal(size=vocab)
        got = np.asarray(fractional_counts(logits, n_paths, g))
        ref = n_paths * _ref_sparsemax(logits / (g * n_paths))
        assert got == pytest.approx(ref, abs=1e-7)
        assert got.sum() == pytest.approx(n_paths)
        assert np.all(got >= -1e-12)


def test_integer_counts_within_one_of_fractional():
    """Paper claim: |c_v - c*_v| < 1. Seed 0 draws, plus ties.

    The brute-force integer identity is the hard test and is not relaxed here.
    """
    cases = []
    rng = np.random.default_rng(0)
    for _ in range(16):
        vocab = int(rng.integers(1, 6))
        n_paths = int(rng.integers(2, 8))
        g = float(10 ** rng.uniform(-2, 1))
        logits = rng.normal(size=vocab) * float(10 ** rng.uniform(-1, 1))
        cases.append((logits, n_paths, g))
    cases.extend(
        [
            (np.array([1.0, 1.0, 0.5]), 4, 1.0),
            (np.array([2.0, 1.0, 0.0]), 3, 1.0),
            (np.zeros(4), 5, 0.25),
            (np.array([100.0, 0.0, -50.0]), 6, 0.01),
        ]
    )
    for logits, n_paths, g in cases:
        _, counts = cascade_allocate(jnp.asarray(logits), n_paths, g)
        frac = np.asarray(fractional_counts(logits, n_paths, g))
        diff = np.abs(np.asarray(counts, dtype=np.float64) - frac)
        assert np.all(diff < 1.0)


def test_algorithm1_calls_and_counts():
    table = jnp.array(
        [
            [2.0, 1.0, 0.0],  # gap beats g = 1/3, one-hot even with penalty
            [1.0, 0.8, 0.0],  # cascade splits
            [0.4, 0.4, 0.4],  # tie, smallest index
            [-0.5, 0.3, 0.9],
        ],
        dtype=jnp.float64,
    )
    n_paths = 4
    seq_len = 4
    vocab = 3
    n_steps = 5
    lam = 1.0
    g = lam / (n_paths - 1)

    model = TableModel(table)
    canvases, _ = self_repulsive_sample(
        model,
        n_paths=n_paths,
        seq_len=seq_len,
        vocab_size=vocab,
        n_steps=n_steps,
        tau=0.0,
        lambda_pen=lam,
        rho=1.0,
    )
    assert model.calls == n_steps
    assert model.batch_sizes == [n_paths] * n_steps
    assert np.all(model.seen[0] == vocab)
    assert np.all(np.asarray(canvases) != vocab)
    counts = _position_counts(canvases, vocab)
    for j in range(seq_len):
        _, casc = cascade_allocate(table[j], n_paths, g)
        assert tuple(int(c) for c in counts[j]) == tuple(int(c) for c in casc)
        # Hard identity, not only the library cascade.
        best_val, optima = _optimal_counts(np.asarray(table[j]), n_paths, g)
        got = tuple(int(c) for c in counts[j])
        assert got in optima
        assert _objective(got, np.asarray(table[j]), g) == pytest.approx(best_val)

    off = TableModel(table)
    canvases_off, _ = self_repulsive_sample(
        off,
        n_paths=n_paths,
        seq_len=seq_len,
        vocab_size=vocab,
        n_steps=n_steps,
        tau=0.0,
        lambda_pen=lam,
        rho=0.0,
    )
    assert off.calls == n_steps
    counts_off = _position_counts(canvases_off, vocab)
    for j in range(seq_len):
        winner = int(jnp.argmax(table[j]))
        assert int(counts_off[j, winner]) == n_paths
        assert int(counts_off[j].sum()) == n_paths


def test_rho_and_confidence_schedule():
    # Equal logits: confidence ties, smallest index is committed first.
    # T = 4, L = 4 commits one new position per step.
    # rho = 0.5 keeps the penalty on for t = 0, 1 only.
    table = jnp.array([[1.0, 0.6]] * 4, dtype=jnp.float64)
    n_paths = 3
    g = 1.0 / (n_paths - 1)
    model = TableModel(table)
    canvases, _ = self_repulsive_sample(
        model,
        n_paths=n_paths,
        seq_len=4,
        vocab_size=2,
        n_steps=4,
        tau=0.0,
        lambda_pen=1.0,
        rho=0.5,
    )
    assert model.calls == 4
    counts = _position_counts(canvases, 2)
    _, on = cascade_allocate(table[0], n_paths, g)
    on = tuple(int(c) for c in on)
    assert on != (n_paths, 0)
    for j in range(2):
        assert tuple(int(c) for c in counts[j]) == on
    for j in range(2, 4):
        assert tuple(int(c) for c in counts[j]) == (n_paths, 0)
    # Step 0 commits only position 0 (index tie break).
    assert np.all(model.seen[1][:, 0] != 2)
    assert np.all(model.seen[1][:, 1:] == 2)

    # Higher untilted confidence is committed first, not the smallest index.
    peaked = jnp.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    peaked_model = TableModel(peaked)
    self_repulsive_sample(
        peaked_model,
        n_paths=2,
        seq_len=2,
        vocab_size=2,
        n_steps=2,
        tau=0.0,
        lambda_pen=1.0,
        rho=1.0,
    )
    assert np.all(peaked_model.seen[1][:, 1] != 2)
    assert np.all(peaked_model.seen[1][:, 0] == 2)


def test_plurality_including_tie():
    assert plurality(["b", "a", "b", "a"]) == "a"
    assert plurality([3, 1, 3, 1, 2]) == 1
    assert plurality(["c", "a", "c"]) == "c"
    assert plurality([-1, -1, 0, 0]) == -1
    assert plurality([(1, 0), (0, 1), (1, 0)]) == (1, 0)

    def answer(canvas):
        return int(canvas[0])

    # K = 2, identical logits, tau 0 cascade splits the position. 1-1 tie.
    table = jnp.array([[0.0, 0.0]], dtype=jnp.float64)
    canvases, vote = self_repulsive_sample(
        TableModel(table),
        n_paths=2,
        seq_len=1,
        vocab_size=2,
        n_steps=1,
        tau=0.0,
        lambda_pen=1.0,
        rho=1.0,
        answer_fn=answer,
    )
    answers = [answer(canvases[k]) for k in range(2)]
    assert sorted(answers) == [0, 1]
    assert vote == 0
    assert vote == plurality(answers)

    # A strict majority.
    majority, vote_m = self_repulsive_sample(
        TableModel(jnp.array([[5.0, 0.0]])),
        n_paths=4,
        seq_len=1,
        vocab_size=2,
        n_steps=1,
        tau=0.0,
        lambda_pen=1.0,
        rho=1.0,
        answer_fn=answer,
    )
    assert vote_m == 0
    assert all(int(majority[k, 0]) == 0 for k in range(4))


def test_tau_positive_sample_matches_tilted_categorical():
    logits = np.array([0.2, -0.4, 1.1, 0.0], dtype=np.float64)
    n_paths = 3
    g = 1.0 / (n_paths - 1)
    tau = 0.5
    key = jax.random.PRNGKey(1)
    _, step_key = jax.random.split(key)
    path_keys = jax.random.split(step_key, n_paths)
    counts = np.zeros(4, dtype=np.float64)
    expected = []
    for k in range(n_paths):
        scores = logits - (g / tau) * counts
        choice = int(jax.random.categorical(path_keys[k], jnp.asarray(scores)))
        expected.append(choice)
        counts[choice] += 1.0

    def model(canvases):
        return jnp.broadcast_to(
            jnp.asarray(logits), (canvases.shape[0], 1, 4)
        )

    canvases, _ = self_repulsive_sample(
        model,
        n_paths=n_paths,
        seq_len=1,
        vocab_size=4,
        n_steps=1,
        tau=tau,
        lambda_pen=1.0,
        rho=1.0,
        key=jax.random.PRNGKey(1),
    )
    assert [int(canvases[k, 0]) for k in range(n_paths)] == expected
