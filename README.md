# Self-repulsion

A small JAX kernel of self-repulsive sampling for masked diffusion language models (arXiv 2609.39560, Helcig and Jaggi). The method is training-free: K paths share one forward pass per step, then a peer-count penalty on the logits pushes later paths off tokens earlier paths just committed.

## Install and run

```
pip install -e .
pytest
python train.py
```

`pytest` reads `pythonpath = ["src"]` from pyproject.toml. `python train.py` needs `PYTHONPATH=src` if the package is not installed. `train.py` is a deterministic demo on a fixed logit table, not a training run.

## What this does not match

This is not a reproduction of the paper's experiments. There is no LLaDA-8B, no GSM8K, MATH, or TruthfulQA run, and no claimed accuracy.

The commit schedule is linear in position count. After step t (0-based) of T steps, `round((t+1)/T * L)` positions are non-mask, chosen by highest mean untilted confidence, with ties broken by the smallest index. `round` is Python's half-to-even round. The paper keeps the original decoder's confidence rule inside semi-autoregressive blocks and turns the penalty off after the first fraction rho of each block. Here there are no blocks: rho is one fraction of the whole run, and the default is rho = 1 (penalty on at every step, since `(t+1)/T <= 1`).

For tau > 0 the tilted law is `softmax(logit - (g/tau) * counts)`. The base logits are not divided by tau. The paper samples `softmax((logit - g * counts) / tau)`. At tau = 0 both use `argmax(logit - g * counts)`, smallest index on ties, and the token choice does not form a softmax. Confidence for the schedule is still the untilted `softmax(logit)` probability of the selected token.

Plurality ties return the smallest answer in sorted order. The paper draws a tied answer uniformly. `answer(canvas)` must return a sortable hashable.

The gate is `g = lambda_pen / (K - 1)` with default `lambda_pen = 1`. Mask token id is V. Logits at a position have length V and do not score the mask. A non-cascade read of the counts (shared penalty) is available for the tau = 0 degeneracy: every path then commits the same token from identical logits.
