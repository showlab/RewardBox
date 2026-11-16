"""Ranking / preference metrics.

"pairwise" has two easily confused meanings when evaluating reward models; they are
split into two clear entry points instead of one vague parameter meaning both:

  paired_accuracy    -- pair by pair: preferred[i] and nonpreferred[i] are paired samples
                        (e.g. the success vs failure variant of the same task)
  all_pairs_accuracy -- between sets: the Cartesian product of the positive and the
                        negative set (e.g. terminal scores of all successful trajectories
                        vs all failed ones, compared pairwise)

Robometer's native ``compare()`` returns the model's own preference direction (a scalar
mapped to [-1,1]); semantically that is not "comparing two independent scores".  Evaluate
it with ``preference_accuracy``, never through the same array interface as the two above.
"""
from __future__ import annotations

import numpy as np


def _validate_1d(name: str, arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf): {a}")
    return a


def paired_accuracy(preferred, nonpreferred, *, tie_value: float = 0.5) -> float:
    """Pair by pair: expects preferred[i] > nonpreferred[i].  Both arrays must have the
    same length and pair one to one (not a Cartesian product; use all_pairs_accuracy).

    Ties score tie_value: tie_value=0.5 is the standard "half each" convention,
    tie_value=0.0 requires strictly higher to count as correct.
    """
    a = _validate_1d("preferred", preferred)
    b = _validate_1d("nonpreferred", nonpreferred)
    if not np.isfinite(tie_value) or not 0.0 <= tie_value <= 1.0:
        raise ValueError(f"tie_value must be in [0,1], got {tie_value!r}")
    if a.size != b.size:
        raise ValueError(f"preferred/nonpreferred must have the same length: {a.size} vs {b.size}")
    scores = (a > b).astype(float)
    scores[a == b] = tie_value
    return float(scores.mean())


def all_pairs_accuracy(positive, negative, *, tie_value: float = 0.5) -> float:
    """Pairwise comparison between sets: for every (i, j) in positive x negative, expects
    positive[i] > negative[j].  A measure of overall separation (mathematically
    equivalent to success_auroc with a different interface: two groups of raw scores are
    passed directly, no scores+labels form needed).
    """
    pos = _validate_1d("positive", positive)
    neg = _validate_1d("negative", negative)
    if not np.isfinite(tie_value) or not 0.0 <= tie_value <= 1.0:
        raise ValueError(f"tie_value must be in [0,1], got {tie_value!r}")
    diff = pos[:, None] - neg[None, :]
    scores = (diff > 0).astype(float)
    scores[diff == 0] = tie_value
    return float(scores.mean())


def preference_accuracy(compare_results) -> float:
    """Evaluate native ``compare()``-style preference outputs (not a comparison of two
    independent scalar scores).

    compare_results: return values of compare(a, b) where a should be better (e.g.
    a=success trajectory, b=failure trajectory).  >0 counts as correct, ==0 scores 0.5
    (tie), <0 counts as wrong.
    """
    r = _validate_1d("compare_results", compare_results)
    scores = (r > 0).astype(float)
    scores[r == 0] = 0.5
    return float(scores.mean())
