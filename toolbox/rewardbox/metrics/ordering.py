"""Ordering metrics.  VOC reproduces the semantics of TOPReward's official metrics/voc.py."""
from __future__ import annotations

import numpy as np


def voc(progress, ground_truth_order=None, degenerate: float = 0.0) -> float:
    """Value-Order Correlation.

    Official semantics (topreward/metrics/voc.py): length < 2 or a constant sequence ->
    NaN, mapped to 0.0 upstream with a note in details.  Here it also returns degenerate,
    and voc_with_note gives the reason.

    ⚠️ Assumes "time order == progress order" by default, which **holds only for
       successful trajectories**.  VOC is meaningless on failed/regressing trajectories;
       a benchmark must split success/failure.
    """
    return voc_with_note(progress, ground_truth_order, degenerate)[0]


def voc_with_note(progress, ground_truth_order=None,
                  degenerate: float = 0.0) -> tuple[float, str]:
    from scipy.stats import spearmanr
    v = np.asarray(progress, dtype=float)
    if v.size < 2:
        return degenerate, "insufficient length"
    t = (np.arange(v.size, dtype=float) if ground_truth_order is None
         else np.asarray(ground_truth_order, dtype=float))
    if v.size != t.size:
        raise ValueError(f"length mismatch: {v.size} vs {t.size}")
    if np.allclose(v, v[0]):
        return degenerate, "constant predictions"
    if np.allclose(t, t[0]):
        return degenerate, "constant ground truth"
    rho = spearmanr(v, t).statistic
    if np.isnan(rho):
        return degenerate, "undefined correlation"
    return float(rho), ""


def _corr(fn, a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    r = fn(a, b)
    r = r.statistic if hasattr(r, "statistic") else r[0]
    return 0.0 if np.isnan(r) else float(r)


def plcc(a, b) -> float:
    from scipy.stats import pearsonr
    return _corr(pearsonr, a, b)


def srcc(a, b) -> float:
    from scipy.stats import spearmanr
    return _corr(spearmanr, a, b)


def krcc(a, b) -> float:
    from scipy.stats import kendalltau
    return _corr(kendalltau, a, b)


def monotonicity(progress) -> float:
    """Fraction of non-negative adjacent differences, in [0,1]."""
    x = np.asarray(progress, float)
    return 1.0 if x.size < 2 else float(np.mean(np.diff(x) >= 0))
