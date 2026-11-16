"""Detection metrics: success_auroc / success_f1 / balanced_accuracy.

Rule: every metric takes raw scores; no min-max normalisation happens here, it
only happens at the consumer (``RewardOutput.normalized()``).  AUROC is invariant
under strictly monotone transforms of the scores (min-max included), which is why it
is the scale-free detection metric; F1/balanced_accuracy need a threshold, and the
caller must give it explicitly on the raw scale.  Raw score ranges differ a lot
between models (TOPReward is a negative log-likelihood, Robometer progress is in
[0,1]); there is no universal default cut-off.
"""
from __future__ import annotations

import numpy as np


class UndefinedMetricError(ValueError):
    """The metric is undefined for this input (e.g. labels have only one class).  No
    plausible-looking number (0.5 / NaN) is returned silently; callers must handle it."""


def _validate_scores_labels(scores, labels):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels)
    if s.ndim != 1 or y.ndim != 1:
        raise ValueError(f"scores/labels must be 1-D, got {s.shape}/{y.shape}")
    if s.size != y.size:
        raise ValueError(f"length mismatch: scores={s.size} labels={y.size}")
    if s.size == 0:
        raise ValueError("scores/labels must not be empty")
    if not np.all(np.isfinite(s)):
        raise ValueError(f"scores contain non-finite values (NaN/Inf): {s}")
    yf = y.astype(float)
    if not np.all(np.isin(yf, (0.0, 1.0))):
        raise ValueError(f"labels accept only 0/1 or bool, got {np.unique(y)}")
    return s, yf.astype(bool)


def success_auroc(raw_scores, labels) -> float:
    """Area under the ROC curve = P(score_pos > score_neg) + 0.5*P(score_pos == score_neg).

    Higher scores should mean label=1 (success).  Invariant under strictly monotone
    transforms of raw_scores (min-max normalisation included).

    Both classes must be present, otherwise AUROC is undefined and
    ``UndefinedMetricError`` is raised (no 0.5 or NaN).

    Ties: tied scores use average ranks as in the standard ROC convention (equivalent
    to Mann-Whitney U / (n_pos*n_neg)), no random tie-breaking; a tied positive/negative
    pair contributes 0.5 and favours neither side.
    """
    from scipy.stats import rankdata

    s, y = _validate_scores_labels(raw_scores, labels)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise UndefinedMetricError(
            f"success_auroc is undefined with a single class (n_pos={n_pos}, n_neg={n_neg})")
    ranks = rankdata(s, method="average")
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def success_f1(raw_scores, labels, *, threshold: float) -> float:
    """F1 = 2*precision*recall / (precision+recall), predicted = raw_scores >= threshold.

    threshold must be given explicitly (no default); see the module docstring.

    With no positive prediction at all (tp=0) the standard convention returns 0.0.  That
    is not a "plausible high score" but F1's standard definition in this degenerate case
    (a zero precision/recall denominator counts as 0, neither an exception nor 1.0).
    """
    s, y = _validate_scores_labels(raw_scores, labels)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be a finite number")
    pred = s >= threshold
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


def balanced_accuracy(raw_scores, labels, *, threshold: float) -> float:
    """(TPR + TNR) / 2, predicted = raw_scores >= threshold.

    threshold must be given explicitly (no default); see the module docstring.

    When one class has no samples its TPR/TNR is undefined and ``UndefinedMetricError``
    is raised (no zero fill, which would bias the result while looking like a normal number).
    """
    s, y = _validate_scores_labels(raw_scores, labels)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be a finite number")
    pred = s >= threshold
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise UndefinedMetricError(
            f"balanced_accuracy is undefined with a single class (n_pos={n_pos}, n_neg={n_neg})")
    tpr = np.sum(pred & y) / n_pos
    tnr = np.sum(~pred & ~y) / n_neg
    return float((tpr + tnr) / 2)
