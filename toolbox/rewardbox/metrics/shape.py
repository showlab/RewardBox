"""Curve-shape metrics, and noise robustness from two real inference outputs."""
from __future__ import annotations

import numpy as np


def smoothness(progress, *, scale: str = "raw") -> float:
    """Negative standard deviation of adjacent differences: smoother is closer to 0,
    more oscillating is more negative.

    ``scale`` only labels whether the curve is raw or consumer-normalised; values under
    different scales are not comparable (TOPReward's raw log-likelihood and Robometer's
    [0,1] progress differ by orders of magnitude).  No implicit conversion happens;
    callers must compare smoothness of different models on the same scale.
    """
    if scale not in ("raw", "normalized"):
        raise ValueError(f"scale must be raw|normalized, got {scale!r}")
    x = np.asarray(progress, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"progress must be 1-D, got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"progress contains non-finite values (NaN/Inf): {x}")
    if x.size == 0:
        raise ValueError("progress must not be empty")
    if x.size < 2:
        return 0.0
    return float(-np.std(np.diff(x)))


def terminal_saturation(progress, *, window: int, target: float) -> float:
    """How close the last ``window`` frames are to ``target``:
    ``1 - mean(|target - progress[-window:]|) / range`` (a constant curve uses 1.0 as the
    range to avoid division by zero; the result is then always 1.0, because target must
    equal that constant).

    ``window`` must be given explicitly: the number of frames that make up "the end"
    differs enormously between sampling densities, and frame counts only come from the
    caller's SamplingSpec, so no default is invented here (rule 3).

    ``target`` must be given explicitly.  Raw scales differ between models; the curve's
    own maximum is never silently taken as the "task completed" value.
    """
    x = np.asarray(progress, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"progress must be 1-D, got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"progress contains non-finite values (NaN/Inf): {x}")
    if x.size == 0:
        raise ValueError("progress must not be empty")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window > x.size:
        raise ValueError(f"window={window} exceeds the curve length {x.size}")
    tail = x[-window:]
    t = float(target)
    if not np.isfinite(t):
        raise ValueError("target must be a finite number")
    rng = float(x.max() - x.min())
    denom = rng if rng > 1e-12 else 1.0
    return float(1.0 - np.mean(np.abs(t - tail)) / denom)


def noise_robustness(clean, noisy, *, tolerance: float) -> float:
    """Fraction of points that change by at most ``tolerance`` between two real
    inferences (clean/noisy) of the same model.

    Returns [0,1], higher is more stable.  Both :class:`RewardOutput` objects must share
    model, SamplingSpec and output frame grid; how the input noise was generated and how
    strong it is belongs in the benchmark record, never synthesised from one curve.
    ``tolerance`` must be given explicitly on the raw score scale.
    """
    from ..core.types import RewardOutput

    if not isinstance(clean, RewardOutput) or not isinstance(noisy, RewardOutput):
        raise TypeError("clean/noisy must be RewardOutputs from two real inferences")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite number >= 0")
    if clean.model_name != noisy.model_name:
        raise ValueError(
            f"clean/noisy models differ: {clean.model_name!r} vs {noisy.model_name!r}")
    if ((clean.sampling is None) != (noisy.sampling is None)
            or (clean.sampling is not None
                and clean.sampling.key() != noisy.sampling.key())):
        raise ValueError("clean/noisy SamplingSpec differ")
    if not np.array_equal(clean.frame_indices, noisy.frame_indices):
        raise ValueError("clean/noisy frame_indices differ; cannot compare point by point")
    return float(np.mean(np.abs(clean.progress - noisy.progress) <= tolerance))
