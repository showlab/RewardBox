"""Cost metrics: latency and GPU memory.

A vague single ``latency_per_frame`` function is a trap: the call granularity of
models differs hugely (Robo-Dopamine makes 3x(T-1) vLLM calls, Robometer one forward
pass), and dividing by the wrong denominator makes benchmark latencies incomparable.
Three explicitly named functions are provided on purpose instead of one with a default
denominator, so callers have to decide which one they mean.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def _validate_latency(total_latency_s: float) -> float:
    value = float(total_latency_s)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"total_latency_s must be finite and >= 0, got {total_latency_s!r}")
    return value


def latency_per_input_frame(total_latency_s: float, num_input_frames: int) -> float:
    """Total latency / input frames (frames sampled by the SamplingSpec, i.e. len(Trajectory))."""
    if num_input_frames <= 0:
        raise ValueError(f"num_input_frames must be > 0, got {num_input_frames}")
    return _validate_latency(total_latency_s) / num_input_frames


def latency_per_output_point(total_latency_s: float, num_output_points: int) -> float:
    """Total latency / model output points (len(RewardOutput.progress)).  May be fewer than
    the input frames, e.g. TOPReward only reports some prefix end points and Robo-Dopamine
    lacks the first frame."""
    if num_output_points <= 0:
        raise ValueError(f"num_output_points must be > 0, got {num_output_points}")
    return _validate_latency(total_latency_s) / num_output_points


def latency_per_backend_call(total_latency_s: float, num_backend_calls: int) -> float:
    """Total latency / official entry-point calls (RewardOutput.meta['n_backend_calls'])."""
    if num_backend_calls <= 0:
        raise ValueError(f"num_backend_calls must be > 0, got {num_backend_calls}")
    return _validate_latency(total_latency_s) / num_backend_calls


@dataclass
class CostBreakdown:
    """Split cost into bridge / inference / total wall-clock time + peak GPU memory instead
    of one blended number.

    **No adapter fills the bridge_s/inference_s split automatically today**: the four
    adapters validated in P0 only measure the overall latency_s wrapped by BaseAdapter.score().
    This class is an optional structure for callers that want finer cost analysis (e.g.
    the benchmark layer) and time things themselves; it does not re-implement or replace
    ``RewardOutput.cost`` (that field's schema is locked into result-v1).

    bridge_s: RewardBox's own input-bridging overhead (e.g. the Robo-Dopamine adapter
      writing frames to a PNG directory).  That is RewardBox I/O cost, not model inference;
      mixing them makes latencies of different adapters incomparable.
    inference_s: the time of the official entry-point call itself.
    total_s: total wall-clock time.  Not forced to equal bridge_s + inference_s; the caller
      may have an independently measured third number (e.g. including overhead in between).
    peak_vram_gb: see ``reset_peak_vram_tracking()``.
    """

    bridge_s: float = 0.0
    inference_s: float = 0.0
    total_s: float = 0.0
    peak_vram_gb: float | None = None

    def __post_init__(self):
        for name in ("bridge_s", "inference_s", "total_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
            setattr(self, name, value)
        if self.peak_vram_gb is not None and (
                not math.isfinite(self.peak_vram_gb) or self.peak_vram_gb < 0):
            raise ValueError("peak_vram_gb must be a finite non-negative number or None")

    def to_dict(self) -> dict:
        return {"bridge_s": self.bridge_s, "inference_s": self.inference_s,
                "total_s": self.total_s, "peak_vram_gb": self.peak_vram_gb}


def reset_peak_vram_tracking(device=None) -> None:
    """Reset the peak GPU memory counter manually.

    ``BaseAdapter.score()`` already performs the same reset on every visible CUDA device
    before each inference; this public helper is for extra cost measurements that do not
    go through BaseAdapter.
    """
    import torch
    if torch.cuda.is_available():
        if device is not None:
            torch.cuda.reset_peak_memory_stats(device)
        else:
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
