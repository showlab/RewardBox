"""Unified input / output / sampling protocol.

Three rules:
  1. no torch / transformers import at module top level (the manifest must build in
     a light environment)
  2. model outputs keep their raw scale; normalisation happens only at the consumer
  3. every "frame count" comes from a SamplingSpec, never hard-coded
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .enums import Normalize

PRIMARY_VIEW = "primary"


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingSpec:
    """The sampling protocol, a first-class object: cross-model comparison requires
    this object to be identical."""

    num_frames: int = 16
    strategy: str = "uniform"            # uniform | uniform_floor | all | fps | interval
    fps: float | None = None             # strategy='fps'
    interval: int | None = None          # strategy='interval' (Robo-Dopamine's official usage)
    resize: tuple[int, int] | None = None  # (w, h); None = no resize
    include_last: bool = True

    def __post_init__(self):
        if not isinstance(self.num_frames, int) or isinstance(self.num_frames, bool):
            raise ValueError(f"num_frames must be an integer, got {self.num_frames!r}")
        if self.strategy not in {
            "uniform", "uniform_floor", "all", "fps", "interval",
        }:
            raise ValueError(f"unknown sampling strategy: {self.strategy!r}")
        if self.strategy in {"uniform", "uniform_floor"} and self.num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        if self.strategy == "fps" and (
                self.fps is None or not np.isfinite(self.fps) or self.fps <= 0):
            raise ValueError("strategy='fps' requires a finite fps > 0")
        if self.strategy == "interval" and (
                not isinstance(self.interval, int) or isinstance(self.interval, bool)
                or self.interval <= 0):
            raise ValueError("strategy='interval' requires an integer interval > 0")
        if self.resize is not None and (
                len(self.resize) != 2
                or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0
                       for x in self.resize)):
            raise ValueError(
                f"resize must be two positive integers (width, height), got {self.resize!r}")

    def indices(self, total: int, source_fps: float | None = None) -> np.ndarray:
        """Return strictly increasing frame numbers.

        Contract (callers must accept it):
          - with ``strategy="uniform"/"uniform_floor"`` the length is <= num_frames,
            possibly fewer (short videos), and a single index when num_frames == 1;
            the two quantise linspace with round/floor respectively.  The floor
            variant expresses the 10-anchor grid of RewardGen's public SOLE-R1
            reference artefacts
          - ``strategy="all"/"fps"/"interval"`` are native protocols (every frame /
            by source_fps / fixed stride); their length follows total/fps/interval
            and is **not** capped by num_frames and may exceed it.  That is the price
            of reproducing the official sampling faithfully; nothing is truncated to
            match older documentation
          - every strategy is strictly increasing without duplicates
        """
        if total <= 0:
            raise ValueError(f"total frame count must be > 0, got {total}")

        if self.strategy == "all":
            idx = np.arange(total, dtype=np.int64)
        elif self.strategy in {"uniform", "uniform_floor"}:
            if self.num_frames == 1:
                return np.array([total - 1 if self.include_last else 0], dtype=np.int64)
            if total <= self.num_frames:
                idx = np.arange(total, dtype=np.int64)
            else:
                grid = np.linspace(0, total - 1, self.num_frames)
                quantized = np.floor(grid) if self.strategy == "uniform_floor" else np.rint(grid)
                idx = np.unique(quantized.astype(np.int64))
        elif self.strategy == "fps":
            if not source_fps or source_fps <= 0:
                raise ValueError("strategy='fps' needs a valid source_fps")
            step = max(1, int(round(source_fps / self.fps)))
            idx = np.arange(0, total, step, dtype=np.int64)
        else:  # interval: reproduces Robo-Dopamine's make_sample_indices_by_interval
            idx = np.arange(0, total, self.interval, dtype=np.int64)

        if self.include_last and len(idx) and idx[-1] != total - 1:
            idx = np.append(idx, total - 1)
        return idx.astype(np.int64)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resize"] = list(self.resize) if self.resize else None
        d["key"] = self.key()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SamplingSpec:
        d = {k: v for k, v in d.items() if k != "key"}
        if d.get("resize") is not None:
            d["resize"] = tuple(d["resize"])
        return cls(**d)

    def key(self) -> str:
        """Canonical, human-readable protocol id; equal keys mean equal specs."""
        fps = "-" if self.fps is None else repr(float(self.fps))
        interval = "-" if self.interval is None else str(self.interval)
        resize = "-" if not self.resize else f"{self.resize[0]}x{self.resize[1]}"
        return (f"{self.strategy}:n={self.num_frames}:fps={fps}:interval={interval}"
                f":resize={resize}:last={int(self.include_last)}")


# --------------------------------------------------------------------------
@dataclass
class Trajectory:
    """The unified input.  Every adapter accepts only this."""

    frames: np.ndarray | dict[str, np.ndarray]   # [T,H,W,3] uint8 RGB; dict for multi-view
    instruction: str
    frame_indices: np.ndarray | None = None
    fps: float | None = None
    goal_image: np.ndarray | None = None
    # Named reference image groups decoupled from the time axis: {name: [K,H,W,3] uint8 RGB}.
    # Every view in ``views`` must have exactly T frames (aligned with frame_indices), but
    # some models also need auxiliary images whose count is independent of T: VLAC's
    # ``ref_image_list`` (official ref_num=9) and Robo-Dopamine's reference-start /
    # reference-end.  These images are not in frame_indices and skip the equal-length
    # view check.  A single goal image can still use ``goal_image``.
    reference_images: dict[str, np.ndarray] | None = None
    success: bool | None = None
    progress_gt: np.ndarray | None = None
    sampling: SamplingSpec | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if not isinstance(self.meta, dict):
            raise ValueError("meta must be a dict")
        if self.success is not None and not isinstance(self.success, (bool, np.bool_)):
            raise ValueError("success must be a bool or None")
        views = self.views
        if not views:
            raise ValueError("a trajectory needs at least one view")
        ref_len = None
        for name, v in views.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("view names must be non-empty strings")
            if not isinstance(v, np.ndarray) or v.ndim != 4 or v.shape[-1] != 3:
                raise ValueError(
                    f"view {name!r} must be [T,H,W,3], got {getattr(v, 'shape', type(v))}")
            if v.dtype != np.uint8:
                raise ValueError(f"view {name!r} must be uint8 RGB [0,255], got {v.dtype}")
            if len(v) == 0:
                raise ValueError(f"view {name!r} has 0 frames")
            if ref_len is None:
                ref_len = len(v)
            elif len(v) != ref_len:
                raise ValueError(
                    f"view {name!r} has {len(v)} frames, the other views have {ref_len}")
        if self.frame_indices is None:
            self.frame_indices = np.arange(ref_len, dtype=np.int64)
        self.frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        if self.frame_indices.ndim != 1:
            raise ValueError(f"frame_indices must be 1-D, got {self.frame_indices.shape}")
        if len(self.frame_indices) != ref_len:
            raise ValueError(
                f"frame_indices length {len(self.frame_indices)} != frame count {ref_len}")
        if np.any(self.frame_indices < 0):
            raise ValueError("frame_indices must not contain negative values")
        if len(self.frame_indices) > 1 and np.any(np.diff(self.frame_indices) <= 0):
            raise ValueError("frame_indices must be strictly increasing")
        if self.fps is not None and (
                not np.isfinite(self.fps) or self.fps <= 0):
            raise ValueError(f"fps must be a finite number > 0, got {self.fps!r}")
        if self.goal_image is not None:
            if (not isinstance(self.goal_image, np.ndarray)
                    or self.goal_image.ndim != 3
                    or self.goal_image.shape[-1] != 3
                    or self.goal_image.dtype != np.uint8):
                raise ValueError(
                    "goal_image must be [H,W,3] uint8 RGB")
        if self.reference_images is not None:
            if not isinstance(self.reference_images, dict):
                raise ValueError("reference_images must be dict[str, [K,H,W,3] uint8]")
            if not self.reference_images:
                raise ValueError("reference_images must not be an empty dict; pass None instead")
            for name, imgs in self.reference_images.items():
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("reference_images names must be non-empty strings")
                if (not isinstance(imgs, np.ndarray) or imgs.ndim != 4
                        or imgs.shape[-1] != 3):
                    raise ValueError(
                        f"reference_images[{name!r}] must be [K,H,W,3], "
                        f"got {getattr(imgs, 'shape', type(imgs))}")
                if imgs.dtype != np.uint8:
                    raise ValueError(
                        f"reference_images[{name!r}] must be uint8 RGB [0,255], got {imgs.dtype}")
                if len(imgs) == 0:
                    raise ValueError(f"reference_images[{name!r}] has 0 images")
        if self.progress_gt is not None:
            self.progress_gt = np.asarray(self.progress_gt, dtype=np.float64)
            if self.progress_gt.ndim != 1:
                raise ValueError(f"progress_gt must be 1-D, got {self.progress_gt.shape}")
            if len(self.progress_gt) != ref_len:
                raise ValueError(
                    f"progress_gt length {len(self.progress_gt)} != frame count {ref_len}; "
                    "manifest labels must be aligned to the actual sampled frames first")
            if not np.all(np.isfinite(self.progress_gt)):
                raise ValueError("progress_gt must not contain NaN/Inf")

    @property
    def views(self) -> dict[str, np.ndarray]:
        return self.frames if isinstance(self.frames, dict) else {PRIMARY_VIEW: self.frames}

    @property
    def primary(self) -> np.ndarray:
        if isinstance(self.frames, dict):
            return self.frames.get(PRIMARY_VIEW) if PRIMARY_VIEW in self.frames \
                else next(iter(self.frames.values()))
        return self.frames

    def view(self, name: str) -> np.ndarray:
        v = self.views
        if name not in v:
            raise KeyError(f"trajectory has no view {name!r}; available: {sorted(v)}")
        return v[name]

    @property
    def reference_names(self) -> list[str]:
        return sorted(self.reference_images) if self.reference_images else []

    def reference(self, name: str) -> np.ndarray:
        """One named reference image group [K,H,W,3].  Missing groups raise; never a
        silent fallback to an empty array."""
        if not self.reference_images or name not in self.reference_images:
            raise KeyError(
                f"trajectory has no reference image group {name!r}; "
                f"available: {self.reference_names}")
        return self.reference_images[name]

    def __len__(self) -> int:
        return len(self.primary)

    # ---- constructors ----
    @classmethod
    def from_video(cls, path, instruction: str, sampling: SamplingSpec, **kw) -> Trajectory:
        from .video import read_frames
        frames, idx, fps = read_frames(path, sampling)
        return cls(frames=frames, instruction=instruction, frame_indices=idx,
                   fps=fps, sampling=sampling, source=str(path), **kw)

    @classmethod
    def from_frames_dir(cls, d, instruction: str, sampling: SamplingSpec, **kw) -> Trajectory:
        from .video import read_frames_dir
        frames, idx = read_frames_dir(d, sampling)
        return cls(frames=frames, instruction=instruction, frame_indices=idx,
                   sampling=sampling, source=str(d), **kw)

    @classmethod
    def from_array(cls, arr, instruction: str, sampling: SamplingSpec | None = None,
                   **kw) -> Trajectory:
        """``arr`` must hold the **original full frame sequence**; it is sampled here
        when ``sampling`` is given.  For already sampled data construct a Trajectory
        directly and pass frame_indices explicitly."""
        from .video import resize_frames
        arr = np.asarray(arr)
        idx = np.arange(len(arr), dtype=np.int64)
        if sampling is not None:
            sel = sampling.indices(len(arr), kw.get("fps"))
            arr, idx = arr[sel], sel
            if sampling.resize:
                arr = resize_frames(arr, sampling.resize)
        return cls(frames=arr, instruction=instruction, frame_indices=idx,
                   sampling=sampling, **kw)

    @classmethod
    def from_views(cls, views: dict, instruction: str, sampling: SamplingSpec,
                   **kw) -> Trajectory:
        """views: {view_name: video path or frame directory}.  All views share one
        sampling protocol."""
        from .video import read_frames
        out, idx, fps = {}, None, None
        for name, p in views.items():
            f, i, fp = read_frames(p, sampling)
            out[name] = f
            if idx is None:
                idx, fps = i, fp
            elif not np.array_equal(i, idx):
                raise ValueError(
                    f"the sampled grid of view {name!r} {i.tolist()} differs from the "
                    f"other views {idx.tolist()}")
            elif fps is not None and fp is not None and not np.isclose(fp, fps):
                raise ValueError(
                    f"view {name!r} has FPS={fp}, the other views have FPS={fps}")
        return cls(frames=out, instruction=instruction, frame_indices=idx, fps=fps,
                   sampling=sampling, source=json.dumps({k: str(v) for k, v in views.items()}),
                   **kw)


# --------------------------------------------------------------------------
@dataclass
class CostRecord:
    latency_s: float = 0.0
    n_calls: int = 0
    peak_vram_gb: float | None = None

    def __post_init__(self):
        if not np.isfinite(self.latency_s) or self.latency_s < 0:
            raise ValueError("latency_s must be finite and >= 0")
        if (not isinstance(self.n_calls, int) or isinstance(self.n_calls, bool)
                or self.n_calls < 0):
            raise ValueError("n_calls must be a non-negative integer")
        if self.peak_vram_gb is not None and (
                not np.isfinite(self.peak_vram_gb) or self.peak_vram_gb < 0):
            raise ValueError("peak_vram_gb must be a finite non-negative number or None")

    def to_dict(self) -> dict:
        return asdict(self)

    def __add__(self, o: CostRecord) -> CostRecord:
        vals = [x for x in (self.peak_vram_gb, o.peak_vram_gb) if x is not None]
        return CostRecord(self.latency_s + o.latency_s, self.n_calls + o.n_calls,
                          max(vals) if vals else None)


@dataclass
class RewardOutput:
    """The unified output.  progress is always on the raw scale (rule 2)."""

    progress: np.ndarray                    # [T]
    frame_indices: np.ndarray               # [T] source-video frame numbers
    weights: np.ndarray | None = None       # [T] per-step quantity (Robo-Dopamine hop, ...)
    success: np.ndarray | None = None       # [T] per-frame success probability (Robometer)
    raw: dict = field(default_factory=dict)
    model_name: str = ""
    sampling: SamplingSpec | None = None
    normalize: str = Normalize.NONE.value
    cost: CostRecord | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.progress = np.asarray(self.progress, dtype=np.float64)
        self.frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        if self.progress.ndim != 1:
            raise ValueError(f"progress must be 1-D, got {self.progress.shape}")
        if self.frame_indices.ndim != 1:
            raise ValueError(f"frame_indices must be 1-D, got {self.frame_indices.shape}")
        if len(self.progress) == 0:
            raise ValueError("progress must not be empty")
        if not np.all(np.isfinite(self.progress)):
            raise ValueError(f"progress contains non-finite values (NaN/Inf): {self.progress}")
        if len(self.progress) != len(self.frame_indices):
            raise ValueError(f"progress length {len(self.progress)} != "
                             f"frame_indices length {len(self.frame_indices)}")
        if np.any(self.frame_indices < 0):
            raise ValueError("frame_indices must not contain negative values")
        if len(self.frame_indices) > 1 and np.any(np.diff(self.frame_indices) <= 0):
            raise ValueError("frame_indices must be strictly increasing")
        for name in ("weights", "success"):
            v = getattr(self, name)
            if v is not None:
                v = np.asarray(v, dtype=np.float64)
                if len(v) != len(self.progress):
                    raise ValueError(f"{name} must have the same length as progress")
                if not np.all(np.isfinite(v)):
                    raise ValueError(f"{name} must not contain NaN/Inf")
                if name == "success" and np.any((v < 0) | (v > 1)):
                    raise ValueError("success must be a probability in [0,1]")
                setattr(self, name, v)

    def __len__(self) -> int:
        return len(self.progress)

    @property
    def terminal_success(self) -> float | None:
        return None if self.success is None else float(self.success[-1])

    def normalized(self, method: str = "minmax") -> np.ndarray:
        """Normalisation happens at the consumer.  Note: this differs from TOPReward's
        official normalize_rewards (which returns 1.0 for a single point / all-equal
        values; this returns 0.0).  Use adapters.topreward.normalize_official for the
        official semantics."""
        x = self.progress
        if method == Normalize.NONE.value:
            return x.copy()
        if method == Normalize.MINMAX.value:
            lo, hi = float(x.min()), float(x.max())
            return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)
        if method == Normalize.ZSCORE.value:
            s = float(x.std())
            return np.zeros_like(x) if s < 1e-12 else (x - float(x.mean())) / s
        if method == Normalize.RANK.value:
            from scipy.stats import rankdata
            return (rankdata(x) - 1) / max(1, len(x) - 1)
        raise ValueError(f"unknown normalisation method: {method!r}")

    def resample_to(self, target, extrapolate: str = "raise") -> RewardOutput:
        if extrapolate not in ("raise", "constant"):
            raise ValueError(f"extrapolate must be 'raise' or 'constant', got {extrapolate!r}")
        t = np.asarray(target, dtype=np.int64)
        lo, hi = int(self.frame_indices[0]), int(self.frame_indices[-1])
        oob = t[(t < lo) | (t > hi)]
        if len(oob) and extrapolate == "raise":
            raise ValueError(f"target points {oob.tolist()[:5]} are outside the frame range "
                             f"[{lo}, {hi}]. Pass extrapolate='constant' to extrapolate.")
        return RewardOutput(
            progress=np.interp(t, self.frame_indices, self.progress),
            frame_indices=t,
            weights=None if self.weights is None else np.interp(t, self.frame_indices, self.weights),
            success=None if self.success is None else np.interp(t, self.frame_indices, self.success),
            raw=dict(self.raw), model_name=self.model_name, sampling=self.sampling,
            normalize=self.normalize, cost=self.cost,
            meta={**self.meta, "resampled_from": self.frame_indices.tolist()},
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "normalize": self.normalize,
            "progress": self.progress.tolist(),
            "frame_indices": self.frame_indices.tolist(),
            "weights": None if self.weights is None else self.weights.tolist(),
            "success": None if self.success is None else self.success.tolist(),
            "raw": _jsonable(self.raw),
            "cost": self.cost.to_dict() if self.cost else None,
            "meta": _jsonable(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RewardOutput:
        return cls(
            progress=np.asarray(d["progress"]),
            frame_indices=np.asarray(d["frame_indices"]),
            weights=None if d.get("weights") is None else np.asarray(d["weights"]),
            success=None if d.get("success") is None else np.asarray(d["success"]),
            raw=d.get("raw", {}), model_name=d.get("model", ""),
            normalize=d.get("normalize", "none"),
            cost=CostRecord(**d["cost"]) if d.get("cost") else None,
            meta=d.get("meta", {}),
        )


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (bool, int, float, str)) or o is None:
        return o
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    return str(o)
