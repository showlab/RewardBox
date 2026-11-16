"""Adapter base class.

**RewardBox is an adapter layer, not a re-implementation.**  Every adapter does
exactly four things:
  1. Trajectory -> the input form the official API expects
  2. call the official entry point
  3. official output -> RewardOutput
  4. record every conversion step in meta

Any code beyond these four needs an explicit reason and must be marked as a
RewardBox extension in meta.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from .spec import DependencyError, ModelSpec
from .types import CostRecord, RewardOutput, SamplingSpec, Trajectory


class BaseAdapter(ABC):
    """Subclass contract:
      - **no** torch / transformers / vllm import at module top level (the manifest
        must build in a light environment)
      - score() returns the official raw scale, no normalisation (rule 2)
      - never read the video or choose the frame count itself (rule 3)
    """

    def __init__(self, spec: ModelSpec, check_deps: bool = True, **opts):
        self.spec = spec
        self.opts = opts
        if check_deps:
            rep = spec.requires.check()
            if not rep.ok:
                raise DependencyError(rep.format(spec.name))

    @property
    def name(self) -> str:
        return self.spec.name

    # ---- implemented by subclasses ----
    @abstractmethod
    def _score(self, traj: Trajectory) -> RewardOutput:
        """Call the official entry point and convert its output."""

    def _compare(self, a: Trajectory, b: Trajectory) -> float:
        raise NotImplementedError(
            f"{self.name} does not implement compare() "
            f"(spec.supports_pairwise={self.spec.supports_pairwise})")

    # ---- public ----
    def score(self, traj: Trajectory) -> RewardOutput:
        self._check_views(traj)
        _reset_peak_vram()
        t0 = time.perf_counter()
        out = self._score(traj)
        elapsed = time.perf_counter() - t0

        out.model_name = out.model_name or self.spec.name
        out.sampling = out.sampling or traj.sampling
        cost = out.cost or CostRecord()
        cost.latency_s = elapsed
        cost.n_calls = int(out.meta.get("n_backend_calls", cost.n_calls or 1))
        cost.peak_vram_gb = _peak_vram()
        out.cost = cost
        out.meta.setdefault("upstream", self.spec.upstream)
        out.meta.setdefault("weights", self.spec.weights)
        return out

    def compare(self, a: Trajectory, b: Trajectory) -> float:
        """>0 means a is better."""
        self._check_views(a)
        self._check_views(b)
        return self._compare(a, b)

    def __call__(self, video, instruction: str | None = None, *,
                 sampling: SamplingSpec | None = None, **trajectory_kw) -> RewardOutput:
        """pyiqa-style shortcut: ``rm("robot.mp4", "open the drawer")``.

        ``video`` may be
          - a :class:`Trajectory` (then ``instruction``/``sampling``/kwargs must be omitted),
          - a video file or a directory of frame images,
          - ``{view_name: path}`` for multi-view models (e.g. ``{"primary": ...,
            "left_wrist": ...}``),
          - a raw ``[T,H,W,3]`` uint8 RGB array of *all* frames (pass ``fps=`` when the
            model samples by time).

        Frames are sampled with the model's own default protocol (``spec.sampling``)
        unless ``sampling`` is given, so the shortcut stays on the upstream recipe.
        Extra keyword arguments (``goal_image``, ``reference_images``, ``fps``, ...)
        go to the :class:`Trajectory` constructor.
        """
        if isinstance(video, Trajectory):
            if instruction is not None or sampling is not None or trajectory_kw:
                raise TypeError(
                    "a Trajectory already carries its instruction and sampling; "
                    "call rm(traj) without extra arguments")
            return self.score(video)
        if not isinstance(instruction, str) or not instruction.strip():
            raise TypeError("instruction is required: rm(video, \"task instruction\")")
        protocol = sampling if sampling is not None else self.spec.sampling
        if isinstance(video, dict):
            traj = Trajectory.from_views(video, instruction, protocol, **trajectory_kw)
        elif isinstance(video, np.ndarray):
            traj = Trajectory.from_array(video, instruction, protocol, **trajectory_kw)
        else:
            traj = Trajectory.from_video(video, instruction, protocol, **trajectory_kw)
        return self.score(traj)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- helpers ----
    def _check_views(self, traj: Trajectory) -> None:
        have, need = set(traj.views), set(self.spec.required_views)
        if not need <= have:
            raise ValueError(
                f"{self.name} needs views {sorted(need)}, the trajectory only has "
                f"{sorted(have)}. Build it with "
                "Trajectory.from_views({'primary': ..., 'left_wrist': ...}).")

    def _out(self, progress, traj: Trajectory, *, frame_indices=None, **kw) -> RewardOutput:
        return RewardOutput(
            progress=np.asarray(progress, dtype=np.float64),
            frame_indices=traj.frame_indices if frame_indices is None else frame_indices,
            model_name=self.spec.name, sampling=traj.sampling, **kw)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.spec.name!r})"


def _peak_vram() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return max(
                torch.cuda.max_memory_allocated(i)
                for i in range(torch.cuda.device_count())
            ) / 1e9
    except Exception:
        pass
    return None


def _reset_peak_vram() -> None:
    """Make the CostRecord peak belong to this score() call, not to the process's
    historical maximum."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
    except Exception:
        # Cost is an optional observation; platforms without memory counters must
        # not block inference.
        pass
