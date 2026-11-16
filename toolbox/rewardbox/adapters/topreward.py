"""TOPReward adapter; keeps apart two upstream protocols that must not share a name.

``topreward`` is the Robometer Table-I baseline: it calls the pinned
``robometer.evals.baselines.topreward.TopReward``, not the TOPReward repository's
``QwenClient`` directly.  They differ materially: the former resizes inputs to 224px,
defaults to ``add_chat_template=True``, and min-max normalises the prefix log-probs before
interpolating them into a full progress curve.  That is exactly the implementation run by
Robometer's ``eval_commands/{reward_alignment,policy_ranking}.sh`` with ``max_frames=64`` /
``model_config.num_prefix_samples=15``.

``topreward-qwen2.5-7b`` keeps TOPReward's native QwenClient protocol (15 prefixes,
``add_chat_template=False``, raw prefix log-probs).  It is not a Robometer Table-I model;
QwenClient also hard-codes the Qwen3 HF model class, so this Qwen2.5 variant must be
validated separately with real weights and must never borrow ``topreward``'s paper-alignment
conclusion.

Neither path outputs weights: TOPReward has no official dense-reward / positive-weight
formula (confirmed by a full grep), and no exp/clip transform is invented.
"""
from __future__ import annotations

import math
import numpy as np

from ..core.base import BaseAdapter
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import RewardOutput, SamplingSpec, Trajectory
from ..upstream import ensure_importable

_REQ = Requirements(
    pip=("torch>=2.8", "transformers>=4.57,<4.58", "qwen-vl-utils>=0.0.14",
         "accelerate", "loguru"),
    python=">=3.11",
    vram_gb=20.0,
    note="TOPReward's native QwenClient hard-codes attn_implementation=\"flash_attention_2\"; "
         "Robometer's Table-I TopReward also tries it first and falls back to sdpa/eager. To keep "
         "the pinned baseline environment identical, the flash-attn==2.8.3 pinned by the official "
         "pyproject is installed from a prebuilt wheel matching torch/cuda/python (see "
         "docs/installation.md and the header of locks/shared-requirements.in).",
)

_ROBOMETER_PROTOCOL = "robometer_table_i"
_NATIVE_PROTOCOL = "topreward_native"


_COMMON = dict(
    cls="TOPRewardAdapter", module=__name__,
    reward_mode=RewardMode.PROGRESS, signal_type=SignalType.TOKEN_LOGPROB,
    requires=_REQ,
    higher_better=True,
    paper="https://arxiv.org/pdf/2602.19313",
    implementation_kind=ImplementationKind.OFFICIAL_ADAPTER,
)

SPECS = [
    # Source: pinned robometer/eval_commands/{reward_alignment,policy_ranking}.sh
    # passes max_frames=64 and model_config.num_prefix_samples=15.  The remaining
    # constructor defaults (mean / chat-template=True / fps=2) come from the pinned
    # robometer/configs/eval_configs.py::TopRewardConfig.  The recipe's top-level
    # max_frames=64 is DataConfig.max_frames: the eval sampler subsamples whole
    # trajectories with robometer/data/datasets/helpers.py::linspace_subsample_frames
    # (np.rint, first/last forced) *before* TopReward sees them, so the baseline's own
    # np.linspace(..., dtype=int) cap never fires in the recipe.  "uniform" (rint) is
    # index-identical to that sampler; "uniform_floor" would not be.  No padding when T<64.
    ModelSpec(
        name="topreward",
        weights="Qwen/Qwen3-VL-8B-Instruct",
        model_opts={"model_name": "Qwen/Qwen3-VL-8B-Instruct", "protocol": _ROBOMETER_PROTOCOL},
        sampling=SamplingSpec(num_frames=64, strategy="uniform"),
        score_range=(0.0, 1.0),
        upstream="robometer.evals.baselines.topreward:TopReward.compute_progress",
        description=(
            "Robometer Table-I TOPReward baseline: official prefix log-probs are min-max "
            "normalised and linearly interpolated to every sampled input frame."
        ),
        **_COMMON,
    ),
    # This was an existing convenience variant, not an upstream Robometer baseline.
    # Keep its source-native defaults visible rather than silently giving it the
    # Table-I protocol or conclusion belonging only to the Qwen3-VL baseline above.
    ModelSpec(name="topreward-qwen2.5-7b", weights="Qwen/Qwen2.5-VL-7B-Instruct",
              model_opts={"model_name": "Qwen/Qwen2.5-VL-7B-Instruct", "protocol": _NATIVE_PROTOCOL},
              sampling=SamplingSpec(num_frames=15, strategy="uniform"),
              score_range=(-40.0, 0.0),
              upstream="topreward.clients.qwen:QwenClient.compute_instruction_rewards_for_prefixes",
              description=(
                  "Unverified non-Table-I TOPReward native-client variant. The vendored "
                  "QwenClient hard-codes Qwen3VLForConditionalGeneration; validate this "
                  "Qwen2.5 checkpoint independently before relying on it."
              ),
              **_COMMON),
]


def normalize_official(rewards) -> np.ndarray:
    """The **exact** semantics of the official QwenClient.normalize_rewards (verified).

    Differs from RewardOutput.normalized('minmax'):
        official: empty -> empty; single point -> [1.0]; all equal -> all 1
        generic:  single point -> [0.0]; all equal -> all 0
    """
    r = np.asarray(rewards, dtype=np.float64)
    if r.size == 0:
        return r
    if r.size == 1:
        return np.array([1.0])
    lo, hi = r.min(), r.max()
    if hi == lo:
        return np.ones_like(r)
    return (r - lo) / (hi - lo)


class TOPRewardAdapter(BaseAdapter):
    def __init__(self, spec, model_name: str | None = None, num_samples: int = 15,
                 reduction: str = "mean", fps: float = 2.0,
                 add_chat_template: bool | None = None,
                 use_video_description: bool = False, protocol: str | None = None,
                 **opts):
        """Build the one pinned protocol that belongs to ``spec``.

        The public ``topreward`` default is deliberately the Robometer Table-I
        protocol, not the similarly named native TOPReward client defaults.  An
        explicit incompatible protocol is rejected instead of silently changing
        what a benchmark number means.
        """
        super().__init__(spec, **opts)
        expected_protocol = spec.model_opts.get("protocol")
        if expected_protocol not in {_ROBOMETER_PROTOCOL, _NATIVE_PROTOCOL}:
            raise ValueError(f"{spec.name} lacks a known TOPReward protocol marker")
        if protocol is not None and protocol != expected_protocol:
            raise ValueError(
                f"{spec.name} only supports protocol={expected_protocol!r}; "
                f"refusing to silently switch to {protocol!r}")
        self.protocol = expected_protocol
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be mean|sum, got {reduction!r}")
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError(f"num_samples must be a positive integer, got {num_samples!r}")
        if (not isinstance(fps, (int, float)) or isinstance(fps, bool)
                or not math.isfinite(float(fps)) or fps <= 0):
            raise ValueError(f"fps must be a finite positive number, got {fps!r}")
        if add_chat_template is not None and not isinstance(add_chat_template, bool):
            raise ValueError("add_chat_template must be a bool or None")
        if not isinstance(use_video_description, bool):
            raise ValueError("use_video_description must be a bool")
        if self.protocol == _ROBOMETER_PROTOCOL and use_video_description:
            raise ValueError(
                "the Robometer Table-I TOPReward baseline has no use_video_description; "
                "native TOPReward extensions cannot be mixed into the paper protocol")

        self.model_name = model_name or spec.model_opts.get("model_name")
        self.num_samples = num_samples
        self.reduction = reduction
        self.fps = float(fps)
        self.add_chat_template = (
            self.protocol == _ROBOMETER_PROTOCOL
            if add_chat_template is None else add_chat_template
        )
        self.use_video_description = use_video_description
        self._client = None

    def client(self):
        if self._client is None:
            if self.protocol == _ROBOMETER_PROTOCOL:
                # This is the exact baseline implementation named by the Robometer
                # recipe, including its 224px frame conversion and minmax/interp
                # output semantics.  Do not substitute TOPReward's QwenClient here.
                ensure_importable("robometer")
                from robometer.evals.baselines.topreward import TopReward
                self._client = TopReward(
                    model_path=self.model_name,
                    max_frames=self.spec.sampling.num_frames,
                    num_prefix_samples=self.num_samples,
                    reduction=self.reduction,
                    add_chat_template=self.add_chat_template,
                    fps=self.fps,
                )
            else:
                ensure_importable("topreward")
                from topreward.clients.qwen import QwenClient
                self._client = QwenClient(model_name=self.model_name)
        return self._client

    @staticmethod
    def prefix_lengths(num_frames: int, num_samples: int) -> list[int]:
        """Reproduce the prefix-length sequence of the official
        compute_instruction_rewards_for_prefixes, to map official results back to source
        video frame numbers."""
        num_samples = min(num_samples, num_frames)
        if num_frames > 2:
            return sorted({int(x) for x in np.linspace(1, num_frames, num_samples, dtype=int)})
        return [num_frames]

    def _robometer_input_indices(self, n_frames: int) -> np.ndarray:
        """Map Robometer's private internal 64-frame cap back to source indices.

        ``TopReward.compute_progress`` itself performs this exact ``np.linspace``
        selection before returning a curve.  RewardBox needs the same map only to
        label its already-official output with original Trajectory frame indices.
        """
        if n_frames <= self.spec.sampling.num_frames:
            return np.arange(n_frames, dtype=np.int64)
        return np.linspace(
            0, n_frames - 1, self.spec.sampling.num_frames, dtype=int
        ).astype(np.int64)

    @staticmethod
    def _robometer_prefix_raw(result) -> dict:
        """Serialize evidence yielded by Robometer's own prefix helper."""
        return {
            "prefix_lengths": list(result.prefix_lengths or []),
            "prefix_rewards": list(result.prefix_rewards or []),
            "normalized_prefix_rewards": list(result.normalized_prefix_rewards or []),
            "reward": float(result.reward),
            "reduction": result.reduction,
            "token_count": int(result.token_count),
        }

    def _score_robometer_table_i(self, traj: Trajectory) -> RewardOutput:
        model = self.client()
        original_prefix_fn = model._compute_instruction_rewards_for_prefixes
        captured = []

        # Call the public compute_progress implementation exactly.  The tiny wrapper
        # merely retains the official helper's result for provenance; it neither
        # changes inputs nor reconstructs its minmax/interpolation computation.
        def capture_prefixes(frames, instruction):
            result = original_prefix_fn(frames, instruction)
            captured.append(result)
            return result

        model._compute_instruction_rewards_for_prefixes = capture_prefixes
        try:
            progress = np.asarray(
                model.compute_progress(traj.primary, traj.instruction), dtype=np.float64
            )
        finally:
            model._compute_instruction_rewards_for_prefixes = original_prefix_fn

        local_indices = self._robometer_input_indices(len(traj))
        if len(progress) != len(local_indices):
            raise RuntimeError(
                "output length of Robometer TOPReward.compute_progress does not match its max_frames grid: "
                f"{len(progress)} vs {len(local_indices)}")
        if len(captured) != 1:
            raise RuntimeError(
                "Robometer TOPReward.compute_progress did not call the official prefix helper exactly once: "
                f"{len(captured)} calls")

        prefix_raw = self._robometer_prefix_raw(captured[0])
        return self._out(
            progress,
            traj,
            frame_indices=traj.frame_indices[local_indices],
            weights=None,
            raw={
                **prefix_raw,
                "official_compute_progress": progress.tolist(),
                "input_local_indices": local_indices.tolist(),
            },
            meta={
                "adapter_calls_official": True,
                "protocol": _ROBOMETER_PROTOCOL,
                "paper_equivalence": "Robometer Table-I baseline recipe",
                "official_args": {
                    "max_frames": self.spec.sampling.num_frames,
                    "num_prefix_samples": self.num_samples,
                    "reduction": self.reduction,
                    "add_chat_template": self.add_chat_template,
                    "fps": self.fps,
                },
                "model_name": self.model_name,
                "progress_semantics": (
                    "official Robometer TopReward: minmax-normalized prefix log-probs "
                    "interpolated to every input frame"
                ),
                "n_backend_calls": len(prefix_raw["prefix_lengths"]),
            },
        )

    def _score_native(self, traj: Trajectory) -> RewardOutput:
        frames = [f for f in traj.primary]        # the official signature wants list[np.ndarray]
        res = self.client().compute_instruction_rewards_for_prefixes(
            frames=frames,
            instruction=traj.instruction,
            num_samples=self.num_samples,
            reduction=self.reduction,
            fps=self.fps,
            use_video_description=self.use_video_description,
            add_chat_template=self.add_chat_template,
        )
        lengths = np.asarray(res.prefix_lengths, dtype=np.int64)
        rewards = np.asarray(res.prefix_rewards, dtype=np.float64)

        return self._out(
            rewards, traj,
            frame_indices=traj.frame_indices[lengths - 1],   # prefix length L -> frame L-1
            weights=None,                                     # no such concept upstream
            raw={"prefix_lengths": res.prefix_lengths,
                 "prefix_rewards": res.prefix_rewards,
                 "normalized_prefix_rewards": res.normalized_prefix_rewards,
                 "reward": res.reward, "token_count": res.token_count},
            meta={"adapter_calls_official": True,
                  "protocol": _NATIVE_PROTOCOL,
                  "paper_equivalence": "not asserted; not a Robometer Table-I baseline",
                  "official_args": {"num_samples": self.num_samples,
                                    "reduction": self.reduction, "fps": self.fps,
                                    "add_chat_template": self.add_chat_template,
                                    "use_video_description": self.use_video_description},
                  "model_name": self.model_name,
                  "n_backend_calls": len(lengths)},
        )

    def _score(self, traj: Trajectory) -> RewardOutput:
        if self.protocol == _ROBOMETER_PROTOCOL:
            return self._score_robometer_table_i(traj)
        return self._score_native(traj)

    def close(self):
        self._client = None
