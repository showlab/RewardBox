"""VLAC official adapter.

Upstream:
  https://github.com/InternRobotics/VLAC (commit dc57e571)
Entry point:
  evo_vlac:GAC_model.get_trajectory_critic

VLAC is natively a pair-wise image critic.  Its official helper turns adjacent
critic scores into a cumulative ``value_list`` whose nominal scale is 0--100.
RewardBox deliberately keeps that list unchanged:

* no ``/ 100`` normalization;
* no padding/truncation to the input length;
* no video encoding/compression round-trip.

The last point is possible because the official method accepts PIL images in
memory.  RewardBox maps its actual output grid back to ``Trajectory.frame_indices``
and records the pair-wise critic scores in ``raw``.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np

from ..core.base import BaseAdapter
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import RewardOutput, SamplingSpec, Trajectory
from ..upstream import UpstreamNotFoundError, upstream_candidates


def _requirements(model_name: str) -> Requirements:
    # Mirrors the upstream pyproject instead of forcing VLAC into RewardBox's
    # shared lock.  ms-swift 3.3 is old enough, and invasive enough, that the
    # recommended deployment is an isolated benchmark worker/environment.
    return Requirements(
        pip=(
            "torch==2.8.0",
            "torchvision==0.23.0",
            "transformers==4.51.3",
            "peft>=0.15.2",
            "ms-swift==3.3",
            "timm",
            "sentencepiece==0.1.99",
            "flash-attn==2.8.3",
            "Pillow",
            "loguru",
            "matplotlib",
            "requests",
            "opencv-python-headless",
        ),
        python=">=3.9",
        note=(
            "The official implementation depends on the evo-vlac source and ms-swift==3.3. Set "
            "REWARDBOX_VLAC_ROOT, or put reward_model_code/VLAC inside the RewardBox "
            "repository. Upstream recommends Python 3.10 / transformers 4.51.3; point "
            "--worker-python at a separate VLAC environment. torch is pinned to 2.8.0 (not "
            "upstream's >=2.0): on 2026-07-26 an unconstrained resolve picked torch 2.13+cu13, "
            "which has no prebuilt flash-attn wheel. init_model fixes attn_impl='flash_attn'; "
            "without flash_attn it is a hard ImportError, never a silent fallback to sdpa (the "
            "same class of problem as TOPReward). sentencepiece is pinned to 0.1.99: 0.2.x "
            "validates InternLM2's tokenizer.model more strictly, which makes transformers turn "
            "a failed tokenizer initialisation silently into a bare bool return value instead "
            "of an error. All three are listed in docs/installation.md."
        ),
        install_hint=(
            "uv venv --python 3.10 .venv-vlac",
            "uv pip install --python .venv-vlac/bin/python "
            "-r locks/vlac-py310-requirements.in",
            "uv pip install --python .venv-vlac/bin/python "
            "https://github.com/Dao-AILab/flash-attention/releases/download/"
            "v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-"
            "linux_x86_64.whl  # choose TRUE/FALSE from torch._C._GLIBCXX_USE_CXX11_ABI",
            "uv pip install --python .venv-vlac/bin/python -e .[dev]",
            "uv pip install --python .venv-vlac/bin/python "
            "--no-deps -e reward_model_code/VLAC",
            "rewardbox bench ... --worker-python "
            f"{model_name}=.venv-vlac/bin/python",
        ),
    )


_COMMON = dict(
    cls="VLACAdapter",
    module=__name__,
    reward_mode=RewardMode.PROGRESS,
    signal_type=SignalType.GENERATED_TEXT,
    # 64 = VLAC's max_frames in the upstream eval_commands, with pad_frames=false; since
    # SamplingSpec.num_frames is a cap (all frames when T<N, no upsampling) the two are equivalent.
    sampling=SamplingSpec(num_frames=64, strategy="uniform"),
    # Upstream calls this a 0--100 value, but its cumulative formula is not
    # clamped and can go below zero for a sufficiently negative critic.
    score_range=(-100.0, 100.0),
    higher_better=True,
    supports_pairwise=True,
    outputs_success=False,
    paper="https://arxiv.org/abs/2509.15937",
    upstream="evo_vlac:GAC_model.get_trajectory_critic",
    description=(
        "Official pair-wise image critic; adjacent-frame critics are accumulated by the "
        "official formula into nominal 0-100 progress. RewardBox uses the in-memory image "
        "entry point and keeps the raw scale."
    ),
    implementation_kind=ImplementationKind.OFFICIAL_ADAPTER,
)

SPECS = [
    ModelSpec(
        name="vlac-2b",
        weights="InternRobotics/VLAC",
        model_opts={"model_path": "InternRobotics/VLAC"},
        requires=_requirements("vlac-2b"),
        **_COMMON,
    ),
    ModelSpec(
        name="vlac-8b",
        weights="InternRobotics/VLAC-8b",
        model_opts={"model_path": "InternRobotics/VLAC-8b"},
        requires=_requirements("vlac-8b"),
        **_COMMON,
    ),
]


def _repo_root() -> Path:
    """Locate an upstream source checkout without importing its heavy package."""
    explicit = os.environ.get("REWARDBOX_VLAC_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if (root / "evo_vlac" / "utils" / "model_utils.py").exists():
            return root
        raise RuntimeError(
            f"REWARDBOX_VLAC_ROOT={explicit!r} has no "
            "evo_vlac/utils/model_utils.py"
        )

    for candidate in upstream_candidates("VLAC"):
        if (candidate / "evo_vlac" / "utils" / "model_utils.py").exists():
            return candidate
    raise UpstreamNotFoundError(
        "official VLAC repository not found. Set REWARDBOX_VLAC_ROOT to its root, "
        "or clone it to reward_model_code/VLAC."
    )


def _pil_images(frames):
    """Convert RGB uint8 arrays to the in-memory type accepted by upstream."""
    from PIL import Image

    return [Image.fromarray(np.ascontiguousarray(frame)) for frame in frames]


class VLACAdapter(BaseAdapter):
    def __init__(
        self,
        spec,
        model_path: str | None = None,
        model_type: str = "internvl2",
        device_map: str = "auto",
        temperature: float = 0.5,
        top_k: int = 1,
        batch_size: int = 5,
        skip: int = 1,
        rich: bool = True,
        think: bool = False,
        value_simple: bool | str = True,
        reference_name: str = "reference",
        **opts,
    ):
        """Configure only parameters exposed by the official image API.

        ``skip`` is measured on the already sampled RewardBox trajectory.
        ``rich`` is upstream's digit-logprob interpolation switch; it does not
        change the cumulative value unit from percentage to [0, 1]. It defaults to
        True to match the official Robometer VLAC baseline, which produced the
        paper's Table I number; upstream's own function default is False, which
        quantises each critic to an integer and measurably costs correlation
        (360-episode reward-alignment membership: rich=False + 16 frames gives
        ID 0.2797 / OOD 0.3841, rich=True + 64 frames gives 0.1137 / 0.1785,
        against Table I's 0.16 / 0.17).
        ``reference_name`` is the key of the reference-trajectory image group in
        ``Trajectory.reference_images``; without it the official ref_num=0 path (no
        reference) is used.
        """
        super().__init__(spec, **opts)
        if not isinstance(reference_name, str) or not reference_name.strip():
            raise ValueError("reference_name must be a non-empty string")
        self.reference_name = reference_name
        for name, value in (("batch_size", batch_size), ("skip", skip), ("top_k", top_k)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or temperature <= 0
        ):
            raise ValueError("temperature must be a finite positive number")
        if not isinstance(rich, bool) or not isinstance(think, bool):
            raise ValueError("rich and think must be bools")
        if not (
            isinstance(value_simple, bool)
            or (type(value_simple) is str and value_simple == "mix_f")
        ):
            raise ValueError("value_simple must be true, false or 'mix_f'")
        if not isinstance(model_type, str) or not model_type.strip():
            raise ValueError("model_type must be a non-empty string")
        if not isinstance(device_map, str) or not device_map.strip():
            raise ValueError("device_map must be a non-empty string")

        self.model_path = model_path or spec.model_opts.get("model_path") or spec.weights
        self.model_type = model_type
        self.device_map = device_map
        self.temperature = float(temperature)
        self.top_k = top_k
        self.batch_size = batch_size
        self.skip = skip
        self.rich = rich
        self.think = think
        self.value_simple = value_simple
        self._critic = None

    def critic(self):
        if self._critic is None:
            # A source checkout is supported without an editable install.  Do
            # not locate/import it during registry discovery.
            if importlib.util.find_spec("evo_vlac") is None:
                root = _repo_root()
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
            from evo_vlac import GAC_model

            critic = GAC_model(tag="critic")
            critic.init_model(
                model_path=self.model_path,
                model_type=self.model_type,
                device_map=self.device_map,
            )
            critic.temperature = self.temperature
            critic.top_k = self.top_k
            critic.set_config()
            critic.set_system_prompt()
            self._critic = critic
        return self._critic

    def _official_call(
        self, frames, instruction: str, *, skip: int, ref_frames=None
    ) -> tuple[list[float], list[float]]:
        # Official semantics (evo_vlac.utils.model_utils.get_trajectory_critic): ref_image_list
        # is one **reference trajectory** and ref_num is how many images are drawn from it at
        # even spacing; with ref_image_list None upstream forces ref_num to 0, and the score
        # prompt's trajectory_len becomes 0 too.  Passing ref_num=len(ref) gives delta=1, i.e.
        # all images in their original order, matching the official image mode.
        ref_list = None if ref_frames is None else _pil_images(ref_frames)
        critic_list, value_list = self.critic().get_trajectory_critic(
            task=instruction,
            image_list=_pil_images(frames),
            ref_image_list=ref_list,
            batch_num=self.batch_size,
            ref_num=0 if ref_list is None else len(ref_list),
            think=self.think,
            skip=skip,
            rich=self.rich,
            reverse_eval=False,
            frame_skip=True,
            value_simple=self.value_simple,
        )
        try:
            critics = [float(value) for value in critic_list]
            values = [float(value) for value in value_list]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "the official VLAC entry point returned critic/value that do not parse as finite floats"
            ) from exc
        if not np.all(np.isfinite(critics)) or not np.all(np.isfinite(values)):
            raise RuntimeError("the official VLAC entry point returned NaN/Inf")
        return critics, values

    def _score(self, traj: Trajectory) -> RewardOutput:
        local_indices = np.asarray(
            [0, *range(self.skip, len(traj), self.skip)], dtype=np.int64
        )
        ref_frames = None
        if traj.reference_images and self.reference_name in traj.reference_images:
            ref_frames = traj.reference(self.reference_name)
        critics, values = self._official_call(
            traj.primary, traj.instruction, skip=self.skip, ref_frames=ref_frames
        )
        if len(values) != len(local_indices):
            raise RuntimeError(
                f"VLAC value_list length {len(values)} does not match the grid length "
                f"{len(local_indices)} of official skip={self.skip}"
            )
        if len(critics) != max(0, len(local_indices) - 1):
            raise RuntimeError(
                f"VLAC critic_list length {len(critics)} does not match value_list length "
                f"{len(values)}"
            )

        pair_local = [
            [int(i - self.skip), int(i)]
            for i in range(self.skip, len(traj), self.skip)
        ]
        pair_frames = [
            [int(traj.frame_indices[a]), int(traj.frame_indices[b])]
            for a, b in pair_local
        ]
        return self._out(
            values,
            traj,
            frame_indices=traj.frame_indices[local_indices],
            weights=None,
            raw={
                "critic_list": critics,
                "official_value_list": values,
                "critic_pair_local_indices": pair_local,
                "critic_pair_frame_indices": pair_frames,
            },
            meta={
                "adapter_calls_official": True,
                "protocol": "controlled_in_memory_images",
                "model_path": self.model_path,
                "model_type": self.model_type,
                "device_map": self.device_map,
                "official_args": {
                    "batch_num": self.batch_size,
                    "skip": self.skip,
                    "rich": self.rich,
                    "think": self.think,
                    "frame_skip": True,
                    "reverse_eval": False,
                    "ref_num": 0,
                    "value_simple": self.value_simple,
                },
                "reference_trajectory": "none",
                "value_scale": "official_nominal_0_to_100_unclamped",
                "normalized": False,
                "padded_or_truncated": False,
                "first_value": "official_zero_without_pair",
                "n_backend_calls": math.ceil(len(critics) / self.batch_size),
            },
        )

    def _compare(self, a: Trajectory, b: Trajectory) -> float:
        """Compare terminal states with the native pair-wise image critic.

        Upstream returns a positive score when its *second* image is closer to
        task completion.  Passing ``b`` first and ``a`` second therefore matches
        RewardBox's contract: positive means ``a`` is better.
        """
        if a.instruction.strip() != b.instruction.strip():
            raise ValueError("VLAC compare() requires both trajectories to use the same instruction")
        # Upstream converts/resizes each image to 448x448; keep a list instead of np.stack,
        # which would force both trajectories to share their original resolution.
        frames = [b.primary[-1], a.primary[-1]]
        critics, values = self._official_call(frames, a.instruction, skip=1)
        if len(critics) != 1 or len(values) != 2:
            raise RuntimeError(
                "a two-image VLAC compare should return 1 critic and 2 values, "
                f"got {len(critics)} / {len(values)}"
            )
        return critics[0]

    def close(self) -> None:
        self._critic = None
