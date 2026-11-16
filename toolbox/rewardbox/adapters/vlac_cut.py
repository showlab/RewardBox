"""VLAC-Cut official adapter.

Upstream:
  https://github.com/InternRobotics/VLAC-cut (commit 354df56f)
Entry points, imported from the pinned checkout instead of re-implemented:
  scripts/run_example.py
      compute_sampled_indices / build_chunk_all_prompt / load_swift_runtime /
      parse_point_blocks / scale_times  (README Quick Start, ms-swift PtEngine)
  scripts/run_vpb_inference.py
      load_model_and_processor / infer_one  (``backend="transformers"``, the
      Video Progress Benchmark inference path).  2026-10-06: blocked upstream —
      under the transformers==4.57.6 pin from upstream's own reward-env recipe,
      ``infer_one`` passes ``return_metadata=True`` and BatchFeature fails to
      tensorise ``VideoMetadata`` before the script pops it.  The upstream
      transformers version this script targets is undocumented; not patched here.
  scripts/evaluate_vpb_predictions.py
      align_curve_to_length  (``curve_alignment="vpb_index"``)
  real_robot_rl/server/vlarlkit/rewards/vlm_progress.py
      interpolate_progress  (``curve_alignment="time"``, the real-robot RL reward path)

VLAC-Cut is a fine-tuned Qwen3-VL-30B-A3B that reads a 2 Hz frame sequence and
*generates* sparse keypoints ``时间: X.Xs, 进度: Y%``.  Progress is signed and
non-monotonic (regression is negative), nominally percent, never clamped here.

Upstream turns keypoints into per-frame values in two different ways, and both
are official: the VPB evaluator stretches the first/last keypoint onto the
first/last input frame (index-normalised), while the RL reward path linearly
interpolates on absolute time with constant ends.  RewardBox computes both,
returns the one selected by ``curve_alignment`` as ``progress``, and keeps the
other plus the raw keypoints and response in ``raw``.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..core.base import BaseAdapter
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import RewardOutput, SamplingSpec, Trajectory
from ..upstream import UpstreamNotFoundError, upstream_candidates

_UPSTREAM_REPO = "https://github.com/InternRobotics/VLAC-cut"
_UPSTREAM_COMMIT = "354df56ff177018f4ed3849996b3b1fc9cadedc8"
_WEIGHTS_REPO = "InternRobotics/VLAC-Cut"
_WEIGHTS_REVISION = "2a809cbb10d63c2e5b1b3f046426c554f437bdb5"

# run_example.py: MODEL_ASSUMED_SAMPLE_HZ.  For image-list videos ms-swift always tells the
# model qwen_vl_utils.vision_process.FPS (default 2.0), so the model's output time axis is 2 Hz.
_MODEL_ASSUMED_SAMPLE_HZ = 2.0
# run_example.py::load_swift_runtime / run_vpb_inference.py::load_transformers_runtime
# values set with os.environ.setdefault; ms-swift writes them into qwen_vl_utils at model load.
_UPSTREAM_VISION_ENV = {
    "IMAGE_MAX_TOKEN_NUM": "256",
    "VIDEO_MAX_TOKEN_NUM": "256",
    "VIDEO_MIN_TOKEN_NUM": "4",
    "QWEN_VL_UTILS_MAX_FRAME_LIST": "0",
}
_BACKENDS = ("swift", "transformers")
_FRAME_SELECTIONS = ("official", "as_given")
_CURVE_ALIGNMENTS = ("vpb_index", "time")

_REQ = Requirements(
    # copied verbatim from the reward-environment recipe in upstream real_robot_rl/server/README.md.
    pip=(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers==4.57.6",
        "ms-swift==3.12.4",
        "qwen-vl-utils==0.0.14",
        "decord==0.6.0",
        "accelerate",
        "Pillow",
    ),
    python=">=3.11,<3.12",
    vram_gb=65.0,
    note=(
        "The upstream reward environment pins torch 2.6 / transformers 4.57.6 / ms-swift 3.12.4, "
        "which conflicts with the shared environment (torch 2.8 / transformers 4.57.1); it lives in "
        "a separate .venv-vlac-cut (Python 3.11). bf16 weights are about 62 GB. Set "
        "REWARDBOX_VLAC_CUT_ROOT, or put the upstream repository at reward_model_code/VLAC-cut."
    ),
    install_hint=(
        "uv venv --python 3.11 .venv-vlac-cut",
        "uv pip install --python .venv-vlac-cut/bin/python "
        "-r locks/vlac-cut-py311-requirements.in",
        "uv pip install --python .venv-vlac-cut/bin/python --no-deps -e .",
        "HF_HOME=$PWD/.hf .venv/bin/python -c \"from huggingface_hub import "
        f"snapshot_download; snapshot_download('{_WEIGHTS_REPO}', "
        f"revision='{_WEIGHTS_REVISION}')\"",
        "rewardbox bench ... --worker-python vlac-cut=.venv-vlac-cut/bin/python",
    ),
)

SPEC = ModelSpec(
    name="vlac-cut",
    cls="VLACCutAdapter",
    module=__name__,
    reward_mode=RewardMode.PROGRESS,
    signal_type=SignalType.GENERATED_TEXT,
    requires=_REQ,
    model_opts={"model_path": _WEIGHTS_REPO},
    # the official entry point takes the whole video and picks frames by sample_hz itself (compute_sampled_indices).
    sampling=SamplingSpec(strategy="all"),
    # signed progress: regression is negative and upstream does not clamp (the VPB evaluator only counts out-of-range values).
    score_range=(-100.0, 100.0),
    higher_better=True,
    supports_pairwise=False,
    outputs_success=False,
    paper="https://arxiv.org/abs/2607.09776",
    weights=f"{_WEIGHTS_REPO}@{_WEIGHTS_REVISION}",
    upstream=(
        f"{_UPSTREAM_REPO}@{_UPSTREAM_COMMIT}; scripts/run_example.py "
        "(ms-swift PtEngine) | scripts/run_vpb_inference.py (transformers)"
    ),
    description=(
        "Qwen3-VL-30B-A3B video progress critic: a 2 Hz frame sequence -> generated, timestamped, "
        "signed progress keypoints (may regress, non-monotonic). The raw response and keypoints are kept."
    ),
    implementation_kind=ImplementationKind.OFFICIAL_ADAPTER,
)


# --------------------------------------------------------------------------
def _repo_root() -> Path:
    """Locate the upstream checkout without importing anything heavy."""
    marker = Path("scripts") / "run_example.py"
    explicit = os.environ.get("REWARDBOX_VLAC_CUT_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if (root / marker).exists():
            return root
        raise RuntimeError(f"REWARDBOX_VLAC_CUT_ROOT={explicit!r} has no {marker}")
    for candidate in upstream_candidates("VLAC-cut"):
        if (candidate / marker).exists():
            return candidate
    raise UpstreamNotFoundError(
        "official VLAC-Cut repository not found. Set REWARDBOX_VLAC_CUT_ROOT, or clone it to "
        "reward_model_code/VLAC-cut."
    )


_UPSTREAM_CACHE: dict[str, object] = {}


def _upstream(name: str):
    """Load one pinned upstream file as a module (cached per process).

    The scripts are not a package, so they are loaded by path.  None of them
    imports torch/transformers/swift at module top level.
    """
    if name in _UPSTREAM_CACHE:
        return _UPSTREAM_CACHE[name]
    root = _repo_root()
    if name == "vlm_progress":
        # vlm_progress.py imports ``vlarlkit.utils.video_annotation`` (numpy/PIL).
        server = root / "real_robot_rl" / "server"
        if str(server) not in sys.path:
            sys.path.insert(0, str(server))
        module = importlib.import_module("vlarlkit.rewards.vlm_progress")
    else:
        path = root / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"_rewardbox_vlac_cut_{name}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load the VLAC-Cut upstream file {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    _UPSTREAM_CACHE[name] = module
    return module


def _resolve_model_dir(model_path: str, revision: str) -> tuple[str, str | None]:
    """Return a local checkpoint directory.

    ms-swift treats a bare model id as a ModelScope id, so a Hugging Face id is
    resolved to the pinned local HF snapshot first and never downloaded here.
    """
    local = Path(model_path).expanduser()
    if local.exists():
        return str(local.resolve()), None
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(model_path, revision=revision, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - re-raised with an actionable message
        raise RuntimeError(
            f"{model_path}@{revision} is not in the local HF cache; download the weights per install_hint, "
            "or pass a local directory as model_path."
        ) from exc
    return str(Path(path).resolve()), revision


class VLACCutAdapter(BaseAdapter):
    def __init__(
        self,
        spec,
        model_path: str | None = None,
        revision: str = _WEIGHTS_REVISION,
        backend: str = "swift",
        sample_hz: float = 2.0,
        frame_selection: str = "official",
        curve_alignment: str = "vpb_index",
        task_plan_key: str = "task_plan",
        prompt_key: str = "vlac_cut_prompt",
        max_new_tokens: int = 1024,
        device_map: str = "auto",
        dtype: str | None = None,
        attn_implementation: str | None = None,
        **opts,
    ):
        """
        ``backend``: ``swift`` = the ``run_example.py`` path of the README Quick Start / RL
        reward server (it produced the official example reference outputs); ``transformers``
        = the VPB inference path of ``run_vpb_inference.py``.  Their visual preprocessing differs.

        ``frame_selection``: ``official`` requires a complete dense trajectory (frames
        contiguous from 0, with fps) and picks frames by ``sample_hz`` with upstream
        ``compute_sampled_indices``; ``as_given`` uses the trajectory frames as they are and
        treats them as a sequence at the nominal ``sample_hz`` (as VPB manifests do).

        ``curve_alignment``: ``vpb_index`` = the official VPB evaluator's alignment; ``time``
        = absolute-time interpolation of the RL reward path.  Both are written to raw.

        ``task_plan_key`` / ``prompt_key``: optional task-plan text / full prompt override
        read from ``Trajectory.meta`` (run_example's --task-plan / --prompt).
        """
        super().__init__(spec, **opts)
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
        if frame_selection not in _FRAME_SELECTIONS:
            raise ValueError(
                f"frame_selection must be one of {_FRAME_SELECTIONS}, got {frame_selection!r}")
        if curve_alignment not in _CURVE_ALIGNMENTS:
            raise ValueError(
                f"curve_alignment must be one of {_CURVE_ALIGNMENTS}, got {curve_alignment!r}")
        if (
            not isinstance(sample_hz, (int, float))
            or isinstance(sample_hz, bool)
            or not math.isfinite(float(sample_hz))
            or sample_hz <= 0
        ):
            raise ValueError("sample_hz must be a finite positive number")
        if (
            backend == "transformers"
            and curve_alignment == "time"
            and not math.isclose(float(sample_hz), _MODEL_ASSUMED_SAMPLE_HZ)
        ):
            # run_vpb_inference tells the processor fps=sample_hz, and upstream never converts
            # its response times to video time (the VPB evaluator aligns by index, independent
            # of time units).  No conversion is invented here on upstream's behalf.
            raise ValueError(
                "backend='transformers' with sample_hz != 2.0 has no upstream time-axis conversion; "
                "use curve_alignment='vpb_index'")
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        for name, value in (
            ("task_plan_key", task_plan_key),
            ("prompt_key", prompt_key),
            ("device_map", device_map),
            ("revision", revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        self.model_path = model_path or spec.model_opts.get("model_path") or spec.weights
        self.revision = revision
        self.backend = backend
        self.sample_hz = float(sample_hz)
        self.frame_selection = frame_selection
        self.curve_alignment = curve_alignment
        self.task_plan_key = task_plan_key
        self.prompt_key = prompt_key
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self._engine = None
        self._runtime: dict = {}

    # ---------------------------------------------------------------- backend
    def _load(self):
        if self._engine is not None:
            return self._engine
        model_dir, revision = _resolve_model_dir(self.model_path, self.revision)
        if self.backend == "swift":
            run_example = _upstream("run_example")
            infer_request, pt_engine, request_config = run_example.load_swift_runtime()
            # argument-for-argument identical to run_example.py::main.
            engine = pt_engine(
                model_dir,
                model_type="qwen3_moe_vl",
                max_batch_size=1,
                device_map=self.device_map,
            )
            from qwen_vl_utils import vision_process

            self._engine = (engine, infer_request, request_config)
            self._runtime = {
                "qwen_vl_utils_effective": {
                    key: getattr(vision_process, key, None)
                    for key in (
                        "IMAGE_MAX_TOKEN_NUM", "IMAGE_MIN_TOKEN_NUM",
                        "VIDEO_MAX_TOKEN_NUM", "VIDEO_MIN_TOKEN_NUM", "FPS",
                    )
                },
            }
        else:
            vpb = _upstream("run_vpb_inference")
            model, processor = vpb.load_model_and_processor(
                model_path=model_dir,
                device_map=self.device_map,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
            )
            self._engine = (model, processor)
            self._runtime = {}
        self._runtime.update({
            "model_dir": model_dir,
            "hf_revision": revision,
            "env": {key: os.environ.get(key) for key in (*_UPSTREAM_VISION_ENV, "FPS")},
        })
        return self._engine

    def _generate(self, image_paths: list[str], prompt: str) -> str:
        engine = self._load()
        if self.backend == "swift":
            pt_engine, infer_request, request_config = engine
            request = infer_request(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "video", "video": image_paths},
                        {"type": "text", "text": prompt},
                    ],
                }]
            )
            # run_example.py's RequestConfig copied as is (greedy).
            response = pt_engine.infer(
                [request],
                request_config(
                    max_tokens=self.max_new_tokens, temperature=0.0, top_k=1, top_p=1.0),
            )[0].choices[0].message.content
        else:
            model, processor = engine
            response = _upstream("run_vpb_inference").infer_one(
                model=model,
                processor=processor,
                image_paths=image_paths,
                prompt=prompt,
                sample_hz=self.sample_hz,
                max_new_tokens=self.max_new_tokens,
            )
        return str(response or "")

    # ---------------------------------------------------------------- score
    def _select(self, traj: Trajectory) -> tuple[np.ndarray, list[float]]:
        """Return (local indices into traj, input timestamps in video seconds)."""
        if self.frame_selection == "as_given":
            local = np.arange(len(traj), dtype=np.int64)
            return local, [round(float(k) / self.sample_hz, 6) for k in local]
        if traj.fps is None:
            raise ValueError(
                "frame_selection='official' needs Trajectory.fps (upstream picks "
                f"{self.sample_hz:g} Hz frames by fps); for image-directory trajectories use frame_selection='as_given'")
        expected = np.arange(len(traj), dtype=np.int64)
        if not np.array_equal(traj.frame_indices, expected):
            raise ValueError(
                "frame_selection='official' needs a complete dense trajectory starting at frame 0 "
                "(SamplingSpec(strategy='all')); for sampled trajectories use frame_selection='as_given'")
        indices, timestamps = _upstream("run_example").compute_sampled_indices(
            len(traj), float(traj.fps), self.sample_hz)
        return np.asarray(indices, dtype=np.int64), [float(t) for t in timestamps]

    def _prompt(self, traj: Trajectory) -> tuple[str, str, str | None]:
        override = traj.meta.get(self.prompt_key)
        if override is not None:
            if not isinstance(override, str) or not override.strip():
                raise ValueError(f"Trajectory.meta[{self.prompt_key!r}] must be a non-empty string")
            return override.strip(), "raw_prompt", None
        plan = traj.meta.get(self.task_plan_key)
        if plan is not None and not isinstance(plan, str):
            raise ValueError(f"Trajectory.meta[{self.task_plan_key!r}] must be a string")
        plan = plan.strip() if plan else None
        prompt = _upstream("run_example").build_chunk_all_prompt(traj.instruction, plan)
        return prompt, "task_instruction_plus_plan", plan

    def _score(self, traj: Trajectory) -> RewardOutput:
        from PIL import Image

        local, input_times = self._select(traj)
        prompt, prompt_source, task_plan = self._prompt(traj)
        global_idx = traj.frame_indices[local]

        # upstream image boundary: decord RGB -> PIL JPEG quality=95 -> list of paths.
        with tempfile.TemporaryDirectory(prefix="rewardbox_vlac_cut_") as tmp:
            paths: list[str] = []
            for i, frame in zip(global_idx, traj.primary[local]):
                path = Path(tmp) / f"frame_{int(i):06d}.jpg"
                Image.fromarray(np.ascontiguousarray(frame)).save(path, quality=95)
                paths.append(str(path))
            response = self._generate(paths, prompt)

        run_example = _upstream("run_example")
        times_model, values = run_example.parse_point_blocks(response)
        if not values:
            raise RuntimeError(f"no keypoint could be parsed from the VLAC-Cut response: {response!r}")
        if self.backend == "swift":
            times_video = run_example.scale_times(
                times_model, from_hz=_MODEL_ASSUMED_SAMPLE_HZ, to_hz=self.sample_hz)
        else:
            times_video = list(times_model)

        n = len(local)
        vpb_curve = _upstream("evaluate_vpb_predictions").align_curve_to_length(
            times_model, values, n)
        time_curve = _upstream("vlm_progress").interpolate_progress(
            times_video, values, np.asarray(input_times, dtype=np.float32))
        vpb_curve = [float(v) for v in vpb_curve]
        time_curve = [float(v) for v in np.asarray(time_curve, dtype=np.float64)]
        if len(vpb_curve) != n or len(time_curve) != n:
            raise RuntimeError(
                f"VLAC-Cut curve alignment lengths {len(vpb_curve)}/{len(time_curve)} do not match the {n} input frames")
        progress = vpb_curve if self.curve_alignment == "vpb_index" else time_curve

        return self._out(
            progress,
            traj,
            frame_indices=global_idx,
            raw={
                "response": response,
                "prompt": prompt,
                "keypoint_times_model_sec": [float(t) for t in times_model],
                "keypoint_times_sec": [float(t) for t in times_video],
                "keypoint_progress": [float(v) for v in values],
                "input_timestamps_sec": input_times,
                "progress_vpb_index": vpb_curve,
                "progress_time": time_curve,
            },
            meta={
                "adapter_calls_official": True,
                "protocol": f"vlac_cut_{self.backend}",
                "backend": self.backend,
                "upstream_commit": _UPSTREAM_COMMIT,
                "model_path": self.model_path,
                **self._runtime,
                "sample_hz": self.sample_hz,
                "model_assumed_sample_hz": _MODEL_ASSUMED_SAMPLE_HZ,
                "frame_selection": self.frame_selection,
                "curve_alignment": self.curve_alignment,
                "prompt_source": prompt_source,
                "task_plan": task_plan,
                "generation": {
                    "max_new_tokens": self.max_new_tokens,
                    "greedy": True,
                },
                "image_boundary": "RGB-to-PIL-JPEG-quality-95",
                # RewardBox decodes videos with cv2; upstream run_example uses decord.
                "video_decoder": "rewardbox.core.video (cv2)",
                "value_scale": "official_signed_percent_unclamped",
                "normalized": False,
                "n_backend_calls": 1,
            },
        )

    def close(self) -> None:
        self._engine = None
