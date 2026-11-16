"""Robo-Dopamine v1/v2 adapter.

v1 is the official GRM-3B / GRM-8B, v2 is GRM-2.0-4B/8B-Preview.
Both generations share the official GRMInference; the old name robodopamine-8b stays as a
compatibility alias of v2.

Official: https://github.com/FlagOpen/Robo-Dopamine  commit 2c714abc  (Apache-2.0)
Entry:    examples.inference.GRMInference.run_pipeline

Official signature (verified; frame_interval=10 is the official function's own default, not a guess):
    run_pipeline(cam_high_path, cam_left_path, cam_right_path, out_root, task,
                 frame_interval=10, batch_size=1, goal_image=None,
                 eval_mode='incremental', visualize=False) -> str (run directory)

Four official rules followed as written:
  1. README §Simple Usage: *"we highly recommend averaging the inference reward
     results from all three modes to use as the final reward in RL"* -> default fuse=True
  2. progress values are read straight from the official pred_vllm.json progress / hop;
     **the accumulation formula is not recomputed**
  3. **no clamp**; the official source comments:
     "Not strictly enforcing clamping to allow debugging"
  4. without a goal image the official examples/blank_goal.png is used (the README stresses this)

Single view is an officially supported usage (README "More Cases" passes the same path for all three cams).

There is only one path: pass all frames and let the official frame_interval sample.  An
earlier "controlled" protocol (16 uniformly sampled frames fed with frame_interval=1, to put
several models on one sampling grid) left the official default frame_interval=10 and moved
the results further from the official benchmark numbers, so it was removed.  When a
comparison needs a common grid, the consumer side (ResultSet/metrics) handles each model's
own sampling grid.
"""
from __future__ import annotations

import json
import importlib
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..core.base import BaseAdapter
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import RewardOutput, SamplingSpec, Trajectory
from ..upstream import UpstreamNotFoundError, upstream_candidates

MODES = ("forward", "incremental", "backward")
VIEW_TO_CAM = {"primary": "cam_high", "left_wrist": "cam_left_wrist",
               "right_wrist": "cam_right_wrist"}
DEFAULT_GPU_MEMORY_UTILIZATION = 0.7

_REQ = Requirements(
    pip=("vllm>=0.11,<0.12", "transformers>=4.57,<4.58", "torch>=2.8",
         "matplotlib", "moviepy", "opencv-python-headless"),
    python=">=3.10",
    vram_gb=24.0,
    note="Upstream infers with vLLM. The adapter must add the Robo-Dopamine repository root to sys.path "
         "(not a pip package), given by the REWARDBOX_ROBODOPAMINE_ROOT environment variable.",
)

SPEC = ModelSpec(
    name="robodopamine-8b", cls="RoboDopamineAdapter", module=__name__,
    reward_mode=RewardMode.PROGRESS, signal_type=SignalType.GENERATED_TEXT,
    requires=_REQ,
    weights="tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview",
    model_opts={"model_path": "tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview",
                "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTILIZATION},
    # Pass all frames; upstream resamples by frame_interval (default 10, its own signature default).
    # This is the only supported protocol; see the module docstring.
    sampling=SamplingSpec(strategy="all"),
    score_range=(-1.0, 2.0),        # backward mode 1+s in [0,2]; upstream does not clip
    higher_better=True,
    required_views=("primary",),
    optional_views=("left_wrist", "right_wrist"),
    paper="https://arxiv.org/pdf/2512.23703",
    upstream="examples.inference:GRMInference.run_pipeline",
    description="BEFORE/AFTER three-view relative progress scoring, averaged over three modes "
                "(forward/incremental/backward), as the official README recommends.",
    implementation_kind=ImplementationKind.OFFICIAL_ADAPTER,
)
V2_8B_SPEC = replace(
    SPEC, name="robodopamine-v2-8b",
    description="v2 / GRM-2.0-8B-Preview. " + SPEC.description,
    model_opts=dict(SPEC.model_opts),
)
V2_4B_SPEC = replace(
    V2_8B_SPEC, name="robodopamine-v2-4b",
    weights="tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview",
    model_opts={**SPEC.model_opts,
                "model_path": "tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview"},
    description="v2 / GRM-2.0-4B-Preview. " + SPEC.description,
)
V1_8B_SPEC = replace(
    SPEC, name="robodopamine-v1-8b",
    weights="tanhuajie2001/Robo-Dopamine-GRM-8B",
    model_opts={**SPEC.model_opts, "model_path": "tanhuajie2001/Robo-Dopamine-GRM-8B"},
    description="v1 / GRM-8B (RoboBrain2.0-8B). " + SPEC.description
                + " The v1 official examples use three views and a reference goal image; quality without a goal image is unverified.",
)
V1_3B_SPEC = replace(
    V1_8B_SPEC, name="robodopamine-v1-3b",
    weights="tanhuajie2001/Robo-Dopamine-GRM-3B",
    model_opts={**SPEC.model_opts, "model_path": "tanhuajie2001/Robo-Dopamine-GRM-3B"},
    description=V1_8B_SPEC.description.replace("GRM-8B", "GRM-3B").replace(
        "RoboBrain2.0-8B", "RoboBrain2.0-3B"),
)
SPEC = replace(SPEC, description="Compatibility name: same as robodopamine-v2-8b. " + SPEC.description)
SPECS = [V1_3B_SPEC, V1_8B_SPEC, V2_4B_SPEC, V2_8B_SPEC, SPEC]
MODEL_NAMES = tuple(spec.name for spec in SPECS)


def _repo_root() -> Path:
    """Robo-Dopamine is not a pip package; the repository root must be located."""
    env = os.environ.get("REWARDBOX_ROBODOPAMINE_ROOT")
    if env:
        p = Path(env)
        if (p / "examples" / "inference.py").exists():
            return p
        raise RuntimeError(f"REWARDBOX_ROBODOPAMINE_ROOT={env} has no examples/inference.py")
    for cand in upstream_candidates("Robo-Dopamine"):
        if (cand / "examples" / "inference.py").exists():
            return cand
    raise UpstreamNotFoundError(
        "Robo-Dopamine repository not found. Set REWARDBOX_ROBODOPAMINE_ROOT to its root.")


class RoboDopamineAdapter(BaseAdapter):
    def __init__(self, spec, model_path: str | None = None, modes=MODES,
                 fuse: bool = True, frame_interval: int = 10, batch_size: int = 1,
                 gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
                 max_model_len: int | None = None, safe_run_names: bool = False, **opts):
        """modes / fuse default to the official README recommendation (average of three modes).

        All frames are passed and upstream resamples by frame_interval; ``frame_interval=10``
        is the default of the official ``run_pipeline`` signature, not our choice.  **Requires**
        ``traj.sampling.strategy == "all"``: the whole point of this protocol is that upstream
        decides how to sample from all frames.  If RewardBox had already sub-sampled
        (uniform/fps/interval), upstream would not get "all frames" and could not be reproduced
        faithfully, so ``_score()`` refuses before calling official code.
        """
        super().__init__(spec, **opts)
        if not modes:
            raise ValueError("modes must not be an empty tuple; give at least one of forward/incremental/backward")
        bad = set(modes) - set(MODES)
        if bad:
            raise ValueError(f"unknown modes: {sorted(bad)}, choose from {MODES}")
        if not isinstance(frame_interval, int) or isinstance(frame_interval, bool) or frame_interval <= 0:
            raise ValueError("frame_interval must be a positive integer")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if (isinstance(gpu_memory_utilization, bool)
                or not isinstance(gpu_memory_utilization, (int, float))
                or not 0 < gpu_memory_utilization <= 1):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        self.model_path = model_path or spec.model_opts.get("model_path") or spec.weights
        self.modes = tuple(modes)
        self.fuse = fuse
        self.frame_interval = frame_interval
        self.batch_size = batch_size
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        if max_model_len is not None and (type(max_model_len) is not int or max_model_len < 8192):
            raise ValueError("max_model_len must be an integer >= the native 8192 limit")
        if not isinstance(safe_run_names, bool):
            raise ValueError("safe_run_names must be bool")
        self.max_model_len = max_model_len
        self.safe_run_names = safe_run_names
        self._grm = None

    def grm(self):
        if self._grm is None:
            root = _repo_root()
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            # The official examples/inference.py unconditionally runs
            # os.environ['CUDA_VISIBLE_DEVICES'] = '0' at module top level (at import time; not
            # algorithm logic we changed, purely its own side effect).  On a shared multi-GPU
            # machine this overrides the caller's GPU choice with card 0 after import; with GPU0
            # busy, vLLM initialisation then either OOMs or crowds out other jobs.  Here
            # CUDA_VISIBLE_DEVICES is restored to the caller's value after import and before
            # GRMInference(...), which is what actually initialises CUDA.  See docs/installation.md.
            wanted = os.environ.get("CUDA_VISIBLE_DEVICES")
            try:
                inference = importlib.import_module("examples.inference")
            finally:
                if wanted is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = wanted
                else:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            # The official constructor hard-codes 0.9 and takes no memory argument.  Only the LLM's
            # memory budget is overridden during construction, then the entry point is restored;
            # sampling, prompts and the output algorithm still run upstream.
            official_llm = inference.LLM

            def configured_llm(*args, **kwargs):
                kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization
                if self.max_model_len is not None:
                    kwargs["max_model_len"] = self.max_model_len
                return official_llm(*args, **kwargs)

            inference.LLM = configured_llm
            try:
                self._grm = inference.GRMInference(self.model_path)
            finally:
                inference.LLM = official_llm
        return self._grm

    # ---- input preparation ----
    def _write_cams(self, traj: Trajectory, workdir: Path) -> dict[str, str]:
        """Write the frame_{:06d}.png directories that the official list_pngs_sorted expects.
        Missing wrist views copy the primary view, the single-view usage the official README supports."""
        from ..core.video import write_png_dir
        out = {}
        written = {}
        for view, cam in VIEW_TO_CAM.items():
            frames = traj.views.get(view, traj.primary)
            destination = workdir / cam
            if id(frames) in written:
                # Replicated primary views use the same immutable pixels.
                # Preserve all native camera paths without re-encoding PNGs.
                destination.mkdir(parents=True, exist_ok=True)
                for source in written[id(frames)].glob("*.png"):
                    try:
                        os.link(source, destination / source.name)
                    except OSError:
                        shutil.copyfile(source, destination / source.name)
            else:
                write_png_dir(frames, destination)
                written[id(frames)] = destination
            out[cam] = str(destination)
        return out

    def _goal_path(self, traj: Trajectory, workdir: Path) -> tuple[str, str]:
        if traj.goal_image is not None:
            from ..core.video import write_png_dir
            write_png_dir(traj.goal_image[None], workdir / "goal", indices=[0])
            return str(workdir / "goal" / "frame_000000.png"), "provided"
        blank = _repo_root() / "examples" / "blank_goal.png"
        if blank.exists():
            dst = workdir / "blank_goal.png"
            shutil.copyfile(blank, dst)
            return str(dst), "official_blank_placeholder"
        raise RuntimeError(f"the official blank_goal.png does not exist: {blank}")

    # ---- main flow ----
    def _score(self, traj: Trajectory) -> RewardOutput:
        if traj.sampling is None or traj.sampling.strategy != "all":
            raise ValueError(
                f"{self.name} requires the Trajectory to be sampled with SamplingSpec(strategy='all') "
                "(all frames). The point of this protocol is that upstream resamples by frame_interval; "
                "if RewardBox already sub-sampled (uniform/fps/interval), upstream would not get "
                f"'all frames' and could not be reproduced faithfully. Current traj.sampling={traj.sampling!r}")
        degraded = not {"left_wrist", "right_wrist"} <= set(traj.views)
        with tempfile.TemporaryDirectory(prefix="rb-dopamine-") as d:
            work = Path(d)
            cams = self._write_cams(traj, work)
            goal, goal_src = self._goal_path(traj, work)

            per_mode, per_hop, raw_preds, per_mode_after = {}, {}, {}, {}
            for mode in self.modes:
                grm = self.grm()
                if self.safe_run_names:
                    from .robodopamine_execution import bounded_run_pipeline
                    run_pipeline = bounded_run_pipeline(grm)
                else:
                    run_pipeline = grm.run_pipeline
                run_dir = run_pipeline(
                    cam_high_path=cams["cam_high"],
                    cam_left_path=cams["cam_left_wrist"],
                    cam_right_path=cams["cam_right_wrist"],
                    out_root=str(work / f"out_{mode}"),
                    task=traj.instruction,
                    frame_interval=self.frame_interval,
                    batch_size=self.batch_size,
                    goal_image=goal,
                    eval_mode=mode,
                    visualize=False,
                )
                items = json.loads((Path(run_dir) / "pred_vllm.json").read_text())
                samples = json.loads((Path(run_dir) / "sample.json").read_text())
                if len(samples) != len(items):
                    raise RuntimeError(
                        f"{mode} sample.json has {len(samples)} items, but pred_vllm.json "
                        f"has {len(items)}")
                after_local: list[int] = []
                for sample in samples:
                    match = re.search(r"-af_(\d+)$", str(sample.get("id", "")))
                    if match is None:
                        raise RuntimeError(
                            f"cannot parse the AFTER frame number from the {mode} sample id: {sample.get('id')!r}")
                    idx = int(match.group(1))
                    if idx < 0 or idx >= len(traj):
                        raise RuntimeError(
                            f"official {mode} AFTER frame number {idx} is outside the input [0, {len(traj) - 1}]")
                    after_local.append(idx)
                # take the officially computed values as is, no recomputed accumulation
                per_mode[mode] = [float(it["progress"]) for it in items]
                per_hop[mode] = [float(it.get("hop", 0.0)) for it in items]
                raw_preds[mode] = [it.get("pred", "") for it in items]
                per_mode_after[mode] = after_local

        lens = {m: len(v) for m, v in per_mode.items()}
        if len(set(lens.values())) != 1:
            raise RuntimeError(f"modes produced different output lengths: {lens}")
        n_steps = next(iter(lens.values()))
        grids = {m: tuple(v) for m, v in per_mode_after.items()}
        if len(set(grids.values())) != 1:
            raise RuntimeError(f"modes produced different official AFTER frame grids: {grids}")

        if self.fuse:
            progress = np.mean([per_mode[m] for m in self.modes], axis=0)
            hops = np.mean([per_hop[m] for m in self.modes], axis=0)
        else:
            progress = np.asarray(per_mode[self.modes[0]], dtype=np.float64)
            hops = np.asarray(per_hop[self.modes[0]], dtype=np.float64)

        # sample.json is the time grid upstream actually used in this run.  With the native
        # interval=10 the output is usually far shorter than T-1; the mapping cannot be guessed
        # from "length equals T or T-1".
        after_local = np.asarray(per_mode_after[self.modes[0]], dtype=np.int64)
        fidx = traj.frame_indices[after_local]

        return self._out(
            progress, traj, frame_indices=fidx, weights=hops,
            raw={"per_mode_progress": per_mode, "per_mode_hop": per_hop,
                 "responses": raw_preds,
                 "official_after_local_indices": per_mode_after},
            meta={"adapter_calls_official": True,
                  "model_version": "v1" if self.name.startswith("robodopamine-v1-") else "v2",
                  "canonical_model_name": ("robodopamine-v2-8b"
                                           if self.name == "robodopamine-8b" else self.name),
                  "model_path": self.model_path,
                  "modes": list(self.modes), "fused": self.fuse,
                  "fusion": "mean(official README recommendation)" if self.fuse else "none",
                  "clamped": False,
                  "goal_image_source": goal_src,
                  "view_mode": "replicated_primary" if degraded else "native_three_view",
                  "frame_interval": self.frame_interval,
                  "gpu_memory_utilization": self.gpu_memory_utilization,
                  "max_model_len_override": self.max_model_len,
                  "bounded_output_directory_names": self.safe_run_names,
                  "first_frame_omitted": 0 not in after_local,
                  "inference_batch_size": self.batch_size,
                  "n_inference_requests": len(self.modes) * n_steps,
                  "n_backend_calls": len(self.modes) * math.ceil(n_steps / self.batch_size)},
        )

    def close(self):
        self._grm = None
