"""NVIDIA Flora-Rewarder (FLORA rewarder, Qwen3.5-27B) terminal reward, isolated runtime.

Upstream ships its inference code inside the model repository:
  nvidia/Flora-Rewarder@031299e46e452fc0cedb4e0975eb9fa95fc37783
  batch_utils/core.py, batch_utils/engine.py, batch_utils/terminal_reward.py
They are imported from the pinned HF snapshot, not re-implemented:
  core.plan_collage / check_synchronized / terminal_frame_indices / video_metadata /
  terminal_messages / terminal_regex / terminal_max_tokens / parse_terminal /
  score_probabilities, engine.load_llm / load_processor / render / COLLAGE_VIDEO_SIZE.
The request dict, greedy regex-constrained SamplingParams, logprob positions and the
output record fields copy ``terminal_reward.main`` line for line.

RewardBox owns only these boundaries (all recorded in ``meta``):
  * Frames come from the Trajectory (RewardBox's decoder) instead of upstream's PyAV
    reader.  Each view is resized to its planned collage cell with the same PIL
    bilinear call and tiled exactly as ``core.Collage.frames`` does.
  * Multimodal requests carry per-request UUIDs, so vLLM does not hash pixel content
    for its caches.  Outputs are unaffected.
  * ``flora-rewarder-27b-prefix`` scores every causal prefix ``[0, k]`` as if it were
    a whole episode of ``k + 1`` frames: upstream frame selection, absolute-time
    metadata and prompt for that shorter episode.  The model card defines only
    whole-episode scoring; prefix curves are a RewardBox extension.  The last prefix
    is the full trajectory, i.e. exactly the official terminal call.

Not wrapped here: the dense head (1-3 action quality per 0.5 s) and the rubric head
(needs a per-task rubric); both are separate upstream scripts.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
from importlib.metadata import version
import math
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..backends.vllm import VLLMAdapter, resolve_cached_repo
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import SamplingSpec

REPO = "nvidia/Flora-Rewarder"
REVISION = "031299e46e452fc0cedb4e0975eb9fa95fc37783"
# terminal_reward.py argparse default; core.py documents the 16,384-token budget.
MAX_MODEL_LEN = 16384
_PACKAGE = "_rewardbox_flora_batch_utils"

REQ = Requirements(
    # Verbatim from the pinned snapshot's batch_utils/requirements.txt (terminal path).
    pip=("vllm==0.24.0", "transformers==5.8.1", "numpy<3", "Pillow>=11,<13"),
    python=">=3.12,<3.13",
    vram_gb=60.0,
    note=("Upstream validated Python 3.12 + vllm 0.24 / transformers 5.8 (Qwen3.5); "
          "isolated .venv-flora. BF16 weights about 54 GB; one H100/H200 per engine."),
    install_hint=(
        "uv venv --python 3.12 .venv-flora",
        "uv pip install --python .venv-flora/bin/python -r locks/flora-py312-requirements.in",
        "uv pip install --python .venv-flora/bin/python --no-deps -e .",
        "HF_HOME=$PWD/.hf HF_HUB_DISABLE_XET=1 .venv/bin/python -c \"from huggingface_hub import "
        f"snapshot_download; snapshot_download('{REPO}', revision='{REVISION}')\"",
    ),
)
COMMON = dict(
    cls="FloraAdapter", module=__name__, reward_mode=RewardMode.PROGRESS,
    signal_type=SignalType.TOKEN_LOGPROB, requires=REQ,
    # Upstream reads the whole native-rate video and selects frames itself.
    sampling=SamplingSpec(strategy="all"), score_range=(0.0, 1.0), higher_better=True,
    optional_views=("left_wrist", "right_wrist", "top", "left", "right", "wrist"),
    supports_pairwise=False, outputs_success=False,
    weights=f"{REPO}@{REVISION}", paper="https://huggingface.co/nvidia/Flora-Rewarder",
    upstream=(f"https://huggingface.co/{REPO}@{REVISION}; "
              "batch_utils/terminal_reward.py + batch_utils/core.py"),
)
SPECS = [
    ModelSpec(name="flora-rewarder-27b", description=(
        "Flora terminal reward: whole episode, up to 64 frames at <=10 FPS, 1-5 score "
        "decoded greedily under a regex; progress=(E[score]-1)/4 from P(1..5) at the score token."),
        implementation_kind=ImplementationKind.OFFICIAL_ADAPTER, **COMMON),
    ModelSpec(name="flora-rewarder-27b-prefix", description=(
        "RewardBox prefix-curve extension of the Flora terminal reward: each causal prefix "
        "is scored as a whole episode with the native sampling, timing and prompt."),
        implementation_kind=ImplementationKind.REWARDBOX_EXTENSION, **COMMON),
]


def upstream(model_dir):
    """The pinned snapshot's ``batch_utils`` package as ``(core, engine)``."""
    root = Path(model_dir) / "batch_utils"
    if not (root / "core.py").is_file() or not (root / "engine.py").is_file():
        raise RuntimeError(f"Flora snapshot lacks batch_utils/: {model_dir}")
    package = sys.modules.get(_PACKAGE)
    if package is None or Path(package.__path__[0]) != root:
        for name in [name for name in sys.modules if name == _PACKAGE or name.startswith(_PACKAGE + ".")]:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(
            _PACKAGE, root / "__init__.py", submodule_search_locations=[str(root)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[_PACKAGE] = package
        spec.loader.exec_module(package)
    return importlib.import_module(_PACKAGE + ".core"), importlib.import_module(_PACKAGE + ".engine")


def prefix_ends(length, fps, prefix_fps):
    """Causal prefix end frames on a ``prefix_fps`` grid (nearest, half up), plus the last frame."""
    if length < 1:
        raise ValueError("trajectory has no frames")
    ends, step = set(), 0
    while (frame := math.floor(step * fps / prefix_fps + 0.5)) <= length - 1:
        ends.add(frame)
        step += 1
    ends.add(length - 1)
    return sorted(ends)


class ArrayCollage:
    """``core.Collage`` over already decoded RGB views instead of video files."""

    def __init__(self, core, views, fps):
        infos = [core.VideoInfo(f"view{i}", float(fps), len(frames), int(frames.shape[2]), int(frames.shape[1]))
                 for i, frames in enumerate(views)]
        self.fps, self.num_frames = core.check_synchronized(infos)
        self.rectangles, self.width, self.height = core.plan_collage(
            tuple((info.width, info.height) for info in infos))
        self.views = views
        self._cache = {}

    def frames(self, indices):
        from PIL import Image

        for index in sorted({int(i) for i in indices} - set(self._cache)):
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for frames, (x, y, w, h) in zip(self.views, self.rectangles):
                # core.read_frames: PIL image, bilinear resize to (w, h) if needed, RGB uint8.
                image = Image.fromarray(np.ascontiguousarray(frames[index]))
                if image.size != (w, h):
                    image = image.resize((w, h), resample=Image.Resampling.BILINEAR)
                canvas[y:y + h, x:x + w] = np.asarray(image.convert("RGB"), dtype=np.uint8)
            self._cache[index] = canvas
        return np.stack([self._cache[int(i)] for i in indices])


class FloraRuntime:
    def __init__(self, model_dir, config):
        self.model_dir, self.config = model_dir, config
        self.core, self.engine = upstream(model_dir)
        self.processor = self.llm = self.tokenizer = self.params = None
        self.batch_count = 0

    def _load(self):
        if self.llm is not None:
            return
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        core, engine = self.core, self.engine
        args = argparse.Namespace(**{key: self.config[key] for key in (
            "tensor_parallel_size", "gpu_memory_utilization", "max_num_seqs", "seed")})
        self.llm = engine.load_llm(self.model_dir, args, max_model_len=MAX_MODEL_LEN,
                                   mm_processor_kwargs={"do_sample_frames": False,
                                                        "size": engine.COLLAGE_VIDEO_SIZE},
                                   max_logprobs=core.TERMINAL_TOP_LOGPROBS)
        self.tokenizer = self.llm.get_tokenizer()
        rationale = self.config["with_rationale"]
        self.params = SamplingParams(
            temperature=0.0, max_tokens=core.terminal_max_tokens(rationale),
            logprobs=core.TERMINAL_TOP_LOGPROBS,
            structured_outputs=StructuredOutputsParams(regex=core.terminal_regex(rationale)))

    def prompt(self, instruction):
        if self.processor is None:
            self.processor = self.engine.load_processor(self.model_dir)
        messages = self.core.terminal_messages(instruction, self.config["with_rationale"])
        return self.engine.render(self.processor, messages, thinking=False)

    def request(self, prompt, collage, end):
        """terminal_reward.main's request for an episode made of source frames [0, end]."""
        indices = self.core.terminal_frame_indices(end + 1, collage.fps)
        return {
            "prompt": prompt,
            "multi_modal_data": {"video": (collage.frames(indices),
                                           self.core.video_metadata(indices, collage.fps, end + 1))},
            "mm_processor_kwargs": {"do_sample_frames": False},
            "multi_modal_uuids": {"video": [uuid4().hex]},
        }, len(indices)

    def generate(self, requests):
        self._load()
        outputs = self.llm.generate(requests, self.params, use_tqdm=False)
        self.batch_count += 1
        if len(outputs) != len(requests):
            raise RuntimeError("vLLM returned an incomplete batch")
        records = []
        for output in outputs:
            completion = output.outputs[0]
            score, rationale = self.core.parse_terminal(completion.text, self.config["with_rationale"])
            positions = [
                (self.tokenizer.decode([token]), float(candidates[token].logprob),
                 [(self.tokenizer.decode([i]), float(c.logprob)) for i, c in candidates.items()])
                for token, candidates in zip(completion.token_ids, completion.logprobs)
            ]
            probabilities, expected = self.core.score_probabilities(positions, score)
            records.append({
                "score": score,
                "expected_score": round(expected, 4),
                "progress": round((expected - 1.0) / 4.0, 4),
                "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
                "rationale": rationale,
                "output": completion.text,
                "expected_score_unrounded": expected,
                "finish_reason": completion.finish_reason,
                "prompt_tokens": len(output.prompt_token_ids or []),
                "completion_tokens": len(completion.token_ids),
            })
        return records

    def close(self):
        if self.llm is not None:
            client = getattr(self.llm.llm_engine, "engine_core", None)
            if client is not None and hasattr(client, "shutdown"):
                client.shutdown()
            self.llm = None
        self.processor = None


class FloraAdapter(VLLMAdapter):
    def __init__(self, spec, model_path=None, tensor_parallel_size=1, gpu_memory_utilization=0.88,
                 max_num_seqs=8, seed=0, with_rationale=False, prefix_fps=1.0,
                 view_order=("primary",), **opts):
        super().__init__(spec, **opts)
        # engine.base_parser defaults: TP 1, 0.88 memory, 8 concurrent requests, seed 0.
        for key, value in dict(tensor_parallel_size=tensor_parallel_size, max_num_seqs=max_num_seqs).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if isinstance(gpu_memory_utilization, bool) or not isinstance(gpu_memory_utilization, (int, float)) \
                or not math.isfinite(gpu_memory_utilization) or not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if not isinstance(with_rationale, bool):
            raise ValueError("with_rationale must be bool")
        if isinstance(prefix_fps, bool) or not isinstance(prefix_fps, (int, float)) \
                or not math.isfinite(prefix_fps) or prefix_fps <= 0:
            raise ValueError("prefix_fps must be positive")
        view_order = tuple(view_order)
        if not 1 <= len(view_order) <= 4 or len(set(view_order)) != len(view_order):
            raise ValueError("view_order must name 1 to 4 distinct views (upstream collage limit)")
        self.config = dict(tensor_parallel_size=tensor_parallel_size,
                           gpu_memory_utilization=float(gpu_memory_utilization),
                           max_num_seqs=max_num_seqs, seed=seed, with_rationale=with_rationale,
                           prefix_fps=float(prefix_fps), view_order=list(view_order))
        self.model_path = model_path
        self.runtime = None

    @property
    def prefix(self):
        return self.name.endswith("-prefix")

    def _runtime(self):
        if self.runtime is None:
            path = self.model_path or resolve_cached_repo(REPO, REVISION)
            self.runtime = FloraRuntime(path, self.config)
        return self.runtime

    def _score(self, traj):
        if traj.sampling is None or traj.sampling.strategy != "all" \
                or traj.frame_indices.tolist() != list(range(len(traj))):
            raise ValueError("Flora requires every source frame; it owns native frame selection")
        if traj.fps is None or not math.isfinite(traj.fps) or traj.fps <= 0:
            raise ValueError("Flora's absolute-time metadata requires Trajectory.fps")
        missing = [name for name in self.config["view_order"] if name not in traj.views]
        if missing:
            raise ValueError(f"view_order names views the trajectory lacks: {missing}")
        runtime = self._runtime()
        collage = ArrayCollage(runtime.core, [traj.view(name) for name in self.config["view_order"]], traj.fps)
        ends = (prefix_ends(collage.num_frames, collage.fps, self.config["prefix_fps"]) if self.prefix
                else [collage.num_frames - 1])
        prompt = runtime.prompt(traj.instruction)
        runtime.batch_count = 0
        records, sampled = [], []
        # terminal_reward.main: one llm.generate call per max_num_seqs requests.
        for offset in range(0, len(ends), self.config["max_num_seqs"]):
            batch = [runtime.request(prompt, collage, end)
                     for end in ends[offset:offset + self.config["max_num_seqs"]]]
            records.extend(runtime.generate([request for request, _ in batch]))
            sampled.extend(count for _, count in batch)
        for record, count, end in zip(records, sampled, ends):
            record.update(end_frame=end, sampled_frames=count,
                          duration_s=round((end + 1) / collage.fps, 3))
        return self._out([record["progress"] for record in records], traj,
                         frame_indices=traj.frame_indices[ends], raw={"points": records}, meta={
            "upstream_revision": REVISION, "weights_revision": REVISION,
            "protocol": "flora_terminal_prefix_v1" if self.prefix else "flora_terminal_v1",
            "adapter_calls_official": True, "mode": "prefix" if self.prefix else "terminal",
            "prefix_curve_is_extension": self.prefix,
            "prefix_grid": (f"nearest-half-up {self.config['prefix_fps']:g} FPS + last frame"
                            if self.prefix else None),
            "progress_definition": "(expected_score - 1) / 4 from P(1..5) at the score token, rounded to 4 dp as upstream writes it",
            "frame_selection": "core.terminal_frame_indices(end + 1, fps) per prefix",
            "frame_reader": "RewardBox Trajectory RGB frames; upstream reads with PyAV frame.to_image()",
            "collage": {"views": self.config["view_order"], "width": collage.width, "height": collage.height,
                        "rectangles": [list(rect) for rect in collage.rectangles]},
            "generation": {"temperature": 0.0, "max_tokens": runtime.core.terminal_max_tokens(self.config["with_rationale"]),
                           "logprobs": runtime.core.TERMINAL_TOP_LOGPROBS,
                           "regex": runtime.core.terminal_regex(self.config["with_rationale"])},
            "engine_options": {**self.config, "max_model_len": MAX_MODEL_LEN},
            "multimodal_identity": "unique_request_uuid; no pixel-content hash",
            "package_versions": {name: version(name) for name in ("vllm", "transformers", "torch")},
            "normalized": False, "model_path": runtime.model_dir,
            "n_backend_calls": runtime.batch_count, "n_inference_requests": len(records),
            "prompt_tokens": sum(record["prompt_tokens"] for record in records),
            "completion_tokens": sum(record["completion_tokens"] for record in records),
            "paper_equivalence": "not asserted",
        })

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
