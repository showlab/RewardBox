"""Pinned PRIMO-R1 interleaved video progress, in its isolated upstream runtime."""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from ..backends.vllm import GeneratedText, VLLMAdapter, resolve_cached_repo
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import SamplingSpec
from ..upstream import UpstreamNotFoundError, upstream_candidates

UPSTREAM_COMMIT = "f746e1f83af001337dabffafa948129353ac962e"
REPO = "LeonOverload/PRIMO-R1-7B"
REVISION = "4d452de3f721ec20e311e3a2dff309df7e022a69"
QUESTION = 'What is the completion percentage of the task "{instruction}" in the video?'
REQ = Requirements(
    pip=("torch==2.5.1", "torchvision==0.20.1", "vllm==0.7.2",
         "transformers==4.49.0.dev0", "qwen-vl-utils==0.0.10", "decord==0.6.0"),
    python=">=3.11,<3.12",
    note="Use .venv-primo and the pinned PRIMO-R1 vendored Transformers/vision code.",
    install_hint=("Use locks/primo-py311-requirements.in with .venv-primo/bin/python.",),
)
COMMON = dict(
    cls="PRIMOAdapter", module=__name__, reward_mode=RewardMode.PROGRESS,
    signal_type=SignalType.GENERATED_TEXT, requires=REQ,
    sampling=SamplingSpec(strategy="all"), score_range=(0., 100.),
    weights=f"{REPO}@{REVISION}", paper="https://arxiv.org/abs/2603.15600",
    upstream="PRIMO-R1/src/eval/eval_interleave.py + src/primo_prompts.py",
    supports_pairwise=False, outputs_success=False,
)
SPECS = [
    ModelSpec(name="primo-r1-7b", description=(
        "PRIMO-R1 native initial image/video/current image input; one terminal "
        "raw percentage. The task instruction is bound into its regression question."),
        implementation_kind=ImplementationKind.OFFICIAL_ADAPTER, **COMMON),
    ModelSpec(name="primo-r1-7b-prefix", description=(
        "Explicit RewardBox prefix-curve extension of PRIMO-R1; each point uses "
        "the native interleaved input, prompt and sampling. No interpolation."),
        implementation_kind=ImplementationKind.REWARDBOX_EXTENSION, **COMMON),
]


def upstream_root():
    explicit = os.environ.get("REWARDBOX_PRIMO_ROOT")
    candidates = [Path(explicit)] if explicit else upstream_candidates("PRIMO-R1")
    for root in candidates:
        if (root / "src/primo_prompts.py").is_file():
            return root.resolve()
    raise UpstreamNotFoundError(
        "Pinned PRIMO-R1 source is missing; run `rewardbox fetch-upstream primo` "
        "(or set REWARDBOX_PRIMO_ROOT) and prepare the isolated runtime first")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_percentage(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if not match:
        raise ValueError("PRIMO generation lacks an <answer> block")
    answer = match.group(1).strip()
    value = float(answer.replace("≈", "").replace("~", "").replace("％", "%")
                  .removesuffix("%").replace(",", "").strip())
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("PRIMO progress percentage is outside [0, 100]")
    # Upstream's evaluation normalizer divides suffixed answers by 100.
    # Preserve that metric input separately; progress exposes the model-card
    # percentage units for both '42.5' and '42.5%'. No clipping or interpolation.
    normalized = value / 100 if answer.replace("％", "%").endswith("%") else value
    return value, answer, normalized


def prefix_ends(length, count):
    if length < 2 or length == 3:
        raise ValueError("PRIMO native video rounding requires two or at least four source frames")
    # The native frame-factor rounding rejects a three-frame video (rounds to
    # four); omit such prefixes instead of duplicating frames or changing it.
    return [int(end) for end in np.unique(np.linspace(1, length - 1, min(count, length - 1),
                                                    dtype=np.int64)) if end != 2]


class PRIMORuntime:
    def __init__(self, model_path, config):
        self.model_path, self.config = model_path, config
        self.llm = self.processor = self.params = self.vision = self.prompts = None
        self.batch_count = 0

    def _prepare_runtime(self):
        if self.processor is not None:
            return
        from transformers import AutoProcessor, AutoTokenizer

        root = upstream_root()
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if commit != UPSTREAM_COMMIT:
            raise RuntimeError("PRIMO upstream commit changed; prepare a separately versioned profile")
        self.prompts = load_module(root / "src/primo_prompts.py", "_rewardbox_primo_prompts")
        self.vision = load_module(root / "src/qwen-vl-utils/src/qwen_vl_utils/vision_process.py",
                                  "_rewardbox_primo_vision")
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        tokenizer.padding_side = "left"
        self.processor.tokenizer = tokenizer

    def prepare(self, traj, end):
        """Use the native reader's index/resize path on already decoded frames."""
        self._prepare_runtime()
        import torch
        from PIL import Image

        total, fps = end + 1, traj.fps
        if fps is None or not math.isfinite(fps) or fps <= 0:
            raise ValueError("PRIMO's native video timing requires Trajectory.fps")
        requested = max(2, min(self.config["nframes"], total, 22))
        native_requested = requested
        # The released reader rounds to an even count, which can exceed a
        # short prefix's length. Recovery is an explicit prefix-only extension;
        # keep the released path unchanged for all previously valid requests.
        if self.config.get("short_prefix_policy") == "floor-on-overflow-v1" \
                and self.vision.round_by_factor(requested, self.vision.FRAME_FACTOR) > total:
            requested = self.vision.floor_by_factor(total, self.vision.FRAME_FACTOR)
        sampled = self.vision.smart_nframes({"nframes": requested}, total, fps)
        local_indices = torch.linspace(0, total - 1, sampled).round().long().numpy()
        video = torch.from_numpy(np.ascontiguousarray(traj.primary[local_indices])).permute(0, 3, 1, 2)
        sample_fps = sampled / total * fps
        question = self.config["question_template"].format(instruction=traj.instruction)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.prompts.SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": Image.fromarray(traj.primary[0])},
                {"type": "video", "video": "rewardbox-decoded-input", "nframes": requested},
                {"type": "image", "image": Image.fromarray(traj.primary[end])},
                {"type": "text", "text": self.prompts.build_question(question, "regression")},
            ]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Patch only a private copy's reader; native fetch_video still owns the
        # same bicubic resize/pixel budgets. No second video decode or pixel hash.
        previous = self.vision.get_video_reader_backend
        self.vision.VIDEO_READER_BACKENDS["rewardbox-decoded"] = lambda element: (video, sample_fps)
        self.vision.get_video_reader_backend = lambda: "rewardbox-decoded"
        try:
            images, videos, kwargs = self.vision.process_vision_info(messages, return_video_kwargs=True)
        finally:
            self.vision.get_video_reader_backend = previous
            self.vision.VIDEO_READER_BACKENDS.pop("rewardbox-decoded", None)
        return {"prompt": prompt, "multi_modal_data": {"image": images, "video": videos[0]},
                "mm_processor_kwargs": {key: value[0] for key, value in kwargs.items()}}, {
                    "video_frame_indices": traj.frame_indices[local_indices].tolist(),
                    "anchor_frame_indices": [int(traj.frame_indices[0]), int(traj.frame_indices[end])],
                    "video_fps": sample_fps,
                    "native_requested_nframes": native_requested,
                    "requested_nframes": requested,
                    "short_prefix_repaired": requested != native_requested,
                }

    def _load(self):
        if self.llm is not None:
            return
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=self.model_path, tensor_parallel_size=self.config["tensor_parallel_size"],
            max_model_len=16384, gpu_memory_utilization=self.config["gpu_memory_utilization"],
            limit_mm_per_prompt={"image": 3, "video": 1}, dtype="bfloat16",
            max_num_seqs=self.config["batch_size"],
            disable_mm_preprocessor_cache=True,
        )
        self.params = SamplingParams(temperature=0.1, top_p=0.001, max_tokens=4096, stop_token_ids=[])

    def generate(self, requests):
        self._load()
        generated = self.llm.generate(requests, sampling_params=self.params, use_tqdm=False)
        self.batch_count += 1
        return [GeneratedText(out.outputs[0].text, out.outputs[0].finish_reason,
                              len(out.prompt_token_ids or []), len(out.outputs[0].token_ids))
                for out in generated]

    def close(self):
        if self.llm is not None:
            # This pinned engine has no public shutdown API; releasing this
            # process-owned engine is completed when its isolated worker exits.
            self.llm = None
        self.processor = None


class PRIMOAdapter(VLLMAdapter):
    def __init__(self, spec, model_path=None, batch_size=4, gpu_memory_utilization=0.5,
                 tensor_parallel_size=1, nframes=32, num_prefixes=16,
                 question_template=QUESTION, short_prefix_policy="native", **opts):
        super().__init__(spec, **opts)
        for key, value in dict(batch_size=batch_size, tensor_parallel_size=tensor_parallel_size,
                               nframes=nframes, num_prefixes=num_prefixes).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if isinstance(gpu_memory_utilization, bool) or not math.isfinite(gpu_memory_utilization) \
                or not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0,1]")
        if "{instruction}" not in question_template:
            raise ValueError("question_template must include {instruction}")
        if short_prefix_policy not in ("native", "floor-on-overflow-v1"):
            raise ValueError("unknown PRIMO short_prefix_policy")
        if short_prefix_policy != "native" and not spec.name.endswith("-prefix"):
            raise ValueError("short-prefix recovery applies only to the explicit prefix extension")
        self.config = dict(batch_size=batch_size, gpu_memory_utilization=gpu_memory_utilization,
                           tensor_parallel_size=tensor_parallel_size, nframes=nframes,
                           num_prefixes=num_prefixes, question_template=question_template,
                           short_prefix_policy=short_prefix_policy)
        self.model_path = model_path
        self.runtime = None

    def _runtime(self):
        if self.runtime is None:
            path = self.model_path or resolve_cached_repo(REPO, REVISION)
            self.runtime = PRIMORuntime(path, self.config)
        return self.runtime

    def _score(self, traj):
        if traj.sampling is None or traj.sampling.strategy != "all" \
                or traj.frame_indices.tolist() != list(range(len(traj))):
            raise ValueError("PRIMO requires every source frame; it owns native video sampling")
        if len(traj) < 2:
            raise ValueError("PRIMO requires at least two source frames")
        runtime = self._runtime()
        ends = (prefix_ends(len(traj), self.config["num_prefixes"])
                if self.name.endswith("-prefix") else [len(traj) - 1])
        texts, records, finishes, prompt_tokens, completion_tokens = [], [], [], 0, 0
        runtime.batch_count = 0
        for offset in range(0, len(ends), self.config["batch_size"]):
            batch = [runtime.prepare(traj, end) for end in ends[offset:offset + self.config["batch_size"]]]
            outputs = runtime.generate([request for request, _ in batch])
            if len(outputs) != len(batch):
                raise RuntimeError("PRIMO output count differs from submitted prefixes")
            texts.extend(output.text for output in outputs)
            records.extend(record for _, record in batch)
            finishes.extend(output.finish_reason for output in outputs)
            prompt_tokens += sum(output.prompt_tokens for output in outputs)
            completion_tokens += sum(output.completion_tokens for output in outputs)
        values, answers, normalized = zip(*(parse_percentage(text) for text in texts))
        return self._out(values, traj, frame_indices=traj.frame_indices[ends], raw={
            "responses": texts, "answers": list(answers), "upstream_normalized_numbers": list(normalized),
            "input_records": records,
        }, meta={
            "upstream_commit": UPSTREAM_COMMIT, "weights_revision": REVISION,
            "protocol": "primo_interleaved_regression_v1",
            "adapter_calls_official": True, "mode": "prefix" if self.name.endswith("-prefix") else "terminal",
            "prefix_curve_is_extension": self.name.endswith("-prefix"), "native_video_frame_cap": 22,
            "generation": {"temperature": 0.1, "top_p": 0.001, "max_tokens": 4096, "stop_token_ids": []},
            "units": "raw percentage as specified by the model card",
            "parser_policy": "numeric answer in percentage units; upstream percent-suffix normalizer retained separately",
            "engine_options": dict(self.config), "normalized": False,
            "n_backend_calls": runtime.batch_count, "n_inference_requests": len(texts),
            "finish_reasons": finishes, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "model_path": runtime.model_path,
            "paper_equivalence": "not asserted; task question binding and prefix protocol are explicit",
        })

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
