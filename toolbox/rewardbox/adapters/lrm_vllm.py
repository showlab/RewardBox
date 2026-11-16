"""Opt-in vLLM execution of the native LRM single-head frame/pair protocol."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..backends.vllm import (
    REQUIREMENTS, VLLMAdapter, VLLMBackend, VLLMConfig, multimodal_request, resolve_cached_repo,
)
from ..core.enums import ImplementationKind
from .lrm import SPECS as HTTP_SPECS
from .lrm import _UPSTREAM_COMMIT, _WEIGHTS_REPO, _WEIGHTS_REVISION
from .lrm_native_protocol import (
    messages_for, parse_comparison, parse_completion, parse_progress, progress_messages,
)

SPECS = [replace(
    spec, name=f"{spec.name}-vllm", cls="LRMVLLMAdapter", module=__name__,
    requires=REQUIREMENTS, implementation_kind=ImplementationKind.REWARDBOX_EXTENSION,
    description=f"{spec.name}: native prompt and frame/pair semantics; opt-in batched vLLM execution.",
) for spec in HTTP_SPECS if spec.name != "lrm-tri"]


class LRMVLLMAdapter(VLLMAdapter):
    def __init__(self, spec, model_path: str | None = None, revision: str = _WEIGHTS_REVISION,
                 batch_size: int = 32, gpu_memory_utilization: float = 0.7,
                 max_model_len: int = 8192, tensor_parallel_size: int = 1,
                 dtype: str = "bfloat16", seed: int = 0, enforce_eager: bool = False,
                 max_num_batched_tokens: int = 8192,
                 include_initial_image: bool = False, include_goal_image: bool = False,
                 request_chunk_size: int | None = None, **opts):
        super().__init__(spec, **opts)
        self.head = spec.name.removeprefix("lrm-").removesuffix("-vllm")
        if self.head not in {"completion", "progress", "contrastive"}:
            raise ValueError("A vLLM instance must use one dedicated LRM head")
        if not isinstance(include_initial_image, bool) or not isinstance(include_goal_image, bool):
            raise ValueError("include_initial_image/include_goal_image must be bool")
        if self.head != "progress" and (include_initial_image or include_goal_image):
            raise ValueError("Reference images apply only to lrm-progress-vllm")
        if request_chunk_size is not None and (
                not isinstance(request_chunk_size, int) or isinstance(request_chunk_size, bool)
                or request_chunk_size < batch_size):
            raise ValueError("request_chunk_size must be an integer >= batch_size")
        self.include_initial_image = include_initial_image
        self.include_goal_image = include_goal_image
        # Requests submitted per LLM.generate call; None keeps one engine batch.
        self.request_chunk_size = request_chunk_size
        self.config = VLLMConfig(batch_size, gpu_memory_utilization, max_model_len,
                                 tensor_parallel_size, dtype, seed, enforce_eager,
                                 max_num_batched_tokens)
        self.model_path = model_path
        self.revision = revision
        self._backend = None

    def _runtime(self):
        if self._backend is None:
            path = self.model_path
            if path is None:
                root = resolve_cached_repo(_WEIGHTS_REPO, self.revision)
                path = str(Path(root) / self.head)
            max_images = (1 + int(self.include_initial_image) + int(self.include_goal_image)
                          if self.head == "progress" else 2 if self.head == "contrastive" else 1)
            self._backend = VLLMBackend(path, self.revision, self.config,
                                       limit_mm_per_prompt={"image": max_images, "video": 0})
        return self._backend

    def _requests(self, traj, runtime):
        from PIL import Image

        processor = None
        initial = Image.fromarray(traj.primary[0]) if self.include_initial_image else None
        goal = Image.fromarray(traj.goal_image) if self.include_goal_image else None
        for index, frame in enumerate(traj.primary):
            if self.head == "contrastive" and index == 0:
                continue
            current = Image.fromarray(frame)
            if self.head == "contrastive":
                messages = messages_for(self.head, traj.instruction,
                                        [Image.fromarray(traj.primary[index - 1]), current])
            elif self.head == "completion":
                messages = messages_for(self.head, traj.instruction, [current])
            else:
                messages = progress_messages(traj.instruction, current, initial=initial, goal=goal)
            if processor is None:
                processor = runtime.processor
            prompt = processor.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
            images = [item["image"] for item in messages[0]["content"] if item["type"] == "image"]
            yield multimodal_request(prompt, {"image": images})

    def _score(self, traj):
        if self.include_goal_image and traj.goal_image is None:
            raise ValueError("include_goal_image=true requires Trajectory.goal_image")
        # A one-frame contrastive trajectory only has the official zero baseline.
        runtime = self._runtime()
        max_tokens = 512 if self.head == "contrastive" else 128
        generated = runtime.generate(self._requests(traj, runtime), max_tokens=max_tokens,
                                     chunk_size=self.request_chunk_size)
        texts = [item.text for item in generated]
        expected = len(traj) - int(self.head == "contrastive")
        if len(texts) != expected:
            raise RuntimeError("LRM vLLM output count does not match supplied frames/pairs")
        if self.head == "contrastive":
            comparisons = [parse_comparison(text) for text in texts]
            values = [0.0] + [{"ImageA": -1.0, "ImageB": 1.0, None: 0.0}[x]
                              for x in comparisons]
            raw = {"comparison_results": ["first_frame"] + comparisons,
                   "responses": [""] + texts}
            success = None
        elif self.head == "completion":
            completed = [parse_completion(text) for text in texts]
            values = [float(value) for value in completed]
            raw = {"completed": completed, "responses": texts}
            success = values
        else:
            values = [parse_progress(text) for text in texts]
            raw, success = {"responses": texts}, None
        return self._out(values, traj, success=success, raw=raw, meta={
            "adapter_calls_official": False,
            "protocol": "lrm_native_prompt_batched_vllm_v1", "profile": self.head,
            "execution_extension": "RewardBox vLLM replaces official Flask/Transformers generation",
            "upstream_commit": _UPSTREAM_COMMIT, "weights_revision": self.revision,
            **runtime.metadata(), "normalized": False,
            "sampling_profile": "all supplied frames; adjacent pairs for contrastive; no interpolation",
            "include_initial_image": self.include_initial_image,
            "include_goal_image": self.include_goal_image,
            "generation": {"max_new_tokens": max_tokens, "greedy": True},
            "n_inference_requests": len(generated), "n_backend_calls": runtime.batch_count,
            "finish_reasons": [item.finish_reason for item in generated],
            "prompt_tokens": sum(item.prompt_tokens for item in generated),
            "completion_tokens": sum(item.completion_tokens for item in generated),
            "paper_equivalence": "not asserted; native prompt parity tested offline, GPU calibration pending",
        })

    def close(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None
