"""Lazy, bounded vLLM generation shared by RewardBox's opt-in adapters.

The caller owns prompts, media preparation, sampling and output semantics.
This module only owns engine lifetime, greedy decoding and request batching.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import islice
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from ..core.base import BaseAdapter
from ..core.spec import Requirements
from ..core.types import CostRecord

REQUIREMENTS = Requirements(
    pip=("vllm>=0.11,<0.12", "torch>=2.8", "transformers>=4.57,<4.58",
         "qwen-vl-utils>=0.0.14", "huggingface-hub>=0.35"),
    python=">=3.11",
    note="Opt-in local vLLM backend; the original HTTP/Swift environments are unchanged.",
    install_hint=("Use the RewardBox shared vLLM environment and locks/shared-py311-linux-x86_64.lock.",),
)


def positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class VLLMConfig:
    batch_size: int = 32
    gpu_memory_utilization: float = 0.7
    max_model_len: int = 32768
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    seed: int = 0
    enforce_eager: bool = False
    max_num_batched_tokens: int = 8192
    skip_mm_profiling: bool = False

    def __post_init__(self):
        for name in ("batch_size", "max_model_len", "tensor_parallel_size", "max_num_batched_tokens"):
            positive_int(name, getattr(self, name))
        if self.max_num_batched_tokens < self.batch_size:
            raise ValueError("max_num_batched_tokens must be >= batch_size")
        fraction = self.gpu_memory_utilization
        if (not isinstance(fraction, (int, float)) or isinstance(fraction, bool)
                or not math.isfinite(fraction) or not 0 < fraction <= 1):
            raise ValueError("gpu_memory_utilization must be finite and in (0, 1]")
        if self.dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("unsupported dtype")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.enforce_eager, bool):
            raise ValueError("enforce_eager must be bool")
        if not isinstance(self.skip_mm_profiling, bool):
            raise ValueError("skip_mm_profiling must be bool")


@dataclass(frozen=True)
class GeneratedText:
    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int


def resolve_cached_repo(repo_id: str, revision: str) -> str:
    """Honor HF's configured cache and the repository-managed .hf/hub cache."""
    repo_cache = Path(__file__).resolve().parents[2] / ".hf" / "hub"
    candidate = repo_cache / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
    if candidate.is_dir():
        return str(candidate.resolve())
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id, revision=revision, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"Checkpoint not cached: {repo_id}@{revision}") from exc


def resolve_checkpoint(model_path: str, revision: str) -> tuple[str, str | None]:
    """Resolve pinned cached weights without downloads or content hashing."""
    path = Path(model_path).expanduser()
    if path.is_dir():
        resolved, cached_revision = path.resolve(), None
    else:
        resolved = Path(resolve_cached_repo(model_path, revision)).resolve()
        cached_revision = revision
    if not (resolved / "config.json").is_file():
        raise ValueError(f"Checkpoint has no config.json: {resolved}")
    return str(resolved), cached_revision


def multimodal_request(prompt: str, data: dict, processor_kwargs: dict | None = None) -> dict:
    """Use unique request IDs instead of hashing fixed input pixels for MM caching."""
    uuids = {}
    for modality, items in data.items():
        count = len(items) if isinstance(items, list) else 1
        uuids[modality] = [uuid4().hex for _ in range(count)]
    return {
        "prompt": prompt,
        "multi_modal_data": data,
        "multi_modal_uuids": uuids,
        "mm_processor_kwargs": dict(processor_kwargs or {}),
    }


class VLLMBackend:
    def __init__(self, model_path: str, revision: str, config: VLLMConfig,
                 *, limit_mm_per_prompt: dict[str, int]):
        self.model_path = model_path
        self.revision = revision
        self.config = config
        self.limit_mm_per_prompt = dict(limit_mm_per_prompt)
        self._llm = None
        self._processor = None
        self._sampling_params = None
        self.model_dir = None
        self.cached_revision = None
        self.batch_count = 0
        self.request_batch_size = config.batch_size

    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self.model_dir, self.cached_revision = resolve_checkpoint(
                self.model_path, self.revision)
            self._processor = AutoProcessor.from_pretrained(
                self.model_dir, local_files_only=True, trust_remote_code=False)
        return self._processor

    def _load(self):
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams

        if self.model_dir is None:
            self.model_dir, self.cached_revision = resolve_checkpoint(
                self.model_path, self.revision)
        self._sampling_params = SamplingParams
        self._llm = LLM(
            model=self.model_dir,
            dtype=self.config.dtype,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            tensor_parallel_size=self.config.tensor_parallel_size,
            max_num_seqs=self.config.batch_size,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            enable_chunked_prefill=True,
            skip_mm_profiling=self.config.skip_mm_profiling,
            seed=self.config.seed,
            enforce_eager=self.config.enforce_eager,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            generation_config="auto",
            trust_remote_code=False,
        )

    def generate(self, requests: Iterable[dict], *, max_tokens: int,
                 batch_size: int | None = None,
                 chunk_size: int | None = None) -> list[GeneratedText]:
        """Greedy-decode requests in input order.

        ``chunk_size`` submits that many requests per LLM.generate call. The
        engine still runs at most ``config.batch_size`` sequences at once, so a
        larger chunk only keeps its queue full instead of idling on each
        chunk's longest generation.
        """
        positive_int("max_tokens", max_tokens)
        if batch_size is not None and chunk_size is not None:
            raise ValueError("pass batch_size or chunk_size, not both")
        self.request_batch_size = self.config.batch_size if batch_size is None else positive_int(
            "batch_size", batch_size)
        if self.request_batch_size > self.config.batch_size:
            raise ValueError("request batch_size cannot exceed the engine's configured batch_size")
        if chunk_size is not None:
            self.request_batch_size = positive_int("chunk_size", chunk_size)
        pending = iter(requests)
        texts = []
        self.batch_count = 0
        while batch := list(islice(pending, self.request_batch_size)):
            self._load()
            params = self._sampling_params(
                n=1, temperature=0.0, top_p=1.0, top_k=-1,
                max_tokens=max_tokens, seed=self.config.seed)
            # LLM.generate returns outputs in the order of its input prompts.
            outputs = self._llm.generate(batch, params, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned an incomplete batch")
            for output in outputs:
                if len(output.outputs) != 1:
                    raise RuntimeError("vLLM must return exactly one completion per request")
                completion = output.outputs[0]
                if not isinstance(completion.text, str) or not completion.text.strip():
                    raise RuntimeError("vLLM returned an empty completion")
                texts.append(GeneratedText(
                    completion.text, completion.finish_reason,
                    len(output.prompt_token_ids or []), len(completion.token_ids)))
            self.batch_count += 1
        return texts

    def metadata(self) -> dict:
        return {
            "backend": "vllm", "engine_config": asdict(self.config),
            "request_batch_size": self.request_batch_size,
            "model_dir": self.model_dir, "cached_revision": self.cached_revision,
            "requested_revision": self.revision,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "package_versions": {name: version(name) for name in
                                 ("vllm", "torch", "transformers", "qwen-vl-utils")},
            "multimodal_identity": "unique_request_uuid; no pixel-content hash",
        }

    def close(self):
        if self._llm is not None:
            # Shut down this engine's own worker process, never global CUDA state.
            engine = self._llm.llm_engine
            client = getattr(engine, "engine_core", None)
            if client is not None:
                client.shutdown()
            self._llm = None
        self._processor = None


class VLLMAdapter(BaseAdapter):
    """Unified score interface for adapters whose model runs in a vLLM engine.

    Same contract as :meth:`BaseAdapter.score` without the torch peak-VRAM probe
    (the engine's memory lives in its own worker process).
    """

    def score(self, traj):
        self._check_views(traj)
        start = time.perf_counter()
        out = self._score(traj)
        out.model_name = out.model_name or self.spec.name
        out.sampling = out.sampling or traj.sampling
        cost = out.cost or CostRecord()
        cost.latency_s = time.perf_counter() - start
        cost.n_calls = int(out.meta["n_backend_calls"])
        out.cost = cost
        out.meta.setdefault("upstream", self.spec.upstream)
        out.meta.setdefault("weights", self.spec.weights)
        return out
