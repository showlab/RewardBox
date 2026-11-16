"""Opt-in vLLM generation with VLAC-Cut's existing native sampling/curve bridge."""
from __future__ import annotations

import importlib.util
from dataclasses import replace
from functools import lru_cache

from ..backends.vllm import REQUIREMENTS, VLLMAdapter, VLLMBackend, VLLMConfig, multimodal_request
from ..backends.vllm import positive_int
from ..backends.vlac_vision import NativeVLACVision
from ..core.enums import ImplementationKind
from .vlac_cut import SPEC as NATIVE_SPEC
from .vlac_cut import VLACCutAdapter, _MODEL_ASSUMED_SAMPLE_HZ, _WEIGHTS_REVISION

SPEC = replace(
    NATIVE_SPEC, name="vlac-cut-vllm", cls="VLACCutVLLMAdapter", module=__name__,
    requires=REQUIREMENTS, implementation_kind=ImplementationKind.REWARDBOX_EXTENSION,
    description="VLAC-Cut native Swift frame/prompt/curve protocol with opt-in vLLM generation.",
)


@lru_cache(maxsize=1)
def _native_vision_process():
    from qwen_vl_utils import vision_process

    # fetch_video calls fetch_image with a different patch factor before its
    # final resize. Passing max_pixels to fetch_video would change that first
    # resize. Give the unmodified upstream utility a private module namespace
    # with Swift's native constants instead, preserving both resize stages.
    spec = importlib.util.spec_from_file_location(
        "_rewardbox_vlac_cut_vision_process", vision_process.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.IMAGE_MAX_TOKEN_NUM = 256
    module.IMAGE_MIN_TOKEN_NUM = 4
    module.VIDEO_MAX_TOKEN_NUM = 256
    module.VIDEO_MIN_TOKEN_NUM = 4
    module.FPS = _MODEL_ASSUMED_SAMPLE_HZ
    return module


def prepare_swift_video(image_paths: list[str]):
    """Mirror Swift's Qwen3-VL frame-list boundary without global env mutations."""
    import torch

    # Upstream uses 256 visual tokens/frame, patch=16 and spatial merge=2.
    # The requested sample_hz is mapped to the model's fixed 2 Hz axis by
    # VLACCutAdapter._score after generation, exactly as in run_example.py.
    (video, metadata), _ = _native_vision_process().fetch_video(
        {"video": image_paths, "sample_fps": _MODEL_ASSUMED_SAMPLE_HZ},
        image_patch_size=16, return_video_metadata=True, return_video_sample_fps=True)
    video = video.to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
    return video, {**metadata, "do_sample_frames": False}


class VLACCutVLLMAdapter(VLACCutAdapter, VLLMAdapter):
    def __init__(self, spec, model_path: str | None = None, revision: str = _WEIGHTS_REVISION,
                 sample_hz: float = 2.0, frame_selection: str = "official",
                 curve_alignment: str = "vpb_index", task_plan_key: str = "task_plan",
                 prompt_key: str = "vlac_cut_prompt", max_new_tokens: int = 1024,
                 batch_size: int = 1, gpu_memory_utilization: float = 0.7,
                 max_model_len: int = 32768, tensor_parallel_size: int = 1,
                 dtype: str = "bfloat16", seed: int = 0, enforce_eager: bool = False,
                 max_num_batched_tokens: int = 8192, skip_mm_profiling: bool = False,
                 max_video_frames: int | None = None,
                 vision_preprocess_python: str | None = None, **opts):
        # Reuse the Swift reference's selection, JPEG boundary, prompt builder,
        # keypoint parser and both official curve alignments. Replace generation.
        super().__init__(spec, model_path=model_path, revision=revision, backend="swift",
                         sample_hz=sample_hz, frame_selection=frame_selection,
                         curve_alignment=curve_alignment, task_plan_key=task_plan_key,
                         prompt_key=prompt_key, max_new_tokens=max_new_tokens, **opts)
        self.config = VLLMConfig(batch_size, gpu_memory_utilization, max_model_len,
                                 tensor_parallel_size, dtype, seed, enforce_eager,
                                 max_num_batched_tokens, skip_mm_profiling)
        if max_video_frames is not None:
            positive_int("max_video_frames", max_video_frames)
        if skip_mm_profiling and max_video_frames is None:
            raise ValueError("skip_mm_profiling requires an explicit max_video_frames bound")
        self.max_video_frames = max_video_frames
        self._vision = NativeVLACVision(vision_preprocess_python) if vision_preprocess_python else None
        self._backend = None
        self._generated = []

    def _load(self):
        if self._backend is None:
            self._backend = VLLMBackend(self.model_path, self.revision, self.config,
                                       limit_mm_per_prompt={"image": 0, "video": 1})
        return self._backend

    def _generate(self, image_paths: list[str], prompt: str) -> str:
        padded_frames = len(image_paths) + len(image_paths) % 2
        if self.max_video_frames is not None and padded_frames > self.max_video_frames:
            raise ValueError("input exceeds max_video_frames after native padding; frames are never dropped")
        runtime = self._load()
        messages = [{"role": "user", "content": [
            {"type": "video", "video": image_paths}, {"type": "text", "text": prompt},
        ]}]
        text = runtime.processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
        video, metadata = (self._vision.prepare(image_paths) if self._vision else
                           prepare_swift_video(image_paths))
        request = multimodal_request(
            text, {"video": [(video, metadata)]},
            {"do_sample_frames": False, "do_resize": False})
        self._generated = runtime.generate([request], max_tokens=self.max_new_tokens)
        if len(self._generated) != 1:
            raise RuntimeError("VLAC-Cut vLLM must return one response per video")
        self._runtime = runtime.metadata()
        return self._generated[0].text

    def _score(self, traj):
        out = super()._score(traj)
        out.meta.update({
            "adapter_calls_official": False, "backend": "vllm",
            "protocol": "vlac_cut_swift_bridge_vllm_v1",
            "reference_backend": "swift",
            "execution_extension": "RewardBox vLLM replaces official Swift PtEngine generation",
            "video_preprocessing": "qwen-vl-utils; JPEG95; patch16; 256 visual tokens/frame",
            "vision_preprocess_runtime": self._vision.runtime if self._vision else "in-process",
            "resampling_in_backend": False,
            "max_video_frames": self.max_video_frames,
            "n_inference_requests": 1,
            "n_backend_calls": self._backend.batch_count,
            "finish_reasons": [item.finish_reason for item in self._generated],
            "prompt_tokens": sum(item.prompt_tokens for item in self._generated),
            "completion_tokens": sum(item.completion_tokens for item in self._generated),
            "paper_equivalence": "not asserted; Swift bridge reused, GPU calibration pending",
        })
        return out

    def close(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        if self._vision is not None:
            self._vision.close()
        self._engine = None
        self._generated = []
