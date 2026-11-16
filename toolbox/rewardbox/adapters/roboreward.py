"""RoboReward adapter.

⚠️ The **only model of the first four without official code**; there is just a model card:
   https://huggingface.co/teetone/RoboReward-4B  (CC-BY-4.0)

So this is the one adapter where RewardBox writes the inference itself, and it must be
extra conservative:
  - the prompt is the model card's original text (prompts/roboreward.txt, snapshot-tested)
  - input follows the model card's "native Qwen3-VL video input" -> standard process_vision_info
  - output follows the model card's ``ANSWER: <score>``, parsed by regex, mapped to (s-1)/4
  - no invented expectation decoding, no change of the default parsing strategy

⚠️ Methodology note (readers must know this):
   the model card defines an **end-of-episode** score ("judge only the final state").
   ``mode="prefix"``, one call per prefix to obtain a curve, is a RewardBox/reward-scope
   **extension**, not prescribed by the model card.  meta marks it; the survey must say so.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..core.base import BaseAdapter
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import RewardOutput, SamplingSpec, Trajectory

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "roboreward.txt"

_REQ = Requirements(
    pip=("torch>=2.8", "transformers>=4.57,<4.58", "qwen-vl-utils>=0.0.14", "accelerate"),
    python=">=3.10", vram_gb=20.0,
    note="No official inference code; this adapter is built from the model card.",
)

_COMMON = dict(
    cls="RoboRewardAdapter", module=__name__,
    reward_mode=RewardMode.PROGRESS, signal_type=SignalType.GENERATED_TEXT,
    requires=_REQ,
    sampling=SamplingSpec(num_frames=16, strategy="uniform"),
    score_range=(0.0, 1.0), higher_better=True,
    paper="https://arxiv.org/abs/2601.00675",
    upstream="(no official code) model card: https://huggingface.co/teetone/RoboReward-4B",
    description="Fine-tuned Qwen3-VL that outputs a discrete 1-5 terminal progress score, mapped to (s-1)/4. "
                "The model card defines an end-of-episode score; per-prefix calls are an extension.",
    implementation_kind=ImplementationKind.MODEL_CARD_REFERENCE,
)

SPECS = [
    ModelSpec(name="roboreward-4b", weights="teetone/RoboReward-4B",
              model_opts={"model_name": "teetone/RoboReward-4B"}, **_COMMON),
    ModelSpec(name="roboreward-8b", weights="teetone/RoboReward-8B",
              model_opts={"model_name": "teetone/RoboReward-8B"}, **_COMMON),
]


def load_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8").rstrip("\n")


def parse_answer(text: str) -> float | None:
    """Parse the model card's ``ANSWER: <score>``.  Returns None on failure (no silent score)."""
    m = re.search(r"ANSWER\s*:\s*([1-5])", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    tail = re.findall(r"\b([1-5])\b", text)
    return float(tail[-1]) if tail else None


class RoboRewardAdapter(BaseAdapter):
    def __init__(self, spec, model_name: str | None = None, device: str | None = None,
                 mode: str = "prefix", max_new_tokens: int = 128,
                 video_fps: float = 1.0, image_patch_size: int = 16, **opts):
        """mode:
          prefix   -- score every prefix to get a curve (RewardBox/reward-scope extension, default)
          terminal -- score the whole video once (the model card's definition; progress is broadcast as a constant)
        """
        super().__init__(spec, **opts)
        if mode not in ("prefix", "terminal"):
            raise ValueError(f"mode must be prefix|terminal, got {mode!r}")
        self.model_name = model_name or spec.model_opts.get("model_name") or spec.weights
        self._device_arg = device
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.video_fps = video_fps
        self.image_patch_size = image_patch_size
        self._loaded = None

    def _ensure_loaded(self):
        if self._loaded is not None:
            return self._loaded
        import torch
        from transformers import AutoProcessor
        ModelCls = _pick_model_class(self.model_name)
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        model = ModelCls.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16,
            device_map=self._device_arg or "auto", trust_remote_code=True)
        model.eval()
        self._loaded = (model, processor, torch)
        return self._loaded

    def _score_clip(self, frames: np.ndarray, instruction: str) -> tuple[float | None, str]:
        """Per the model card: native Qwen3-VL video input + ANSWER: <score> output."""
        from qwen_vl_utils import process_vision_info

        from ..core.video import to_pil
        model, processor, torch = self._ensure_loaded()

        # Qwen3-VL's video channel needs >= 2 frames
        pil = [to_pil(f) for f in frames]
        if len(pil) == 1:
            pil = pil * 2

        messages = [{"role": "user", "content": [
            {"type": "video", "video": pil, "fps": self.video_fps},
            {"type": "text", "text": load_prompt().format(instruction=instruction)},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        # return_video_metadata=True is required: Qwen3-VL builds the prompt from timestamps,
        # and without video_metadata it falls back to fps=24 with a warning, changing the real input.
        images, videos, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True, return_video_metadata=True,
            image_patch_size=self.image_patch_size)
        video_metadata = None
        if videos is not None:
            vids, metas = zip(*videos)
            videos, video_metadata = list(vids), list(metas)
        inputs = processor(text=[text], images=images, videos=videos,
                           video_metadata=video_metadata, padding=True,
                           return_tensors="pt", do_resize=False, **video_kwargs)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                     do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        text_out = processor.decode(out_ids[0][n_in:], skip_special_tokens=True).strip()
        return parse_answer(text_out), text_out

    def _score(self, traj: Trajectory) -> RewardOutput:
        frames = traj.primary
        scores, texts, failures = [], [], []

        clips = ([frames] if self.mode == "terminal"
                 else [frames[: k + 1] for k in range(len(frames))])
        for i, clip in enumerate(clips):
            s, txt = self._score_clip(clip, traj.instruction)
            texts.append(txt)
            if s is None:
                failures.append(i)
                s = scores[-1] if scores else 1.0     # conservative: reuse the previous score, lowest for the first
            scores.append(float(s))

        raw = np.asarray(scores, dtype=np.float64)
        progress = (raw - 1.0) / 4.0                  # model card: 1 -> 0.0, 5 -> 1.0
        if self.mode == "terminal":
            progress = np.full(len(traj), progress[0])

        return self._out(
            progress, traj,
            raw={"discrete_scores": scores, "responses": texts,
                 "parse_failures": failures},
            meta={"adapter_calls_official": False,
                  "implemented_from": "model card (no official code)",
                  "mode": self.mode,
                  "extension_notice": (
                      "The model card defines an end-of-episode score; mode='prefix' (one call per prefix) "
                      "is a RewardBox extension, not prescribed by the model card"
                      if self.mode == "prefix" else ""),
                  "n_parse_failures": len(failures),
                  "video_fps": self.video_fps,
                  "image_patch_size": self.image_patch_size,
                  "n_backend_calls": len(clips)},
        )

    def close(self):
        self._loaded = None


def _pick_model_class(model_name: str):
    """Capability probe; no transformers version comparison."""
    import transformers
    for cand in ("Qwen3VLForConditionalGeneration",
                 "Qwen2_5_VLForConditionalGeneration"):
        cls = getattr(transformers, cand, None)
        if cls is not None:
            return cls
    from transformers import AutoModelForVision2Seq
    return AutoModelForVision2Seq
