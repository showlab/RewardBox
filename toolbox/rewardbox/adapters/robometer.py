"""Robometer adapter.

Official: https://github.com/robometer/robometer  commit 5b815254
Entry:    robometer.utils.save.load_model_from_hf
      robometer.utils.setup_utils.setup_batch_collator
      robometer.evals.eval_server.compute_batch_outputs

⚠️ Fundamental difference from the others: Robometer goes through **no VLM
   chat/generate interface**.  It loads a custom reward_model (progress head + success
   head); one forward pass yields per-frame progress and per-frame success probability
   for the whole trajectory, with no prefix sampling.

⚠️ Side-effect warning: importing robometer.utils.setup_utils runs its line 7
   ``from unsloth import FastVisionModel``, and unsloth **monkeypatches**
   torch/transformers globally.  Other models in the same process are affected; see
   docs/installation.md.
"""
from __future__ import annotations

import numpy as np

from ..core.base import BaseAdapter
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import RewardOutput, SamplingSpec, Trajectory
from ..upstream import ensure_importable

_REQ = Requirements(
    pip=("torch>=2.8", "transformers>=4.57", "unsloth>=2025.10", "trl",
         "codetiming", "tensorboard", "peft", "sentence-transformers"),
    python=">=3.10",     # the official pyproject says ==3.10.* without a code reason; setup.py says >=3.10
    vram_gb=14.0,
    note="Do not install tensorflow / tensorflow-datasets / protobuf<3.21: "
         "upstream comments them as 'just for oxe dataset uploading', and they hard-conflict with vllm.",
)

_COMMON = dict(
    cls="RobometerAdapter", module=__name__,
    reward_mode=RewardMode.PROGRESS, signal_type=SignalType.VALUE_HEAD,
    requires=_REQ,
    # 8 frames = the checkpoint config's max_frames, also the value used by all three eval
    # commands in the upstream README (reward_alignment / policy_ranking / confusion_matrix).
    # The adapter does not clamp: num_frames decides how many frames the model sees, so it
    # must match upstream.  Measured (paired-200, same membership, upstream Pearson): 8
    # frames ID mean 0.9219 vs 16 frames 0.8510, paired difference -0.0651, 95% CI
    # [-0.0739, -0.0559]; on all 19,796 trajectories 8 frames give ID 0.9170 / OOD 0.9502,
    # -0.0023 / +0.0057 from the paper's Table XII 0.9193 / 0.9445.
    sampling=SamplingSpec(num_frames=8, strategy="uniform"),
    score_range=(0.0, 1.0), higher_better=True,
    supports_pairwise=True, outputs_success=True,
    paper="https://arxiv.org/abs/2603.02115",
    upstream="robometer.evals.eval_server:compute_batch_outputs",
    description="Dedicated reward model with a progress head and a success head; one forward "
                "pass outputs per-frame progress and success probability for the whole trajectory; "
                "natively supports trajectory comparison.",
    implementation_kind=ImplementationKind.OFFICIAL_ADAPTER,
)

SPECS = [
    ModelSpec(name="robometer-4b", weights="aliangdw/Robometer-4B",
              model_opts={"model_path": "aliangdw/Robometer-4B"}, **_COMMON),
    ModelSpec(name="robometer-4b-libero", weights="aliangdw/Robometer-4B-LIBERO",
              model_opts={"model_path": "aliangdw/Robometer-4B-LIBERO"}, **_COMMON),
]


class RobometerAdapter(BaseAdapter):
    def __init__(self, spec, model_path: str | None = None, device: str | None = None,
                 **opts):
        super().__init__(spec, **opts)
        self.model_path = model_path or spec.model_opts.get("model_path") or spec.weights
        self._device_arg = device
        self._loaded = None

    # ---- official model loading ----
    def _ensure_loaded(self):
        if self._loaded is not None:
            return self._loaded
        import torch
        ensure_importable("robometer")
        from robometer.utils.save import load_model_from_hf
        from robometer.utils.setup_utils import setup_batch_collator

        dev = torch.device(self._device_arg or
                           ("cuda" if torch.cuda.is_available() else "cpu"))
        exp_config, tokenizer, processor, model = load_model_from_hf(
            model_path=self.model_path, device=dev)
        if model is None:
            raise RuntimeError(f"load_model_from_hf returned no model: {self.model_path}")
        model.eval()
        collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
        self._loaded = (exp_config, tokenizer, processor, model, collator, dev)
        return self._loaded

    def _wrap(self, traj: Trajectory, tid: str):
        from robometer.data.dataset_types import Trajectory as RmTrajectory
        f = np.ascontiguousarray(traj.primary)
        return RmTrajectory(frames=f, frames_shape=tuple(f.shape), task=traj.instruction,
                            id=tid, metadata={"subsequence_length": int(f.shape[0])},
                            video_embeddings=None)

    def _to_device(self, inputs, dev, dtype=None):
        """Move to device + align floating tensors with the model dtype.

        The official example_inference_local.py only calls ``.to(device)``, which on
        transformers 4.57.1 raises ``expected scalar type BFloat16 but found Float``
        (vision tower layer_norm): the model loads with config.yaml torch_dtype=bfloat16 while
        the processor yields float32 pixel_values.  One dtype alignment is added here; it is
        device/precision placement, not an algorithm change.  Integer tensors (input_ids /
        grid_thw / attention_mask) must not be converted and stay as they are.
        """
        import torch
        out = {}
        for k, v in inputs.items():
            if not hasattr(v, "to"):
                out[k] = v
            elif dtype is not None and getattr(v, "is_floating_point", None) \
                    and v.is_floating_point():
                out[k] = v.to(device=dev, dtype=dtype)
            else:
                out[k] = v.to(dev)
        return out

    @staticmethod
    def _model_dtype(model):
        try:
            return next(model.parameters()).dtype
        except StopIteration:
            return None

    # ---- progress ----
    def _forward_progress(self, traj: Trajectory):
        # _ensure_loaded() runs first: it puts the pinned checkout on sys.path, which
        # ``import robometer.*`` needs when the package is not pip-installed.
        exp_config, tokenizer, _, model, collator, dev = self._ensure_loaded()
        from robometer.data.dataset_types import ProgressSample
        from robometer.evals.eval_server import compute_batch_outputs
        batch = collator([ProgressSample(trajectory=self._wrap(traj, "0"),
                                         sample_type="progress")])
        inputs = self._to_device(batch["progress_inputs"], dev)   # autocast handles precision
        import torch

        loss_cfg = getattr(exp_config, "loss", None)
        is_discrete = (getattr(loss_cfg, "progress_loss_type", "l2").lower() == "discrete"
                       if loss_cfg else False)
        num_bins = (getattr(loss_cfg, "progress_discrete_bins", None)
                    or getattr(exp_config.model, "progress_discrete_bins", 10))

        # autocast: this checkpoint is mixed precision (LayerNorm fp32 / Linear bf16, from
        # transformers' _keep_in_fp32_modules); a plain forward raises
        # "expected scalar type BFloat16 but found Float".  autocast does not change weights.
        with torch.autocast("cuda", dtype=self._model_dtype(model)) if dev.type == "cuda" \
                else _nullctx():
            res = compute_batch_outputs(model, tokenizer, inputs, sample_type="progress",
                                        is_discrete_mode=is_discrete, num_bins=num_bins)

        prog = res.get("progress_pred") or []
        prog = np.asarray(prog[0], dtype=np.float64) if len(prog) else np.array([], np.float64)
        succ_out = res.get("outputs_success") or {}
        succ = succ_out.get("success_probs") or []
        succ = np.asarray(succ[0], dtype=np.float64) if len(succ) else None
        if prog.size == 0:
            raise RuntimeError("Robometer returned no progress_pred")
        # attn_implementation is chosen by the official setup_utils depending on whether
        # flash_attn imports (not by this adapter); it is recorded so that "the environment
        # changed, so the output changed" stays traceable, see docs/installation.md.
        # Note: the outer RBM wrapper's
        # model.config holds the wrapper's own default (always "sdpa"); self-attention runs in
        # model.model (the Qwen3VLModel backbone), and measurements confirm that only
        # backbone.config._attn_implementation changes with flash_attn availability.
        backbone = getattr(model, "model", model)
        return prog, succ, {"is_discrete": is_discrete, "num_bins": int(num_bins),
                            "autocast_dtype": str(self._model_dtype(model)),
                            "attn_implementation": getattr(backbone.config, "_attn_implementation", None)}

    def _score(self, traj: Trajectory) -> RewardOutput:
        prog, succ, info = self._forward_progress(traj)
        T = len(traj)
        native_len = int(prog.size)

        # The official output length may differ from our sampled frame count (the model has
        # its own internal frame rate).  The official rewardscope integration script resamples
        # with np.interp; we do the same and mark it explicitly.
        resampled = native_len != T
        if resampled:
            src = np.linspace(0, native_len - 1, T)
            base = np.arange(native_len)
            if succ is not None and succ.size == native_len:
                succ = np.interp(src, base, succ)
            else:
                succ = None
            prog = np.interp(src, base, prog)
        elif succ is not None and succ.size != T:
            succ = None

        return self._out(
            prog, traj, success=succ,
            raw={"native_length": native_len, **info},
            meta={"adapter_calls_official": True,
                  "model_path": self.model_path,
                  "resampled_to_sampling_grid": resampled,
                  "unsloth_loaded": "unsloth" in _loaded_modules(),
                  "autocast_dtype": info.get("autocast_dtype"),
                  "n_backend_calls": 1},
        )

    # ---- preference ----
    def _compare(self, a: Trajectory, b: Trajectory) -> float:
        """Native preference head.  >0 means a is better.

        Official semantics (eval_server.py:547): pref_logits -> sigmoid -> prob, and
        prob > 0.5 means chosen (the first trajectory) is better.  Mapped to [-1, 1] here.
        """
        _, tokenizer, _, model, collator, dev = self._ensure_loaded()   # sys.path first
        from robometer.data.dataset_types import PreferenceSample
        from robometer.evals.eval_server import compute_batch_outputs
        sample = PreferenceSample(chosen_trajectory=self._wrap(a, "chosen"),
                                  rejected_trajectory=self._wrap(b, "rejected"))
        batch = collator([sample])
        if "preference_inputs" not in batch:
            raise RuntimeError(
                f"the collator produced no preference_inputs (got {sorted(batch)}); "
                "this checkpoint may not have a trained preference head")
        import torch
        inputs = self._to_device(batch["preference_inputs"], dev)
        with torch.autocast("cuda", dtype=self._model_dtype(model)) if dev.type == "cuda" \
                else _nullctx():
            res = compute_batch_outputs(model, tokenizer, inputs, sample_type="preference")
        probs = res.get("prediction_probs")
        if not probs:
            raise RuntimeError("preference head returned no prediction_probs")
        return (float(probs[0]) - 0.5) * 2.0

    def close(self):
        self._loaded = None


def _nullctx():
    import contextlib
    return contextlib.nullcontext()


def _loaded_modules() -> set[str]:
    import sys
    return set(sys.modules)
