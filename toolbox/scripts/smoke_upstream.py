#!/usr/bin/env python3
"""Shared-environment smoke test: official entry points exist and match adapter assumptions.

The official entry points of the integrated models must import and have the signatures the
adapters assume.  No weight download, no inference: it only checks that
"what the adapter intends to call really exists and looks like this".  The last gate before
writing an adapter.

VLAC pins ms-swift 3.3 and deliberately stays out of this shared environment; check its official
signatures in a separate environment with ``python -m rewardbox.cli doctor vlac-2b --deep``.

Usage:
    .venv/bin/python scripts/smoke_upstream.py
"""
from __future__ import annotations

import inspect
import subprocess
import sys
import os
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE = Path(os.environ.get("REWARDBOX_UPSTREAM_DIR") or REPO_ROOT / "reward_model_code").expanduser()

OK, FAIL, SKIP = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[33m-\033[0m"
results: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator: run one check and record its result."""
    def deco(fn):
        try:
            detail = fn() or ""
            results.append((name, True, str(detail)))
            print(f"  {OK} {name}" + (f"  → {detail}" if detail else ""))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            results.append((name, False, msg))
            print(f"  {FAIL} {name}\n      {msg}")
            if "-v" in sys.argv:
                traceback.print_exc()
        return fn
    return deco


def sig(fn) -> str:
    return f"{fn.__name__}{inspect.signature(fn)}"


def has_params(fn, *names) -> list[str]:
    """Return the missing parameter names."""
    p = inspect.signature(fn).parameters
    return [n for n in names if n not in p]


# ---------------------------------------------------------------- base environment
print("\n=== 0. base environment ===")


@check("python 3.11")
def _():
    v = sys.version_info
    assert v[:2] == (3, 11), f"expected 3.11, actual {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"


@check("torch + CUDA")
def _():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    return f"torch {torch.__version__}, {torch.cuda.device_count()} GPU"


@check("transformers")
def _():
    import transformers
    return transformers.__version__


# ---------------------------------------------------------------- qwen-vl-utils
print("\n=== 1. qwen-vl-utils (RoboReward adapter dependency) ===")


@check("process_vision_info signature")
def _():
    from qwen_vl_utils import process_vision_info
    s = inspect.signature(process_vision_info)
    return sig(process_vision_info)


@check("process_vision_info supports return_video_kwargs")
def _():
    from qwen_vl_utils import process_vision_info
    missing = has_params(process_vision_info, "return_video_kwargs")
    assert not missing, f"missing parameters: {missing} (needed by RoboReward's native video path)"
    return "yes"


@check("process_vision_info supports image_patch_size / return_video_metadata")
def _():
    from qwen_vl_utils import process_vision_info
    p = inspect.signature(process_vision_info).parameters
    have = [n for n in ("image_patch_size", "return_video_metadata") if n in p]
    # no assert: reward-scope uses both parameters, but 0.0.14 may not have them.
    # The result decides whether the adapter passes them.
    return f"supports {have}" if have else "supports neither -> the adapter must not pass them"


# ---------------------------------------------------------------- TOPReward
print("\n=== 2. TOPReward ===")
sys.path.insert(0, str(CODE / "TOPReward"))


@check("import topreward")
def _():
    import topreward
    return getattr(topreward, "__file__", "?")


@check("QwenClient importable")
def _():
    from topreward.clients.qwen import QwenClient
    return QwenClient.__mro__[1].__name__


@check("compute_instruction_rewards_for_prefixes signature")
def _():
    from topreward.clients.qwen import QwenClient
    fn = QwenClient.compute_instruction_rewards_for_prefixes
    missing = has_params(fn, "frames", "instruction", "num_samples",
                         "reduction", "fps", "add_chat_template")
    assert not missing, f"missing parameters: {missing}"
    return sig(fn)


@check("compute_instruction_reward signature")
def _():
    from topreward.clients.qwen import QwenClient
    return sig(QwenClient.compute_instruction_reward)


@check("normalize_rewards official semantics (single point -> 1.0, all equal -> all 1)")
def _():
    import numpy as np
    from topreward.clients.qwen import QwenClient
    f = QwenClient.normalize_rewards
    assert np.allclose(f([-3.0]), [1.0]), "a single point should return 1.0"
    assert np.allclose(f([-3.0, -3.0, -3.0]), [1.0, 1.0, 1.0]), "all-equal input should return all 1"
    assert np.allclose(f([-4.0, -3.0, -2.0]), [0.0, 0.5, 1.0])
    return "confirmed (differs from RewardOutput.normalized semantics, see R2 §1.2)"


@check("InstructionRewardResult fields")
def _():
    from topreward.metrics.instruction_reward import InstructionRewardResult
    names = set(InstructionRewardResult.__dataclass_fields__)
    need = {"prefix_lengths", "prefix_rewards", "normalized_prefix_rewards", "reward"}
    assert need <= names, f"missing fields: {need - names}"
    return f"{len(names)} fields"


@check("QwenClient constructor signature (the adapter passes model_name)")
def _():
    from topreward.clients.qwen import QwenClient
    return sig(QwenClient.__init__)


# ---------------------------------------------------------------- Robometer
print("\n=== 3. Robometer ===")
sys.path.insert(0, str(CODE / "robometer"))


@check("import robometer")
def _():
    import robometer
    return getattr(robometer, "__file__", "?")


@check("load_model_from_hf signature")
def _():
    from robometer.utils.save import load_model_from_hf
    missing = has_params(load_model_from_hf, "model_path")
    assert not missing, f"missing parameters: {missing}"
    return sig(load_model_from_hf)


@check("setup_batch_collator signature")
def _():
    from robometer.utils.setup_utils import setup_batch_collator
    missing = has_params(setup_batch_collator, "processor", "tokenizer", "cfg", "is_eval")
    assert not missing, f"missing parameters: {missing}"
    return sig(setup_batch_collator)


@check("compute_batch_outputs signature")
def _():
    from robometer.evals.eval_server import compute_batch_outputs
    missing = has_params(compute_batch_outputs, "model", "tokenizer",
                         "batch_inputs", "sample_type")
    assert not missing, f"missing parameters: {missing}"
    return sig(compute_batch_outputs)


@check("ProgressSample / PreferenceSample / Trajectory")
def _():
    from robometer.data.dataset_types import (PreferenceSample, ProgressSample,
                                              Trajectory)
    tf = set(Trajectory.model_fields)
    need = {"frames", "frames_shape", "task", "id", "metadata"}
    assert need <= tf, f"Trajectory lacks fields: {need - tf}"
    pf = set(PreferenceSample.model_fields)
    assert {"chosen_trajectory", "rejected_trajectory"} <= pf
    return "all three types present"


# ---------------------------------------------------------------- Robo-Dopamine
print("\n=== 4. Robo-Dopamine ===")
sys.path.insert(0, str(CODE / "Robo-Dopamine"))


@check("vllm importable")
def _():
    import vllm
    return vllm.__version__


@check("GRMInference importable")
def _():
    from examples.inference import GRMInference
    return "ok"


@check("run_pipeline signature")
def _():
    from examples.inference import GRMInference
    fn = GRMInference.run_pipeline
    missing = has_params(fn, "cam_high_path", "cam_left_path", "cam_right_path",
                         "out_root", "task", "frame_interval", "goal_image", "eval_mode")
    assert not missing, f"missing parameters: {missing}"
    return sig(fn)


@check("SYSTEM_PROMPT contains 8 <image> placeholders")
def _():
    from examples.inference import SYSTEM_PROMPT
    n = SYSTEM_PROMPT.count("<image>")
    assert n == 8, f"expected 8 <image>, actual {n}"
    return f"{len(SYSTEM_PROMPT)} characters, 8 placeholders"


@check("run_pipeline accepts PNG directories (the adapter relies on this)")
def _():
    import inspect as _i
    from examples.inference import get_frame_count
    src = _i.getsource(get_frame_count)
    assert "is_dir()" in src, "get_frame_count does not support directory input; the adapter would need video files"
    return "supported"


@check("blank_goal.png exists (upstream requires it without a goal image)")
def _():
    p = CODE / "Robo-Dopamine" / "examples" / "blank_goal.png"
    assert p.exists(), f"does not exist: {p}"
    return str(p.relative_to(REPO_ROOT))


# ---------------------------------------------------------------- SOLE-R1 (separate service; the shared environment only does a static contract smoke)
print("\n=== 5. SOLE-R1 (official reward server boundary) ===")


@check("SOLE-R1 vendored commit")
def _():
    root = CODE / "sole-r1-model"
    got = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = "2a1286b205577a38a2ef5d9d13efa76878485f87"
    assert got == expected, f"expected {expected}, actual {got}"
    return got[:12]


@check("SOLE-R1 official client payload / transport contract")
def _():
    root = CODE / "sole-r1-model" / "reward_client"
    main_source = (root / "main.py").read_text(encoding="utf-8")
    utils_source = (root / "utils.py").read_text(encoding="utf-8")
    required_main = (
        '"front_images": [front_frames]',
        '"wrist_images": [wrist_frames if wrist_frames is not None else front_frames]',
        '"temperature": temperature',
        '"from_zero": from_zero',
        '"external_only": external_only',
        'default=1.0, metavar="HZ"',
    )
    missing = [needle for needle in required_main if needle not in main_source]
    assert not missing, f"client payload/default contract lacks: {missing}"
    assert "class InferenceClient" in utils_source
    assert '"type": "call_interface"' in utils_source
    return "1 Hz; front + optional wrist; Agentlace call_interface"


@check("SOLE-R1 official server response / environment contract")
def _():
    root = CODE / "sole-r1-model" / "reward_server"
    server_source = (root / "src" / "reward_server" / "main.py").read_text(
        encoding="utf-8"
    )
    config_source = (root / "src" / "reward_server" / "config" / "default.yaml").read_text(
        encoding="utf-8"
    )
    pyproject_source = (root / "pyproject.toml").read_text(encoding="utf-8")
    required_server = (
        'valid_answers_ = [0]',
        '"valid_answers": valid_answers',
        '"text_outputs": text_output_list_list',
        '"text_inputs": text_input_list_list',
        'payload.get("from_zero", False)',
        'payload.get("external_only", False)',
    )
    missing = [needle for needle in required_server if needle not in server_source]
    assert not missing, f"server response contract lacks: {missing}"
    assert "service_name: rewards" in config_source
    for pin in ('"torch==2.9.0"', '"transformers==4.57.3"', '"vllm==0.12.0"'):
        assert pin in pyproject_source, f"server dependency pin drifted: {pin}"
    return "rewards:8001; first frame 0; progress + raw text; separate vLLM 0.12 environment"


# ---------------------------------------------------------------- Large-Reward-Models (separate Flask service)
print("\n=== 6. Large-Reward-Models (official HTTP server boundary) ===")


@check("Large-Reward-Models vendored commit")
def _():
    root = CODE / "Large-Reward-Models"
    got = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = "5845ab0f7c3c6a0bef8bd37bf7f07e9533f7ed2f"
    assert got == expected, f"expected {expected}, actual {got}"
    return got[:12]


@check("LRM official HTTP client method signatures")
def _():
    from rewardbox.adapters.lrm import _upstream_client_class

    client = _upstream_client_class()
    required = {
        "compute_comparison": ("image_a", "image_b", "task_description"),
        "compute_completion": ("image", "task_description"),
        "compute_reward": (
            "image", "task_description", "reward_type", "goal_image", "initial_image",
        ),
    }
    missing = {
        method: has_params(getattr(client, method), *parameters)
        for method, parameters in required.items()
    }
    missing = {method: values for method, values in missing.items() if values}
    assert not missing, f"LRM client signatures missing: {missing}"
    return "comparison/completion/progress; requests HTTP"


@check("LRM server endpoints / tri adapter activation contract")
def _():
    source = (
        CODE / "Large-Reward-Models" / "vlm_reward" / "vlm_reward_server.py"
    ).read_text(encoding="utf-8")
    required = (
        "@app.route('/health', methods=['GET'])",
        "@app.route('/compute_reward', methods=['POST'])",
        "@app.route('/compute_completion', methods=['POST'])",
        "@app.route('/compute_comparison', methods=['POST'])",
        '_activate_adapter("contrastive")',
        '_activate_adapter("progress")',
        '_activate_adapter("completion")',
        "def load_vlm_multi_adapter(",
        "delta = (lora_B @ lora_A) * scaling",
        "max_new_tokens=512",
        "max_new_tokens=128",
        "do_sample=False",
    )
    missing = [needle for needle in required if needle not in source]
    assert not missing, f"LRM server contract lacks: {missing}"
    return "HTTP :5002; 3 endpoints; deterministic generation; tri PEFT switch"


@check("LRM closed/open-loop official protocol files")
def _():
    root = CODE / "Large-Reward-Models"
    closed = (root / "config" / "eval" / "closed_loop_eval.yaml").read_text(
        encoding="utf-8"
    )
    tri = (root / "config" / "tri.yaml").read_text(encoding="utf-8")
    runner = (root / "eval" / "run_closed_loop_seeds.sh").read_text(encoding="utf-8")
    metrics = (root / "eval" / "compute_openloop_metrics.py").read_text(
        encoding="utf-8"
    )
    closed_required = (
        "total_num_envs: 320",
        "use_fixed_reset_state_ids: True",
        "do_sample: True",
        "temperature_eval: 0.6",
    )
    assert all(item in closed for item in closed_required)
    assert "vlm_tri_weights: [0.333333333333, 0.333333333333, 0.333333333334]" in tri
    assert 'SEEDS="0 1 2 3 4"' in runner and "TOTAL_NUM_ENVS=320" in runner
    for metric in (
        "roc_auc", "pairwise_acc_pct", "global_pearson", "per_traj_pearson",
        "direction_acc_pct", "progress_recall_pct", "monotonicity_success",
    ):
        assert metric in metrics, f"open-loop metric drifted: {metric}"
    return "320 episodes/seed; 5 seeds default; official mode-specific metrics"


# ---------------------------------------------------------------- RynnValue (separate Robometer service)
print("\n=== 7. RynnValue (official Robometer HTTP server boundary) ===")


@check("RynnValue vendored commit")
def _():
    root = CODE / "RynnValue"
    got = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = "10e0d333f5f3811d0d130587e50f1faf48da49e5"
    assert got == expected, f"expected {expected}, actual {got}"
    return got[:12]


@check("RynnValue official multipart/server endpoint contract")
def _():
    root = CODE / "RynnValue" / "robometer"
    client = (root / "scripts" / "example_inference.py").read_text(encoding="utf-8")
    server = (
        root / "robometer" / "evals" / "baseline_eval_server.py"
    ).read_text(encoding="utf-8")
    required_client = (
        'file_key = f"sample_{i}_trajectory_{field}"',
        'data[f"sample_{i}"] = json.dumps(sample_copy)',
        'url = eval_server_url.rstrip("/") + "/evaluate_batch_npy"',
        'data["use_frame_steps"] = "true" if use_frame_steps else "false"',
    )
    required_server = (
        '@app.post("/evaluate_batch_npy")',
        '@app.get("/health")',
        '@app.get("/model_info")',
        'elif self.reward_model == "rynnvalue":',
    )
    missing = [item for item in required_client if item not in client]
    missing += [item for item in required_server if item not in server]
    assert not missing, f"RynnValue HTTP contract lacks: {missing}"
    return "multipart .npy; evaluate_batch_npy; health + model_info"


@check("RynnValue released-checkpoint official profile")
def _():
    root = CODE / "RynnValue"
    readme = (root / "README.md").read_text(encoding="utf-8")
    model = (
        root / "robometer" / "robometer" / "evals" / "baselines" / "rynnvalue.py"
    ).read_text(encoding="utf-8")
    config = (root / "rynn_value" / "configuration_rynn_value_lang.py").read_text(
        encoding="utf-8"
    )
    for item in (
        "'max_frames=8'",
        "'custom_eval.use_frame_steps=false'",
        "'custom_eval.pad_frames=false'",
        "'model_config.stride=2'",
        "'model_config.num_frames=8'",
        "pred_slot_isolated_eager",
    ):
        assert item in readme, f"README released profile drifted: {item}"
    for item in (
        'if mode == "absolute"',
        "max_new_tokens=self.max_new_tokens",
        "do_sample=False",
        "num_beams=1",
        "result = [-100.0] * len(result)",
    ):
        assert item in model, f"RynnValue inference contract drifted: {item}"
    assert 'attn_implementation = "pred_slot_isolated_eager"' in config
    return (
        "8 frames; no frame-step/padding; absolute=-remaining seconds; "
        "greedy 128; isolated attention"
    )


# ---------------------------------------------------------------- Robo-ValueRL (separate OpenPI service)
print("\n=== 8. Robo-ValueRL (official OpenPI critic isolation boundary) ===")


def _robovaluerl_git_show(path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(CODE / "Robo-ValueRL"), "show", f"HEAD:{path}"],
        text=True,
    )


@check("Robo-ValueRL vendored commit / Git objects")
def _():
    root = CODE / "Robo-ValueRL"
    got = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = "1618bcb01212eb747a5d2659228b7fb5db87a563"
    assert got == expected, f"expected {expected}, actual {got}"
    # Read from Git objects, so that the check also works on a checkout whose
    # working tree is incomplete, and never touches it.
    assert "class OpenPi_X_Humanoid_Critic" in _robovaluerl_git_show(
        "robo_valuerl/agents/openpi_x_humanoid_critic.py"
    )
    return got[:12]


@check("Robo-ValueRL released critic / annotation drift is pinned")
def _():
    critic = _robovaluerl_git_show(
        "openpi/src/openpi/models_pytorch/"
        "x_humanoid_pi0_pytorch_critic_with_history.py"
    )
    annotation = _robovaluerl_git_show(
        "robo_valuerl/generate_action_quality/"
        "annotate_value_function_sparse_part.py"
    )
    config = _robovaluerl_git_show("openpi/src/openpi/training/config.py")
    policy = _robovaluerl_git_show("openpi/src/openpi/policies/x_humaniod_policy.py")
    assert 'return {\n            "remain_time": F.softmax' in critic
    for field in ("preds['value']", "preds['rtg']", "preds['remain_time']"):
        assert field in annotation, f"annotation field drifted: {field}"
    assert "history_idx = max(0, i - history_offset)" in annotation
    assert "interpolation_method: str = 'cubic'" in annotation
    assert "smooth: bool = True" in annotation
    assert 'name="all_value_function_training"' in config
    assert "only_predict_remain_time = True" in config
    assert "/mnt/dataset/toago6/workspace/code/hierarchical_rl_in_real_world/" in policy
    return "annotation=3 heads; critic=remain_time only; absolute-path config blocker retained"


@check("Robo-ValueRL adapter/server immutable profile pins")
def _():
    from rewardbox.adapters import robovaluerl as adapter

    server = (REPO_ROOT / "scripts" / "run_robovaluerl_server.py").read_text(
        encoding="utf-8"
    )
    assert adapter._UPSTREAM_COMMIT in server
    assert adapter._WEIGHTS_REVISION in server
    assert adapter._WEIGHTS_SHA256 in server
    for item in (
        '"history_length": 5',
        '"stride": 10',
        '"num_bins": 256',
        '"interpolation_method": "cubic"',
        '"smooth_window": 21',
        '"released_output_field": "remain_time"',
        '"config_loader": "exact-release-Pi0Config-fields"',
        '"interpolation_loader": "AST-executed-exact-pinned-functions"',
    ):
        assert item in server, f"server profile pin drifted: {item}"
    return "3 views; history 5; stride 10; 256 bins; cubic+SG; content-pinned"


# ---------------------------------------------------------------- ProcVLM (separate vLLM service)
print("\n=== 9. ProcVLM (official vLLM progress isolation boundary) ===")


@check("ProcVLM vendored commit")
def _():
    root = CODE / "ProcVLM"
    got = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = "377523a31f05bab9c0db5ac8b9edfa7b7f03968a"
    assert got == expected, f"expected {expected}, actual {got}"
    dirty = subprocess.check_output(
        ["git", "-c", "core.fileMode=false", "-C", str(root),
         "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    assert not dirty, f"ProcVLM tracked checkout has modifications: {dirty}"
    return got[:12]


@check("ProcVLM official video inference profile")
def _():
    root = CODE / "ProcVLM"
    inference = (root / "evqa" / "inference.py").read_text(encoding="utf-8")
    model = (root / "evqa" / "model.py").read_text(encoding="utf-8")
    required_inference = (
        'default=4096, help="Max generation tokens per frame"',
        'default=0.0, help="Generation temperature"',
        'default=8, help="Number of images per inference window"',
        'default=512,',
        'default=False,',
        'img.save(str(frame_path), quality=95)',
        'selected_indices = build_uniform_indices(source_frames, sampled_count)',
        'progress = last_progress',
    )
    missing = [item for item in required_inference if item not in inference]
    for item in (
        "def batch_chat_with_vllm(",
        'process_workers = int(runtime_engine_kwargs.pop("process_workers", 16)',
        'pipeline_chunk_size = int(runtime_engine_kwargs.pop("pipeline_chunk_size", 8)',
    ):
        if item not in model:
            missing.append(item)
    assert not missing, f"ProcVLM official profile drifted: {missing}"
    return "8-frame left-pad; uniform cap 512; BF16 vLLM; greedy 4096; JPEG95"


@check("ProcVLM adapter/server immutable identity pins")
def _():
    from rewardbox.adapters import procvlm as adapter

    server = (REPO_ROOT / "scripts" / "run_procvlm_server.py").read_text(
        encoding="utf-8"
    )
    assert adapter._UPSTREAM_COMMIT in server
    assert adapter._WEIGHTS_REVISION in server
    for value in adapter._WEIGHT_FILES.values():
        assert value["sha256"] in server
    assert adapter._PROMPT_SHA256 in server
    assert adapter._PROFILE_FIELDS["window_size"] == 8
    assert adapter._PROFILE_FIELDS["max_sampled_frames"] == 512
    assert adapter._PROFILE_FIELDS["enable_value_head"] is False
    return "source/prompt/2 safetensors/head hashes; profile and provenance fail closed"


# ---------------------------------------------------------------- summary
print("\n" + "=" * 62)
n_ok = sum(1 for _, ok, _ in results if ok)
n_bad = len(results) - n_ok
print(f"passed {n_ok}/{len(results)}" + (f", failed {n_bad}" if n_bad else ", all passed"))
if n_bad:
    print("\nfailed checks:")
    for name, ok, msg in results:
        if not ok:
            print(f"  - {name}: {msg}")
print("=" * 62)
sys.exit(1 if n_bad else 0)
