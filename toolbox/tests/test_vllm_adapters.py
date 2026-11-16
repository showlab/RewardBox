"""CPU contracts for batching, native LRM messages and VLAC curve/time semantics."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
from PIL import Image

import rewardbox
from rewardbox.backends.vllm import VLLMBackend, VLLMConfig
from rewardbox.core.types import SamplingSpec, Trajectory
from rewardbox.upstream import find_upstream

ROOT = Path(__file__).resolve().parents[1]
SERVER = (find_upstream("lrm") or ROOT / "reward_model_code/Large-Reward-Models") / (
    "vlm_reward/vlm_reward_server.py")


class FakeProcessor:
    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        self.messages.append(messages)
        return "native chat template"


class FakeParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLLM:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.calls = []
        self.closed = False
        self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=self.shutdown))

    def shutdown(self):
        self.closed = True

    def generate(self, requests, params, *, use_tqdm):
        assert use_tqdm is False
        self.calls.append((requests, params.kwargs))
        return [SimpleNamespace(prompt_token_ids=[1, 2], outputs=[SimpleNamespace(
            text=next(self.texts), token_ids=[3], finish_reason="stop")]) for _ in requests]


def fake_runtime(adapter, texts):
    config = adapter.config
    runtime = VLLMBackend("unused", "pinned", config, limit_mm_per_prompt={"image": 3})
    runtime._llm = FakeLLM(texts)
    runtime._sampling_params = FakeParams
    runtime._processor = FakeProcessor()
    runtime.metadata = lambda: {"backend": "vllm", "model_dir": "fake-pinned-checkpoint"}
    adapter._backend = runtime
    return runtime


def traj(n=5, *, goal=False):
    return Trajectory(
        frames=np.stack([np.full((8, 12, 3), index * 20, np.uint8) for index in range(n)]),
        instruction="put the block in the bowl", frame_indices=np.arange(n) * 7 + 3,
        sampling=SamplingSpec(strategy="all"),
        goal_image=np.full((8, 12, 3), 255, np.uint8) if goal else None)


@pytest.mark.parametrize("name", ["lrm-completion-vllm", "lrm-progress-vllm",
                                 "lrm-contrastive-vllm", "vlac-cut-vllm"])
def test_opt_in_factory_is_lazy_and_distinct(name):
    adapter = rewardbox.create_reward(name, check_deps=False)
    assert adapter._backend is None
    assert adapter.spec.implementation_kind.value == "rewardbox_extension"
    assert rewardbox.get_spec(name.removesuffix("-vllm")).implementation_kind.value == "official_adapter"
    assert adapter.config.gpu_memory_utilization == 0.7
    adapter.close()


def test_new_manifest_and_constructors_do_not_import_heavy_modules():
    code = """
import sys, rewardbox
for name in rewardbox.list_models(filter='*vllm'):
    rewardbox.create_reward(name, check_deps=False)
assert not {'torch', 'transformers', 'vllm'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


@pytest.mark.parametrize("opts", [{"batch_size": 0}, {"batch_size": True},
                                 {"gpu_memory_utilization": float("nan")},
                                 {"gpu_memory_utilization": 1.1}, {"max_model_len": 0},
                                 {"tensor_parallel_size": -1}, {"seed": True}])
def test_invalid_engine_options_rejected(opts):
    with pytest.raises(ValueError):
        rewardbox.create_reward("lrm-completion-vllm", check_deps=False, **opts)


def test_unknown_engine_option_rejected():
    with pytest.raises(TypeError, match="does not support model options"):
        rewardbox.create_reward("vlac-cut-vllm", check_deps=False, batch_sze=4)


def test_skipping_visual_profile_requires_a_bound_and_never_drops_frames():
    with pytest.raises(ValueError, match="explicit max_video_frames"):
        rewardbox.create_reward("vlac-cut-vllm", check_deps=False, skip_mm_profiling=True)
    adapter = rewardbox.create_reward("vlac-cut-vllm", check_deps=False,
                                     skip_mm_profiling=True, max_video_frames=2)
    with pytest.raises(ValueError, match="frames are never dropped"):
        adapter._generate(["frame1", "frame2", "frame3"], "task")
    assert adapter._backend is None


def test_native_cpu_bridge_is_persistent_and_never_exposes_gpus(tmp_path):
    for module in ("torch", "torchvision", "qwen_vl_utils"):   # imported by the worker process
        pytest.importorskip(module)
    from rewardbox.backends.vlac_vision import NativeVLACVision
    from rewardbox.adapters.vlac_cut_vllm import prepare_swift_video

    path = tmp_path / "frame.jpg"
    Image.fromarray(np.full((70, 90, 3), 123, np.uint8)).save(path, quality=95)
    bridge = NativeVLACVision(sys.executable)
    try:
        actual, metadata = bridge.prepare([str(path)] * 3)
        first_pid = bridge.process.pid
        expected, expected_meta = prepare_swift_video([str(path)] * 3)
        np.testing.assert_array_equal(actual, expected)
        assert metadata == expected_meta
        assert bridge.runtime["cuda_visible_devices"] == ""
        bridge.prepare([str(path)] * 2)
        assert bridge.process.pid == first_pid
        process = bridge.process
    finally:
        bridge.close()
    assert process.returncode == 0 and bridge.process is None


def test_native_cpu_bridge_rejects_a_missing_interpreter():
    with pytest.raises(ValueError, match="executable"):
        rewardbox.create_reward("vlac-cut-vllm", check_deps=False,
                                vision_preprocess_python="/missing/native/python")


def test_completion_batches_frames_preserves_order_without_hashing(monkeypatch):
    adapter = rewardbox.create_reward("lrm-completion-vllm", check_deps=False, batch_size=2)
    runtime = fake_runtime(adapter, ['{"task_completed":"no"}', '{"task_completed":"yes"}',
                                     '{"task_completed":"no"}', '{"task_completed":"yes"}',
                                     '{"task_completed":"yes"}'])
    trajectory = traj()
    monkeypatch.setattr(hashlib, "sha256", lambda *a: pytest.fail("pixel hash was called"))
    out = adapter.score(trajectory)
    assert out.progress.tolist() == [0, 1, 0, 1, 1]
    assert out.success.tolist() == out.progress.tolist()
    assert out.frame_indices.tolist() == [3, 10, 17, 24, 31]
    assert [len(call[0]) for call in runtime._llm.calls] == [2, 2, 1]
    values = [int(np.asarray(request["multi_modal_data"]["image"][0])[0, 0, 0])
              for requests, _ in runtime._llm.calls for request in requests]
    assert values == [0, 20, 40, 60, 80]
    assert out.cost.n_calls == 3 and out.meta["n_inference_requests"] == 5
    uuids = [request["multi_modal_uuids"]["image"][0]
             for requests, _ in runtime._llm.calls for request in requests]
    assert len(set(uuids)) == 5
    for _, params in runtime._llm.calls:
        assert params["temperature"] == 0 and params["max_tokens"] == 128
    llm = runtime._llm
    adapter.close()
    assert llm.closed


def test_contrastive_batches_adjacent_pairs_and_keeps_first_zero():
    adapter = rewardbox.create_reward("lrm-contrastive-vllm", check_deps=False, batch_size=2)
    runtime = fake_runtime(adapter, ['{"more_complete_image":"ImageB"}',
                                     '{"more_complete_image":"ImageA"}', "uncertain"])
    out = adapter.score(traj(4))
    assert out.progress.tolist() == [0, 1, -1, 0]
    pairs = [[int(np.asarray(img)[0, 0, 0]) for img in request["multi_modal_data"]["image"]]
             for requests, _ in runtime._llm.calls for request in requests]
    assert pairs == [[0, 20], [20, 40], [40, 60]]
    assert out.raw["responses"][0] == ""
    assert out.meta["n_inference_requests"] == 3 and out.cost.n_calls == 2
    assert all(params["max_tokens"] == 512 for _, params in runtime._llm.calls)


def test_request_chunk_submits_beyond_one_engine_batch_in_order():
    adapter = rewardbox.create_reward("lrm-contrastive-vllm", check_deps=False, batch_size=2,
                                      request_chunk_size=8)
    runtime = fake_runtime(adapter, ['{"more_complete_image":"ImageB"}',
                                     '{"more_complete_image":"ImageA"}', "uncertain"])
    out = adapter.score(traj(4))
    assert out.progress.tolist() == [0, 1, -1, 0]
    assert len(runtime._llm.calls) == 1 and out.cost.n_calls == 1
    assert runtime.config.batch_size == 2 and runtime.request_batch_size == 8
    with pytest.raises(ValueError, match="request_chunk_size"):
        rewardbox.create_reward("lrm-contrastive-vllm", check_deps=False, batch_size=8,
                                request_chunk_size=4)


def test_single_frame_contrastive_does_not_load_an_engine():
    adapter = rewardbox.create_reward("lrm-contrastive-vllm", check_deps=False)
    runtime = fake_runtime(adapter, [])
    runtime._load = lambda: pytest.fail("engine loaded for zero requests")
    out = adapter.score(traj(1))
    assert out.progress.tolist() == [0]
    assert out.cost.n_calls == 0


def test_progress_reference_order_is_goal_initial_current():
    adapter = rewardbox.create_reward("lrm-progress-vllm", check_deps=False,
                                     include_initial_image=True, include_goal_image=True)
    runtime = fake_runtime(adapter, ['{"completion_progress":0.1}', '{"completion_progress":0.8}'])
    out = adapter.score(traj(2, goal=True))
    assert out.progress.tolist() == [0.1, 0.8]
    images = runtime._llm.calls[0][0][1]["multi_modal_data"]["image"]
    assert [int(np.asarray(image)[0, 0, 0]) for image in images] == [255, 0, 20]


def test_missing_goal_rejected_before_runtime_load():
    adapter = rewardbox.create_reward("lrm-progress-vllm", check_deps=False, include_goal_image=True)
    with pytest.raises(ValueError, match="goal_image"):
        adapter.score(traj())
    assert adapter._backend is None


@pytest.mark.parametrize("kind", ["partial", "empty", "multiple"])
def test_backend_rejects_incomplete_or_invalid_outputs(kind):
    adapter = rewardbox.create_reward("lrm-completion-vllm", check_deps=False)
    runtime = fake_runtime(adapter, ["yes"])
    output = SimpleNamespace(prompt_token_ids=[], outputs=[SimpleNamespace(
        text="" if kind == "empty" else "yes", token_ids=[], finish_reason="stop")])
    if kind == "multiple":
        output.outputs *= 2
    runtime._llm.generate = lambda *a, **kw: [] if kind == "partial" else [output]
    with pytest.raises(RuntimeError):
        adapter.score(traj(1))


def test_engine_load_passes_only_explicit_budget_and_greedy_params(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    constructed = []
    fake = FakeLLM(["yes"])

    def constructor(**kwargs):
        constructed.append(kwargs)
        return fake

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=constructor, SamplingParams=FakeParams))
    backend = VLLMBackend(str(tmp_path), "pin", VLLMConfig(batch_size=8, gpu_memory_utilization=0.3),
                          limit_mm_per_prompt={"image": 1, "video": 0})
    assert constructed == []
    backend.generate([{"prompt": "test"}], max_tokens=128)
    assert constructed[0]["gpu_memory_utilization"] == 0.3
    assert constructed[0]["max_num_seqs"] == 8
    assert constructed[0]["generation_config"] == "auto"
    assert constructed[0]["model"] == str(tmp_path)
    backend.close()
    assert fake.closed


def test_backend_submission_override_keeps_engine_config_and_bounds():
    adapter = rewardbox.create_reward("lrm-completion-vllm", check_deps=False, batch_size=8)
    runtime = fake_runtime(adapter, ["yes", "no", "yes"])
    generated = runtime.generate([{"prompt": "test"}] * 3, max_tokens=128, batch_size=1)
    assert len(generated) == 3 and runtime.batch_count == 3
    assert runtime.config.batch_size == 8 and runtime.request_batch_size == 1
    with pytest.raises(ValueError, match="cannot exceed"):
        runtime.generate([], max_tokens=128, batch_size=9)


def test_repo_managed_cache_resolves_without_download(monkeypatch, tmp_path):
    from rewardbox.backends import vllm

    module_path = tmp_path / "rewardbox/backends/vllm.py"
    module_path.parent.mkdir(parents=True)
    checkpoint = tmp_path / ".hf/hub/models--org--model/snapshots/pinned"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    monkeypatch.setattr(vllm, "__file__", str(module_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    path, revision = vllm.resolve_checkpoint("org/model", "pinned")
    assert path == str(checkpoint) and revision == "pinned"


def test_vlac_reuses_native_jpeg_selection_signed_curves_and_time_scale(monkeypatch):
    from rewardbox.adapters import vlac_cut_vllm

    adapter = rewardbox.create_reward("vlac-cut-vllm", check_deps=False,
                                     frame_selection="as_given", sample_hz=5.0, curve_alignment="time")
    runtime = fake_runtime(adapter, ["时间: 0.0s, 进度: -20%\n时间: 1.0s, 进度: 120%"])
    observed = []

    def prepare(paths):
        observed.append([Path(path).name for path in paths])
        assert all(Path(path).exists() for path in paths)
        return np.zeros((4, 8, 12, 3), np.uint8), {
            "fps": 2.0, "frames_indices": [0, 1, 2, 3], "total_num_frames": 4,
            "do_sample_frames": False}

    monkeypatch.setattr(vlac_cut_vllm, "prepare_swift_video", prepare)
    out = adapter.score(traj(3))
    assert observed == [["frame_000003.jpg", "frame_000010.jpg", "frame_000017.jpg"]]
    assert out.raw["keypoint_times_sec"] == [0.0, 0.4]
    np.testing.assert_allclose(out.progress, [-20, 50, 120])
    assert out.frame_indices.tolist() == [3, 10, 17]
    assert out.meta["backend"] == "vllm" and out.meta["reference_backend"] == "swift"
    assert out.meta["normalized"] is False and out.cost.n_calls == 1
    request = runtime._llm.calls[0][0][0]
    assert request["mm_processor_kwargs"] == {"do_sample_frames": False, "do_resize": False}
    assert request["multi_modal_data"]["video"][0][1]["fps"] == 2.0


class Recorded(Exception):
    pass


def official_namespace():
    """Load only reference functions; never import/run the Flask model server."""
    names = {"compute_vlm_completion", "compute_vlm_comparison", "compute_vlm_reward",
             "extract_comparison_result", "extract_progress_score"}
    if not SERVER.is_file():
        pytest.skip("Large-Reward-Models upstream checkout missing (rewardbox fetch-upstream lrm)")
    tree = ast.parse(SERVER.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    for node in functions:
        node.decorator_list = []
    namespace = {"np": np, "Image": Image, "re": re, "json": json, "Optional": Optional,
                 "_activate_adapter": lambda head: None, "device": "cpu",
                 "torch": SimpleNamespace(no_grad=nullcontext, is_tensor=lambda value: True)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SERVER), "exec"), namespace)
    return namespace


def recorded_official(head, trajectory, *, initial=False, goal=False):
    class Tensor:
        def to(self, device):
            return self

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return Inputs(input_ids=Tensor()) if kwargs.get("tokenize") else "chat"

        def __call__(self, **kwargs):
            return Inputs(input_ids=Tensor())

    processor = Processor()
    namespace = official_namespace()

    def generate(**kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == (512 if head == "contrastive" else 128)
        raise Recorded

    namespace.update(vlm_processor=processor, vlm_model=SimpleNamespace(generate=generate))
    with pytest.raises(Recorded):
        if head == "completion":
            namespace["compute_vlm_completion"](trajectory.primary[-1], trajectory.instruction)
        elif head == "contrastive":
            namespace["compute_vlm_comparison"](trajectory.primary[0], trajectory.primary[1],
                                                 trajectory.instruction)
        else:
            namespace["compute_vlm_reward"](
                trajectory.primary[-1], trajectory.instruction,
                initial_image=trajectory.primary[0] if initial else None,
                goal_image=trajectory.goal_image if goal else None)
    return processor.messages


@pytest.mark.parametrize("head,initial,goal", [("completion", False, False),
    ("contrastive", False, False), ("progress", False, False), ("progress", True, False),
    ("progress", False, True), ("progress", True, True)])
def test_lrm_messages_match_actual_official_function(head, initial, goal):
    from rewardbox.adapters.lrm_native_protocol import messages_for, progress_messages

    trajectory = traj(2, goal=True)
    pil = [Image.fromarray(frame) for frame in trajectory.primary]
    if head == "progress":
        actual = progress_messages(trajectory.instruction, pil[-1], initial=pil[0] if initial else None,
                                   goal=Image.fromarray(trajectory.goal_image) if goal else None)
    else:
        actual = messages_for(head, trajectory.instruction, pil if head == "contrastive" else [pil[-1]])
    expected = recorded_official(head, trajectory, initial=initial, goal=goal)
    assert len(actual[0]["content"]) == len(expected[0]["content"])
    for a, b in zip(actual[0]["content"], expected[0]["content"]):
        assert a["type"] == b["type"]
        if a["type"] == "text":
            assert a["text"] == b["text"]
        else:
            np.testing.assert_array_equal(np.asarray(a["image"]), np.asarray(b["image"]))


@pytest.mark.parametrize("text", ['{"completion_progress":0.7}', '{"completion_progress":"NA"}',
    '{"completion_progress":1.2}', "Progress: 0.3", "unparseable", "ImageA then ImageB",
    '{"more_complete_image":"ImageA"}', '{"more_complete_image":"ImageB"}'])
def test_progress_and_comparison_parsers_match_official(text):
    from rewardbox.adapters.lrm_native_protocol import parse_comparison, parse_progress

    original = official_namespace()
    assert parse_progress(text) == original["extract_progress_score"](text)
    assert parse_comparison(text) == original["extract_comparison_result"](text)


@pytest.mark.parametrize("n", [1, 3, 4])
def test_vlac_actual_cpu_preprocessing_matches_swift_constants(monkeypatch, tmp_path, n):
    import torch
    from qwen_vl_utils import vision_process
    from rewardbox.adapters.vlac_cut_vllm import prepare_swift_video

    paths = []
    rng = np.random.default_rng(0)
    for index in range(n):
        path = tmp_path / f"frame_{index}.jpg"
        shape = (900, 1400, 3) if n == 3 else (150, 180, 3)
        Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(path, quality=95)
        paths.append(str(path))
    for name, value in {"IMAGE_MAX_TOKEN_NUM": 256, "IMAGE_MIN_TOKEN_NUM": 4,
                        "VIDEO_MAX_TOKEN_NUM": 256, "VIDEO_MIN_TOKEN_NUM": 4}.items():
        monkeypatch.setattr(vision_process, name, value)
    (reference, metadata), _ = vision_process.fetch_video(
        {"video": paths, "sample_fps": 2.0}, image_patch_size=16,
        return_video_metadata=True, return_video_sample_fps=True)
    actual, actual_meta = prepare_swift_video(paths)
    np.testing.assert_array_equal(actual, reference.to(torch.uint8).permute(0, 2, 3, 1).numpy())
    assert actual_meta == {**metadata, "do_sample_frames": False}
    assert len(actual) == n + n % 2
