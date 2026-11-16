"""PRIMO contracts: native prompts/sampling, sparse grids and bounded batching."""
from __future__ import annotations

import hashlib
import subprocess
import sys
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import rewardbox
from rewardbox.adapters.primo import PRIMORuntime, QUESTION, parse_percentage, prefix_ends
from rewardbox.backends.vllm import GeneratedText
from rewardbox.core.types import SamplingSpec, Trajectory


def trajectory(length=10):
    return Trajectory(frames=np.zeros((length, 28, 28, 3), dtype=np.uint8),
                             instruction="put the block in the bowl", fps=30.,
                             frame_indices=np.arange(length), sampling=SamplingSpec(strategy="all"))


class Runtime:
    def __init__(self):
        self.model_path = "pinned-local-checkpoint"
        self.prepared = []
        self.calls = []
        self.batch_count = 0
        self.closed = False

    def prepare(self, traj, end):
        self.prepared.append(end)
        return {"end": end}, {"anchor_frame_indices": [0, end]}

    def generate(self, requests):
        self.calls.append(requests)
        self.batch_count += 1
        return [GeneratedText("<think>inspection</think><answer>42.5</answer>", "stop", 400, 20)
                for request in requests]

    def close(self):
        self.closed = True


def test_registration_and_creation_remain_lazy():
    code = """import sys,rewardbox
for name in rewardbox.list_models(filter='primo*'):
    a=rewardbox.create_reward(name,check_deps=False)
    assert a.runtime is None
assert not {'torch','transformers','vllm'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("text,value,native", [
    ("<answer>42.5</answer>", 42.5, 42.5),
    ("<answer>42.5%</answer>", 42.5, .425),
    ("<answer>≈ 42.5％</answer>", 42.5, .425),
])
def test_answer_percentage_units_and_upstream_normalizer_are_separate(text, value, native):
    actual, _, normalized = parse_percentage(text)
    assert actual == value and normalized == native


@pytest.mark.parametrize("text", ["42.5", "<answer>garbled</answer>", "<answer>nan</answer>",
                                 "<answer>101</answer>", "<answer>-1</answer>"])
def test_invalid_answer_never_becomes_a_zero_score(text):
    with pytest.raises(ValueError):
        parse_percentage(text)


def test_terminal_score_uses_only_endpoint_without_hashing(monkeypatch):
    adapter = rewardbox.create_reward("primo-r1-7b", check_deps=False)
    runtime = Runtime()
    adapter.runtime = runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("content hashing is forbidden")

    monkeypatch.setattr(hashlib, "sha256", forbidden)
    output = adapter.score(trajectory())
    assert runtime.prepared == [9]
    assert output.frame_indices.tolist() == [9]
    assert output.progress.tolist() == [42.5]
    assert output.meta["generation"] == {"temperature": .1, "top_p": .001,
                                          "max_tokens": 4096, "stop_token_ids": []}
    assert output.meta["native_video_frame_cap"] == 22
    adapter.close()
    assert runtime.closed


def test_prefix_profile_has_distinct_identity_bounded_batches_and_no_interpolation():
    adapter = rewardbox.create_reward("primo-r1-7b-prefix", check_deps=False,
                                      batch_size=2, num_prefixes=8)
    runtime = Runtime()
    adapter.runtime = runtime
    output = adapter.score(trajectory())
    expected = prefix_ends(10, 8)
    assert output.frame_indices.tolist() == expected == runtime.prepared
    assert len(output.progress) < 10
    assert all(len(batch) <= 2 for batch in runtime.calls)
    assert output.meta["n_backend_calls"] == len(runtime.calls)
    assert output.meta["prefix_curve_is_extension"] is True
    assert 2 not in expected and expected[-1] == 9


@pytest.mark.parametrize("options", [{"batch_size": True}, {"batch_size": 0},
                                     {"gpu_memory_utilization": 2}, {"num_prefixes": -1},
                                     {"question_template": "no task binding"}])
def test_bad_options_fail_before_model_loading(options):
    with pytest.raises(ValueError):
        rewardbox.create_reward("primo-r1-7b", check_deps=False, **options)


def test_sparse_input_and_invalid_native_video_length_are_rejected():
    adapter = rewardbox.create_reward("primo-r1-7b", check_deps=False)
    traj = trajectory()
    traj.frame_indices = np.arange(len(traj)) * 2
    with pytest.raises(ValueError, match="every source frame"):
        adapter.score(traj)
    with pytest.raises(ValueError):
        prefix_ends(3, 16)


def native_frame_functions():
    """Exercise the pinned reader's real rounding without loading a checkpoint."""
    from rewardbox.upstream import find_upstream

    root = find_upstream("primo") or Path(__file__).resolve().parents[1] / "reward_model_code/PRIMO-R1"
    source = root / "src/qwen-vl-utils/src/qwen_vl_utils/vision_process.py"
    if not source.is_file():
        pytest.skip("pinned PRIMO source is required for native-reader regression checks")
    pytest.importorskip("torch")
    names = {"round_by_factor", "floor_by_factor", "ceil_by_factor", "smart_nframes"}
    tree = ast.parse(source.read_text())
    functions = ast.Module(body=[node for node in tree.body
                                if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {"math": math, "FRAME_FACTOR": 2}
    exec(compile(functions, str(source), "exec"), namespace)
    return namespace


def prepare_without_checkpoint(length, policy):
    native = native_frame_functions()
    runtime = PRIMORuntime("unused", {"nframes": 32, "question_template": QUESTION,
                                      "short_prefix_policy": policy})
    runtime.processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "native prompt")
    runtime.prompts = SimpleNamespace(SYSTEM_PROMPT="native system", build_question=lambda q, kind: q)
    runtime.vision = SimpleNamespace(
        **{name: native[name] for name in ("smart_nframes", "round_by_factor", "floor_by_factor")},
        FRAME_FACTOR=2, VIDEO_READER_BACKENDS={}, get_video_reader_backend=lambda: "original",
        process_vision_info=lambda *args, **kwargs: ([], [None], {"fps": [30.]}))
    return runtime.prepare(trajectory(length), length - 1)[1]


@pytest.mark.parametrize("length,frames", [(7, 6), (11, 10), (15, 14), (19, 18)])
def test_explicit_prefix_recovery_repairs_native_rounding_overflow(length, frames):
    with pytest.raises(ValueError, match="nframes should in interval"):
        prepare_without_checkpoint(length, "native")
    record = prepare_without_checkpoint(length, "floor-on-overflow-v1")
    assert len(record["video_frame_indices"]) == frames
    assert record["video_frame_indices"][0] == 0
    assert record["video_frame_indices"][-1] == length - 1
    assert record["anchor_frame_indices"] == [0, length - 1]
    assert record["short_prefix_repaired"] is True


@pytest.mark.parametrize("length", [2, 4, 5, 6, 8, 9, 10, 12, 13, 16, 17, 20, 21, 22, 30, 41])
def test_prefix_recovery_keeps_every_previously_valid_native_request(length):
    assert prepare_without_checkpoint(length, "native") == prepare_without_checkpoint(length, "floor-on-overflow-v1")


def test_short_prefix_repair_cannot_silently_change_terminal_protocol():
    with pytest.raises(ValueError, match="only to the explicit prefix extension"):
        rewardbox.create_reward("primo-r1-7b", check_deps=False,
                                short_prefix_policy="floor-on-overflow-v1")
