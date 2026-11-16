"""Flora contracts: pinned upstream helpers, causal prefixes, bounded batches, no hashing."""
from __future__ import annotations

import hashlib
import subprocess
import sys

import numpy as np
import pytest

import rewardbox
from rewardbox.adapters.flora import REPO, REVISION, ArrayCollage, FloraRuntime, prefix_ends, upstream
from rewardbox.backends.vllm import resolve_cached_repo
from rewardbox.core.enums import ImplementationKind
from rewardbox.core.types import SamplingSpec, Trajectory


def snapshot():
    try:
        return resolve_cached_repo(REPO, REVISION)
    except RuntimeError:
        pytest.skip("pinned Flora snapshot is not cached")


def trajectory(length=95, views=None, fps=30.):
    frames = views or np.random.default_rng(0).integers(0, 255, (length, 48, 64, 3), dtype=np.uint8)
    return Trajectory(frames=frames, instruction="  put the  cup on the tray ", fps=fps,
                             frame_indices=np.arange(length), sampling=SamplingSpec(strategy="all"))


class Runtime(FloraRuntime):
    """Real upstream request building; generation replaced by a fixed answer."""

    def __init__(self, model_dir, config):
        super().__init__(model_dir, config)
        self.requests, self.closed = [], False

    def prompt(self, instruction):
        return self.core.terminal_messages(instruction, False)

    def generate(self, requests):
        self.requests.append(requests)
        self.batch_count += 1
        return [{"score": 4, "expected_score": 3.5, "progress": .625, "prompt_tokens": 100,
                 "completion_tokens": 4} for _ in requests]

    def close(self):
        self.closed = True


def attach(adapter):
    adapter.runtime = Runtime(snapshot(), adapter.config)
    return adapter.runtime


def test_registration_and_creation_remain_lazy():
    code = """import sys,rewardbox
names = rewardbox.list_models(filter='flora*')
assert names == ['flora-rewarder-27b', 'flora-rewarder-27b-prefix'], names
for name in names:
    assert rewardbox.create_reward(name, check_deps=False).runtime is None
assert not {'torch','transformers','vllm'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
    assert rewardbox.get_spec("flora-rewarder-27b").implementation_kind is ImplementationKind.OFFICIAL_ADAPTER
    assert rewardbox.get_spec("flora-rewarder-27b-prefix").implementation_kind is ImplementationKind.REWARDBOX_EXTENSION


def test_prefix_grid_is_nearest_half_up_plus_last_frame():
    assert prefix_ends(1095, 30, 1.0)[:3] == [0, 30, 60]
    assert prefix_ends(1095, 30, 1.0)[-2:] == [1080, 1094]
    assert prefix_ends(1, 30, 1.0) == [0]
    assert prefix_ends(31, 30, 1.0) == [0, 30]
    assert prefix_ends(100, 25, 2.0) == [0, 13, 25, 38, 50, 63, 75, 88, 99]


def test_terminal_is_one_official_call_on_the_whole_trajectory(monkeypatch):
    adapter = rewardbox.create_reward("flora-rewarder-27b", check_deps=False)
    runtime = attach(adapter)
    monkeypatch.setattr(hashlib, "sha256", lambda *a: pytest.fail("content hashing is forbidden"))
    output = adapter.score(trajectory())
    (request,), = runtime.requests
    frames, metadata = request["multi_modal_data"]["video"]
    indices = runtime.core.terminal_frame_indices(95, 30.)
    assert len(indices) == 30 and metadata == runtime.core.video_metadata(indices, 30., 95)
    assert frames.shape[0] == len(indices) and frames.dtype == np.uint8
    assert request["mm_processor_kwargs"] == {"do_sample_frames": False}
    assert set(request["multi_modal_uuids"]) == {"video"}
    assert request["prompt"][1]["content"][0]["text"].startswith("Task: put the cup on the tray")
    assert output.frame_indices.tolist() == [94] and output.progress.tolist() == [.625]
    assert output.meta["prefix_curve_is_extension"] is False
    assert output.meta["generation"]["regex"] == "Score: [1-5]"
    adapter.close()
    assert runtime.closed


def test_prefix_requests_are_whole_episodes_of_each_prefix_in_bounded_batches():
    adapter = rewardbox.create_reward("flora-rewarder-27b-prefix", check_deps=False, max_num_seqs=2)
    runtime = attach(adapter)
    output = adapter.score(trajectory())
    ends = [0, 30, 60, 90, 94]
    assert output.frame_indices.tolist() == ends
    assert [len(batch) for batch in runtime.requests] == [2, 2, 1]
    assert output.meta["n_backend_calls"] == 3 and output.meta["n_inference_requests"] == 5
    for end, request in zip(ends, [request for batch in runtime.requests for request in batch]):
        _, metadata = request["multi_modal_data"]["video"]
        assert metadata["total_num_frames"] == end + 1
        assert metadata["frames_indices"] == runtime.core.terminal_frame_indices(end + 1, 30.)
        assert max(metadata["frames_indices"]) == end
    assert [point["end_frame"] for point in output.raw["points"]] == ends
    assert output.meta["prefix_curve_is_extension"] is True


def test_collage_matches_upstream_layout_and_resize():
    core, _ = upstream(snapshot())
    rng = np.random.default_rng(1)
    views = [rng.integers(0, 255, (6, 480, 640, 3), dtype=np.uint8) for _ in range(3)]
    collage = ArrayCollage(core, views, 30.)
    rectangles, width, height = core.plan_collage(((640, 480),) * 3)
    assert (collage.rectangles, collage.width, collage.height) == (rectangles, width, height)
    from PIL import Image

    x, y, w, h = rectangles[1]
    expected = np.asarray(Image.fromarray(views[1][4]).resize((w, h), resample=Image.Resampling.BILINEAR))
    assert np.array_equal(collage.frames([4])[0][y:y + h, x:x + w], expected)


def test_multiview_order_and_missing_views():
    rng = np.random.default_rng(2)
    views = {name: rng.integers(0, 255, (40, 48, 64, 3), dtype=np.uint8)
             for name in ("primary", "left_wrist", "right_wrist")}
    adapter = rewardbox.create_reward("flora-rewarder-27b", check_deps=False,
                                      view_order=["left_wrist", "primary", "right_wrist"])
    attach(adapter)
    output = adapter.score(trajectory(40, views))
    assert output.meta["collage"]["views"] == ["left_wrist", "primary", "right_wrist"]
    assert len(output.meta["collage"]["rectangles"]) == 3
    adapter = rewardbox.create_reward("flora-rewarder-27b", check_deps=False, view_order=["top"])
    attach(adapter)
    with pytest.raises(ValueError, match="lacks"):
        adapter.score(trajectory(40, views))


@pytest.mark.parametrize("options", [{"max_num_seqs": 0}, {"max_num_seqs": True},
                                     {"gpu_memory_utilization": 1.5}, {"prefix_fps": 0},
                                     {"view_order": []}, {"view_order": ["a", "a"]},
                                     {"with_rationale": "yes"}, {"seed": -1}])
def test_bad_options_fail_before_model_loading(options):
    with pytest.raises(ValueError):
        rewardbox.create_reward("flora-rewarder-27b", check_deps=False, **options)


def test_sparse_or_untimed_input_is_rejected():
    adapter = rewardbox.create_reward("flora-rewarder-27b", check_deps=False)
    traj = trajectory()
    traj.frame_indices = np.arange(len(traj)) * 2
    with pytest.raises(ValueError, match="every source frame"):
        adapter.score(traj)
    traj = trajectory()
    traj.fps = None
    with pytest.raises(ValueError, match="fps"):
        adapter.score(traj)
