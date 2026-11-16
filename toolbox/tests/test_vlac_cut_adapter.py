"""Offline contracts for the VLAC-Cut adapter.

The model call is faked; everything else (frame selection, prompt, response
parsing, both curve alignments) runs through the pinned upstream files, which
must be checked out (``rewardbox fetch-upstream vlac-cut``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rewardbox.core.types import SamplingSpec, Trajectory

ROOT = Path(__file__).resolve().parents[1]
from rewardbox.upstream import find_upstream  # noqa: E402

UPSTREAM = find_upstream("vlac-cut") or ROOT / "reward_model_code" / "VLAC-cut"
pytestmark = pytest.mark.skipif(
    not (UPSTREAM / "scripts" / "run_example.py").exists(),
    reason="VLAC-Cut upstream checkout missing",
)


def _reference(name: str) -> dict:
    path = UPSTREAM / "examples" / "reference_outputs" / f"{name}.jsonl"
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def _dense(n: int, fps: float = 30.0, **meta) -> Trajectory:
    frames = np.stack([np.full((6, 8, 3), i % 256, np.uint8) for i in range(n)])
    return Trajectory(
        frames=frames, instruction="pick the cube", fps=fps,
        sampling=SamplingSpec(strategy="all"), meta=meta)


def _adapter(**kw):
    from rewardbox.adapters.vlac_cut import VLACCutAdapter
    from rewardbox.registry import get_spec

    return VLACCutAdapter(get_spec("vlac-cut"), check_deps=False, **kw)


def _fake(adapter, response: str):
    calls = []

    def generate(paths, prompt):
        calls.append({"paths": list(paths), "prompt": prompt,
                      "exist": all(Path(p).exists() for p in paths)})
        return response

    adapter._generate = generate
    return calls


@pytest.mark.parametrize(
    "kw, match",
    (
        ({"backend": "vllm"}, "backend"),
        ({"frame_selection": "uniform"}, "frame_selection"),
        ({"curve_alignment": "minmax"}, "curve_alignment"),
        ({"sample_hz": 0}, "sample_hz"),
        ({"sample_hz": True}, "sample_hz"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"task_plan_key": " "}, "task_plan_key"),
        ({"backend": "transformers", "curve_alignment": "time", "sample_hz": 5.0},
         "sample_hz != 2.0"),
    ),
)
def test_invalid_options_rejected(kw, match):
    with pytest.raises(ValueError, match=match):
        _adapter(**kw)


def test_reference_example_01_contract():
    """The official example_01 reference output: frame selection, prompt and keypoint parsing reproduced item by item."""
    ref = _reference("example_01")
    n = int(ref["decoded_video_stats"]["decoded_frame_count"])
    traj = _dense(n, fps=ref["decoded_video_stats"]["decoded_avg_fps"],
                  task_plan=ref["task_plan"])
    traj = Trajectory(frames=traj.frames, instruction=ref["task_instruction"],
                      fps=traj.fps, sampling=traj.sampling, meta=traj.meta)
    adapter = _adapter()
    calls = _fake(adapter, ref["response"])

    out = adapter.score(traj)

    assert out.frame_indices.tolist() == ref["input_frame_indices"]
    assert out.raw["input_timestamps_sec"] == ref["input_timestamps_sec"]
    assert calls[0]["prompt"] == ref["prompt"]
    assert calls[0]["exist"] and len(calls[0]["paths"]) == ref["input_frame_count"]
    assert Path(calls[0]["paths"][1]).name == "frame_000015.jpg"
    assert out.raw["keypoint_times_sec"] == ref["pred_curve_point_times_sec"]
    assert out.raw["keypoint_progress"] == ref["pred_curve_point_progress"]
    assert out.meta["prompt_source"] == ref["prompt_source"]
    # default vpb_index: the first/last keypoints stretch to the first/last input frames, raw scale not clamped.
    assert out.progress[0] == ref["pred_curve_point_progress"][0]
    assert out.progress[-1] == ref["pred_curve_point_progress"][-1]
    assert out.progress.tolist() == out.raw["progress_vpb_index"]
    assert out.progress.min() == pytest.approx(-25.0, abs=5.0)
    assert out.meta["normalized"] is False


def test_time_alignment_interpolates_on_input_timestamps():
    adapter = _adapter(curve_alignment="time")
    _fake(adapter, "时间: 1.0s, 进度: 0%\n时间: 2.0s, 进度: 50%\n时间: 3.0s, 进度: -10%")
    out = adapter.score(_dense(121))   # 0..4 s @30fps → 2 Hz: 0.0,0.5,...,4.0

    assert out.frame_indices.tolist() == list(range(0, 121, 15))
    np.testing.assert_allclose(
        out.progress, [0, 0, 0, 25, 50, 20, -10, -10, -10], atol=1e-5)
    assert out.progress.tolist() == out.raw["progress_time"]
    # the other official alignment is still kept.
    assert out.raw["progress_vpb_index"][0] == 0.0
    assert out.raw["progress_vpb_index"][-1] == -10.0


def test_non_integer_fps_uses_upstream_nearest_frame_grid():
    adapter = _adapter()
    calls = _fake(adapter, "时间: 0.0s, 进度: 0%\n时间: 1.0s, 进度: 100%")
    out = adapter.score(_dense(60, fps=25.0))

    assert out.frame_indices.tolist() == [0, 12, 25, 37, 50]
    assert len(calls[0]["paths"]) == 5


def test_as_given_uses_frames_verbatim_with_nominal_timestamps():
    frames = np.zeros((4, 6, 8, 3), np.uint8)
    traj = Trajectory(frames=frames, instruction="pick", frame_indices=[3, 9, 20, 41],
                      sampling=SamplingSpec(num_frames=4))
    adapter = _adapter(frame_selection="as_given", curve_alignment="time")
    calls = _fake(adapter, "时间: 0.0s, 进度: 0%\n时间: 1.5s, 进度: 30%")
    out = adapter.score(traj)

    assert out.frame_indices.tolist() == [3, 9, 20, 41]
    assert out.raw["input_timestamps_sec"] == [0.0, 0.5, 1.0, 1.5]
    np.testing.assert_allclose(out.progress, [0, 10, 20, 30], atol=1e-5)
    assert [Path(p).name for p in calls[0]["paths"]] == [
        "frame_000003.jpg", "frame_000009.jpg", "frame_000020.jpg", "frame_000041.jpg"]


def test_official_selection_rejects_sampled_or_fpsless_trajectories():
    adapter = _adapter()
    _fake(adapter, "时间: 0.0s, 进度: 0%")
    sampled = Trajectory(frames=np.zeros((3, 6, 8, 3), np.uint8), instruction="pick",
                         frame_indices=[0, 10, 20], fps=30.0)
    with pytest.raises(ValueError, match="as_given"):
        adapter.score(sampled)
    no_fps = Trajectory(frames=np.zeros((3, 6, 8, 3), np.uint8), instruction="pick")
    with pytest.raises(ValueError, match="fps"):
        adapter.score(no_fps)


def test_prompt_override_from_meta():
    adapter = _adapter()
    calls = _fake(adapter, "时间: 0.0s, 进度: 0%")
    out = adapter.score(_dense(31, vlac_cut_prompt="  custom prompt  "))
    assert calls[0]["prompt"] == "custom prompt"
    assert out.meta["prompt_source"] == "raw_prompt"


def test_unparseable_response_fails_closed():
    adapter = _adapter()
    _fake(adapter, "I cannot tell.")
    with pytest.raises(RuntimeError, match="no keypoint could be parsed"):
        adapter.score(_dense(31))


def test_dependency_error_points_to_isolated_environment():
    import rewardbox

    spec = rewardbox.get_spec("vlac-cut")
    message = spec.requires.check().format(spec.name)
    assert ".venv-vlac-cut" in message
