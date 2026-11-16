"""Regression tests: adapter outputs must match the verified results in calibration/records/.

This is the only meaningful correctness proof for the adapter layer; right shapes and no
NaN prove nothing about faithful reproduction.

How to run (needs a GPU + downloaded weights):
    HF_HOME=$PWD/.hf \\
        PYTHONPATH=$PWD/reward_model_code/robometer:$PWD/reward_model_code/TOPReward \\
        REWARDBOX_ROBODOPAMINE_ROOT=$PWD/reward_model_code/Robo-Dopamine \\
        .venv/bin/python -m pytest tests/test_calibration.py -m calibration -v

In an ordinary development environment these tests skip without weights/dependencies;
``test_records_are_wellformed`` always runs.  **The formal acceptance gate** must set
``REWARDBOX_REQUIRE_MODELS=1``: model creation failures then ``pytest.fail()`` instead of
skipping, so "all green because packages/weights were missing, nothing actually verified"
never slips into a formal acceptance record.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from rewardbox.core.types import SamplingSpec
from rewardbox.registry import _CALIBRATION_RECORDS as RECORDS  # noqa: E402  (packaged records)
VIDEO = Path(__file__).resolve().parent.parent / "calibration" / "synthetic_pick.mp4"
VIDEO_FAIL = Path(__file__).resolve().parent.parent / "calibration" / "synthetic_pick_fail.mp4"
COMPARE_RECORDS = RECORDS / "compare"

# Per-model tolerances.  VLM sampling is not strictly deterministic: discrete scores must match exactly, continuous values get a small tolerance.
TOL = {
    "roboreward-4b": {"atol": 1e-9, "rtol": 0.0},  # discrete 1-5 -> must match bit for bit
    "robometer-4b": {"atol": 2e-2, "rtol": 0.0},   # C51 continuous values, bf16 autocast jitter
    # robodopamine-8b is not listed: the official SamplingParams(temperature=0.1, ...) does
    # sample rather than decode greedily, but vLLM LLM() defaults to seed=0 (not overridden by
    # GRMInference), and measurements reproduce bit for bit at the default atol=1e-3.
    "vlac-2b": {"atol": 1.5, "rtol": 0.0},
    "vlac-8b": {"atol": 1.5, "rtol": 0.0},
    # temperature=0 removes sampling randomness, but the pinned vLLM 0.12 /
    # bf16 server still produced one observed 1-point parser-output change on
    # the synthetic fixture ([0,5,9,13] once, then [0,5,8,13] in six runs).
    # Keep the tolerance at exactly that observed integer boundary.
    "sole-r1-8b": {"atol": 1.0, "rtol": 0.0},
    # top_k=1 should be bit-exact in theory, but on 2026-07-26
    # test_vlac_official_example_compare_direction_is_correct occasionally moved by ±1
    # (e.g. -61 vs the recorded -62); suspected bf16/fp16 numeric noise on a borderline logit
    # flipping the top-1 digit token, not the model drifting by an order of magnitude.
    # vlac-8b uses the same top_k=1 decoding and the same noise risk, so it reuses this tolerance.
}

# Measured 2026-07-26: for vlac-2b (a 2B model) the synthetic red/green block videos in
# calibration/synthetic_pick*.mp4 are out-of-distribution input: score()'s per-frame critics
# are almost all "+0"/"-0" (signal near 0) and compare() points against the intuition
# "success > fail".  The first/last frames of both videos were exported and their pixels
# checked (no frame-extraction/channel bug), and re-running VLAC's own real robot-arm example
# images (with/without a reference trajectory) gives correct, monotone critic/value, which
# shows the adapter call and the environment are fine and the model lacks a reliable prior
# for such synthetic frames.  Evidence of correctness on real data is in
# calibration/records/official_examples/ (vlac-2b.json and compare/vlac-2b.json).  Only the
# "output is reproducible" regression is kept here, with no direction assertion for vlac-2b.
DIRECTION_ASSERTION_EXEMPT = {
    "vlac-2b": (
        "the 2B model sees RewardBox's synthetic fixture as out-of-distribution input, so the "
        "compare() direction is unreliable; correctness evidence is in calibration/records/official_examples/"
    ),
}


def _records():
    return sorted(RECORDS.glob("*.json")) if RECORDS.exists() else []


def _require_or_skip(name: str, exc: Exception) -> None:
    """Ordinary dev machines may skip; the formal REWARDBOX_REQUIRE_MODELS=1 gate must fail."""
    if os.environ.get("REWARDBOX_REQUIRE_MODELS") == "1":
        pytest.fail(f"REWARDBOX_REQUIRE_MODELS=1: cannot create {name}; strict mode must fail "
                    f"here instead of skipping silently: {exc}")
    pytest.skip(f"cannot create {name}: {exc}")


def test_records_are_wellformed():
    """Records must carry all metadata needed to reproduce them."""
    recs = _records()
    assert recs, f"no calibration records: {RECORDS}"
    for p in recs:
        d = json.loads(p.read_text())
        for k in ("model", "video", "instruction", "sampling", "environment", "output"):
            assert k in d, f"{p.name} lacks field {k}"
        for k in ("python", "torch", "transformers", "gpu"):
            assert k in d["environment"], f"{p.name} environment lacks {k}"
        assert isinstance(d.get("model_opts", {}), dict), (
            f"{p.name} model_opts must be an object")
        assert d["sampling"]["key"] == SamplingSpec.from_dict(d["sampling"]).key()
        assert not any(k.startswith("trajectory_hash") for k in d)
        assert len(d["output"]["progress"]) == len(d["output"]["frame_indices"])


@pytest.mark.calibration
@pytest.mark.gpu
@pytest.mark.parametrize("record", _records(), ids=lambda p: p.stem)
def test_matches_recorded_output(record):
    import rewardbox
    from rewardbox.core.types import SamplingSpec, Trajectory

    d = json.loads(record.read_text())
    name = d["model"]
    spec = SamplingSpec.from_dict(d["sampling"])
    traj = Trajectory.from_video(VIDEO, d["instruction"], spec)


    if name == "sole-r1-8b" and os.environ.get(
            "REWARDBOX_SOLE_R1_CALIBRATION") != "1":
        message = (
            "sole-r1-8b calibration needs the pinned separate service running and an explicit "
            "REWARDBOX_SOLE_R1_CALIBRATION=1")
        if os.environ.get("REWARDBOX_REQUIRE_MODELS") == "1":
            pytest.fail(message)
        pytest.skip(message)

    kw = dict(d.get("model_opts", {}))
    if name.startswith("roboreward"):
        kw.setdefault("mode", "prefix")
    try:
        rm = rewardbox.create_reward(name, **kw)
    except Exception as e:
        _require_or_skip(name, e)
    try:
        out = rm.score(traj)
    finally:
        rm.close()

    tol = TOL.get(name, {"atol": 1e-3, "rtol": 0.0})
    exp = np.asarray(d["output"]["progress"], dtype=float)
    assert len(out.progress) == len(exp), f"length changed: {len(out.progress)} vs {len(exp)}"
    assert np.allclose(out.progress, exp, **tol), (
        f"{name} progress deviates from the record:\n  now    {np.round(out.progress, 4).tolist()}\n"
        f"  record {np.round(exp, 4).tolist()}\n  tolerance {tol}")

    if d["output"].get("success") is not None:
        assert out.success is not None, "the record has success, the output does not"
        assert np.allclose(out.success, np.asarray(d["output"]["success"], float), **tol)


def _compare_records():
    return sorted(COMPARE_RECORDS.glob("*.json")) if COMPARE_RECORDS.exists() else []


@pytest.mark.calibration
@pytest.mark.gpu
@pytest.mark.parametrize("record", _compare_records(), ids=lambda p: p.stem)
def test_compare_matches_recorded_output(record):
    """compare()'s preference direction and value must be reproducible (see calibration/records/compare/)."""
    import rewardbox
    from rewardbox.core.types import SamplingSpec, Trajectory

    d = json.loads(record.read_text())
    name = d["model"]
    spec = SamplingSpec.from_dict(d["sampling"])
    a = Trajectory.from_video(VIDEO, d["instruction"], spec)
    b = Trajectory.from_video(VIDEO_FAIL, d["instruction"], spec)

    try:
        rm = rewardbox.create_reward(name)
    except Exception as e:
        _require_or_skip(name, e)
    try:
        compare_ab = rm.compare(a, b)
        compare_ba = rm.compare(b, a)
    finally:
        rm.close()

    tol = TOL.get(name, {"atol": 1e-3, "rtol": 0.0})
    if name not in DIRECTION_ASSERTION_EXEMPT:
        assert compare_ab > 0, f"{name}.compare(success, failure) should be >0, got {compare_ab}"
        assert compare_ba < 0, f"{name}.compare(failure, success) should be <0, got {compare_ba}"
    assert np.allclose(compare_ab, d["output"]["compare_a_vs_b"], **tol)
    assert np.allclose(compare_ba, d["output"]["compare_b_vs_a"], **tol)


OFFICIAL_EXAMPLES = RECORDS / "official_examples"
def _vlac_root() -> Path:
    from rewardbox.upstream import find_upstream

    return find_upstream("vlac") or Path(
        os.environ.get("REWARDBOX_VLAC_ROOT")
        or (Path(__file__).resolve().parent.parent / "reward_model_code" / "VLAC"))


VLAC_ROOT = _vlac_root()


def _load_official_example_frames(image_names: list[str]) -> np.ndarray:
    from PIL import Image as PILImage

    images_dir = VLAC_ROOT / "evo_vlac" / "examples" / "images" / "test"
    return np.stack([
        np.asarray(PILImage.open(images_dir / name).convert("RGB"), dtype=np.uint8)
        for name in image_names
    ])


def _official_example_records():
    return sorted(OFFICIAL_EXAMPLES.glob("*.json")) if OFFICIAL_EXAMPLES.exists() else []


def _official_example_compare_records():
    d = OFFICIAL_EXAMPLES / "compare"
    return sorted(d.glob("*.json")) if d.exists() else []


@pytest.mark.calibration
@pytest.mark.gpu
@pytest.mark.parametrize("record", _official_example_records(), ids=lambda p: p.stem)
def test_vlac_official_example_score_matches_recorded_output(record):
    """vlac-2b's out-of-distribution behaviour on the synthetic fixture (see
    DIRECTION_ASSERTION_EXEMPT) needs in-distribution counter-evidence: the real robot-arm
    example images shipped with the official VLAC repository.  vlac-8b's compare() direction
    on the synthetic fixture is already right and needs no such evidence, but it runs too as
    an in-distribution consistency record."""
    import rewardbox
    from rewardbox.core.types import SamplingSpec, Trajectory

    if not (VLAC_ROOT / "evo_vlac" / "utils" / "model_utils.py").exists():
        pytest.skip(f"official VLAC repository not found: {VLAC_ROOT}")

    d = json.loads(record.read_text())
    name = d["model"]
    spec = SamplingSpec.from_dict(d["sampling"])
    frames = _load_official_example_frames(d["images"])
    traj = Trajectory(frames=frames, instruction=d["instruction"],
                      frame_indices=np.arange(len(frames), dtype=np.int64), sampling=spec)


    try:
        rm = rewardbox.create_reward(name)
    except Exception as e:
        _require_or_skip(name, e)
    try:
        out = rm.score(traj)
    finally:
        rm.close()

    tol = TOL.get(name, {"atol": 1e-3, "rtol": 0.0})
    exp = np.asarray(d["output"]["progress"], dtype=float)
    assert np.allclose(out.progress, exp, **tol), (
        f"{name} progress on the official real example images deviates from the record:\n"
        f"  now    {np.round(out.progress, 4).tolist()}\n  record {np.round(exp, 4).tolist()}")
    # This is a "correctness" record (in-distribution data), so the direction assertion is
    # enforced here, the opposite of the synthetic-fixture exemption, to show that vlac-2b
    # itself is not broken and the synthetic frames are just out of distribution for it.
    assert np.all(np.diff(out.progress) >= 0), (
        f"the official real example images should be monotone non-decreasing, got {out.progress.tolist()}")


@pytest.mark.calibration
@pytest.mark.gpu
@pytest.mark.parametrize("record", _official_example_compare_records(), ids=lambda p: p.stem)
def test_vlac_official_example_compare_direction_is_correct(record):
    import rewardbox
    from rewardbox.core.types import SamplingSpec, Trajectory

    if not (VLAC_ROOT / "evo_vlac" / "utils" / "model_utils.py").exists():
        pytest.skip(f"official VLAC repository not found: {VLAC_ROOT}")

    d = json.loads(record.read_text())
    name = d["model"]
    spec = SamplingSpec.from_dict(d["sampling"])
    late = Trajectory(frames=_load_official_example_frames([d["image_late"]]),
                      instruction=d["instruction"],
                      frame_indices=np.array([0], dtype=np.int64), sampling=spec)
    early = Trajectory(frames=_load_official_example_frames([d["image_early"]]),
                       instruction=d["instruction"],
                       frame_indices=np.array([0], dtype=np.int64), sampling=spec)

    try:
        rm = rewardbox.create_reward(name)
    except Exception as e:
        _require_or_skip(name, e)
    try:
        compare_late_early = rm.compare(late, early)
        compare_early_late = rm.compare(early, late)
    finally:
        rm.close()

    tol = TOL.get(name, {"atol": 1e-3, "rtol": 0.0})
    assert compare_late_early > 0, (
        f"{name}.compare(late, early) should be >0 on real images, got {compare_late_early}")
    assert compare_early_late < 0, (
        f"{name}.compare(early, late) should be <0 on real images, got {compare_early_late}")
    assert np.allclose(compare_late_early, d["output"]["compare_late_vs_early"], **tol)
    assert np.allclose(compare_early_late, d["output"]["compare_early_vs_late"], **tol)
