"""Core-layer tests.  None needs a GPU or weights."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import rewardbox
from rewardbox.core.types import RewardOutput, SamplingSpec, Trajectory

ROOT = Path(rewardbox.__file__).parent


def frames(t=8, h=16, w=16):
    return np.stack([np.full((h, w, 3), min(255, i * 20), np.uint8) for i in range(t)])


# ---------------------------------------------------------------- SamplingSpec
class TestSamplingSpec:
    def test_uniform_basic(self):
        idx = SamplingSpec(num_frames=5).indices(100)
        assert len(idx) == 5 and idx[0] == 0 and idx[-1] == 99
        assert np.all(np.diff(idx) > 0)

    def test_uniform_floor_matches_sole_r1_rewardgen_reference_grid(self):
        rounded = SamplingSpec(num_frames=10).indices(25)
        floored = SamplingSpec(
            num_frames=10, strategy="uniform_floor",
        ).indices(25)
        assert rounded.tolist() == [0, 3, 5, 8, 11, 13, 16, 19, 21, 24]
        assert floored.tolist() == [0, 2, 5, 8, 10, 13, 16, 18, 21, 24]

    def test_num_frames_one_returns_one(self):
        """Regression: include_last used to append the last frame and return 2."""
        for total in (1, 2, 10, 100):
            assert len(SamplingSpec(num_frames=1).indices(total)) == 1

    @pytest.mark.parametrize("total", range(1, 40))
    @pytest.mark.parametrize("n", [1, 2, 3, 7, 16])
    def test_never_exceeds_and_strictly_increasing(self, total, n):
        idx = SamplingSpec(num_frames=n).indices(total)
        assert len(idx) <= n, f"total={total} n={n} → {idx}"
        assert len(idx) >= 1
        if len(idx) > 1:
            assert np.all(np.diff(idx) > 0)
        assert idx[-1] <= total - 1

    def test_fewer_frames_than_requested(self):
        assert SamplingSpec(num_frames=10).indices(3).tolist() == [0, 1, 2]

    def test_interval_strategy_matches_upstream(self):
        """Reproduces Robo-Dopamine's make_sample_indices_by_interval."""
        idx = SamplingSpec(strategy="interval", interval=10).indices(35)
        assert idx.tolist() == [0, 10, 20, 30, 34]

    def test_all_strategy(self):
        assert SamplingSpec(strategy="all").indices(5).tolist() == [0, 1, 2, 3, 4]

    def test_all_strategy_can_exceed_num_frames(self):
        """all/fps/interval are native protocols not capped by num_frames (the docstring once
        wrongly claimed every strategy returns "at most num_frames")."""
        idx = SamplingSpec(strategy="all", num_frames=4).indices(10)
        assert len(idx) == 10

    def test_interval_strategy_can_exceed_num_frames(self):
        idx = SamplingSpec(strategy="interval", interval=2, num_frames=2).indices(21)
        assert len(idx) > 2

    def test_zero_frames_raises(self):
        with pytest.raises(ValueError):
            SamplingSpec().indices(0)

    def test_key_stable_and_discriminating(self):
        a, b, c = (SamplingSpec(num_frames=10), SamplingSpec(num_frames=10),
                   SamplingSpec(num_frames=16))
        assert a.key() == b.key() != c.key()

    def test_roundtrip(self):
        s = SamplingSpec(num_frames=8, resize=(480, 270))
        assert SamplingSpec.from_dict(s.to_dict()) == s


# ---------------------------------------------------------------- Trajectory
class TestTrajectory:
    def test_dtype_validated(self):
        with pytest.raises(ValueError, match="uint8"):
            Trajectory(frames=frames().astype(np.float32), instruction="x")

    def test_shape_validated(self):
        with pytest.raises(ValueError, match=r"\[T,H,W,3\]"):
            Trajectory(frames=np.zeros((4, 16, 16), np.uint8), instruction="x")

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="has 0 frames"):
            Trajectory(frames=np.zeros((0, 8, 8, 3), np.uint8), instruction="x")

    def test_all_views_validated(self):
        """Regression: only primary used to be validated."""
        with pytest.raises(ValueError, match="the other views have"):
            Trajectory(frames={"primary": frames(8), "left_wrist": frames(5)},
                       instruction="x")
        with pytest.raises(ValueError, match="uint8"):
            Trajectory(frames={"primary": frames(4),
                               "left_wrist": frames(4).astype(np.float32)},
                       instruction="x")

    def test_trajectory_has_no_content_hash(self):
        """Inputs are trusted fixed files; trajectories are never content-hashed."""
        assert not hasattr(Trajectory, "content_hash")

    def test_view_accessors(self):
        t = Trajectory(frames={"primary": frames(), "left_wrist": frames()},
                       instruction="x")
        assert len(t) == 8 and set(t.views) == {"primary", "left_wrist"}
        with pytest.raises(KeyError):
            t.view("nope")


# ---------------------------------------------------------------- RewardOutput
class TestRewardOutput:
    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            RewardOutput(progress=[1, 2, 3], frame_indices=[0, 1])

    def test_non_increasing_indices(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            RewardOutput(progress=[1, 2, 3], frame_indices=[0, 2, 2])

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            RewardOutput(progress=[], frame_indices=[])

    def test_single_point_ok(self):
        assert len(RewardOutput(progress=[0.5], frame_indices=[3])) == 1

    def test_normalized_does_not_mutate(self):
        o = RewardOutput(progress=[-4.0, -2.0, 0.0], frame_indices=[0, 1, 2])
        assert np.allclose(o.normalized("minmax"), [0.0, 0.5, 1.0])
        assert o.progress[0] == -4.0

    def test_normalized_constant_no_nan(self):
        o = RewardOutput(progress=[3.0] * 5, frame_indices=range(5))
        assert np.allclose(o.normalized("minmax"), 0.0)

    def test_resample_rejects_extrapolation(self):
        """Regression: np.interp's default constant extrapolation would add values nobody observed."""
        o = RewardOutput(progress=[0.0, 1.0], frame_indices=[2, 10])
        with pytest.raises(ValueError, match="outside the frame range"):
            o.resample_to([0, 5, 12])
        assert np.allclose(o.resample_to([2, 6, 10]).progress, [0.0, 0.5, 1.0])

    def test_resample_rejects_invalid_extrapolate_value(self):
        """extrapolate is only raise|constant; other strings used to be silently treated as constant."""
        o = RewardOutput(progress=[0.0, 1.0], frame_indices=[2, 10])
        with pytest.raises(ValueError, match="extrapolate"):
            o.resample_to([2, 10], extrapolate="typo")

    def test_nonfinite_progress_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            RewardOutput(progress=[0.0, float("nan")], frame_indices=[0, 1])
        with pytest.raises(ValueError, match="non-finite"):
            RewardOutput(progress=[0.0, float("inf")], frame_indices=[0, 1])

    def test_success_and_weights_length_checked(self):
        with pytest.raises(ValueError, match="success"):
            RewardOutput(progress=[1, 2], frame_indices=[0, 1], success=[1.0])

    def test_terminal_success(self):
        o = RewardOutput(progress=[0, 1], frame_indices=[0, 1], success=[0.2, 0.9])
        assert o.terminal_success == 0.9
        assert RewardOutput(progress=[0], frame_indices=[0]).terminal_success is None

    def test_roundtrip(self):
        o = RewardOutput(progress=[0.1, 0.2], frame_indices=[0, 3], model_name="m",
                         success=[0.4, 0.6], raw={"x": np.array([1.0])})
        b = RewardOutput.from_dict(o.to_dict())
        assert np.allclose(b.progress, o.progress) and np.allclose(b.success, o.success)


class TestCostRecord:
    def test_zero_vram_not_swallowed(self):
        """Regression: ``max(...) or None`` used to turn a legitimate 0.0 into None."""
        from rewardbox.core.types import CostRecord
        a = CostRecord(peak_vram_gb=0.0)
        b = CostRecord(peak_vram_gb=0.0)
        assert (a + b).peak_vram_gb == 0.0
        assert (CostRecord() + CostRecord()).peak_vram_gb is None


# ---------------------------------------------------------------- registry
class TestRegistry:
    def test_lists_all_model_families(self):
        names = rewardbox.list_models()
        for expect in (
            "topreward", "roboreward-4b", "robodopamine-8b",
            "robometer-4b", "gvl", "gvl-gemini-3.5-flash-lite", "sole-r1-8b",
            "lrm-contrastive", "lrm-completion", "lrm-progress", "lrm-tri",
            "rynnvalue-4b", "rynnvalue-8b",
            "robo-valuerl-value",
            "procvlm-2b",
            "vlac-2b", "vlac-8b", "vlac-cut",
        ):
            assert expect in names, names
        assert "rewind-rbm1m" not in names

    def test_did_you_mean(self):
        with pytest.raises(KeyError, match="Did you mean"):
            rewardbox.get_spec("topreward1")

    def test_filter_and_mode(self):
        assert set(rewardbox.list_models(filter="robo*")) >= {"robometer-4b",
                                                              "robodopamine-8b"}
        assert rewardbox.list_models(mode="progress")

    def test_specs_self_consistent(self):
        for name, s in rewardbox.load_manifest().items():
            assert s.name == name
            assert s.module.startswith("rewardbox.adapters.")
            assert s.upstream, f"{name} records no upstream entry point"
            if s.supports_pairwise:
                import importlib
                cls = getattr(importlib.import_module(s.module), s.cls)
                assert "_compare" in cls.__dict__, f"{name} claims pairwise but does not implement it"

    def test_model_info(self):
        info = rewardbox.get_model_info("topreward")
        assert info["name"] == "topreward" and "available" in info

    def test_vlac_dependency_error_points_to_isolated_environment(self):
        spec = rewardbox.get_spec("vlac-2b")
        message = spec.requires.check().format(spec.name)
        assert ".venv-vlac" in message
        assert "shared-py311-linux-x86_64.lock" not in message


# ---------------------------------------------------------------- rule 1
HEAVY = {"torch", "torchvision", "transformers", "vllm", "unsloth", "qwen_vl_utils",
         "accelerate", "topreward", "robometer", "timm", "swift", "evo_vlac"}
TARGETS = sorted(list((ROOT / "core").glob("*.py")) +
                 list((ROOT / "adapters").glob("*.py")) +
                 list((ROOT / "schemas").glob("*.py")) +
                 list((ROOT / "cli").glob("*.py")) +
                 list((ROOT / "metrics").glob("*.py")) +
                 list((ROOT / "data").glob("*.py")) +
                 list((ROOT / "bench").glob("*.py")) +
                 [ROOT / "registry.py", ROOT / "__init__.py"])


def _toplevel_imports(path: Path) -> set[str]:
    names = set()
    for node in ast.parse(path.read_text()).body:      # module top level only
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("path", TARGETS, ids=lambda p: p.name)
def test_no_heavy_toplevel_import(path):
    """Rule 1: discipline fails, tests do not."""
    bad = _toplevel_imports(path) & HEAVY
    assert not bad, (f"{path.name} imports heavy dependencies {sorted(bad)} at top level; "
                     f"move them into functions/methods")


def test_list_models_does_not_load_heavy():
    code = ("import sys, rewardbox; rewardbox.list_models(); "
            "leaked={m for m in ('torch','transformers','vllm','unsloth') if m in sys.modules}; "
            "assert not leaked, leaked; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ---------------------------------------------------------------- metrics
class TestMetrics:
    def test_voc_perfect_and_reversed(self):
        from rewardbox.metrics import voc
        assert voc([0.0, 0.3, 0.6, 1.0]) == 1.0
        assert voc([1.0, 0.6, 0.3, 0.0]) == -1.0

    def test_voc_degenerate_matches_upstream(self):
        """Official voc.py: constant / too short -> NaN -> upstream notes it and maps 0.0."""
        from rewardbox.metrics.ordering import voc_with_note
        assert voc_with_note([0.5] * 5) == (0.0, "constant predictions")
        assert voc_with_note([0.3]) == (0.0, "insufficient length")

    def test_voc_invariant_under_minmax(self):
        """Evidence for A7: min-max is a positive affine map, strictly order-preserving, so
        Spearman is unchanged bit for bit."""
        from rewardbox.metrics import voc
        raw = np.array([-4.56, -3.17, -3.34, -3.23, -2.81, -0.24])
        mm = (raw - raw.min()) / (raw.max() - raw.min())
        assert voc(raw) == voc(mm)

    def test_monotonicity(self):
        from rewardbox.metrics import monotonicity
        assert monotonicity([0, 1, 2, 3]) == 1.0
        assert monotonicity([0, 1, 0, 1]) == pytest.approx(2 / 3)


# ---------------------------------------------------------------- prompt snapshot
def test_roboreward_prompt_matches_model_card():
    """The prompt is part of the reproduction and must be locked; the text comes from the HF model card."""
    from rewardbox.adapters.roboreward import load_prompt
    p = load_prompt()
    assert "ANSWER: <score>" in p
    assert "judge only the final state without time limits" in p
    assert p.count("\n1 - No Success") == 1
    assert "{instruction}" in p
    assert p.format(instruction="stack cubes").endswith("Task: stack cubes")
