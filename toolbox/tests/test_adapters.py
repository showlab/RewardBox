"""Adapter construction / precondition tests: only input checks that must reject before
official code is called, no GPU or real weights (regressions that run official code are in
test_calibration.py).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from rewardbox.core.types import SamplingSpec, Trajectory


def _traj(sampling, n=8, instruction="pick"):
    frames = np.stack([np.full((8, 8, 3), min(255, i * 20), np.uint8) for i in range(n)])
    return Trajectory(frames=frames, instruction=instruction, sampling=sampling)


class TestRoboDopamineGuards:
    """B4: preconditions of the only supported protocol (all frames, upstream samples by
    frame_interval) must reject before the official GRMInference is called."""

    def _adapter(self, **kw):
        from rewardbox.adapters.robodopamine import RoboDopamineAdapter
        from rewardbox.registry import get_spec
        return RoboDopamineAdapter(get_spec("robodopamine-8b"), **kw)

    def test_empty_modes_rejected_at_construction(self):
        with pytest.raises(ValueError, match="modes must not be an empty tuple"):
            self._adapter(modes=())

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="unknown modes"):
            self._adapter(modes=("sideways",))

    def test_default_frame_interval_is_official_ten(self):
        a = self._adapter()
        assert a.frame_interval == 10

    def test_explicit_frame_interval_overrides_default(self):
        a = self._adapter(frame_interval=5)
        assert a.frame_interval == 5

    @pytest.mark.parametrize("value", [0, -0.1, 1.1, True, float("nan")])
    def test_invalid_gpu_memory_budget_rejected(self, value):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            self._adapter(gpu_memory_utilization=value)

    @pytest.mark.upstream("robodopamine")
    @pytest.mark.parametrize("budget", [None, 0.5])
    @pytest.mark.parametrize("failed", [False, True])
    def test_official_constructor_receives_budget_and_is_restored(self, monkeypatch, budget, failed):
        adapter = self._adapter(**({} if budget is None else {"gpu_memory_utilization": budget}))
        module = types.ModuleType("examples.inference")
        calls = []

        def official_llm(**kwargs):
            calls.append(kwargs)
            return object()

        class GRM:
            def __init__(self, model_path):
                self.model = module.LLM(model=model_path, gpu_memory_utilization=0.9,
                                        max_model_len=8192)
                if failed:
                    raise RuntimeError("startup failed")

        module.LLM = official_llm
        module.GRMInference = GRM
        monkeypatch.setitem(sys.modules, "examples.inference", module)
        if failed:
            with pytest.raises(RuntimeError, match="startup failed"):
                adapter.grm()
            assert adapter._grm is None
        else:
            assert adapter.grm() is adapter.grm()  # one engine per adapter
        assert calls == [{"model": adapter.model_path,
                          "gpu_memory_utilization": 0.7 if budget is None else budget,
                          "max_model_len": 8192}]
        assert module.LLM is official_llm

    @pytest.mark.parametrize("kw", ({"frame_interval": 0}, {"batch_size": 0}))
    def test_positive_integer_runtime_options(self, kw):
        with pytest.raises(ValueError, match="positive integer"):
            self._adapter(**kw)

    def test_rejects_non_all_sampling_before_calling_official_code(self):
        a = self._adapter()
        traj = _traj(SamplingSpec(num_frames=8, strategy="uniform"))
        with pytest.raises(ValueError, match="strategy='all'"):
            a.score(traj)

    @pytest.mark.upstream("robodopamine")
    def test_accepts_all_strategy_sampling(self, monkeypatch):
        """Only checks that the guard lets valid input through.  The official vLLM inference
        does not run: the step after grm() is monkeypatched, short-circuiting right before the
        vLLM engine would be built (not bypassing the argument path), so this CPU-only test
        never pulls in heavy dependencies."""
        a = self._adapter()
        traj = _traj(SamplingSpec(strategy="all"))

        def fake_grm():
            raise RuntimeError("preconditions passed; normally the official GRMInference is called here")

        monkeypatch.setattr(a, "grm", fake_grm)
        with pytest.raises(RuntimeError, match="preconditions passed"):
            a.score(traj)

    @pytest.mark.parametrize("batch_size,backend_calls", [(1, 4), (3, 2), (4, 1)])
    def test_maps_official_sample_grid_to_original_frames(self, monkeypatch, batch_size, backend_calls):
        """The four steps of interval=10 must map to AFTER=[10,20,30,39], without guessing from
        T/T-1 lengths."""
        a = self._adapter(modes=("incremental",), batch_size=batch_size)
        traj = _traj(SamplingSpec(strategy="all"), n=40)

        class FakeGRM:
            def run_pipeline(self, *, cam_high_path, out_root, eval_mode,
                             frame_interval, **kwargs):
                run_dir = Path(out_root) / "run"
                run_dir.mkdir(parents=True)
                after = [10, 20, 30, 39]
                samples = [{"id": f"step-run-{i:04d}-bf-af_{idx:06d}"}
                           for i, idx in enumerate(after)]
                preds = [
                    {"progress": float(i) / 3, "hop": 0.1, "pred": "x"}
                    for i in range(4)
                ]
                (run_dir / "sample.json").write_text(json.dumps(samples))
                (run_dir / "pred_vllm.json").write_text(json.dumps(preds))
                return str(run_dir)

        monkeypatch.setattr(a, "grm", lambda: FakeGRM())
        out = a.score(traj)
        assert out.frame_indices.tolist() == [10, 20, 30, 39]
        assert len(out.progress) == 4
        assert out.meta["n_inference_requests"] == 4
        assert out.meta["inference_batch_size"] == batch_size
        assert out.meta["n_backend_calls"] == backend_calls


class TestTOPRewardAdapter:
    """Protocol selection and output mapping without importing/loading a VLM."""

    def _adapter(self, name="topreward", **kw):
        from rewardbox.adapters.topreward import TOPRewardAdapter
        from rewardbox.registry import get_spec

        return TOPRewardAdapter(get_spec(name), check_deps=False, **kw)

    def test_qwen3_default_is_the_pinned_robometer_table_i_recipe(self):
        adapter = self._adapter()

        assert adapter.protocol == "robometer_table_i"
        assert adapter.spec.sampling == SamplingSpec(num_frames=64, strategy="uniform")
        assert adapter.num_samples == 15
        assert adapter.reduction == "mean"
        assert adapter.fps == 2.0
        assert adapter.add_chat_template is True
        assert adapter.use_video_description is False

    def test_qwen25_is_explicitly_native_and_non_table_i(self):
        adapter = self._adapter("topreward-qwen2.5-7b")

        assert adapter.protocol == "topreward_native"
        assert adapter.spec.sampling == SamplingSpec(num_frames=15, strategy="uniform")
        assert adapter.add_chat_template is False
        assert "non-Table-I" in adapter.spec.description

    @pytest.mark.parametrize(
        "name,kw",
        (
            ("topreward", {"protocol": "topreward_native"}),
            ("topreward-qwen2.5-7b", {"protocol": "robometer_table_i"}),
            ("topreward", {"use_video_description": True}),
        ),
    )
    def test_cross_protocol_or_nonrecipe_options_are_rejected(self, name, kw):
        with pytest.raises(ValueError):
            self._adapter(name, **kw)

    @pytest.mark.upstream("robometer")
    def test_robometer_backend_receives_exact_recipe_constructor_args(self, monkeypatch):
        calls = []

        class FakeTopReward:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        # The adapter imports lazily so registry discovery never imports either
        # Robometer or torch.  Supply only the module chain it asks for.
        robometer = types.ModuleType("robometer")
        robometer.__path__ = []
        evals = types.ModuleType("robometer.evals")
        evals.__path__ = []
        baselines = types.ModuleType("robometer.evals.baselines")
        baselines.__path__ = []
        module = types.ModuleType("robometer.evals.baselines.topreward")
        module.TopReward = FakeTopReward
        for name, value in {
            "robometer": robometer,
            "robometer.evals": evals,
            "robometer.evals.baselines": baselines,
            "robometer.evals.baselines.topreward": module,
        }.items():
            monkeypatch.setitem(sys.modules, name, value)

        adapter = self._adapter()
        adapter.client()

        assert calls == [{
            "model_path": "Qwen/Qwen3-VL-8B-Instruct",
            "max_frames": 64,
            "num_prefix_samples": 15,
            "reduction": "mean",
            "add_chat_template": True,
            "fps": 2.0,
        }]

    def test_robometer_progress_is_passed_through_and_maps_internal_cap_grid(self):
        adapter = self._adapter()

        class FakeTopReward:
            def __init__(self):
                self.called = None

            def _compute_instruction_rewards_for_prefixes(self, frames, instruction):
                self.called = (frames, instruction)
                return types.SimpleNamespace(
                    prefix_lengths=[1, 32, 64],
                    prefix_rewards=[-7.0, -4.0, -1.0],
                    normalized_prefix_rewards=[0.0, 0.5, 1.0],
                    reward=-1.0,
                    reduction="mean",
                    token_count=1,
                )

            def compute_progress(self, frames, task_description):
                # Simulate the exact upstream control flow: it internally caps to
                # 64, asks its own prefix helper once, then returns a dense curve.
                selected = frames if len(frames) <= 64 else frames[np.linspace(
                    0, len(frames) - 1, 64, dtype=int)]
                self._compute_instruction_rewards_for_prefixes(["official-prefix-input"], task_description)
                return np.linspace(0.0, 1.0, len(selected), dtype=np.float64)

        fake = FakeTopReward()
        adapter._client = fake
        traj = _traj(SamplingSpec(num_frames=70), n=70)
        traj.frame_indices = np.arange(70, dtype=np.int64) * 3 + 10

        out = adapter._score(traj)

        expected_local = np.linspace(0, 69, 64, dtype=int)
        assert np.allclose(out.progress, np.linspace(0.0, 1.0, 64))
        assert out.frame_indices.tolist() == traj.frame_indices[expected_local].tolist()
        assert out.raw["prefix_lengths"] == [1, 32, 64]
        assert out.raw["normalized_prefix_rewards"] == [0.0, 0.5, 1.0]
        assert out.raw["official_compute_progress"] == out.progress.tolist()
        assert out.raw["input_local_indices"] == expected_local.tolist()
        assert out.meta["protocol"] == "robometer_table_i"
        assert out.meta["official_args"]["add_chat_template"] is True
        assert out.meta["n_backend_calls"] == 3
        # The temporary provenance wrapper must not leak into the cached backend.
        assert fake._compute_instruction_rewards_for_prefixes.__name__ == (
            "_compute_instruction_rewards_for_prefixes")

    def test_native_qwen25_keeps_raw_prefix_outputs_and_nonbenchmark_provenance(self):
        adapter = self._adapter("topreward-qwen2.5-7b")

        class FakeQwenClient:
            def __init__(self):
                self.kwargs = None

            def compute_instruction_rewards_for_prefixes(self, **kwargs):
                self.kwargs = kwargs
                return types.SimpleNamespace(
                    prefix_lengths=[1, 3, 4],
                    prefix_rewards=[-9.0, -6.0, -3.0],
                    normalized_prefix_rewards=[0.0, 0.5, 1.0],
                    reward=-3.0,
                    token_count=1,
                )

        fake = FakeQwenClient()
        adapter._client = fake
        traj = _traj(SamplingSpec(num_frames=4), n=4)

        out = adapter._score(traj)

        assert out.progress.tolist() == [-9.0, -6.0, -3.0]
        assert out.frame_indices.tolist() == [0, 2, 3]
        assert out.weights is None
        assert out.meta["protocol"] == "topreward_native"
        assert out.meta["paper_equivalence"].startswith("not asserted")
        assert fake.kwargs["add_chat_template"] is False
        assert fake.kwargs["use_video_description"] is False


class TestVLACAdapter:
    def _adapter(self, **kw):
        from rewardbox.adapters.vlac import VLACAdapter
        from rewardbox.registry import get_spec

        return VLACAdapter(get_spec("vlac-2b"), check_deps=False, **kw)

    @pytest.mark.parametrize(
        "kw",
        ({"skip": 0}, {"batch_size": 0}, {"top_k": 0}, {"temperature": 0}),
    )
    def test_positive_runtime_options(self, kw):
        with pytest.raises(ValueError, match="positive"):
            self._adapter(**kw)

    def test_invalid_value_simple_rejected(self):
        with pytest.raises(ValueError, match="value_simple"):
            self._adapter(value_simple="typo")

    @pytest.mark.parametrize("value", (0, 1, None))
    def test_value_simple_does_not_accept_bool_lookalikes(self, value):
        with pytest.raises(ValueError, match="value_simple"):
            self._adapter(value_simple=value)

    def test_blank_model_type_rejected(self):
        with pytest.raises(ValueError, match="model_type"):
            self._adapter(model_type=" ")

    def test_blank_device_map_rejected(self):
        with pytest.raises(ValueError, match="device_map"):
            self._adapter(device_map="")

    def test_score_preserves_official_scale_and_maps_skip_grid(self):
        adapter = self._adapter(skip=2, batch_size=2, rich=False)

        class FakeCritic:
            def __init__(self):
                self.kwargs = None

            def get_trajectory_critic(self, **kwargs):
                self.kwargs = kwargs
                return ["50", "-20"], [0, 50, 40]

        fake = FakeCritic()
        adapter._critic = fake
        traj = _traj(SamplingSpec(num_frames=5), n=5)
        traj.frame_indices = np.asarray([10, 20, 30, 40, 50], dtype=np.int64)

        out = adapter.score(traj)

        assert out.progress.tolist() == [0.0, 50.0, 40.0]
        assert out.frame_indices.tolist() == [10, 30, 50]
        assert out.weights is None
        assert out.raw["critic_list"] == [50.0, -20.0]
        assert out.raw["critic_pair_frame_indices"] == [[10, 30], [30, 50]]
        assert out.meta["normalized"] is False
        assert out.meta["padded_or_truncated"] is False
        assert out.cost.n_calls == 1
        assert fake.kwargs["skip"] == 2
        assert fake.kwargs["batch_num"] == 2
        assert fake.kwargs["frame_skip"] is True
        assert fake.kwargs["ref_image_list"] is None
        assert len(fake.kwargs["image_list"]) == 5

    def test_output_length_drift_is_rejected(self):
        adapter = self._adapter(skip=2)

        class FakeCritic:
            def get_trajectory_critic(self, **kwargs):
                return [10.0], [0.0, 10.0]

        adapter._critic = FakeCritic()
        with pytest.raises(RuntimeError, match="value_list length"):
            adapter.score(_traj(SamplingSpec(num_frames=5), n=5))

    def test_compare_uses_b_then_a_terminal_frames(self):
        adapter = self._adapter()

        class FakeCritic:
            def __init__(self):
                self.images = None

            def get_trajectory_critic(self, **kwargs):
                self.images = kwargs["image_list"]
                return [7.5], [0.0, 7.5]

        fake = FakeCritic()
        adapter._critic = fake
        a = _traj(SamplingSpec(num_frames=2), n=2)
        b = _traj(SamplingSpec(num_frames=2), n=2)
        a.primary[-1].fill(200)
        b.primary[-1].fill(20)

        assert adapter.compare(a, b) == 7.5
        assert np.asarray(fake.images[0]).mean() == 20
        assert np.asarray(fake.images[1]).mean() == 200

    def test_compare_rejects_different_tasks(self):
        adapter = self._adapter()
        a = _traj(SamplingSpec(num_frames=2), n=2, instruction="pick")
        b = _traj(SamplingSpec(num_frames=2), n=2, instruction="place")
        with pytest.raises(ValueError, match="same instruction"):
            adapter.compare(a, b)

    def test_compare_allows_different_source_resolutions(self):
        adapter = self._adapter()

        class FakeCritic:
            def get_trajectory_critic(self, **kwargs):
                assert [image.size for image in kwargs["image_list"]] == [
                    (11, 9), (8, 8),
                ]
                return [2.0], [0.0, 2.0]

        adapter._critic = FakeCritic()
        a = _traj(SamplingSpec(num_frames=2), n=2)
        b = Trajectory(
            frames=np.zeros((2, 9, 11, 3), dtype=np.uint8),
            instruction="pick",
            sampling=SamplingSpec(num_frames=2),
        )
        assert adapter.compare(a, b) == 2.0
