"""Offline contract tests for the pinned Large-Reward-Models HTTP adapters."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from rewardbox.core.types import SamplingSpec, Trajectory


def _traj(n=3, *, goal=False):
    frames = np.stack(
        [np.full((8, 10, 3), i * 20, dtype=np.uint8) for i in range(n)]
    )
    return Trajectory(
        frames=frames,
        instruction="put the object on the plate",
        frame_indices=np.arange(n, dtype=np.int64) * 10 + 5,
        sampling=SamplingSpec(strategy="all"),
        goal_image=(np.full((8, 10, 3), 255, np.uint8) if goal else None),
    )


def _adapter(name, **kwargs):
    from rewardbox.adapters.lrm import LRMAdapter
    from rewardbox.registry import get_spec

    return LRMAdapter(
        get_spec(name), check_deps=False, validate_health=False, **kwargs
    )


class FakeClient:
    def __init__(self, *, comparisons=(), progress=(), completion=()):
        self.comparisons = iter(comparisons)
        self.progress = iter(progress)
        self.completion = iter(completion)
        self.calls = []
        self.closed = False

    def compute_comparison(self, earlier, later, task):
        self.calls.append(("contrastive", earlier.copy(), later.copy(), task))
        result, score = next(self.comparisons)
        return {
            "result": result,
            "score": score,
            "response": f"comparison:{result}",
            "success": True,
        }

    def compute_reward(self, frame, task, **kwargs):
        self.calls.append(("progress", frame.copy(), task, kwargs))
        score = next(self.progress)
        return {"score": score, "response": f"progress:{score}", "success": True}

    def compute_completion(self, frame, task):
        self.calls.append(("completion", frame.copy(), task))
        score = next(self.completion)
        return {
            "score": score,
            "completed": bool(score),
            "response": f"completion:{score}",
            "success": True,
        }

    def close(self):
        self.closed = True


def _attach(adapter, fake):
    for url in set(adapter.server_urls.values()):
        adapter._clients[url] = fake


class TestSpecs:
    def test_four_profiles_are_registered_with_official_all_frame_default(self):
        import rewardbox

        names = {name for name in rewardbox.list_models()
                 if name.startswith("lrm-") and not name.endswith("-vllm")}
        assert names == {
            "lrm-contrastive",
            "lrm-completion",
            "lrm-progress",
            "lrm-tri",
        }
        for name in names:
            spec = rewardbox.get_spec(name)
            assert spec.sampling.strategy == "all"
            assert "5845ab0" not in spec.weights  # code and weights revisions stay distinct
            assert "5f8da7f3934f5d658256d77552927a9f76ad1c0e" in spec.weights

    def test_tri_has_no_runtime_weight_option(self):
        import rewardbox

        with pytest.raises(TypeError, match="does not support model options"):
            rewardbox.create_reward(
                "lrm-tri", check_deps=False, tri_weights=[0.5, 0.25, 0.25]
            )
        signature = inspect.signature(
            __import__("rewardbox.adapters.lrm", fromlist=["LRMAdapter"]).LRMAdapter
        )
        assert "tri_weights" not in signature.parameters

    def test_official_client_source_is_loaded_without_server_import(self):
        from rewardbox.adapters.lrm import _upstream_client_class

        cls = _upstream_client_class()
        assert cls.__name__ == "VLMRewardClient"
        assert cls.__module__ == "_rewardbox_vendored_lrm_vlm_reward_client"


class TestContrastive:
    def test_first_frame_baseline_and_adjacent_signed_scores(self):
        adapter = _adapter("lrm-contrastive")
        fake = FakeClient(comparisons=[("ImageB", 1.0), ("ImageA", -1.0)])
        _attach(adapter, fake)

        out = adapter.score(_traj())

        assert out.progress.tolist() == [0.0, 1.0, -1.0]
        assert out.frame_indices.tolist() == [5, 15, 25]
        assert out.raw["comparison_results"] == ["first_frame", "ImageB", "ImageA"]
        assert out.cost.n_calls == 2
        assert len(fake.calls) == 2
        assert np.array_equal(fake.calls[0][1], _traj().primary[0])
        assert np.array_equal(fake.calls[0][2], _traj().primary[1])

    def test_parse_score_disagreement_is_rejected(self):
        adapter = _adapter("lrm-contrastive")
        fake = FakeClient(comparisons=[("ImageA", 1.0)])
        _attach(adapter, fake)
        with pytest.raises(RuntimeError, match="inconsistent with"):
            adapter.score(_traj(n=2))

    def test_official_client_failure_is_not_imputed_as_zero(self):
        adapter = _adapter("lrm-contrastive")

        class FailedClient:
            def compute_comparison(self, *args):
                return {"score": 0.0, "result": None, "success": False}

        _attach(adapter, FailedClient())
        with pytest.raises(RuntimeError, match="fallback 0"):
            adapter.score(_traj(n=2))


class TestPointwise:
    def test_completion_maps_native_binary_output_to_success(self):
        adapter = _adapter("lrm-completion")
        fake = FakeClient(completion=[0.0, 1.0, 1.0])
        _attach(adapter, fake)

        out = adapter.score(_traj())

        assert out.progress.tolist() == [0.0, 1.0, 1.0]
        assert out.success.tolist() == [0.0, 1.0, 1.0]
        assert out.raw["completed"] == [False, True, True]
        assert out.cost.n_calls == 3

    def test_progress_default_is_current_frame_only(self):
        adapter = _adapter("lrm-progress")
        fake = FakeClient(progress=[0.1, 0.4, 0.9])
        _attach(adapter, fake)

        out = adapter.score(_traj())

        assert out.progress.tolist() == [0.1, 0.4, 0.9]
        kwargs = [call[3] for call in fake.calls]
        assert all(item["reward_type"] == "progress" for item in kwargs)
        assert all(item["initial_image"] is None for item in kwargs)
        assert all(item["goal_image"] is None for item in kwargs)

    def test_progress_can_reproduce_closed_loop_initial_image_profile(self):
        adapter = _adapter(
            "lrm-progress", include_initial_image=True, include_goal_image=True
        )
        fake = FakeClient(progress=[0.2, 0.5])
        _attach(adapter, fake)
        traj = _traj(n=2, goal=True)

        adapter.score(traj)

        for call in fake.calls:
            assert np.array_equal(call[3]["initial_image"], traj.primary[0])
            assert np.array_equal(call[3]["goal_image"], traj.goal_image)

    def test_missing_requested_goal_is_rejected_before_http(self):
        adapter = _adapter("lrm-progress", include_goal_image=True)
        fake = FakeClient(progress=[0.1])
        _attach(adapter, fake)
        with pytest.raises(ValueError, match="has no goal_image"):
            adapter.score(_traj(n=1))
        assert not fake.calls


class TestTri:
    def test_fixed_official_fusion_and_component_audit(self):
        adapter = _adapter("lrm-tri")
        fake = FakeClient(
            comparisons=[("ImageB", 1.0), ("ImageA", -1.0)],
            progress=[0.2, 0.4, 0.6],
            completion=[0.0, 1.0, 1.0],
        )
        _attach(adapter, fake)

        out = adapter.score(_traj())

        weights = np.asarray([0.333333333333, 0.333333333333, 0.333333333334])
        expected = (
            weights[0] * np.asarray([0.0, 1.0, 0.0])
            + weights[1] * np.asarray([0.2, 0.4, 0.6])
            + weights[2] * np.asarray([0.0, 1.0, 1.0])
        )
        assert np.allclose(out.progress, expected)
        assert out.raw["contrastive_signed"] == [0.0, 1.0, -1.0]
        assert out.raw["contrastive_unit"] == [0.0, 1.0, 0.0]
        assert out.raw["progress"] == [0.2, 0.4, 0.6]
        assert out.raw["completion"] == [0.0, 1.0, 1.0]
        assert out.meta["official_args"]["tri_weights"] == weights.tolist()
        assert out.cost.n_calls == 8

    def test_shared_url_requires_all_three_adapters(self, monkeypatch):
        import rewardbox.adapters.lrm as lrm

        adapter = lrm.LRMAdapter(
            lrm.SPECS[-1], check_deps=False, validate_health=True
        )
        monkeypatch.setattr(
            lrm,
            "_fetch_health",
            lambda url, timeout: {"model_loaded": True, "adapters": []},
        )
        with pytest.raises(RuntimeError, match="refusing to silently reuse the wrong checkpoint"):
            adapter._ensure_health()

    def test_shared_url_accepts_advertised_multi_adapter_server(self, monkeypatch):
        import rewardbox.adapters.lrm as lrm

        adapter = lrm.LRMAdapter(
            lrm.SPECS[-1], check_deps=False, validate_health=True
        )
        monkeypatch.setattr(
            lrm,
            "_fetch_health",
            lambda url, timeout: {
                "model_loaded": True,
                "adapters": ["progress", "completion", "contrastive"],
            },
        )
        adapter._ensure_health()
        assert set(adapter._health[adapter.server_url]["adapters"]) == {
            "contrastive",
            "progress",
            "completion",
        }

    def test_distinct_full_model_servers_may_have_empty_adapter_lists(self, monkeypatch):
        import rewardbox.adapters.lrm as lrm

        adapter = lrm.LRMAdapter(
            lrm.SPECS[-1],
            check_deps=False,
            validate_health=True,
            contrastive_url="http://localhost:5101",
            progress_url="http://localhost:5102",
            completion_url="http://localhost:5103",
        )
        monkeypatch.setattr(
            lrm,
            "_fetch_health",
            lambda url, timeout: {"model_loaded": True, "adapters": []},
        )
        adapter._ensure_health()
        assert len(adapter._health) == 3


class TestRuntimeIdentity:
    def test_provenance_file_is_copied_to_output_meta(self, tmp_path):
        provenance = tmp_path / "server.json"
        provenance.write_text(
            json.dumps({"weights_revision": "pinned", "mode": "progress"}),
            encoding="utf-8",
        )
        adapter = _adapter(
            "lrm-progress", server_provenance_file=str(provenance)
        )
        fake = FakeClient(progress=[0.5])
        _attach(adapter, fake)
        out = adapter.score(_traj(n=1))
        assert out.meta["server_checkpoint_provenance"]["mode"] == "progress"
        assert out.meta["server_provenance_file"] == str(provenance.resolve())

    def test_tri_preserves_one_provenance_file_per_dedicated_server(self, tmp_path):
        files = {}
        for head in ("contrastive", "progress", "completion"):
            path = tmp_path / f"{head}.json"
            path.write_text(json.dumps({"mode": head}), encoding="utf-8")
            files[head] = str(path)
        adapter = _adapter(
            "lrm-tri",
            contrastive_url="http://localhost:5101",
            progress_url="http://localhost:5102",
            completion_url="http://localhost:5103",
            server_provenance_files=files,
        )
        fake = FakeClient(
            comparisons=[("ImageB", 1.0)],
            progress=[0.1, 0.2],
            completion=[0.0, 1.0],
        )
        _attach(adapter, fake)

        out = adapter.score(_traj(n=2))

        assert out.meta["server_checkpoint_provenance"] == {
            head: {"mode": head} for head in files
        }
        assert out.meta["server_provenance_files"] == {
            head: str(Path(path).resolve()) for head, path in files.items()
        }

    def test_tri_provenance_map_requires_exact_head_set(self, tmp_path):
        path = tmp_path / "one.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="must contain exactly"):
            _adapter(
                "lrm-tri",
                server_provenance_files={"progress": str(path)},
            )

    def test_wrapper_provenance_with_wrong_head_is_rejected(self, tmp_path):
        path = tmp_path / "wrong.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "rewardbox.lrm-server-provenance.v1",
                    "mode": "completion",
                    "upstream": {
                        "commit": "5845ab0f7c3c6a0bef8bd37bf7f07e9533f7ed2f"
                    },
                    "weights": {"checkpoints": {"completion": {}}},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="mode='completion'"):
            _adapter("lrm-progress", server_provenance_file=str(path))

    def test_valid_wrapper_provenance_is_accepted(self, tmp_path):
        path = tmp_path / "progress.json"
        record = {
            "schema": "rewardbox.lrm-server-provenance.v1",
            "mode": "progress",
            "upstream": {
                "commit": "5845ab0f7c3c6a0bef8bd37bf7f07e9533f7ed2f"
            },
            "weights": {
                "checkpoints": {
                    "progress": {
                        "pinned_public_snapshot": True,
                        "snapshot_revision": (
                            "5f8da7f3934f5d658256d77552927a9f76ad1c0e"
                        ),
                    }
                }
            },
        }
        path.write_text(json.dumps(record), encoding="utf-8")
        adapter = _adapter("lrm-progress", server_provenance_file=str(path))
        fake = FakeClient(progress=[0.5])
        _attach(adapter, fake)
        out = adapter.score(_traj(n=1))
        assert out.meta["server_checkpoint_provenance"] == record

    @pytest.mark.parametrize(
        "url",
        (
            "ftp://localhost:5002",
            "http://user:secret@localhost:5002",
            "http://localhost:5002/path",
            "http://localhost:99999",
        ),
    )
    def test_unsafe_or_invalid_server_url_is_rejected(self, url):
        with pytest.raises(ValueError, match="server_url"):
            _adapter("lrm-progress", server_url=url)

    def test_close_releases_official_clients(self):
        adapter = _adapter("lrm-progress")
        fake = FakeClient(progress=[0.1])
        _attach(adapter, fake)
        adapter.close()
        assert fake.closed is True
        assert adapter._clients == {}
