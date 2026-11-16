"""Offline contracts for the pinned official RynnValue HTTP adapters."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rewardbox.core.types import SamplingSpec, Trajectory


def _trajectory(n: int = 3, *, meta: dict | None = None) -> Trajectory:
    frames = np.stack(
        [np.full((8, 10, 3), 20 * index, dtype=np.uint8) for index in range(n)]
    )
    return Trajectory(
        frames=frames,
        instruction="put the box in the drawer",
        frame_indices=np.arange(n, dtype=np.int64) * 7 + 2,
        sampling=SamplingSpec(num_frames=8, strategy="uniform"),
        meta=(
            meta
            if meta is not None
            else {"camera_description": "fixed third-person RGB camera"}
        ),
    )


def _provenance(
    path: Path,
    *,
    model_size: str = "8b",
    profile: str = "official-isolated",
    pinned: bool = True,
) -> Path:
    from rewardbox.adapters.rynnvalue import _MODELS, _UPSTREAM_COMMIT

    identity = _MODELS[f"rynnvalue-{model_size}"]
    effective = (
        "pred_slot_isolated_eager" if profile == "official-isolated" else "eager"
    )
    record = {
        "schema": "rewardbox.rynnvalue-server-provenance.v1",
        "status": "ready",
        "upstream": {"commit": _UPSTREAM_COMMIT},
        "weights": {
            "model_id": identity["model_id"],
            "expected_revision": identity["revision"],
            "snapshot_revision": identity["revision"] if pinned else None,
            "pinned_public_snapshot": pinned,
        },
        "profile": {
            "name": profile,
            "effective_attention": effective,
            "max_frames": 8,
            "use_frame_steps": False,
            "pad_frames": False,
            "mode": "absolute",
            "stride": 2,
            "num_frames": 8,
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": 1,
            "use_fuse": False,
        },
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _model_info(*, attention=None, mode="absolute") -> dict:
    return {
        "reward_model": "rynnvalue",
        "config": {
            "reward_model": "rynnvalue",
            "max_frames": 8,
            "use_frame_steps": False,
            "custom_eval": {
                "use_frame_steps": False,
                "pad_frames": False,
            },
            "model_path": "/pinned/snapshot",
            "model_config": {
                "checkpoint_path": None,
                "mode": mode,
                "stride": 2,
                "num_frames": 8,
                "max_new_tokens": 128,
                "use_fuse": False,
                "attn_implementation": attention,
            },
        },
    }


class FakeResponse:
    def __init__(self, payload, *, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.text = json.dumps(payload)

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, progress=None, *, model_info=None, health=None, output=None):
        self.progress = progress if progress is not None else [-5.0, -2.0, -0.5]
        self.info = model_info or _model_info()
        self.health = health or {"status": "healthy", "reward_model": "rynnvalue"}
        self.output = output
        self.get_calls = []
        self.post_calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return FakeResponse(self.health if url.endswith("/health") else self.info)

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        payload = self.output or {
            "outputs_preference": None,
            "outputs_progress": {"progress_pred": [self.progress]},
        }
        return FakeResponse(payload)

    def close(self):
        self.closed = True


def _adapter(tmp_path: Path, *, model_size="8b", fake=None, **kwargs):
    from rewardbox.adapters.rynnvalue import RynnValueAdapter
    from rewardbox.registry import get_spec

    if "server_provenance_file" not in kwargs and not kwargs.get(
        "allow_unverified_server", False
    ):
        kwargs["server_provenance_file"] = str(
            _provenance(tmp_path / f"{model_size}.json", model_size=model_size)
        )
    adapter = RynnValueAdapter(
        get_spec(f"rynnvalue-{model_size}"), check_deps=False, **kwargs
    )
    adapter._session = fake or FakeSession()
    return adapter


class TestSpecs:
    def test_both_public_models_use_official_eight_frame_grid(self):
        import rewardbox

        names = {name for name in rewardbox.list_models() if name.startswith("rynnvalue-")}
        assert names == {"rynnvalue-4b", "rynnvalue-8b"}
        revisions = {
            "rynnvalue-4b": "3f73b5d2b5e53b21f248c8791004dde6a8cf2b92",
            "rynnvalue-8b": "8738c5e4ce4418ea0266e9fbeffae0eb9bbb230e",
        }
        for name, revision in revisions.items():
            spec = rewardbox.get_spec(name)
            assert spec.sampling == SamplingSpec(num_frames=8, strategy="uniform")
            assert revision in spec.weights
            assert spec.score_range == (-512.0, 0.0)

    def test_model_registry_does_not_import_torch_or_transformers(self, monkeypatch):
        import sys
        import rewardbox.registry as registry

        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.delitem(sys.modules, "transformers", raising=False)
        registry.load_manifest(force=True)
        assert "torch" not in sys.modules
        assert "transformers" not in sys.modules


class TestTransport:
    def test_official_multipart_payload_and_raw_output_mapping(self, tmp_path):
        fake = FakeSession(progress=[-9.0, -3.0, -0.25])
        adapter = _adapter(tmp_path, fake=fake)
        trajectory = _trajectory(meta={"camera_description": "overhead camera", "id": "ep-7"})

        output = adapter.score(trajectory)

        assert output.progress.tolist() == [-9.0, -3.0, -0.25]
        assert output.frame_indices.tolist() == [2, 9, 16]
        assert output.cost.n_calls == 2
        assert output.meta["n_http_calls"] == 1
        assert output.meta["official_profile"]["effective_attention"] == (
            "pred_slot_isolated_eager"
        )
        assert output.meta["official_profile"]["pad_frames"] is False
        url, request = fake.post_calls[0]
        assert url.endswith("/evaluate_batch_npy")
        assert request["data"]["use_frame_steps"] == "false"
        sample = json.loads(request["data"]["sample_0"])
        assert sample["sample_type"] == "progress"
        assert sample["trajectory"]["task"] == trajectory.instruction
        assert sample["trajectory"]["id"] == "ep-7"
        assert sample["trajectory"]["metadata"]["camera_description"] == (
            "overhead camera"
        )
        file_tuple = request["files"]["sample_0_trajectory_frames"]
        file_tuple[1].seek(0)
        assert np.array_equal(np.load(file_tuple[1], allow_pickle=False), trajectory.primary)

    def test_explicit_description_options_fill_metadata_without_invented_defaults(
        self, tmp_path
    ):
        fake = FakeSession(progress=[-1.0])
        adapter = _adapter(
            tmp_path,
            fake=fake,
            robot_description="7-DoF arm with parallel gripper",
            camera_description="front fixed RGB camera",
            sample_id="fixture-1",
        )
        adapter.score(_trajectory(n=1, meta={"quality": "success"}))
        sample = json.loads(fake.post_calls[0][1]["data"]["sample_0"])
        metadata = sample["trajectory"]["metadata"]
        assert metadata["robot_description"].startswith("7-DoF")
        assert metadata["camera_description"].startswith("front")
        assert sample["trajectory"]["id"] == "fixture-1"

    def test_use_meta_checkpoint_rejects_description_free_input_before_post(self, tmp_path):
        fake = FakeSession(progress=[-1.0])
        adapter = _adapter(tmp_path, fake=fake)
        with pytest.raises(ValueError, match="processor.use_meta=true"):
            adapter.score(_trajectory(n=1, meta={}))
        assert fake.post_calls == []

    def test_optional_success_probabilities_are_preserved(self, tmp_path):
        fake = FakeSession(
            output={
                "outputs_progress": {
                    "progress_pred": [[-2.0, -0.5]],
                    "success_probs": [[0.1, 0.8]],
                }
            }
        )
        output = _adapter(tmp_path, fake=fake).score(_trajectory(n=2))
        assert output.success.tolist() == [0.1, 0.8]

    def test_official_match_no_sentinel_is_preserved_and_marked(self, tmp_path):
        fake = FakeSession(progress=[-100.0, -100.0, -100.0])
        output = _adapter(tmp_path, fake=fake).score(_trajectory())
        assert output.progress.tolist() == [-100.0, -100.0, -100.0]
        assert output.meta["official_mismatch_sentinel_pattern"] is True
        assert output.meta["official_mismatch_sentinel_value"] == -100.0

        ordinary = _adapter(
            tmp_path,
            fake=FakeSession(progress=[-100.0, -3.0, -0.5]),
        ).score(_trajectory())
        assert ordinary.meta["official_mismatch_sentinel_pattern"] is False

    @pytest.mark.parametrize(
        "payload,match",
        (
            ({"outputs_progress": {"progress_pred": []}}, "exactly one"),
            ({"outputs_progress": {"progress_pred": [[-1.0]]}}, "length/dimension"),
            (
                {"outputs_progress": {"progress_pred": [[-1.0, float("nan"), 0.0]]}},
                "NaN/Inf",
            ),
            ({"outputs_progress": {"progress_pred": [[-1.0, 0.2, 0.0]]}}, r"\[-512,0\]"),
        ),
    )
    def test_malformed_or_out_of_range_server_output_is_rejected(
        self, tmp_path, payload, match
    ):
        adapter = _adapter(tmp_path, fake=FakeSession(output=payload))
        with pytest.raises(RuntimeError, match=match):
            adapter.score(_trajectory())


class TestIdentity:
    def test_provenance_is_required_by_default(self):
        from rewardbox.adapters.rynnvalue import RynnValueAdapter
        from rewardbox.registry import get_spec

        with pytest.raises(ValueError, match="server_provenance_file"):
            RynnValueAdapter(get_spec("rynnvalue-8b"), check_deps=False)

    def test_unverified_diagnostic_requires_explicit_flag(self, tmp_path):
        adapter = _adapter(
            tmp_path,
            fake=FakeSession(progress=[-1.0]),
            allow_unverified_server=True,
        )
        output = adapter.score(_trajectory(n=1))
        assert output.meta["server_identity_verified"] is False

    def test_wrong_model_or_profile_provenance_is_rejected(self, tmp_path):
        wrong_model = _provenance(tmp_path / "4b.json", model_size="4b")
        with pytest.raises(ValueError, match="model_id"):
            _adapter(tmp_path, server_provenance_file=str(wrong_model))

        wrong_profile = _provenance(
            tmp_path / "eager.json", profile="paper-eager"
        )
        with pytest.raises(ValueError, match="profile"):
            _adapter(tmp_path, server_provenance_file=str(wrong_profile))

    def test_non_ready_or_padding_drift_provenance_is_rejected(self, tmp_path):
        stopped = _provenance(tmp_path / "stopped.json")
        record = json.loads(stopped.read_text(encoding="utf-8"))
        record["status"] = "interrupted"
        stopped.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError, match="ready"):
            _adapter(tmp_path, server_provenance_file=str(stopped))

        drifted = _provenance(tmp_path / "drifted.json")
        record = json.loads(drifted.read_text(encoding="utf-8"))
        record["profile"]["pad_frames"] = True
        drifted.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError, match="official profile"):
            _adapter(tmp_path, server_provenance_file=str(drifted))

    def test_paper_eager_profile_requires_matching_server_info(self, tmp_path):
        provenance = _provenance(
            tmp_path / "eager.json", profile="paper-eager"
        )
        fake = FakeSession(model_info=_model_info(attention="eager"))
        adapter = _adapter(
            tmp_path,
            fake=fake,
            profile="paper-eager",
            server_provenance_file=str(provenance),
        )
        output = adapter.score(_trajectory())
        assert output.meta["official_profile"]["effective_attention"] == "eager"

    def test_server_profile_drift_is_rejected_before_scoring(self, tmp_path):
        fake = FakeSession(model_info=_model_info(mode="normalized"))
        adapter = _adapter(tmp_path, fake=fake)
        with pytest.raises(RuntimeError, match="pinned official profile"):
            adapter.score(_trajectory())
        assert fake.post_calls == []

    def test_server_padding_profile_drift_is_rejected_before_scoring(self, tmp_path):
        info = _model_info()
        info["config"]["custom_eval"]["pad_frames"] = True
        fake = FakeSession(model_info=info)
        adapter = _adapter(tmp_path, fake=fake)
        with pytest.raises(RuntimeError, match="pad_frames"):
            adapter.score(_trajectory())
        assert fake.post_calls == []

    def test_wrong_health_identity_is_rejected(self, tmp_path):
        fake = FakeSession(health={"status": "healthy", "reward_model": "roboreward"})
        adapter = _adapter(tmp_path, fake=fake)
        with pytest.raises(RuntimeError, match="/health identity"):
            adapter.score(_trajectory())

    def test_close_releases_http_session(self, tmp_path):
        fake = FakeSession()
        adapter = _adapter(tmp_path, fake=fake)
        adapter.close()
        assert fake.closed is True
        assert adapter._session is None
