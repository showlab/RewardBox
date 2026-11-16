"""Offline contracts for the pinned Robo-ValueRL HTTP adapter."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from rewardbox.core.types import SamplingSpec, Trajectory


def _trajectory(length: int = 12, *, indices: np.ndarray | None = None) -> Trajectory:
    def frames(offset: int) -> np.ndarray:
        return np.stack(
            [
                np.full((8, 10, 3), (offset + index * 7) % 256, dtype=np.uint8)
                for index in range(length)
            ]
        )

    return Trajectory(
        frames={
            "primary": frames(0),
            "left_wrist": frames(30),
            "right_wrist": frames(60),
        },
        instruction="sort the blocks by color",
        frame_indices=np.arange(length) if indices is None else indices,
        sampling=SamplingSpec(strategy="all"),
    )


def _provenance(path: Path, *, ready: bool = True, verified: bool = True) -> Path:
    from rewardbox.adapters import robovaluerl as module

    record = {
        "schema": module._PROVENANCE_SCHEMA,
        "status": "ready" if ready else "loading",
        "upstream": {
            "commit": module._UPSTREAM_COMMIT,
            "source_verified": verified,
        },
        "weights": {
            "repo_id": module._WEIGHTS_REPO,
            "expected_revision": module._WEIGHTS_REVISION,
            "snapshot_revision": module._WEIGHTS_REVISION if verified else None,
            "filename": module._WEIGHTS_FILE,
            "sha256": module._WEIGHTS_SHA256,
            "size_bytes": module._WEIGHTS_SIZE,
            "content_verified": verified,
            "pinned_public_snapshot": verified,
        },
        "profile": module._PROFILE_FIELDS,
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _model_info() -> dict:
    from rewardbox.adapters import robovaluerl as module

    return {
        "reward_model": "robo-valuerl",
        "upstream_commit": module._UPSTREAM_COMMIT,
        "weights": {
            "repo_id": module._WEIGHTS_REPO,
            "revision": module._WEIGHTS_REVISION,
            "filename": module._WEIGHTS_FILE,
            "sha256": module._WEIGHTS_SHA256,
        },
        "profile": module._PROFILE_FIELDS,
    }


def _score_payload(length: int = 12, *, remaining: list[float] | None = None) -> dict:
    from rewardbox.adapters import robovaluerl as module

    if remaining is None:
        remaining = np.linspace(4500.0, 500.0, length).tolist()
    key_indices = list(range(0, length, 10))
    if key_indices[-1] != length - 1:
        key_indices.append(length - 1)
    probabilities = np.zeros((len(key_indices), 256), dtype=np.float64)
    probabilities[:, 20] = 1.0
    return {
        "schema": module._SCORE_SCHEMA,
        "profile": module._PROFILE,
        "frame_indices": list(range(length)),
        "progress": np.clip(1.0 - np.asarray(remaining) / 5000.0, 0.0, 1.0).tolist(),
        "remaining_steps": remaining,
        "key_frame_indices": key_indices,
        "key_remaining_steps": [392.156862745] * len(key_indices),
        "key_probabilities": probabilities.tolist(),
        "model_forwards": len(key_indices),
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
    def __init__(self, *, score=None, model_info=None, health=None):
        self.score = score or _score_payload()
        self.info = model_info or _model_info()
        self.health = health or {
            "status": "healthy",
            "reward_model": "robo-valuerl",
        }
        self.get_calls = []
        self.post_calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return FakeResponse(self.health if url.endswith("/health") else self.info)

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return FakeResponse(self.score)

    def close(self):
        self.closed = True


def _adapter(tmp_path: Path, *, fake=None, **kwargs):
    from rewardbox.adapters.robovaluerl import RoboValueRLAdapter
    from rewardbox.registry import get_spec

    if "server_provenance_file" not in kwargs and not kwargs.get(
        "allow_unverified_server", False
    ):
        kwargs["server_provenance_file"] = str(_provenance(tmp_path / "server.json"))
    adapter = RoboValueRLAdapter(
        get_spec("robo-valuerl-value"), check_deps=False, **kwargs
    )
    adapter._session = fake or FakeSession()
    return adapter


class TestSpecAndTransport:
    def test_registered_spec_is_contiguous_three_view_official_adapter(self):
        import rewardbox

        spec = rewardbox.get_spec("robo-valuerl-value")
        assert spec.sampling == SamplingSpec(strategy="all")
        assert spec.required_views == ("primary", "left_wrist", "right_wrist")
        assert spec.score_range == (0.0, 1.0)
        assert spec.zero_shot is False
        assert "b44f33eaa5892233174d596cc7d8a7e685967eef" in spec.weights

    def test_manifest_does_not_import_official_heavy_stack(self, monkeypatch):
        import sys
        import rewardbox.registry as registry

        for name in ("torch", "transformers", "jax", "flax", "albumentations"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        registry.load_manifest(force=True)
        for name in ("torch", "transformers", "jax", "flax", "albumentations"):
            assert name not in sys.modules

    def test_npz_contract_progress_mapping_and_raw_distribution(self, tmp_path):
        fake = FakeSession()
        trajectory = _trajectory()
        output = _adapter(tmp_path, fake=fake).score(trajectory)

        assert np.allclose(output.progress, np.linspace(0.1, 0.9, 12))
        assert output.frame_indices.tolist() == list(range(12))
        assert output.cost.n_calls == 3
        assert output.meta["n_http_calls"] == 1
        assert output.meta["server_identity_verified"] is True
        assert np.asarray(output.raw["key_probabilities"]).shape == (3, 256)

        url, request = fake.post_calls[0]
        assert url.endswith("/score_npz")
        assert request["headers"]["Content-Type"] == "application/x-npz"
        with np.load(io.BytesIO(request["data"]), allow_pickle=False) as archive:
            assert set(archive.files) == {
                "primary",
                "left_wrist",
                "right_wrist",
                "frame_indices",
                "instruction",
            }
            assert np.array_equal(archive["primary"], trajectory.view("primary"))
            assert np.array_equal(archive["left_wrist"], trajectory.view("left_wrist"))
            assert archive["instruction"].item() == trajectory.instruction

    def test_single_frame_has_one_key_frame(self, tmp_path):
        fake = FakeSession(score=_score_payload(length=1, remaining=[2500.0]))
        output = _adapter(tmp_path, fake=fake).score(_trajectory(length=1))
        assert output.progress.tolist() == [0.5]
        assert output.cost.n_calls == 1


class TestInputAndOutputValidation:
    def test_missing_view_is_rejected_before_http(self, tmp_path):
        trajectory = _trajectory()
        trajectory.frames.pop("right_wrist")
        fake = FakeSession()
        with pytest.raises(ValueError, match="needs views"):
            _adapter(tmp_path, fake=fake).score(trajectory)
        assert fake.get_calls == [] and fake.post_calls == []

    def test_sparse_or_shifted_episode_is_not_silently_treated_as_history(self, tmp_path):
        fake = FakeSession()
        trajectory = _trajectory(indices=np.arange(12) * 2)
        with pytest.raises(ValueError, match="complete contiguous episode"):
            _adapter(tmp_path, fake=fake).score(trajectory)
        assert fake.get_calls == [] and fake.post_calls == []

        trajectory = _trajectory()
        trajectory.sampling = SamplingSpec(num_frames=8, strategy="uniform")
        with pytest.raises(ValueError, match="strategy='all'"):
            _adapter(tmp_path, fake=fake).score(trajectory)

    @pytest.mark.parametrize(
        "mutate,match",
        (
            (lambda payload: payload.update(schema="wrong"), "schema"),
            (
                lambda payload: payload.update(frame_indices=list(range(1, 13))),
                "frame_indices",
            ),
            (
                lambda payload: payload.update(progress=[0.2] * 12),
                r"clip\(1-remaining_steps/5000",
            ),
            (
                lambda payload: payload.update(key_probabilities=[[0.0] * 256] * 3),
                "softmax",
            ),
        ),
    )
    def test_malformed_server_output_is_rejected(self, tmp_path, mutate, match):
        payload = _score_payload()
        mutate(payload)
        with pytest.raises(RuntimeError, match=match):
            _adapter(tmp_path, fake=FakeSession(score=payload)).score(_trajectory())


class TestIdentity:
    def test_provenance_is_required_by_default(self):
        from rewardbox.adapters.robovaluerl import RoboValueRLAdapter
        from rewardbox.registry import get_spec

        with pytest.raises(ValueError, match="server_provenance_file"):
            RoboValueRLAdapter(get_spec("robo-valuerl-value"), check_deps=False)

    def test_non_ready_source_or_weight_drift_is_rejected(self, tmp_path):
        loading = _provenance(tmp_path / "loading.json", ready=False)
        with pytest.raises(ValueError, match="ready"):
            _adapter(tmp_path, server_provenance_file=str(loading))

        unverified = _provenance(tmp_path / "unverified.json", verified=False)
        with pytest.raises(ValueError, match="did not verify the pinned upstream"):
            _adapter(tmp_path, server_provenance_file=str(unverified))

        drifted = _provenance(tmp_path / "drifted.json")
        record = json.loads(drifted.read_text(encoding="utf-8"))
        record["weights"]["sha256"] = "0" * 64
        drifted.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError, match="weight identity"):
            _adapter(tmp_path, server_provenance_file=str(drifted))

    def test_server_profile_drift_is_rejected_before_scoring(self, tmp_path):
        info = _model_info()
        info["profile"] = {**info["profile"], "history_length": 30}
        fake = FakeSession(model_info=info)
        with pytest.raises(RuntimeError, match="official profile"):
            _adapter(tmp_path, fake=fake).score(_trajectory())
        assert fake.post_calls == []

    def test_explicit_unverified_diagnostic_is_marked(self, tmp_path):
        output = _adapter(
            tmp_path,
            fake=FakeSession(),
            allow_unverified_server=True,
        ).score(_trajectory())
        assert output.meta["server_identity_verified"] is False

    def test_close_releases_session(self, tmp_path):
        fake = FakeSession()
        adapter = _adapter(tmp_path, fake=fake)
        adapter.close()
        assert fake.closed is True
