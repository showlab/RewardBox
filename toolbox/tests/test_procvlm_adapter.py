"""Offline contracts for the pinned ProcVLM HTTP adapter."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from rewardbox.core.types import SamplingSpec, Trajectory


def _trajectory(length: int = 5, *, indices: np.ndarray | None = None) -> Trajectory:
    frames = np.stack(
        [np.full((8, 10, 3), index * 20, dtype=np.uint8) for index in range(length)]
    )
    return Trajectory(
        frames=frames,
        instruction="close the oven door",
        frame_indices=np.arange(length) if indices is None else indices,
        fps=30.0,
        sampling=SamplingSpec(strategy="all"),
    )


def _provenance(path: Path, *, ready: bool = True, verified: bool = True) -> Path:
    from rewardbox.adapters import procvlm as module

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
            "files": module._WEIGHT_FILES,
            "content_verified": verified,
            "pinned_public_snapshot": verified,
        },
        "profile": module._PROFILE_FIELDS,
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _model_info() -> dict:
    from rewardbox.adapters import procvlm as module

    return {
        "reward_model": "procvlm",
        "upstream_commit": module._UPSTREAM_COMMIT,
        "weights": {
            "repo_id": module._WEIGHTS_REPO,
            "revision": module._WEIGHTS_REVISION,
            "files": module._WEIGHT_FILES,
        },
        "profile": module._PROFILE_FIELDS,
    }


def _score_payload(length: int = 5) -> dict:
    from rewardbox.adapters import procvlm as module

    parsed = [10.0, None, 30.0, None, 50.0][:length]
    if length > 5:
        parsed.extend([50.0] * (length - 5))
    progress: list[float] = []
    previous = 0.0
    records = []
    for index, value in enumerate(parsed):
        if value is None:
            current = previous
            source = "previous"
            output = "unparseable"
        else:
            current = value
            previous = value
            source = "model"
            output = f"reason <progress>{value}%</progress>"
        progress.append(current)
        records.append(
            {
                "sample_index": index,
                "source_position": index,
                "frame_index": index,
                "window_frame_indices": [max(0, index - 7)] * 8,
                "progress": current,
                "parsed_progress": value,
                "progress_source": source,
                "reasoning": "reason" if value is not None else "unparseable",
                "model_output": output,
            }
        )
    return {
        "schema": module._SCORE_SCHEMA,
        "profile": module._PROFILE,
        "source_frame_count": length,
        "source_positions": list(range(length)),
        "frame_indices": list(range(length)),
        "progress": progress,
        "records": records,
        "vllm_batches": 1,
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
        self.health = health or {"status": "healthy", "reward_model": "procvlm"}
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
    from rewardbox.adapters.procvlm import ProcVLMAdapter
    from rewardbox.registry import get_spec

    if "server_provenance_file" not in kwargs and not kwargs.get(
        "allow_unverified_server", False
    ):
        kwargs["server_provenance_file"] = str(_provenance(tmp_path / "server.json"))
    adapter = ProcVLMAdapter(get_spec("procvlm-2b"), check_deps=False, **kwargs)
    adapter._session = fake or FakeSession()
    return adapter


class TestSpecAndTransport:
    def test_registered_spec_is_official_full_episode_percentage_model(self):
        import rewardbox

        spec = rewardbox.get_spec("procvlm-2b")
        assert spec.sampling == SamplingSpec(strategy="all")
        assert spec.required_views == ("primary",)
        assert spec.score_range == (0.0, 100.0)
        assert spec.zero_shot is True
        assert "a7f5c18c6128a9c2a478a62ecc356f19f0e0bcf5" in spec.weights

    def test_manifest_does_not_import_heavy_server_stack(self, monkeypatch):
        import sys

        import rewardbox.registry as registry

        for name in ("torch", "transformers", "vllm", "qwen_vl_utils"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        registry.load_manifest(force=True)
        for name in ("torch", "transformers", "vllm", "qwen_vl_utils"):
            assert name not in sys.modules

    def test_npz_contract_and_official_parse_fallback(self, tmp_path):
        trajectory = _trajectory()
        fake = FakeSession()
        output = _adapter(tmp_path, fake=fake).score(trajectory)

        assert output.progress.tolist() == [10.0, 10.0, 30.0, 30.0, 50.0]
        assert output.frame_indices.tolist() == list(range(5))
        assert output.cost.n_calls == 5
        assert output.meta["n_http_calls"] == 1
        assert output.meta["server_identity_verified"] is True
        assert output.raw["records"][1]["progress_source"] == "previous"

        url, request = fake.post_calls[0]
        assert url.endswith("/score_npz")
        assert request["headers"]["Content-Type"] == "application/x-npz"
        with np.load(io.BytesIO(request["data"]), allow_pickle=False) as archive:
            assert set(archive.files) == {
                "primary", "frame_indices", "instruction", "fps",
            }
            assert np.array_equal(archive["primary"], trajectory.primary)
            assert archive["instruction"].item() == trajectory.instruction
            assert archive["fps"].item() == 30.0

    def test_official_uniform_cap_helper_matches_expected_grid(self):
        from rewardbox.adapters.procvlm import _official_uniform_positions

        positions = _official_uniform_positions(513, 512)
        assert len(positions) == 512
        assert positions[0] == 0 and positions[-1] == 512
        assert positions == sorted(set(positions))


class TestInputAndOutputValidation:
    def test_sparse_or_presampled_episode_is_rejected_before_http(self, tmp_path):
        fake = FakeSession()
        sparse = _trajectory(indices=np.arange(5) * 2)
        with pytest.raises(ValueError, match="complete contiguous episode"):
            _adapter(tmp_path, fake=fake).score(sparse)
        assert fake.get_calls == [] and fake.post_calls == []

        sampled = _trajectory()
        sampled.sampling = SamplingSpec(num_frames=4, strategy="uniform")
        with pytest.raises(ValueError, match="strategy='all'"):
            _adapter(tmp_path, fake=fake).score(sampled)

    @pytest.mark.parametrize(
        "mutate,match",
        (
            (lambda payload: payload.update(schema="wrong"), "schema"),
            (lambda payload: payload.update(frame_indices=list(range(1, 6))), "frame_indices"),
            (lambda payload: payload.update(progress=[101.0] * 5), r"\[0,100\]"),
            (
                lambda payload: payload["records"][1].update(progress_source="model"),
                "previous fallback",
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
        from rewardbox.adapters.procvlm import ProcVLMAdapter
        from rewardbox.registry import get_spec

        with pytest.raises(ValueError, match="server_provenance_file"):
            ProcVLMAdapter(get_spec("procvlm-2b"), check_deps=False)

    def test_non_ready_source_or_weight_drift_is_rejected(self, tmp_path):
        loading = _provenance(tmp_path / "loading.json", ready=False)
        with pytest.raises(ValueError, match="ready"):
            _adapter(tmp_path, server_provenance_file=str(loading))

        unverified = _provenance(tmp_path / "unverified.json", verified=False)
        with pytest.raises(ValueError, match="did not verify the pinned upstream"):
            _adapter(tmp_path, server_provenance_file=str(unverified))

        drifted = _provenance(tmp_path / "drifted.json")
        record = json.loads(drifted.read_text(encoding="utf-8"))
        record["weights"]["files"]["model-00001-of-00002.safetensors"][
            "sha256"
        ] = "0" * 64
        drifted.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError, match="weight identity"):
            _adapter(tmp_path, server_provenance_file=str(drifted))

    def test_server_profile_drift_is_rejected_before_scoring(self, tmp_path):
        info = _model_info()
        info["profile"] = {**info["profile"], "window_size": 4}
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
