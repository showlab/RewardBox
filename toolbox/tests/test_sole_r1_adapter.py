"""Offline contract tests for the official SOLE-R1 reward-server adapter.

The tests replace only the transport client.  They do not claim model
calibration: a real run additionally needs the official 8B checkpoint and
server environment on a suitable GPU.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from rewardbox.adapters import sole_r1
from rewardbox.core.types import SamplingSpec, Trajectory


def _frames(n: int, offset: int = 0) -> np.ndarray:
    return np.stack(
        [np.full((5, 7, 3), i + offset, dtype=np.uint8) for i in range(n)]
    )


def _trajectory(n: int = 4, wrist: bool = False) -> Trajectory:
    views = {"primary": _frames(n)}
    if wrist:
        views["left_wrist"] = _frames(n, offset=50)
    return Trajectory(
        frames=views,
        instruction="open the drawer",
        frame_indices=np.arange(n, dtype=np.int64) * 10 + 100,
        sampling=sole_r1.SPEC.sampling,
    )


def _response(progress=(0, 10, 20, 30)) -> dict:
    n = len(progress)
    return {
        "success": True,
        "data": {
            "valid_answers": [np.asarray(progress)],
            "text_outputs": [[f"reasoning-{i}" for i in range(n - 1)]],
            "text_inputs": [[f"prompt-{i}" for i in range(n - 1)]],
        },
    }


_DEFAULT_RESPONSE = object()


class _FakeClient:
    def __init__(self, response=_DEFAULT_RESPONSE):
        self.response = _response() if response is _DEFAULT_RESPONSE else response
        self.calls = []
        self.closed = False

    def call(self, name, payload):
        self.calls.append((name, payload))
        return self.response

    def close(self):
        self.closed = True


def _adapter(**kwargs) -> sole_r1.SoleR1Adapter:
    return sole_r1.SoleR1Adapter(sole_r1.SPEC, check_deps=False, **kwargs)


def test_spec_matches_official_client_sampling_and_server_boundary():
    assert sole_r1.SPEC.name == "sole-r1-8b"
    assert sole_r1.SPEC.weights == "Philip-MIT/SOLE-R1-8B"
    assert sole_r1.SPEC.sampling == SamplingSpec(
        strategy="fps", fps=1.0, include_last=False
    )
    assert sole_r1.SPEC.score_range == (-100.0, 100.0)
    assert sole_r1.SPEC.required_views == ("primary",)
    assert sole_r1.SPEC.optional_views == ("left_wrist",)


def test_constructs_the_official_client_with_connection_options(monkeypatch):
    calls = []

    class FakeOfficialClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(sole_r1, "_upstream_client_class", lambda: FakeOfficialClient)
    adapter = _adapter(
        host="reward-host",
        port=8123,
        timeout_ms=12345,
        wait_for_server=True,
    )

    assert isinstance(adapter.client(), FakeOfficialClient)
    assert calls == [{
        "server_ip": "reward-host",
        "port_num": 8123,
        "timeout_ms": 12345,
        "wait_for_server": True,
    }]


def test_external_only_payload_and_output_mapping_are_exact():
    adapter = _adapter(server_provenance="Philip-MIT/SOLE-R1-8B@test-revision")
    client = _FakeClient()
    adapter._client = client
    traj = _trajectory()

    out = adapter.score(traj)

    assert len(client.calls) == 1
    service, payload = client.calls[0]
    assert service == "rewards"
    assert payload["task"] == "open the drawer"
    assert payload["external_only"] is True
    assert payload["from_zero"] is False
    assert payload["temperature"] == 1.0
    assert len(payload["front_images"]) == len(payload["wrist_images"]) == 1
    assert len(payload["front_images"][0]) == len(traj)
    for front, wrist in zip(
        payload["front_images"][0], payload["wrist_images"][0], strict=True,
    ):
        assert front is wrist
        assert front.flags.c_contiguous

    assert out.progress.tolist() == [0.0, 10.0, 20.0, 30.0]
    assert out.frame_indices.tolist() == [100, 110, 120, 130]
    assert out.raw["text_outputs"] == ["reasoning-0", "reasoning-1", "reasoning-2"]
    assert out.raw["text_inputs"] == ["prompt-0", "prompt-1", "prompt-2"]
    assert out.meta["protocol"] == "official_reward_server_zmq"
    assert out.meta["official_args"]["external_only"] is True
    assert out.meta["server_checkpoint_provenance"].endswith("@test-revision")
    assert out.meta["n_transport_calls"] == 1
    assert out.cost.n_calls == 3


def test_left_wrist_is_sent_without_modifying_or_stacking_views():
    adapter = _adapter(from_zero=True, temperature=0.0)
    client = _FakeClient()
    adapter._client = client
    traj = _trajectory(wrist=True)

    out = adapter.score(traj)

    payload = client.calls[0][1]
    assert payload["external_only"] is False
    assert payload["from_zero"] is True
    assert payload["temperature"] == 0.0
    for got, expected in zip(
        payload["front_images"][0], traj.view("primary"), strict=True,
    ):
        assert np.array_equal(got, expected)
    for got, expected in zip(
        payload["wrist_images"][0], traj.view("left_wrist"), strict=True,
    ):
        assert np.array_equal(got, expected)
    assert out.meta["wrist_view"] == "left_wrist"


@pytest.mark.parametrize(
    "response,match",
    (
        (None, "returned no response"),
        ({"success": False, "message": "server boom"}, "server boom"),
        ({"success": True}, "data dict"),
        (_response((0, 10, 20)), "maps one to one"),
        (_response((1, 10, 20, 30)), "first-timestep baseline"),
        (_response((0, 10, 20, 101)), r"\[-100, 100\]"),
        (_response((0, 10, np.nan, 30)), "NaN/Inf"),
        (_response((0, 10, np.inf, 30)), "NaN/Inf"),
    ),
)
def test_malformed_or_failed_server_response_is_rejected(response, match):
    adapter = _adapter()
    adapter._client = _FakeClient(response=response)
    with pytest.raises(RuntimeError, match=match):
        adapter.score(_trajectory())


def test_reasoning_length_drift_is_rejected():
    response = _response()
    response["data"]["text_outputs"] = [["too short"]]
    adapter = _adapter()
    adapter._client = _FakeClient(response=response)

    with pytest.raises(RuntimeError, match="text_outputs length"):
        adapter.score(_trajectory())


def test_multi_episode_response_is_rejected():
    response = _response()
    for key in ("valid_answers", "text_outputs", "text_inputs"):
        response["data"][key].append(response["data"][key][0])
    adapter = _adapter()
    adapter._client = _FakeClient(response=response)

    with pytest.raises(RuntimeError, match="is not a single-episode batch"):
        adapter.score(_trajectory())


def test_structured_server_provenance_file_is_preserved(tmp_path):
    provenance = {
        "checkpoint": {"revision": sole_r1._CHECKPOINT_REVISION},
        "startup": {"cwd": "/server", "command": ["uv", "run"]},
    }
    path = tmp_path / "server.json"
    path.write_text(json.dumps(provenance), encoding="utf-8")
    adapter = _adapter(server_provenance_file=str(path))
    adapter._client = _FakeClient()

    out = adapter.score(_trajectory())

    assert out.meta["server_checkpoint_provenance"] == provenance
    assert out.meta["server_provenance_file"] == str(path.resolve())


@pytest.mark.parametrize(
    "kwargs,match",
    (
        ({"host": " "}, "host"),
        ({"port": 0}, "port"),
        ({"timeout_ms": 0}, "timeout_ms"),
        ({"service_name": ""}, "service_name"),
        ({"temperature": -0.1}, "temperature"),
        ({"wrist_view": ""}, "wrist_view"),
        ({"server_provenance": ""}, "server_provenance"),
        ({"server_provenance_file": ""}, "server_provenance_file"),
        ({"server_provenance": "x", "server_provenance_file": "y"}, "cannot both be given"),
    ),
)
def test_invalid_runtime_options_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _adapter(**kwargs)


def test_invalid_server_provenance_file_is_rejected(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="does not exist"):
        _adapter(server_provenance_file=str(missing))

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty JSON object"):
        _adapter(server_provenance_file=str(malformed))


def test_close_delegates_when_supported_and_drops_client():
    adapter = _adapter()
    client = _FakeClient()
    adapter._client = client

    adapter.close()

    assert client.closed is True
    assert adapter._client is None
