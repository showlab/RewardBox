"""Native HTTP transport contracts around the real pinned GVL computation."""
from __future__ import annotations

import json
import random

import numpy as np
import pytest
requests = pytest.importorskip("requests")

from rewardbox.adapters import gvl
from rewardbox.core.enums import ImplementationKind
from rewardbox.core.types import SamplingSpec, Trajectory
from rewardbox.registry import create_reward, get_spec


def _trajectory(n=8):
    return Trajectory(
        frames=np.stack([np.full((4, 5, 3), i * 20, np.uint8) for i in range(n)]),
        instruction="move the block",
        frame_indices=np.arange(n, dtype=np.int64) * 10 + 100,
        sampling=gvl.HTTP_SPEC.sampling,
    )


class _Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status
        self.closed = False

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


def _answer(client, *, omit_gt=()):
    return json.dumps([
        {
            "frame_number": frame["shuffled_index"],
            "frame_description": "offline transport test",
            "task_completion_percentage": frame["gt_index"] * 10,
        }
        for frame in client.frames_info if frame["gt_index"] not in omit_gt
    ])


def _payload(answer):
    return {"candidates": [{"content": {"parts": [{"text": answer}]}, "finishReason": "STOP"}]}


def test_native_transport_preserves_actual_gvl_prompt_shuffle_and_sparse_mapping(monkeypatch):
    secret = "gemini-test-key"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-send-this-openai-key")
    adapter = create_reward(gvl.HTTP_SPEC.name, check_deps=False)
    client = adapter.client()
    upstream = gvl._upstream_gvl_class()
    monkeypatch.setattr(
        upstream, "_stream_inference_gemini",
        lambda *args: pytest.fail("must not use the upstream SSE transport"),
    )
    calls = []
    responses = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        assert kwargs["json"]["contents"][0]["parts"] == client.build_prompt_parts("move the block")
        answer = _answer(client, omit_gt=(2,))
        payload = _payload("")
        payload["candidates"][0]["content"]["parts"] = [
            {"thought": True, "text": "this reasoning is not the JSON answer"},
            {"text": "```json\n" + answer[:len(answer) // 2]},
            {"text": answer[len(answer) // 2:] + "\n```"},
        ]
        payload.update({
            "modelVersion": "gemini-3.8-flash-provider-revision",
            "responseId": "http-response-123",
            "usageMetadata": {"promptTokenCount": 123, "totalTokenCount": 456},
        })
        response = _Response(payload)
        responses.append(response)
        return response

    monkeypatch.setattr(requests, "post", fake_post)
    out = adapter._score(_trajectory())

    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
    assert kwargs["headers"] == {"Content-Type": "application/json", "x-goog-api-key": secret}
    assert kwargs["allow_redirects"] is False
    assert kwargs["timeout"] == (10.0, 180.0)
    assert set(kwargs["json"]) == {"contents"}  # No added generation settings.
    parts = kwargs["json"]["contents"][0]["parts"]
    assert sum("inline_data" in part for part in parts) == 9  # Initial anchor + 8 frames.
    assert all(response.closed for response in responses)
    assert out.progress.tolist() == pytest.approx([.1, .3, .4, .5, .6, .7, .8])
    assert out.frame_indices.tolist() == [100, 120, 130, 140, 150, 160, 170]
    assert out.meta["imputed_missing_predictions"] is False
    assert out.meta["provider"] == "gemini"
    assert out.meta["model_name"] == "gemini-3.8-flash"
    assert out.meta["api_protocol"] == "gemini_native"
    assert out.meta["api_method"] == "generateContent"
    assert out.meta["upstream_transport_reused"] is False
    assert out.meta["credential_source"] == "GEMINI_API_KEY_environment"
    assert out.meta["base_url"] == "https://generativelanguage.googleapis.com"
    assert out.meta["auth_header"] == "x-goog-api-key"
    assert out.meta["provider_reported_model_version"] == "gemini-3.8-flash-provider-revision"
    assert out.meta["provider_response_id"] == "http-response-123"
    assert out.meta["provider_usage_metadata"]["totalTokenCount"] == 456
    assert out.meta["n_http_attempts"] == 1
    assert out.meta["provider_managed_model_revision_pinned"] is False
    assert adapter.spec.implementation_kind is ImplementationKind.REWARDBOX_EXTENSION
    serialized = json.dumps(out.to_dict())
    assert secret not in serialized
    assert "do-not-send-this-openai-key" not in serialized
    assert "base64" not in serialized
    adapter.close()
    assert adapter._client is None
    assert adapter._http_transport is None


def test_gateway_options_send_only_the_named_key_as_bearer(monkeypatch):
    monkeypatch.setenv("MY_GATEWAY_KEY", "gateway-key")
    monkeypatch.setenv("GEMINI_API_KEY", "do-not-send-this-google-key")
    adapter = create_reward(
        gvl.HTTP_SPEC.name, base_url="https://gateway.example/v1beta",
        api_key_env="MY_GATEWAY_KEY", auth_header="bearer", check_deps=False,
    )
    client = adapter.client()
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return _Response(_payload(_answer(client)))

    monkeypatch.setattr(requests, "post", fake_post)
    out = adapter._score(_trajectory())
    url, kwargs = calls[0]
    assert url == "https://gateway.example/v1beta/models/gemini-3.8-flash:generateContent"
    assert kwargs["headers"] == {"Content-Type": "application/json", "Authorization": "Bearer gateway-key"}
    assert out.meta["base_url"] == "https://gateway.example"
    assert out.meta["auth_header"] == "bearer"
    assert out.meta["credential_source"] == "MY_GATEWAY_KEY_environment"
    assert "do-not-send-this-google-key" not in json.dumps(out.to_dict())


@pytest.mark.parametrize("opts", [{"auth_header": "x-api-key"}, {"api_key_env": ""}])
def test_unknown_auth_header_or_empty_key_variable_is_rejected(opts):
    with pytest.raises(ValueError, match=next(iter(opts))):
        create_reward(gvl.HTTP_SPEC.name, check_deps=False, **opts)


def test_key_is_required_without_using_other_provider_credentials(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    adapter = create_reward(gvl.HTTP_SPEC.name, check_deps=False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("no HTTP without a key"))
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        adapter._score(_trajectory())


@pytest.mark.parametrize("n", [7, 9])
def test_http_profile_rejects_non_exact_grid_before_http(monkeypatch, n):
    adapter = create_reward(gvl.HTTP_SPEC.name, api_key="offline-key", check_deps=False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("no HTTP for invalid input"))
    with pytest.raises(ValueError, match="exactly 8 frames"):
        adapter._score(_trajectory(n))


def test_transient_http_errors_retry_without_recomputing_gvl_parts(monkeypatch):
    adapter = create_reward(gvl.HTTP_SPEC.name, api_key="offline-key", check_deps=False)
    client = adapter.client()
    calls = []
    delays = []

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        if len(calls) < 3:
            return _Response({}, status=(429 if len(calls) == 1 else 503))
        return _Response(_payload(_answer(client)))

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(gvl.time, "sleep", delays.append)
    out = adapter._score(_trajectory())
    assert len(calls) == 3
    assert calls[0] == calls[1] == calls[2]
    assert delays == [1.0, 2.0]
    assert out.meta["n_http_attempts"] == 3


@pytest.mark.upstream("robometer")
@pytest.mark.parametrize("status", [400, 401, 403, 404, 302, 503])
def test_http_failures_are_bounded_and_redacted(monkeypatch, capsys, status):
    secret = "private-http-key"
    adapter = create_reward(gvl.HTTP_SPEC.name, api_key=secret, check_deps=False)
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        return _Response({"error": {"message": f"rejected Bearer {secret}"}}, status=status)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(gvl.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match=f"Gemini API HTTP {status}") as exc:
        adapter._score(_trajectory())
    assert len(calls) == (6 if status == 503 else 1)
    captured = capsys.readouterr()
    assert secret not in str(exc.value) + captured.out + captured.err


@pytest.mark.parametrize("payload", [
    ValueError("not JSON"),
    [],
    {"choices": [{"message": {"content": "OpenAI format is not native Gemini"}}]},
    {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "[]"}]}}]},
    {"candidates": [{"content": {"parts": [{"text": "reasoning only", "thought": True}]}}]},
])
@pytest.mark.upstream("robometer")
def test_invalid_or_incomplete_native_responses_never_become_scores(monkeypatch, payload):
    adapter = create_reward(gvl.HTTP_SPEC.name, api_key="offline-key", check_deps=False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Response(payload))
    with pytest.raises(RuntimeError, match="Gemini API"):
        adapter._score(_trajectory())


@pytest.mark.parametrize("base_url", [
    "http://gateway.example", "https://secret@gateway.example",
    "https://gateway.example?key=secret", "https://gateway.example/#secret",
    "https://gateway.example/v1",
])
def test_invalid_native_base_urls_are_rejected_without_echoing_credentials(base_url):
    with pytest.raises(ValueError) as exc:
        create_reward(gvl.HTTP_SPEC.name, base_url=base_url, check_deps=False)
    assert "secret" not in str(exc.value)


def test_native_base_url_can_include_v1beta():
    adapter = create_reward(
        gvl.HTTP_SPEC.name, base_url="https://gateway.example/v1beta/", check_deps=False,
    )
    assert adapter.base_url == "https://gateway.example"


def _dense_trajectory(total=3271, sampling=None):
    sampling = sampling or gvl.DENSE_HTTP_SPEC.sampling
    indices = sampling.indices(total, source_fps=30.0)
    return Trajectory(
        frames=np.stack([np.full((4, 5, 3), i % 256, np.uint8) for i in range(len(indices))]),
        instruction="move the block",
        frame_indices=indices, fps=30.0, sampling=sampling,
        source="private-source-with-chronological-filename.mp4",
    )


def _dense_answer(client, omit_gt=()):
    return json.dumps([
        {"frame_number": frame["shuffled_index"], "frame_description": "offline dense test",
         "task_completion_percentage": 100 * (frame["gt_index"] - 1) / max(1, len(client.frames_info) - 1)}
        for frame in client.frames_info if frame["gt_index"] not in omit_gt
    ])


@pytest.mark.parametrize("profile", [gvl.DENSE_HTTP_SPEC.name])
@pytest.mark.parametrize("fps,n", [(1.0, 110), (2.0, 219)])
def test_dense_long_video_preserves_native_pipeline_in_one_globally_shuffled_request(monkeypatch, fps, n, profile):
    # This synthetic grid covers 109 seconds without decoding any benchmark video.
    sampling = SamplingSpec(strategy="fps", fps=fps, include_last=False)
    traj = _dense_trajectory(sampling=sampling)
    adapter = create_reward(profile, api_key="offline-key", check_deps=False)
    monkeypatch.setattr(random, "shuffle", lambda indices: indices.reverse())
    calls = []

    def fake_post(url, **kwargs):
        assert url == f"https://generativelanguage.googleapis.com/v1beta/models/{gvl.provider_model_for_profile(profile)}:generateContent"
        calls.append(kwargs["json"])
        return _Response(_payload(_dense_answer(adapter.client(), omit_gt=(2,))))

    monkeypatch.setattr(requests, "post", fake_post)
    out = adapter._score(traj)

    # An independent invocation of the unmodified upstream computation is the
    # oracle for the entire image/prompt/parser contract, including sparse output.
    reference = gvl._upstream_gvl_class()(api_key="offline-key", max_frames=n)
    reference_parts = []

    def reference_inference(parts):
        reference_parts.append(parts)
        return _dense_answer(reference, omit_gt=(2,))

    reference.stream_inference = reference_inference
    expected = reference.compute_progress(traj.primary, task_description=traj.instruction)
    assert len(traj) == n > 15
    assert len(calls) == 1
    assert calls[0] == {"contents": [{"parts": reference_parts[0]}]}
    parts = reference_parts[0]
    assert sum("inline_data" in part for part in parts) == n + 1
    assert traj.source not in json.dumps(parts)
    assert [frame["shuffled_index"] for frame in adapter.client().frames_info] == list(range(n, 0, -1))
    assert out.progress.tolist() == pytest.approx(expected)
    assert out.frame_indices.tolist() == np.delete(traj.frame_indices, 1).tolist()
    assert out.raw["sampled_frame_indices"] == traj.frame_indices.tolist()
    assert out.raw["returned_timestamps_s"] == (out.frame_indices / 30.0).tolist()
    assert out.meta["official_args"]["max_frames"] == n
    assert out.meta["upstream_secondary_subsampling"] is False
    assert out.meta["chronological_metadata_sent_to_model"] is False
    assert out.meta["chunking"] is False
    assert out.meta["protocol"] == "gvl_time_sampled_globally_shuffled_v1"


def test_dense_reuses_one_client_without_capping_later_videos_or_padding_short_ones(monkeypatch):
    adapter = create_reward(gvl.DENSE_HTTP_SPEC.name, api_key="offline-key", check_deps=False)
    counts = []

    def fake_post(url, **kwargs):
        client = adapter.client()
        counts.append((id(client), len(client.frames_info)))
        return _Response(_payload(_dense_answer(client)))

    monkeypatch.setattr(requests, "post", fake_post)
    for total, expected_count in [(1, 1), (181, 7), (3271, 110), (61, 3)]:
        traj = _dense_trajectory(total)
        out = adapter._score(traj)
        assert len(out.frame_indices) == expected_count
        assert out.frame_indices.tolist() == traj.frame_indices.tolist()
        assert adapter.client().max_frames == expected_count
    assert len({client_id for client_id, count in counts}) == 1
    assert [count for client_id, count in counts] == [1, 7, 110, 3]


@pytest.mark.parametrize("sampling", [
    SamplingSpec(strategy="interval", interval=10),
    SamplingSpec(strategy="all"),
    SamplingSpec(num_frames=8, strategy="fps", fps=2.0),
])
def test_dense_uses_the_runners_complete_grid_without_a_hidden_budget(monkeypatch, sampling):
    adapter = create_reward(gvl.DENSE_HTTP_SPEC.name, api_key="offline-key", check_deps=False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Response(_payload(_dense_answer(adapter.client()))))
    traj = _dense_trajectory(311, sampling)
    out = adapter._score(traj)
    assert out.frame_indices.tolist() == traj.frame_indices.tolist()
    assert len(out.frame_indices) > 15


def test_dense_default_matches_existing_sole_sampling_and_has_its_own_identity():
    assert gvl.DENSE_HTTP_SPEC.sampling == get_spec("sole-r1-8b").sampling
    assert gvl.DENSE_HTTP_SPEC.name != gvl.HTTP_SPEC.name
    assert gvl.credential_env_for_profile(gvl.DENSE_HTTP_SPEC.name) == "GEMINI_API_KEY"


@pytest.mark.parametrize("invalid", ["fixed_count", "missing_fps", "initial_anchor", "grid"])
def test_dense_invalid_grid_fails_before_an_api_call(monkeypatch, invalid):
    adapter = create_reward(gvl.DENSE_HTTP_SPEC.name, api_key="offline-key", check_deps=False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("no HTTP for an invalid grid"))
    traj = _dense_trajectory(181)
    if invalid == "fixed_count":
        traj.sampling = SamplingSpec(num_frames=8)
    elif invalid == "missing_fps":
        traj.fps = None
    elif invalid == "initial_anchor":
        traj.frame_indices += 1
    else:
        traj.frame_indices[2] += 1
    with pytest.raises(ValueError, match="Dense GVL"):
        adapter._score(traj)
