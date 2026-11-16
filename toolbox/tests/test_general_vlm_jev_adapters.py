"""Offline contracts of the general-VLM (Sequence / Pair) and Jev adapters.

Only ``requests.post`` is faked; prompt construction, image encoding, wire formats,
answer parsing and the benchmark path are the real code.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

requests = pytest.importorskip("requests")

import rewardbox
from rewardbox.adapters import general_vlm as gv
from rewardbox.adapters import jev
from rewardbox.cli.main import EXIT_USAGE_ERROR, main
from rewardbox.core.enums import ImplementationKind
from rewardbox.core.types import SamplingSpec, Trajectory
from rewardbox.registry import validate_model_opts

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "datasets" / "local_example" / "manifest.jsonl"
SECRET = "test-key-must-not-leak"
JEV_URL = "https://jev.example"
NAMES = ("vlm-sequence", "vlm-pair", "jev-1.14-sequence", "jev-1.14-pair")


def _trajectory(n=4, views=("primary",)):
    frames = {v: np.stack([np.full((6, 8, 3), 10 * i + k, np.uint8) for i in range(n)])
              for k, v in enumerate(views)}
    return Trajectory(
        frames=frames if len(views) > 1 else frames[views[0]],
        instruction="put the cup on the coaster",
        frame_indices=np.arange(n, dtype=np.int64) * 30,
        fps=30.0,
        sampling=gv.SEQUENCE_SPEC.sampling,
    )


class _Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        if isinstance(self.payload, str):
            raise ValueError("not JSON")
        return self.payload


class _FakePost:
    """Records every request and answers from a callable ``reply(url, body) -> _Response``."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, url, headers=None, json=None, timeout=None, allow_redirects=None):
        self.calls.append({"url": url, "headers": headers, "body": json,
                           "allow_redirects": allow_redirects})
        return self.reply(url, json)


@pytest.fixture
def post(monkeypatch):
    def install(reply):
        fake = _FakePost(reply)
        monkeypatch.setattr(requests, "post", fake)
        monkeypatch.setattr(gv.time, "sleep", lambda seconds: None)
        return fake
    return install


def _openai(text, finish="stop"):
    return _Response({"id": "r1", "model": "served-model", "usage": {"total_tokens": 7},
                      "choices": [{"finish_reason": finish, "message": {"content": text}}]})


def _progress_json(n):
    return json.dumps({"progress": [10 * i for i in range(n)]})


# ---------------------------------------------------------------- registry
def test_profiles_are_registered_as_extensions_on_the_1fps_grid():
    for name in NAMES:
        spec = rewardbox.get_spec(name)
        assert spec.implementation_kind is ImplementationKind.REWARDBOX_EXTENSION
        assert spec.sampling == SamplingSpec(strategy="fps", fps=1.0, include_last=False)
        assert spec.required_views == ("primary",)
        assert spec.optional_views == ("left_wrist", "right_wrist")
    assert rewardbox.get_spec("vlm-sequence").score_range == (0.0, 100.0)
    assert rewardbox.get_spec("jev-1.14-pair").score_range == (-1.0, 1.0)


def test_only_named_options_are_accepted():
    validate_model_opts("vlm-pair", {"model": "m", "api": "gemini", "max_concurrency": 8})
    validate_model_opts("vlm-sequence", {"model": "m", "extra_body": {"temperature": 0}})
    with pytest.raises(TypeError, match="max_concurrency"):
        validate_model_opts("vlm-sequence", {"max_concurrency": 8})
    with pytest.raises(TypeError, match="model"):
        validate_model_opts("jev-1.14-sequence", {"model": "other"})


def test_model_id_and_option_values_are_checked_at_construction():
    with pytest.raises(ValueError, match="model=NAME"):
        rewardbox.create_reward("vlm-sequence")
    with pytest.raises(ValueError, match="api must be one of"):
        rewardbox.create_reward("vlm-sequence", model="m", api="cohere")
    with pytest.raises(ValueError, match="must not set"):
        rewardbox.create_reward("vlm-sequence", model="m", extra_body={"messages": []})
    with pytest.raises(ValueError, match="base_url"):
        rewardbox.create_reward("vlm-sequence", model="m", base_url="https://user:pw@host/v1")


# ---------------------------------------------------------------- Sequence, three wire formats
def test_sequence_openai_request_layout_and_output(post, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    fake = post(lambda url, body: _openai(_progress_json(4)))
    traj = _trajectory(4, views=("left_wrist", "primary"))
    rm = rewardbox.create_reward("vlm-sequence", model="some-vlm",
                                 extra_body={"reasoning_effort": "low"})
    out = rm.score(traj)

    call = fake.calls[0]
    assert call["url"] == "https://api.openai.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert call["allow_redirects"] is False
    body = call["body"]
    assert body["model"] == "some-vlm" and body["reasoning_effort"] == "low"
    system, user = body["messages"]
    assert system == {"role": "system", "content": gv.load_prompt("system")}
    content = user["content"]
    assert "You will receive 4 observations" in content[0]["text"]
    assert "put the cup on the coaster" in content[0]["text"]
    # label, then per view "View: name" + image; primary before left_wrist
    assert [c.get("text") for c in content[1:6]] == [
        "Observation 0", "View: primary", None, "View: left_wrist", None]
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 8
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    png = base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    assert out.progress.tolist() == [0.0, 10.0, 20.0, 30.0]
    assert out.frame_indices.tolist() == [0, 30, 60, 90]
    assert out.cost.n_calls == 1
    assert out.meta["model_name"] == "some-vlm" and out.meta["api"] == "openai"
    assert out.meta["views_in_request_order"] == ["primary", "left_wrist"]
    assert out.meta["credential_source"] == "OPENAI_API_KEY_environment"
    assert out.raw["response"]["reported_model"] == "served-model"
    assert SECRET not in json.dumps(out.to_dict())


def test_sequence_anthropic_wire_format(post, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    reply = {"id": "msg_1", "model": "claude-x", "stop_reason": "end_turn", "usage": {},
             "content": [{"type": "thinking", "thinking": ""},
                         {"type": "text", "text": _progress_json(3)}]}
    fake = post(lambda url, body: _Response(reply))
    rm = rewardbox.create_reward(
        "vlm-sequence", api="anthropic", model="claude-x", max_tokens=2048,
        extra_body={"output_config": {"effort": "medium"}})
    out = rm.score(_trajectory(3))

    call = fake.calls[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["headers"]["x-api-key"] == SECRET
    assert call["headers"]["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in call["headers"]
    body = call["body"]
    assert body["system"] == gv.load_prompt("system") and body["max_tokens"] == 2048
    assert body["output_config"] == {"effort": "medium"}
    assert [m["role"] for m in body["messages"]] == ["user"]
    image = next(c for c in body["messages"][0]["content"] if c["type"] == "image")
    assert image["source"]["type"] == "base64" and image["source"]["media_type"] == "image/png"
    assert out.progress.tolist() == [0.0, 10.0, 20.0]


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_anthropic_truncated_or_refused_answer_is_an_error(post, monkeypatch, stop_reason):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    post(lambda url, body: _Response({"stop_reason": stop_reason, "content": []}))
    rm = rewardbox.create_reward("vlm-sequence", api="anthropic", model="claude-x")
    with pytest.raises(RuntimeError, match=f"stop_reason={stop_reason}"):
        rm.score(_trajectory(3))


def test_sequence_gemini_wire_format_through_a_gateway(post, monkeypatch):
    monkeypatch.setenv("MY_GATEWAY_KEY", SECRET)
    reply = {"modelVersion": "gemini-x-001", "responseId": "g1", "usageMetadata": {},
             "candidates": [{"finishReason": "STOP", "content": {"parts": [
                 {"text": "thinking...", "thought": True}, {"text": _progress_json(3)}]}}]}
    fake = post(lambda url, body: _Response(reply))
    rm = rewardbox.create_reward(
        "vlm-sequence", api="gemini", model="gemini-x",
        base_url="https://gateway.example/v1beta", api_key_env="MY_GATEWAY_KEY",
        auth_header="bearer", image_format="jpeg", image_max_side=4)
    out = rm.score(_trajectory(3))

    call = fake.calls[0]
    assert call["url"] == "https://gateway.example/v1beta/models/gemini-x:generateContent"
    assert call["headers"]["Authorization"] == f"Bearer {SECRET}"
    body = call["body"]
    assert body["systemInstruction"] == {"parts": [{"text": gv.load_prompt("system")}]}
    parts = body["contents"][0]["parts"]
    inline = next(p["inlineData"] for p in parts if "inlineData" in p)
    assert inline["mimeType"] == "image/jpeg"
    assert base64.b64decode(inline["data"])[:2] == b"\xff\xd8"
    assert out.progress.tolist() == [0.0, 10.0, 20.0]
    assert out.raw["response"]["reported_model"] == "gemini-x-001"
    assert out.meta["image_policy"] == {"format": "jpeg", "jpeg_quality": 90, "max_side": 4}


# ---------------------------------------------------------------- answers and failures
def test_one_wrapping_markdown_fence_is_accepted_and_recorded(post, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    post(lambda url, body: _openai("```json\n" + _progress_json(3) + "\n```"))
    out = rewardbox.create_reward("vlm-sequence", model="m").score(_trajectory(3))
    assert out.progress.tolist() == [0.0, 10.0, 20.0]
    assert out.meta["markdown_fence_removed"] is True


@pytest.mark.parametrize("answer, message", [
    ('{"progress": [0, 50]}', "expected 3 progress values"),
    ('{"progress": [0, 50, 101]}', r"finite numbers in \[0, 100\]"),
    ('{"progress": [0, 50, 100], "note": "x"}', 'exactly the key "progress"'),
    ("The progress is 0, 50, 100.", "not a JSON object"),
])
def test_sequence_answer_in_the_wrong_format_fails_without_imputation(post, monkeypatch, answer, message):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    post(lambda url, body: _openai(answer))
    with pytest.raises(RuntimeError, match=message):
        rewardbox.create_reward("vlm-sequence", model="m").score(_trajectory(3))


def test_truncated_generation_is_an_error(post, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    post(lambda url, body: _openai(_progress_json(3), finish="length"))
    with pytest.raises(RuntimeError, match="finish_reason=length"):
        rewardbox.create_reward("vlm-sequence", model="m").score(_trajectory(3))


def test_rate_limits_are_retried_and_client_errors_are_not(post, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    replies = iter([_Response({"error": "slow down"}, 429), _Response("bad gateway", 502),
                    _openai(_progress_json(3))])
    fake = post(lambda url, body: next(replies))
    out = rewardbox.create_reward("vlm-sequence", model="m").score(_trajectory(3))
    assert len(fake.calls) == 3 and out.raw["response"]["http_attempts"] == 3

    fake = post(lambda url, body: _Response({"error": f"bad key {SECRET}"}, 401))
    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        rewardbox.create_reward("vlm-sequence", model="m").score(_trajectory(3))
    assert len(fake.calls) == 1
    assert SECRET not in str(error.value)


def test_missing_key_names_the_variable_and_the_cli_rejects_keys(post, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake = post(lambda url, body: _openai(_progress_json(3)))
    with pytest.raises(ValueError, match="export OPENAI_API_KEY"):
        rewardbox.create_reward("vlm-sequence", model="m").score(_trajectory(3))
    with pytest.raises(ValueError, match="export OPENAI_API_KEY"):
        rewardbox.create_reward("vlm-pair", model="m").score(_trajectory(3))
    assert fake.calls == []

    video = str(ROOT / "datasets" / "local_example" / "pick_success.mp4")
    rc = main(["run", "vlm-sequence", video, "--instruction", "pick",
               "--model-opt", "model=m", "--model-opt", f"api_key={SECRET}"])
    captured = capsys.readouterr()
    assert rc == EXIT_USAGE_ERROR and SECRET not in captured.out + captured.err


# ---------------------------------------------------------------- Pair
def test_pair_asks_every_adjacent_pair_and_keeps_format_errors_as_missing(post, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    answers = {1: "<score>+1</score>", 2: "It went down.", 3: " <score>-1</score>\n"}

    def reply(url, body):
        images = [c for c in body["messages"][1]["content"] if c["type"] == "image_url"]
        after = images[1]["image_url"]["url"]
        position = next(i for i in range(1, 4) if after == encoded[i])
        return _openai(answers[position])

    traj = _trajectory(4)
    rm = rewardbox.create_reward("vlm-pair", model="m", max_concurrency=2)
    image = rm._image_source(traj)
    encoded = {i: "data:image/png;base64," + image(i, "primary")[1] for i in range(4)}
    fake = post(reply)
    out = rm.score(traj)

    assert len(fake.calls) == 3
    content = fake.calls[0]["body"]["messages"][1]["content"]
    texts = [c["text"] for c in content if c["type"] == "text"]
    assert texts[1:] == ["BEFORE", "View: primary", "AFTER", "View: primary"]
    assert "labeled BEFORE and AFTER" in texts[0]
    assert out.raw["transition_directions"] == [1, None, -1]
    assert out.progress.tolist() == [0.0, 1.0, 0.0, -1.0]
    assert out.frame_indices.tolist() == [0, 30, 60, 90]
    assert out.meta["output_kind"] == "transition_directions"
    assert out.meta["n_format_errors"] == 1 and out.cost.n_calls == 3
    assert out.raw["pairs"][1]["answer"] == "It went down."


def test_pair_transport_failure_fails_the_trajectory(post, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    post(lambda url, body: _Response({"error": "no"}, 400))
    with pytest.raises(RuntimeError, match="3 of 3 pair requests failed"):
        rewardbox.create_reward("vlm-pair", model="m").score(_trajectory(4))
    with pytest.raises(ValueError, match="at least two"):
        rewardbox.create_reward("vlm-pair", model="m").score(_trajectory(1))


# ---------------------------------------------------------------- benchmark path
def test_bench_writes_a_resumable_shard_without_the_key(post, monkeypatch, tmp_path):
    from rewardbox.bench import run_benchmark

    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    fake = post(lambda url, body: _openai("<score>+1</score>"))
    sampling = SamplingSpec(strategy="fps", fps=1.0, include_last=False)
    opts = {"vlm-pair": {"model": "some-vlm", "base_url": "http://localhost:8000/v1"}}
    shards = run_benchmark(["vlm-pair"], MANIFEST, sampling, tmp_path, model_opts=opts,
                           log=lambda message: None)
    shard = shards["vlm-pair"]
    assert not shard.errors and len(shard.results) == 2
    assert shard.identity["model_opts"] == opts["vlm-pair"]
    for result in shard.results.values():
        n = len(result["frame_indices"])
        assert result["raw"]["transition_directions"] == [1] * (n - 1)
        assert result["meta"]["output_kind"] == "transition_directions"
    assert all(call["url"] == "http://localhost:8000/v1/chat/completions" for call in fake.calls)
    assert SECRET not in (tmp_path / "vlm-pair.json").read_text()

    n_calls = len(fake.calls)
    run_benchmark(["vlm-pair"], MANIFEST, sampling, tmp_path, model_opts=opts,
                  log=lambda message: None)
    assert len(fake.calls) == n_calls                       # resumed, nothing asked again


# ---------------------------------------------------------------- Jev
def _jev_score(level_probabilities):
    score = sum(int(k) * v for k, v in level_probabilities.items())
    return {"type": "score", "score": score, "probabilities": level_probabilities}


def _levels(**given):
    return {str(i): float(given.get(f"p{i}", 0.0)) for i in range(5)}


def test_jev_sequence_request_and_expected_level_progress(post, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", SECRET)
    answers = {"progress_0000": _jev_score(_levels(p0=1.0)),
               "progress_0001": _jev_score(_levels(p1=0.5, p2=0.5)),
               "progress_0002": _jev_score(_levels(p4=1.0))}
    fake = post(lambda url, body: _Response({"model": jev.MODEL, "answers": answers,
                                             "usage": {"input_tokens": 9}}))
    out = rewardbox.create_reward("jev-1.14-sequence", base_url=JEV_URL).score(
        _trajectory(3, views=("primary", "right_wrist")))

    call = fake.calls[0]
    assert call["url"] == "https://jev.example/v1/systemone"
    assert call["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert call["headers"]["TypeSafe-Danger-Zone-Images"] == "true"
    body = call["body"]
    assert body["model"] == "jev-1.14.0"
    state = body["state"]
    assert "Follow the requested output format" not in state["evaluation_instructions"]
    assert "Return a JSON object" not in state["task_and_scoring_instructions"]
    assert "put the cup on the coaster" in state["task_and_scoring_instructions"]
    assert [[v["view"] for v in o["views"]] for o in state["observations"]] == [
        ["primary", "right_wrist"]] * 3
    references = [v["image"] for o in state["observations"] for v in o["views"]]
    assert references == list(body["assets"]) and len(set(references)) == 6
    asset = body["assets"][references[0]]
    assert asset["type"] == "image" and asset["format"] == "jpg"
    assert base64.b64decode(asset["data_base64"])[:2] == b"\xff\xd8"
    assert list(body["questions"]) == ["progress_0000", "progress_0001", "progress_0002"]
    assert body["questions"]["progress_0001"]["criteria"] == jev.LEVELS

    assert out.progress.tolist() == [0.0, 37.5, 100.0]
    assert out.meta["protocol"] == jev.SEQUENCE_PROTOCOL_ID
    assert out.meta["image_policy"]["side"] == 280
    assert SECRET not in json.dumps(out.to_dict())


def test_jev_images_are_letterboxed_to_280_square():
    import cv2

    rm = rewardbox.create_reward("jev-1.14-sequence", base_url=JEV_URL, api_key="k")
    frames = np.zeros((1, 480, 640, 3), np.uint8)
    frames[..., 0] = 255                                       # red in RGB
    traj = Trajectory(frames=frames, instruction="task", sampling=jev.SEQUENCE_SPEC.sampling)
    blob = base64.b64decode(rm._image_source(traj)(0, "primary"))
    decoded = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (280, 280, 3)
    assert decoded[5, 140].max() < 16                          # black bar above the 280x210 picture
    assert decoded[140, 140, 2] > 200 and decoded[140, 140, 0] < 60   # still red (BGR order)


def test_jev_wrong_reported_model_is_an_error(post, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", SECRET)
    post(lambda url, body: _Response({"model": "jev-2", "answers": {}}))
    with pytest.raises(RuntimeError, match="reported model mismatch"):
        rewardbox.create_reward("jev-1.14-sequence", base_url=JEV_URL).score(_trajectory(2))


def test_jev_pair_takes_the_more_probable_option_and_keeps_ties(post, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", SECRET)
    probabilities = iter([{"rising": 0.7, "falling": 0.3}, {"rising": 0.5, "falling": 0.5},
                          {"rising": 0.1, "falling": 0.9}])

    def reply(url, body):
        return _Response({"model": jev.MODEL, "usage": {}, "answers": {"direction": {
            "type": "choice", "choice": "rising", "probabilities": next(probabilities)}}})

    fake = post(reply)
    out = rewardbox.create_reward("jev-1.14-pair", base_url=JEV_URL, max_concurrency=1).score(_trajectory(4))

    body = fake.calls[0]["body"]
    assert [o["label"] for o in body["state"]["observations"]] == ["BEFORE", "AFTER"]
    assert body["questions"] == {"direction": jev.PAIR_QUESTION}
    assert "Return exactly one" not in body["state"]["task_and_scoring_instructions"]
    assert out.raw["transition_directions"] == [1, 0, -1]
    assert out.progress.tolist() == [0.0, 1.0, 0.0, -1.0]
    assert out.meta["output_kind"] == "transition_directions"
    assert out.meta["protocol"] == jev.PAIR_PROTOCOL_ID


@pytest.mark.parametrize("name", ["jev-1.14-sequence", "jev-1.14-pair"])
def test_jev_has_no_default_endpoint(name):
    with pytest.raises(ValueError, match="base_url"):
        rewardbox.create_reward(name, api_key="k")
