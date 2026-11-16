"""Offline contract tests for the isolated upstream-GVL wrapper.

These tests deliberately replace the API client.  A real GVL calibration needs
an externally billed, versioned provider endpoint and is therefore a separate
evidence step rather than an ordinary unit test.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from rewardbox.adapters import gvl
from rewardbox.core.types import SamplingSpec, Trajectory


def _trajectory(n: int = 8) -> Trajectory:
    frames = np.stack(
        [np.full((4, 5, 3), i, dtype=np.uint8) for i in range(n)], axis=0
    )
    return Trajectory(
        frames=frames,
        instruction="move the block",
        frame_indices=np.arange(n, dtype=np.int64) * 10 + 100,
        sampling=SamplingSpec(num_frames=8, strategy="uniform"),
    )


class _FakeGVL:
    def __init__(self, predictions, frames_info):
        self.predictions = predictions
        self.frames_info = frames_info
        self.frames = None
        self.task = None

    def compute_progress(self, frames, task_description):
        self.frames = frames
        self.task = task_description
        return self.predictions


def _adapter(**kwargs):
    return gvl.GVLAdapter(gvl.SPEC, check_deps=False, **kwargs)


def _current_adapter(**kwargs):
    return gvl.GVLAdapter(gvl.CURRENT_SPEC, check_deps=False, **kwargs)


@pytest.mark.parametrize("n", (7, 9))
def test_rejects_non_exact_grid_before_api_invocation(n):
    adapter = _adapter()
    client = _FakeGVL([0.2], [])
    adapter._client = client

    with pytest.raises(ValueError, match="exactly 8 frames"):
        adapter.score(_trajectory(n=n))

    assert client.frames is None


def test_rejects_a_non_command_sampling_contract_before_api_invocation():
    adapter = _adapter()
    client = _FakeGVL([0.2], [])
    adapter._client = client
    traj = _trajectory()
    traj.sampling = SamplingSpec(strategy="all")

    with pytest.raises(ValueError, match="Trajectory.sampling"):
        adapter.score(traj)

    assert client.frames is None


def test_constructs_the_exact_shipped_confusion_matrix_client(monkeypatch):
    calls = []

    class FakeUpstreamGVL:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(gvl, "_upstream_gvl_class", lambda: FakeUpstreamGVL)
    adapter = _adapter(api_key="unit-test-secret")

    adapter.client()

    assert calls == [{
        "api_key": "unit-test-secret",
        "max_frames": 8,
        "offset": 0.5,
        "model_name": "gemini-2.5-flash-lite",
        "provider": "gemini",
        "max_retries": 5,
        "base_delay": 1.0,
        "max_delay": 60.0,
    }]


def test_constructs_provider_migration_as_a_separate_model_identity(monkeypatch):
    calls = []

    class FakeUpstreamGVL:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(gvl, "_upstream_gvl_class", lambda: FakeUpstreamGVL)
    adapter = _current_adapter(api_key="unit-test-secret")

    adapter.client()

    assert adapter.spec.name == "gvl-gemini-3.5-flash-lite"
    assert calls[0]["model_name"] == "gemini-3.5-flash-lite"
    assert calls[0]["max_frames"] == 8


def test_preserves_official_sparse_mapping_without_imputation_or_secret_leakage():
    # The official class returns values in chronological gt_index order but
    # omits entries whose model response did not map back to a shuffled frame.
    # The wrapper must retain that sparse grid rather than fill missing values.
    infos = [
        {"gt_index": 1, "shuffled_index": 4, "model_output": {"frame_number": 4}},
        {"gt_index": 2, "shuffled_index": 1, "model_output": None},
        {"gt_index": 3, "shuffled_index": 8, "model_output": None},
        {"gt_index": 4, "shuffled_index": 2, "model_output": {"frame_number": 2}},
        {"gt_index": 5, "shuffled_index": 5, "model_output": None},
        {"gt_index": 6, "shuffled_index": 3, "model_output": None},
        {"gt_index": 7, "shuffled_index": 6, "model_output": None},
        {"gt_index": 8, "shuffled_index": 7, "model_output": {"frame_number": 7}},
    ]
    client = _FakeGVL([0.1, 0.4, 0.8], infos)
    secret = "do-not-serialize-this-api-key"
    adapter = _adapter(api_key=secret)
    adapter._client = client

    out = adapter.score(_trajectory())

    assert client.task == "move the block"
    assert client.frames.flags.c_contiguous
    assert out.progress.tolist() == [0.1, 0.4, 0.8]
    # Input positions [0, 3, 7] mapped to the caller's already-upstream-sampled grid.
    assert out.frame_indices.tolist() == [100, 130, 170]
    assert out.raw["sampled_local_indices"] == list(range(8))
    assert out.raw["returned_local_indices"] == [0, 3, 7]
    assert out.raw["returned_shuffled_indices"] == [4, 2, 7]
    assert out.raw["official_parsed_frames"][0] == {
        "gt_index": 1,
        "shuffled_index": 4,
        "model_output": {"frame_number": 4},
    }
    assert out.meta["imputed_missing_predictions"] is False
    assert out.meta["protocol"] == "upstream_confusion_matrix_recipe_strict_8_frame"
    assert out.meta["paper_alignment_equivalence"].startswith("not asserted:")
    assert out.meta["source_sampler_padding"] == "not needed: strict input has exactly 8 frames"
    assert out.meta["shuffle"] == "upstream random.shuffle (no seed parameter)"
    assert out.meta["remote_model_revision"] is None
    assert out.meta["provider_managed_model_revision_pinned"] is False
    assert out.meta["upstream_recipe_model_name"] == "gemini-2.5-flash-lite"
    assert out.meta["provider_model_migration"] is False
    assert out.meta["credential_source"] == "explicit_constructor"
    assert secret not in json.dumps(out.to_dict())


def test_provider_migration_is_explicit_in_result_metadata():
    infos = [
        {"gt_index": i, "shuffled_index": i, "model_output": {"frame_number": i}}
        for i in range(1, 9)
    ]
    adapter = _current_adapter(api_key="unit-test-secret")
    adapter._client = _FakeGVL(np.linspace(0.0, 1.0, 8), infos)

    out = adapter.score(_trajectory())

    assert out.model_name == "gvl-gemini-3.5-flash-lite"
    assert out.meta["model_name"] == "gemini-3.5-flash-lite"
    assert out.meta["upstream_recipe_model_name"] == "gemini-2.5-flash-lite"
    assert out.meta["provider_model_migration"] is True
    assert "provider model differs" in out.meta["paper_alignment_equivalence"]


def test_redacts_gemini_query_key_from_upstream_logs_and_errors(capsys):
    secret = "gemini-unit-test-secret"

    class FailingGVL:
        api_key = secret
        frames_info = []

        def compute_progress(self, frames, task_description):
            url = f"https://example.invalid/generate?alt=sse&key={secret}"
            print(f"retrying request: {url}")
            raise RuntimeError(f"request failed for {url}")

    adapter = _adapter()
    adapter._client = FailingGVL()

    with pytest.raises(RuntimeError, match="official GVL API call failed") as exc_info:
        adapter.score(_trajectory())

    captured = capsys.readouterr()
    combined = captured.out + captured.err + str(exc_info.value)
    assert captured.out == ""
    assert secret not in combined
    assert "key=[REDACTED]" in combined


def test_rejects_ambiguous_upstream_output_instead_of_guessing_missing_frames():
    infos = [
        {"gt_index": i, "shuffled_index": i, "model_output": {"frame_number": i}}
        for i in range(1, 9)
    ]
    adapter = _adapter()
    # Three frame records say they mapped, but the output list only contains two
    # values.  Carry-forward, truncation, or a guessed index would all be wrong.
    adapter._client = _FakeGVL([0.2, 0.7], infos[:3] + [
        {"gt_index": i, "shuffled_index": i, "model_output": None}
        for i in range(4, 9)
    ])

    with pytest.raises(RuntimeError, match="refusing to guess the missing frames"):
        adapter.score(_trajectory(n=8))


def test_empty_official_response_is_not_converted_to_a_fake_score():
    adapter = _adapter()
    adapter._client = _FakeGVL([], [])

    with pytest.raises(RuntimeError, match="non-empty progress"):
        adapter.score(_trajectory(n=8))


def test_real_pinned_class_keeps_its_sampling_shuffle_and_json_mapping(monkeypatch):
    """Exercise upstream's actual JPEG/shuffle/parser path without an API call."""
    pytest.importorskip("cv2")
    UpstreamGVL = gvl._upstream_gvl_class()
    client = UpstreamGVL(
        api_key="offline-test-key",
        max_frames=8,
        offset=0.5,
        model_name="gemini-2.5-flash-lite",
        provider="gemini",
    )

    def fake_stream_inference(_parts):
        # GVL has already extracted and randomly shuffled frames at this point.
        # Return a response keyed by its actual shuffled labels, so its own parser
        # must restore chronological gt_index order before the wrapper sees it.
        return json.dumps([
            {
                "frame_number": frame["shuffled_index"],
                "frame_description": "offline test",
                "task_completion_percentage": frame["gt_index"] * 10,
            }
            for frame in client.frames_info
        ])

    monkeypatch.setattr(client, "stream_inference", fake_stream_inference)
    adapter = _adapter()
    adapter._client = client

    out = adapter.score(_trajectory())

    assert out.progress.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert out.frame_indices.tolist() == [100, 110, 120, 130, 140, 150, 160, 170]
