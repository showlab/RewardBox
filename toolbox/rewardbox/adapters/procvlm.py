"""ProcVLM released 2B progress-reward adapter.

The pinned ProcVLM runtime uses Python 3.10, torch 2.10, Transformers 4.57,
and vLLM 0.18.  Those packages intentionally stay in the upstream project's
isolated environment.  This module is a small HTTP/NPZ client and never imports
torch, Transformers, vLLM, or the upstream package.

The registered profile follows ``evqa/inference.py`` at the pinned upstream
commit: all source frames (uniformly capped at 512), an eight-observation
left-padded recent window, the released ``procedural_c`` prompt, BF16 vLLM,
greedy generation with at most 4096 new tokens, textual ``<progress>`` parsing,
and carry-forward from zero when parsing fails.  Progress remains on the
official raw percentage scale ``[0, 100]``.  RewardBox supplies already-decoded
RGB trajectory frames at the transport boundary; the service then uses the
same quality-95 JPEG image boundary as the official MoviePy video entrypoint.
"""
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from ..core.base import BaseAdapter
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import RewardOutput, SamplingSpec, Trajectory

_UPSTREAM_REPO = "https://github.com/RUCKBReasoning/ProcVLM"
_UPSTREAM_COMMIT = "377523a31f05bab9c0db5ac8b9edfa7b7f03968a"
_WEIGHTS_REPO = "ce-amtic/ProcVLM-2B"
_WEIGHTS_REVISION = "a7f5c18c6128a9c2a478a62ecc356f19f0e0bcf5"
_WEIGHT_FILES = {
    "model-00001-of-00002.safetensors": {
        "sha256": "b56d35efd5c2d5ec38f6cbb2bd83c337a1425acf1ec5e45ffc52f7ee260715a5",
        "size_bytes": 3_106_948_088,
    },
    "model-00002-of-00002.safetensors": {
        "sha256": "00addfac3617d77d3db4f0957201824700ebb3ed32ef7a684d894af5e38e6055",
        "size_bytes": 1_770_807_984,
    },
    "procvlm_extra/procvlm_head.pt": {
        "sha256": "a1ae24192f79e63e42846b23aed82820f73e473fc9c799d6e73a0b39e1afb875",
        "size_bytes": 69_278_406,
    },
}
_PROFILE = "official-vllm-text-progress-v1"
_DEFAULT_SERVER_URL = "http://127.0.0.1:8013"
_PROVENANCE_SCHEMA = "rewardbox.procvlm-server-provenance.v1"
_SCORE_SCHEMA = "rewardbox.procvlm-score.v1"
_PROMPT_SHA256 = "0dd106e10c778ab3b963d23af18ccde5a16a4023695ffa0f8a3f2e079db52c69"

_PROFILE_FIELDS = {
    "name": _PROFILE,
    "backend": "vllm",
    "model_class": "Qwen3VLForConditionalGeneration",
    "torch_dtype": "bfloat16",
    "window_size": 8,
    "max_sampled_frames": 512,
    "max_new_tokens": 4096,
    "temperature": 0.0,
    "tp": 1,
    "enable_value_head": False,
    "use_lora": False,
    "frame_selection": "evqa.inference.build_uniform_indices",
    "window_policy": "evqa.inference.build_window_values-left-repeat",
    "prompt_template": "evqa/templates/procedural_c.txt",
    "prompt_sha256": _PROMPT_SHA256,
    "image_boundary": "RGB-to-JPEG-quality-95",
    "progress_parser": "evqa.inference.extract_progress-clamp-0-100",
    "parse_failure": "carry-previous-initial-0",
}

_REQ = Requirements(
    pip=("requests>=2",),
    python=">=3.10",
    vram_gb=12.0,
    note=(
        "RewardBox only sends NPZ/HTTP; ProcVLM's torch 2.10, Transformers "
        "4.57, vLLM 0.18 and the checkpoint live in a separate Python 3.10 environment."
    ),
    install_hint=(
        "uv pip install --python .venv/bin/python -e '.[procvlm-client]'",
        "uv sync --project reward_model_code/ProcVLM --python 3.10 --frozen",
        "start the pinned official-vllm service with scripts/run_procvlm_server.py",
    ),
)

SPEC = ModelSpec(
    name="procvlm-2b",
    cls="ProcVLMAdapter",
    module=__name__,
    reward_mode=RewardMode.PROGRESS,
    signal_type=SignalType.GENERATED_TEXT,
    requires=_REQ,
    # The official inference function receives a complete episode and performs
    # its own uniform cap at 512 before building recent-frame windows.
    sampling=SamplingSpec(strategy="all"),
    score_range=(0.0, 100.0),
    higher_better=True,
    required_views=("primary",),
    supports_pairwise=False,
    outputs_success=False,
    zero_shot=True,
    paper="https://arxiv.org/abs/2605.08774",
    weights=f"{_WEIGHTS_REPO}@{_WEIGHTS_REVISION}",
    upstream=(
        f"{_UPSTREAM_REPO}@{_UPSTREAM_COMMIT}; "
        "evqa.inference + evqa.model.batch_chat_with_vllm"
    ),
    description=(
        "Released ProcVLM-2B procedure-grounded progress estimator. Returns "
        "the official textual completion percentage and preserves every raw "
        "reasoning/output record."
    ),
    implementation_kind=ImplementationKind.OFFICIAL_ADAPTER,
)


def _validate_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("server_url must be a non-empty HTTP(S) URL")
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "server_url must be an HTTP(S) server root without credentials or path/query/fragment"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("server_url has an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("the server_url port must be in 1..65535")
    return value


def _read_provenance(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"server_provenance_file does not exist or is not a file: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"server_provenance_file is not readable JSON: {path}: {exc}") from exc
    if not isinstance(record, dict) or not record:
        raise ValueError("server_provenance_file must have a non-empty JSON object at top level")
    return record


def _profile_mismatches(actual: Any) -> dict:
    if not isinstance(actual, dict):
        return {"profile": {"expected": _PROFILE_FIELDS, "actual": actual}}
    return {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in _PROFILE_FIELDS.items()
        if actual.get(key) != expected
    }


def _validate_weights_identity(weights: Any, *, allow_unverified: bool) -> None:
    if not isinstance(weights, dict):
        raise ValueError("ProcVLM provenance lacks a weights object")
    expected = {
        "repo_id": _WEIGHTS_REPO,
        "expected_revision": _WEIGHTS_REVISION,
    }
    mismatches = {
        key: {"expected": value, "actual": weights.get(key)}
        for key, value in expected.items()
        if weights.get(key) != value
    }
    files = weights.get("files")
    if not isinstance(files, dict):
        mismatches["files"] = {"expected": _WEIGHT_FILES, "actual": files}
    else:
        for name, expected_file in _WEIGHT_FILES.items():
            actual = files.get(name)
            if not isinstance(actual, dict):
                mismatches[f"files.{name}"] = {
                    "expected": expected_file,
                    "actual": actual,
                }
                continue
            for field, value in expected_file.items():
                if actual.get(field) != value:
                    mismatches[f"files.{name}.{field}"] = {
                        "expected": value,
                        "actual": actual.get(field),
                    }
    if mismatches:
        raise ValueError(
            "ProcVLM server weight identity mismatch: "
            + json.dumps(mismatches, sort_keys=True, ensure_ascii=False)
        )
    pinned = (
        weights.get("snapshot_revision") == _WEIGHTS_REVISION
        and weights.get("pinned_public_snapshot") is True
        and weights.get("content_verified") is True
    )
    if not pinned and not allow_unverified:
        raise ValueError("the ProcVLM server did not verify the pinned public checkpoint content")


def _validate_provenance(record: dict, *, allow_unverified: bool) -> None:
    if record.get("schema") != _PROVENANCE_SCHEMA:
        raise ValueError(f"the ProcVLM provenance schema must be {_PROVENANCE_SCHEMA!r}")
    if record.get("status") != "ready":
        raise ValueError(
            "ProcVLM server_provenance_file must be in the ready state; "
            f"actual {record.get('status')!r}"
        )
    upstream = record.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("commit") != _UPSTREAM_COMMIT:
        raise ValueError(f"the ProcVLM server upstream commit must be {_UPSTREAM_COMMIT}")
    if upstream.get("source_verified") is not True and not allow_unverified:
        raise ValueError("the ProcVLM server did not verify the pinned upstream source")
    _validate_weights_identity(record.get("weights"), allow_unverified=allow_unverified)
    mismatches = _profile_mismatches(record.get("profile"))
    if mismatches:
        raise ValueError(
            "ProcVLM server provenance profile drifted: "
            + json.dumps(mismatches, sort_keys=True, ensure_ascii=False)
        )


def _official_uniform_positions(total: int, target_count: int) -> list[int]:
    """Mirror the tiny index-only contract of upstream build_uniform_indices."""
    if total <= 0 or target_count <= 0:
        return []
    if target_count >= total:
        return list(range(total))
    if target_count == 1:
        return [0]
    values = np.linspace(0, total - 1, num=target_count)
    candidates = np.round(values).astype(np.int64).tolist()
    unique: list[int] = []
    seen: set[int] = set()
    for value in candidates:
        if value not in seen:
            seen.add(value)
            unique.append(int(value))
    cursor = 0
    while len(unique) < target_count and cursor < total:
        if cursor not in seen:
            seen.add(cursor)
            unique.append(cursor)
        cursor += 1
    unique.sort()
    return unique[:target_count]


class ProcVLMAdapter(BaseAdapter):
    """HTTP client for the pinned isolated ProcVLM vLLM service."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        server_url: str | None = None,
        timeout_s: float = 3600.0,
        server_provenance_file: str | None = None,
        allow_unverified_server: bool = False,
        **opts: Any,
    ):
        super().__init__(spec, **opts)
        if spec.name != SPEC.name:
            raise ValueError(f"unknown ProcVLM spec: {spec.name!r}")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError("timeout_s must be a finite positive number")
        if not math.isfinite(float(timeout_s)) or timeout_s <= 0:
            raise ValueError("timeout_s must be a finite positive number")
        if not isinstance(allow_unverified_server, bool):
            raise ValueError("allow_unverified_server must be a bool")
        if server_provenance_file is not None and (
            not isinstance(server_provenance_file, str)
            or not server_provenance_file.strip()
        ):
            raise ValueError("server_provenance_file, when given, must be a non-empty path string")

        self.server_url = _validate_url(
            server_url
            or os.environ.get("REWARDBOX_PROCVLM_SERVER_URL")
            or _DEFAULT_SERVER_URL
        )
        self.timeout_s = float(timeout_s)
        self.allow_unverified_server = allow_unverified_server
        self.server_provenance_file = (
            Path(server_provenance_file).expanduser().resolve()
            if server_provenance_file is not None
            else None
        )
        if self.server_provenance_file is None:
            if not allow_unverified_server:
                raise ValueError(
                    "ProcVLM requires server_provenance_file by default, binding the official source, "
                    "weight revision and inference profile. Pass allow_unverified_server=true "
                    "explicitly only for temporary diagnostics."
                )
            self._server_provenance = None
        else:
            self._server_provenance = _read_provenance(self.server_provenance_file)
            _validate_provenance(
                self._server_provenance,
                allow_unverified=self.allow_unverified_server,
            )
        self._session = None
        self._health: dict | None = None
        self._model_info: dict | None = None

    def _client(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    @staticmethod
    def _response_json(response, *, endpoint: str) -> dict:
        try:
            response.raise_for_status()
        except Exception as exc:
            body = getattr(response, "text", "")
            excerpt = body[:500] if isinstance(body, str) else ""
            raise RuntimeError(
                f"ProcVLM {endpoint} HTTP failure: {exc}; body={excerpt!r}"
            ) from exc
        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"ProcVLM {endpoint} returned non-JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"ProcVLM {endpoint} JSON top level is not an object")
        return payload

    def _ensure_server_identity(self) -> None:
        if self._health is not None and self._model_info is not None:
            return
        try:
            health_response = self._client().get(
                f"{self.server_url}/health", timeout=min(self.timeout_s, 30.0)
            )
            info_response = self._client().get(
                f"{self.server_url}/model_info", timeout=min(self.timeout_s, 30.0)
            )
        except Exception as exc:
            raise RuntimeError(
                f"cannot connect to the ProcVLM server {self.server_url}: {exc}"
            ) from exc
        health = self._response_json(health_response, endpoint="/health")
        info = self._response_json(info_response, endpoint="/model_info")
        if health.get("status") != "healthy" or health.get("reward_model") != "procvlm":
            raise RuntimeError(f"invalid ProcVLM /health identity: {health!r}")
        if info.get("reward_model") != "procvlm":
            raise RuntimeError(f"invalid ProcVLM /model_info identity: {info!r}")
        if info.get("upstream_commit") != _UPSTREAM_COMMIT:
            raise RuntimeError("ProcVLM /model_info upstream commit drifted")
        weights = info.get("weights")
        if not isinstance(weights, dict) or (
            weights.get("repo_id") != _WEIGHTS_REPO
            or weights.get("revision") != _WEIGHTS_REVISION
        ):
            raise RuntimeError("ProcVLM /model_info checkpoint identity drifted")
        mismatches = _profile_mismatches(info.get("profile"))
        if mismatches:
            raise RuntimeError(
                "the ProcVLM server does not match the pinned official profile: "
                + json.dumps(mismatches, sort_keys=True, ensure_ascii=False)
            )
        self._health = health
        self._model_info = info

    @staticmethod
    def _request_body(traj: Trajectory) -> bytes:
        if traj.sampling is not None and traj.sampling.strategy != "all":
            raise ValueError(
                "ProcVLM official inference needs a complete episode; Trajectory.sampling "
                "must be strategy='all' or None"
            )
        if len(traj.frame_indices) > 1 and not np.all(np.diff(traj.frame_indices) == 1):
            raise ValueError(
                "ProcVLM official inference needs a complete contiguous episode; sparse pre-sampled "
                "frames are never silently treated as the source video"
            )
        buffer = io.BytesIO()
        payload = {
            "primary": np.ascontiguousarray(traj.primary),
            "frame_indices": np.asarray(traj.frame_indices, dtype=np.int64),
            "instruction": np.asarray(traj.instruction),
        }
        if traj.fps is not None:
            payload["fps"] = np.asarray(float(traj.fps), dtype=np.float64)
        np.savez_compressed(buffer, **payload)
        return buffer.getvalue()

    def _score(self, traj: Trajectory) -> RewardOutput:
        packing_started = perf_counter()
        body = self._request_body(traj)
        packing_seconds = perf_counter() - packing_started
        self._ensure_server_identity()
        request_started = perf_counter()
        try:
            response = self._client().post(
                f"{self.server_url}/score_npz",
                data=body,
                headers={"Content-Type": "application/x-npz"},
                timeout=self.timeout_s,
            )
        except Exception as exc:
            raise RuntimeError(f"ProcVLM /score_npz request failed: {exc}") from exc
        payload = self._response_json(response, endpoint="/score_npz")
        request_seconds = perf_counter() - request_started
        if payload.get("schema") != _SCORE_SCHEMA:
            raise RuntimeError(f"the ProcVLM score schema must be {_SCORE_SCHEMA!r}")
        if payload.get("profile") != _PROFILE:
            raise RuntimeError("ProcVLM score response profile drifted")

        try:
            positions = np.asarray(payload.get("source_positions"), dtype=np.int64)
            frame_indices = np.asarray(payload.get("frame_indices"), dtype=np.int64)
            progress = np.asarray(payload.get("progress"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("ProcVLM score response contains non-numeric arrays") from exc

        expected_positions = np.asarray(
            _official_uniform_positions(len(traj), min(len(traj), 512)),
            dtype=np.int64,
        )
        if positions.ndim != 1 or not np.array_equal(positions, expected_positions):
            raise RuntimeError("ProcVLM source_positions do not match the official uniform cap=512")
        expected_indices = traj.frame_indices[expected_positions]
        if frame_indices.ndim != 1 or not np.array_equal(frame_indices, expected_indices):
            raise RuntimeError("ProcVLM response frame_indices do not match the official sampling request")
        if progress.shape != frame_indices.shape or len(progress) == 0:
            raise RuntimeError("invalid ProcVLM progress length/dimension")
        if not np.all(np.isfinite(progress)):
            raise RuntimeError("ProcVLM progress contains NaN/Inf")
        if np.any(progress < -1e-8) or np.any(progress > 100.0 + 1e-8):
            raise RuntimeError("ProcVLM textual progress must be on the official [0,100] scale")

        records = payload.get("records")
        if not isinstance(records, list) or len(records) != len(progress):
            raise RuntimeError("ProcVLM records must map one to one onto progress")
        previous = 0.0
        for index, (record, value) in enumerate(zip(records, progress, strict=True)):
            if not isinstance(record, dict):
                raise RuntimeError("a ProcVLM record must be an object")
            parsed = record.get("parsed_progress")
            if parsed is None:
                expected = previous
                if record.get("progress_source") != "previous":
                    raise RuntimeError("a ProcVLM parse failure is not marked as previous fallback")
            else:
                if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
                    raise RuntimeError("ProcVLM parsed_progress must be a number or null")
                expected = float(parsed)
                if not math.isfinite(expected) or not 0.0 <= expected <= 100.0:
                    raise RuntimeError("ProcVLM parsed_progress is outside [0,100]")
                previous = expected
                if record.get("progress_source") != "model":
                    raise RuntimeError("ProcVLM parsed progress is not marked as model source")
            if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-8):
                raise RuntimeError(f"ProcVLM record {index} carry-forward semantics drifted")

        pinned_weights = bool(
            self._server_provenance
            and self._server_provenance.get("weights", {}).get("content_verified")
            and self._server_provenance.get("weights", {}).get(
                "pinned_public_snapshot"
            )
        )
        return RewardOutput(
            progress=progress,
            frame_indices=frame_indices,
            raw={"records": records},
            model_name=self.spec.name,
            sampling=traj.sampling,
            meta={
                "n_backend_calls": len(progress),
                "n_http_calls": 1,
                "transport_timing": {"npz_encode_seconds": packing_seconds,
                                     "server_request_seconds": request_seconds,
                                     "payload_bytes": len(body)},
                "vllm_batches": payload.get("vllm_batches", 1),
                "model_generations": len(progress),
                "source_frame_count": len(traj),
                "sampled_frame_count": len(progress),
                "source_positions": positions.tolist(),
                "server_url": self.server_url,
                "server_health": self._health,
                "server_model_info": self._model_info,
                "server_checkpoint_provenance": self._server_provenance,
                "server_provenance_file": (
                    str(self.server_provenance_file)
                    if self.server_provenance_file is not None
                    else None
                ),
                "server_identity_verified": pinned_weights,
                "official_profile": _PROFILE_FIELDS,
                "upstream_commit": _UPSTREAM_COMMIT,
                "weights_revision": _WEIGHTS_REVISION,
                "raw_semantics": (
                    "official text-generated completion percentage [0,100]; "
                    "unparsed outputs carry the previous value from initial zero"
                ),
            },
        )

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
