"""Robo-ValueRL released value-estimator adapter.

The released Robo-ValueRL inference stack patches Transformers in-place and
pins a substantially different torch/JAX environment.  RewardBox therefore
keeps that stack behind ``scripts/run_robovaluerl_server.py``.  This module is
deliberately a light HTTP client and does not import torch, transformers, JAX,
or any upstream package.

Official release profile (upstream commit pinned below):

* all three RGB streams, one current frame at a time;
* base-camera history image at ``max(0, i - 5)``;
* key-frame inference every 10 frames plus the terminal frame;
* 256-bin HL-Gauss remaining-step distribution over ``[0, 5000]``;
* cubic interpolation plus Savitzky-Golay smoothing (21, 3);
* paper value ``clip(1 - remaining_steps / 5000, 0, 1)``.

The upstream annotation script currently asks the history critic for
``value`` and ``rtg`` fields even though the released critic returns only
``remain_time``.  The isolated server calls the released critic's actual
``remain_time`` field, while retaining the annotation script's official
remaining-time decoder and interpolation helper.  This compatibility choice
is explicit in server provenance and output metadata.
"""
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from ..core.base import BaseAdapter
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import RewardOutput, SamplingSpec, Trajectory

_UPSTREAM_COMMIT = "1618bcb01212eb747a5d2659228b7fb5db87a563"
_WEIGHTS_REPO = "X-Humanoid/Robo-ValueRL"
_WEIGHTS_REVISION = "b44f33eaa5892233174d596cc7d8a7e685967eef"
_WEIGHTS_FILE = "value_estimator/model.safetensors"
_WEIGHTS_SHA256 = "97c8d7060349fbc675a76cd43006fc52267647cbe67e092096ceac78d550c09b"
_WEIGHTS_SIZE = 6_267_669_712
_PROFILE = "official-release-short-history-v1"
_DEFAULT_SERVER_URL = "http://127.0.0.1:8012"
_PROVENANCE_SCHEMA = "rewardbox.robovaluerl-server-provenance.v1"
_SCORE_SCHEMA = "rewardbox.robovaluerl-score.v1"

_PROFILE_FIELDS = {
    "name": _PROFILE,
    "config_name": "all_value_function_training",
    "history_length": 5,
    "stride": 10,
    "interpolation_method": "cubic",
    "smooth": True,
    "smooth_window": 21,
    "smooth_polyorder": 3,
    "bc_type": "natural",
    "num_bins": 256,
    "remaining_steps_min": 0.0,
    "remaining_steps_max": 5000.0,
    "hl_gauss_sigma": 3.0,
    "progress_transform": "clip(1-remaining_steps/5000,0,1)",
    "history_view": "primary",
    "views": {
        "primary": "base_0_rgb",
        "left_wrist": "left_wrist_0_rgb",
        "right_wrist": "right_wrist_0_rgb",
    },
    "released_output_field": "remain_time",
    "config_loader": "exact-release-Pi0Config-fields",
    "interpolation_loader": "AST-executed-exact-pinned-functions",
}

_REQ = Requirements(
    pip=("requests>=2",),
    python=">=3.11",
    vram_gb=32.0,
    note=(
        "RewardBox only sends NPZ/HTTP; Robo-ValueRL's official torch/JAX, customised "
        "Transformers and 6.27GB checkpoint must live in a separate environment."
    ),
    install_hint=(
        "uv pip install --python .venv/bin/python -e '.[robovaluerl-client]'",
        "create the isolated environment (docs/installation.md, section 5)",
        "start the pinned official service with scripts/run_robovaluerl_server.py",
    ),
)

SPEC = ModelSpec(
    name="robo-valuerl-value",
    cls="RoboValueRLAdapter",
    module=__name__,
    reward_mode=RewardMode.PROGRESS,
    signal_type=SignalType.VALUE_HEAD,
    requires=_REQ,
    # The released annotation path needs the complete contiguous episode to
    # construct its 5-frame history and 10-frame key grid.
    sampling=SamplingSpec(strategy="all"),
    score_range=(0.0, 1.0),
    higher_better=True,
    required_views=("primary", "left_wrist", "right_wrist"),
    supports_pairwise=False,
    outputs_success=False,
    # The public checkpoint is task/data trained; do not advertise universal
    # zero-shot behavior without a dedicated held-out-task validation.
    zero_shot=False,
    paper="https://arxiv.org/abs/2607.09866",
    weights=f"{_WEIGHTS_REPO}@{_WEIGHTS_REVISION}:{_WEIGHTS_FILE}",
    upstream=(
        "Open-X-Humanoid/Robo-ValueRL@1618bcb; "
        "OpenPi_X_Humanoid_Critic.predict_value -> remain_time; "
        "official HL-Gauss decode and annotation interpolation"
    ),
    description=(
        "Released Robo-ValueRL short-history remaining-step estimator. "
        "Returns the paper-defined [0,1] progress value and preserves decoded "
        "remaining steps plus all key-frame distributions as raw evidence."
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


def _validate_provenance(record: dict, *, allow_unverified: bool) -> None:
    if record.get("schema") != _PROVENANCE_SCHEMA:
        raise ValueError(
            f"the Robo-ValueRL provenance schema must be {_PROVENANCE_SCHEMA!r}"
        )
    if record.get("status") != "ready":
        raise ValueError(
            "Robo-ValueRL server_provenance_file must be in the ready state; "
            f"actual {record.get('status')!r}"
        )
    upstream = record.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("commit") != _UPSTREAM_COMMIT:
        raise ValueError(
            f"the Robo-ValueRL server upstream commit must be {_UPSTREAM_COMMIT}"
        )
    if upstream.get("source_verified") is not True and not allow_unverified:
        raise ValueError("the Robo-ValueRL server did not verify the pinned upstream source")

    weights = record.get("weights")
    expected_weights = {
        "repo_id": _WEIGHTS_REPO,
        "expected_revision": _WEIGHTS_REVISION,
        "filename": _WEIGHTS_FILE,
        "sha256": _WEIGHTS_SHA256,
        "size_bytes": _WEIGHTS_SIZE,
    }
    if not isinstance(weights, dict):
        raise ValueError("Robo-ValueRL provenance lacks a weights object")
    mismatches = {
        key: {"expected": expected, "actual": weights.get(key)}
        for key, expected in expected_weights.items()
        if weights.get(key) != expected
    }
    pinned = (
        weights.get("snapshot_revision") == _WEIGHTS_REVISION
        and weights.get("pinned_public_snapshot") is True
        and weights.get("content_verified") is True
    )
    if mismatches:
        raise ValueError(
            "Robo-ValueRL server weight identity mismatch: "
            + json.dumps(mismatches, sort_keys=True, ensure_ascii=False)
        )
    if not pinned and not allow_unverified:
        raise ValueError(
            "the Robo-ValueRL server weights are not a content-verified pinned public snapshot; "
            "pass allow_unverified_server=true explicitly only for diagnostic weight copies"
        )

    profile_mismatches = _profile_mismatches(record.get("profile"))
    if profile_mismatches:
        raise ValueError(
            "Robo-ValueRL provenance official profile fields mismatch: "
            + json.dumps(profile_mismatches, sort_keys=True, ensure_ascii=False)
        )


class RoboValueRLAdapter(BaseAdapter):
    """Thin client for the pinned official Robo-ValueRL value estimator."""

    def __init__(
        self,
        spec: ModelSpec,
        server_url: str | None = None,
        timeout_s: float = 1800.0,
        server_provenance_file: str | None = None,
        allow_unverified_server: bool = False,
        **opts: Any,
    ):
        super().__init__(spec, **opts)
        if spec.name != SPEC.name:
            raise ValueError(f"unknown Robo-ValueRL spec: {spec.name!r}")
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
            or os.environ.get("REWARDBOX_ROBOVALUERL_SERVER_URL")
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
                    "Robo-ValueRL requires server_provenance_file by default, binding the official source, "
                    "weight revision/content and inference profile. Pass allow_unverified_server=true "
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
                f"Robo-ValueRL {endpoint} HTTP failure: {exc}; body={excerpt!r}"
            ) from exc
        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Robo-ValueRL {endpoint} returned non-JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Robo-ValueRL {endpoint} JSON top level is not an object")
        return payload

    def _ensure_server_identity(self) -> None:
        if self._health is not None and self._model_info is not None:
            return
        client = self._client()
        try:
            health_response = client.get(
                f"{self.server_url}/health", timeout=min(self.timeout_s, 30.0)
            )
            info_response = client.get(
                f"{self.server_url}/model_info", timeout=min(self.timeout_s, 30.0)
            )
        except Exception as exc:
            raise RuntimeError(
                f"cannot connect to the Robo-ValueRL server {self.server_url}: {exc}"
            ) from exc
        health = self._response_json(health_response, endpoint="/health")
        info = self._response_json(info_response, endpoint="/model_info")
        if (
            health.get("status") != "healthy"
            or health.get("reward_model") != "robo-valuerl"
        ):
            raise RuntimeError(f"invalid Robo-ValueRL /health identity: {health!r}")
        if info.get("reward_model") != "robo-valuerl":
            raise RuntimeError(f"invalid Robo-ValueRL /model_info identity: {info!r}")
        if info.get("upstream_commit") != _UPSTREAM_COMMIT:
            raise RuntimeError("Robo-ValueRL /model_info upstream commit drifted")
        weights = info.get("weights")
        expected_weights = {
            "repo_id": _WEIGHTS_REPO,
            "revision": _WEIGHTS_REVISION,
            "filename": _WEIGHTS_FILE,
            "sha256": _WEIGHTS_SHA256,
        }
        if not isinstance(weights, dict) or any(
            weights.get(key) != expected
            for key, expected in expected_weights.items()
        ):
            raise RuntimeError("Robo-ValueRL /model_info weight identity drifted")
        mismatches = _profile_mismatches(info.get("profile"))
        if mismatches:
            raise RuntimeError(
                "the Robo-ValueRL server does not match the pinned official profile: "
                + json.dumps(mismatches, sort_keys=True, ensure_ascii=False)
            )
        self._health = health
        self._model_info = info

    @staticmethod
    def _validate_trajectory(traj: Trajectory) -> None:
        expected_indices = np.arange(len(traj), dtype=np.int64)
        if not np.array_equal(traj.frame_indices, expected_indices):
            raise ValueError(
                "the official Robo-ValueRL history/stride profile needs a complete contiguous episode from 0; "
                f"actual frame_indices={traj.frame_indices.tolist()}"
            )
        if traj.sampling is not None and (
            traj.sampling.strategy != "all" or not traj.sampling.include_last
        ):
            raise ValueError(
                "the official Robo-ValueRL profile must use SamplingSpec(strategy='all', "
                "include_last=True); sparse sampling cannot pose as a 5-frame history"
            )

    @staticmethod
    def _npz_payload(traj: Trajectory) -> bytes:
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            primary=np.ascontiguousarray(traj.view("primary")),
            left_wrist=np.ascontiguousarray(traj.view("left_wrist")),
            right_wrist=np.ascontiguousarray(traj.view("right_wrist")),
            frame_indices=np.ascontiguousarray(traj.frame_indices),
            instruction=np.asarray(traj.instruction),
        )
        return buffer.getvalue()

    def _score(self, traj: Trajectory) -> RewardOutput:
        self._validate_trajectory(traj)
        self._ensure_server_identity()
        try:
            response = self._client().post(
                f"{self.server_url}/score_npz",
                data=self._npz_payload(traj),
                headers={"Content-Type": "application/x-npz"},
                timeout=self.timeout_s,
            )
        except Exception as exc:
            raise RuntimeError(f"Robo-ValueRL /score_npz request failed: {exc}") from exc
        payload = self._response_json(response, endpoint="/score_npz")
        if payload.get("schema") != _SCORE_SCHEMA:
            raise RuntimeError(
                f"the Robo-ValueRL score schema must be {_SCORE_SCHEMA!r}"
            )
        if payload.get("profile") != _PROFILE:
            raise RuntimeError("Robo-ValueRL score response profile drifted")

        try:
            frame_indices = np.asarray(payload.get("frame_indices"), dtype=np.int64)
            progress = np.asarray(payload.get("progress"), dtype=np.float64)
            remaining = np.asarray(payload.get("remaining_steps"), dtype=np.float64)
            key_indices = np.asarray(payload.get("key_frame_indices"), dtype=np.int64)
            key_remaining = np.asarray(
                payload.get("key_remaining_steps"), dtype=np.float64
            )
            key_probabilities = np.asarray(
                payload.get("key_probabilities"), dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Robo-ValueRL score response contains non-numeric arrays") from exc

        if frame_indices.ndim != 1 or not np.array_equal(
            frame_indices, traj.frame_indices
        ):
            raise RuntimeError("Robo-ValueRL response frame_indices do not match the request")
        expected_shape = (len(traj),)
        if progress.shape != expected_shape or remaining.shape != expected_shape:
            raise RuntimeError(
                "wrong Robo-ValueRL progress/remaining_steps length: "
                f"{progress.shape}/{remaining.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(progress)) or not np.all(np.isfinite(remaining)):
            raise RuntimeError("Robo-ValueRL progress/remaining_steps contain NaN/Inf")
        if np.any(progress < -1e-8) or np.any(progress > 1.0 + 1e-8):
            raise RuntimeError("Robo-ValueRL paper progress must be in [0,1]")
        expected_progress = np.clip(1.0 - remaining / 5000.0, 0.0, 1.0)
        if not np.allclose(progress, expected_progress, rtol=0.0, atol=1e-7):
            raise RuntimeError(
                "Robo-ValueRL progress does not equal the paper's clip(1-remaining_steps/5000,0,1)"
            )

        expected_key_indices = list(range(0, len(traj), 10))
        if expected_key_indices[-1] != len(traj) - 1:
            expected_key_indices.append(len(traj) - 1)
        expected_key_indices_array = np.asarray(expected_key_indices, dtype=np.int64)
        if key_indices.ndim != 1 or not np.array_equal(
            key_indices, expected_key_indices_array
        ):
            raise RuntimeError("Robo-ValueRL key_frame_indices do not match the official stride=10")
        if key_remaining.shape != (len(key_indices),):
            raise RuntimeError("invalid Robo-ValueRL key_remaining_steps shape")
        if key_probabilities.shape != (len(key_indices), 256):
            raise RuntimeError("Robo-ValueRL key_probabilities must be [K,256]")
        if not np.all(np.isfinite(key_remaining)) or not np.all(
            np.isfinite(key_probabilities)
        ):
            raise RuntimeError("Robo-ValueRL key-frame raw output contains NaN/Inf")
        if np.any(key_remaining < -1e-5) or np.any(key_remaining > 5000.0 + 1e-5):
            raise RuntimeError("Robo-ValueRL decoded key remaining steps are outside [0,5000]")
        if np.any(key_probabilities < -1e-7) or not np.allclose(
            key_probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-5
        ):
            raise RuntimeError("Robo-ValueRL key_probabilities are not a valid softmax distribution")

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
            raw={
                "remaining_steps": remaining.tolist(),
                "key_frame_indices": key_indices.tolist(),
                "key_remaining_steps": key_remaining.tolist(),
                "key_probabilities": key_probabilities.tolist(),
                "server_response_metadata": {
                    key: value
                    for key, value in payload.items()
                    if key
                    not in {
                        "progress",
                        "remaining_steps",
                        "key_frame_indices",
                        "key_remaining_steps",
                        "key_probabilities",
                        "frame_indices",
                    }
                },
            },
            model_name=self.spec.name,
            sampling=traj.sampling,
            meta={
                "n_backend_calls": len(key_indices),
                "n_http_calls": 1,
                "model_forwards": len(key_indices),
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
                "weights_sha256": _WEIGHTS_SHA256,
                "raw_semantics": (
                    "key-frame 256-bin remain_time probabilities; decoded/interpolated "
                    "remaining steps; paper-defined progress transform"
                ),
                "state_semantics": (
                    "server supplies structural zero state; released value forward "
                    "extracts but does not consume state"
                ),
                "release_annotation_contract": (
                    "published annotation script requests value/rtg/remain_time, but "
                    "released history critic returns remain_time only; server uses the "
                    "released field and official remaining-time decode/interpolation"
                ),
            },
        )

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
