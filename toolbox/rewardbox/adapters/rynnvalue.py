"""Official RynnValue Robometer-server adapters.

Pinned upstream:
  https://github.com/alibaba-damo-academy/RynnValue
  commit 10e0d333f5f3811d0d130587e50f1faf48da49e5

RynnValue's released inference stack requires Python 3.10, torch 2.8 and a
recent Qwen3-VL-capable Transformers build.  RewardBox deliberately keeps that
stack in the upstream Robometer server environment.  This module is only the
light HTTP boundary: it serializes one already-sampled ``Trajectory`` using
the multipart ``.npy`` protocol from the official ``example_inference.py``,
calls ``/evaluate_batch_npy``, validates the official raw value output, and
records endpoint/provenance identity.

The default profile is the released model's current official setting:

* eight uniformly sampled frames (NumPy ``rint`` linspace, first/last kept);
* one whole-trajectory call (``use_frame_steps=false``, ``pad_frames=false``);
* ``mode=absolute``: the raw score is negative remaining time in seconds, with
  the official all-``-100`` sentinel retained when Analysis reports Match: No;
* ``stride=2`` and ``num_frames=8`` as in the released RBM-EVAL-OOD command;
* model-config ``pred_slot_isolated_eager`` attention;
* greedy 128-token Analysis generation owned entirely by the server.

The upstream reproduction table also reports a legacy ``eager`` profile.  It
is available as the explicit ``paper-eager`` server profile, but is never
silently substituted for the model-default isolation profile.
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
from ..upstream import UpstreamNotFoundError, upstream_candidates

_UPSTREAM_COMMIT = "10e0d333f5f3811d0d130587e50f1faf48da49e5"
_DEFAULT_SERVER_URL = "http://127.0.0.1:8001"
_PROVENANCE_SCHEMA = "rewardbox.rynnvalue-server-provenance.v1"

_MODELS = {
    "rynnvalue-4b": {
        "model_id": "Alibaba-DAMO-Academy/RynnValue-4B",
        "revision": "3f73b5d2b5e53b21f248c8791004dde6a8cf2b92",
        "vram_gb": 24.0,
    },
    "rynnvalue-8b": {
        "model_id": "Alibaba-DAMO-Academy/RynnValue-8B",
        "revision": "8738c5e4ce4418ea0266e9fbeffae0eb9bbb230e",
        "vram_gb": 40.0,
    },
}

_PROFILES = {
    "official-isolated": {
        "effective_attention": "pred_slot_isolated_eager",
        "model_info_attention": {None, "pred_slot_isolated_eager"},
    },
    "paper-eager": {
        "effective_attention": "eager",
        "model_info_attention": {"eager"},
    },
}

_REQ = Requirements(
    pip=("requests>=2",),
    python=">=3.10",
    note=(
        "The RewardBox process only sends official multipart HTTP requests; RynnValue's Python "
        "3.10 / torch / transformers / Robometer dependencies must stay in the separate upstream environment."
    ),
    install_hint=(
        "uv pip install --python .venv/bin/python -e '.[rynnvalue-client]'",
        "uv sync --frozen --extra robometer --project reward_model_code/RynnValue/robometer",
        "start the pinned official service with scripts/run_rynnvalue_server.py",
    ),
)

_COMMON = dict(
    cls="RynnValueAdapter",
    module=__name__,
    reward_mode=RewardMode.PROGRESS,
    signal_type=SignalType.VALUE_HEAD,
    requires=_REQ,
    # Official RBM-EVAL-OOD released-checkpoint recipe: max_frames=8.  The
    # upstream helper uses np.rint(np.linspace(...)), exactly RewardBox's
    # ``uniform`` SamplingSpec semantics.
    sampling=SamplingSpec(num_frames=8, strategy="uniform", include_last=True),
    score_range=(-512.0, 0.0),
    higher_better=True,
    required_views=("primary",),
    supports_pairwise=False,
    outputs_success=False,
    zero_shot=True,
    paper="https://arxiv.org/abs/2608.09853",
    upstream=(
        "RynnValue/robometer/scripts/example_inference.py multipart client -> "
        "robometer/evals/baseline_eval_server.py -> "
        "baselines/rynnvalue.py:RynnValue.compute_progress"
    ),
    implementation_kind=ImplementationKind.OFFICIAL_ADAPTER,
)

SPECS = [
    ModelSpec(
        name=name,
        weights=f"{identity['model_id']}@{identity['revision']}",
        requires=Requirements(
            **{
                **_REQ.__dict__,
                "vram_gb": identity["vram_gb"],
            }
        ),
        description=(
            f"RynnValue {name.removeprefix('rynnvalue-').upper()} temporal-distance "
            "value model.  Returns the official raw negative remaining-time curve "
            "through the isolated Robometer HTTP server."
        ),
        **{key: value for key, value in _COMMON.items() if key != "requires"},
    )
    for name, identity in _MODELS.items()
]


def _repo_root() -> Path:
    """Locate the pinned official checkout without importing its dependencies."""
    explicit = os.environ.get("REWARDBOX_RYNNVALUE_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if (
            root / "robometer" / "robometer" / "evals" / "baseline_eval_server.py"
        ).is_file():
            return root
        raise RuntimeError(
            f"REWARDBOX_RYNNVALUE_ROOT={explicit!r} has no official baseline_eval_server.py"
        )
    for candidate in upstream_candidates("RynnValue"):
        if (
            candidate
            / "robometer"
            / "robometer"
            / "evals"
            / "baseline_eval_server.py"
        ).is_file():
            return candidate
    raise UpstreamNotFoundError(
        "vendored RynnValue not found. Set REWARDBOX_RYNNVALUE_ROOT, "
        "or put the official repository at reward_model_code/RynnValue."
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


def _validate_provenance(
    record: dict,
    *,
    expected_model: dict,
    profile: str,
    allow_unverified: bool,
) -> None:
    if record.get("schema") != _PROVENANCE_SCHEMA:
        raise ValueError(
            f"the RynnValue provenance schema must be {_PROVENANCE_SCHEMA!r}"
        )
    if record.get("status") != "ready":
        raise ValueError(
            "RynnValue server_provenance_file must be in the ready state; "
            f"actual {record.get('status')!r}"
        )
    upstream = record.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("commit") != _UPSTREAM_COMMIT:
        raise ValueError(
            f"the RynnValue server upstream commit must be {_UPSTREAM_COMMIT}"
        )
    weights = record.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("RynnValue provenance lacks a weights object")
    if weights.get("model_id") != expected_model["model_id"]:
        raise ValueError(
            "RynnValue server model_id does not match the adapter: "
            f"{weights.get('model_id')!r} != {expected_model['model_id']!r}"
        )
    if weights.get("expected_revision") != expected_model["revision"]:
        raise ValueError("RynnValue server expected_revision does not match the adapter pin")
    pinned = (
        weights.get("snapshot_revision") == expected_model["revision"]
        and weights.get("pinned_public_snapshot") is True
    )
    if not pinned and not allow_unverified:
        raise ValueError(
            "the RynnValue server weights are not a pinned public snapshot; private/copied weights "
            "require passing allow_unverified_server=true explicitly"
        )
    profile_record = record.get("profile")
    if not isinstance(profile_record, dict) or profile_record.get("name") != profile:
        raise ValueError(
            f"the RynnValue server profile must be {profile!r}, actual "
            f"{profile_record.get('name') if isinstance(profile_record, dict) else None!r}"
        )
    expected_attention = _PROFILES[profile]["effective_attention"]
    if profile_record.get("effective_attention") != expected_attention:
        raise ValueError(
            "RynnValue server effective_attention does not match the profile: "
            f"{profile_record.get('effective_attention')!r} != {expected_attention!r}"
        )
    expected_profile = {
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
    }
    mismatches = {
        key: {"expected": value, "actual": profile_record.get(key)}
        for key, value in expected_profile.items()
        if profile_record.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "RynnValue provenance official profile fields mismatch: "
            + json.dumps(mismatches, sort_keys=True, ensure_ascii=False)
        )


def _json_metadata(meta: dict) -> dict:
    try:
        # Round-trip also detaches mutable caller-owned metadata from the request.
        return json.loads(json.dumps(meta, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Trajectory.meta must encode as finite JSON to be sent to the official RynnValue server"
        ) from exc


class RynnValueAdapter(BaseAdapter):
    """Thin client for the pinned official RynnValue Robometer server."""

    def __init__(
        self,
        spec: ModelSpec,
        server_url: str | None = None,
        timeout_s: float = 600.0,
        profile: str = "official-isolated",
        server_provenance_file: str | None = None,
        allow_unverified_server: bool = False,
        robot_description: str | None = None,
        camera_description: str | None = None,
        data_source: str | None = None,
        sample_id: str | None = None,
        **opts: Any,
    ):
        super().__init__(spec, **opts)
        if spec.name not in _MODELS:
            raise ValueError(f"unknown RynnValue spec: {spec.name!r}")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError("timeout_s must be a finite positive number")
        if not math.isfinite(float(timeout_s)) or timeout_s <= 0:
            raise ValueError("timeout_s must be a finite positive number")
        if profile not in _PROFILES:
            raise ValueError(f"profile must be one of {sorted(_PROFILES)}")
        if not isinstance(allow_unverified_server, bool):
            raise ValueError("allow_unverified_server must be a bool")
        if server_provenance_file is not None and (
            not isinstance(server_provenance_file, str)
            or not server_provenance_file.strip()
        ):
            raise ValueError("server_provenance_file, when given, must be a non-empty path string")
        for name, value in {
            "robot_description": robot_description,
            "camera_description": camera_description,
            "data_source": data_source,
            "sample_id": sample_id,
        }.items():
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name}, when given, must be a non-empty string")

        self.server_url = _validate_url(
            server_url
            or os.environ.get("REWARDBOX_RYNNVALUE_SERVER_URL")
            or _DEFAULT_SERVER_URL
        )
        self.timeout_s = float(timeout_s)
        self.profile = profile
        self.allow_unverified_server = allow_unverified_server
        self.robot_description = (
            robot_description.strip() if robot_description is not None else None
        )
        self.camera_description = (
            camera_description.strip() if camera_description is not None else None
        )
        self.data_source = data_source.strip() if data_source is not None else None
        self.sample_id = sample_id.strip() if sample_id is not None else None
        self.expected_model = _MODELS[spec.name]
        self.server_provenance_file = (
            Path(server_provenance_file).expanduser().resolve()
            if server_provenance_file is not None
            else None
        )
        if self.server_provenance_file is None:
            if not allow_unverified_server:
                raise ValueError(
                    "RynnValue requires server_provenance_file by default, binding the official source, "
                    "weight revision and inference profile. Pass allow_unverified_server=true "
                    "explicitly only to diagnose an unpinned service."
                )
            self._server_provenance = None
        else:
            self._server_provenance = _read_provenance(self.server_provenance_file)
            _validate_provenance(
                self._server_provenance,
                expected_model=self.expected_model,
                profile=self.profile,
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
                f"RynnValue {endpoint} HTTP failure: {exc}; body={excerpt!r}"
            ) from exc
        try:
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"RynnValue {endpoint} returned non-JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"RynnValue {endpoint} JSON top level is not an object")
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
                f"cannot connect to the RynnValue server {self.server_url}: {exc}"
            ) from exc
        health = self._response_json(health_response, endpoint="/health")
        info = self._response_json(info_response, endpoint="/model_info")
        if health.get("status") != "healthy" or health.get("reward_model") != "rynnvalue":
            raise RuntimeError(f"invalid RynnValue /health identity: {health!r}")
        if info.get("reward_model") != "rynnvalue":
            raise RuntimeError(f"invalid RynnValue /model_info identity: {info!r}")
        config = info.get("config")
        model_config = config.get("model_config") if isinstance(config, dict) else None
        if not isinstance(model_config, dict):
            raise RuntimeError("RynnValue /model_info lacks config.model_config")

        expected = {
            "mode": "absolute",
            "stride": 2,
            "num_frames": 8,
            "max_new_tokens": 128,
            "use_fuse": False,
        }
        mismatches = {
            key: {"expected": value, "actual": model_config.get(key)}
            for key, value in expected.items()
            if model_config.get(key) != value
        }
        if config.get("use_frame_steps") is not False:
            mismatches["use_frame_steps"] = {
                "expected": False,
                "actual": config.get("use_frame_steps"),
            }
        if config.get("max_frames") != 8:
            mismatches["max_frames"] = {
                "expected": 8,
                "actual": config.get("max_frames"),
            }
        custom_eval = config.get("custom_eval")
        if not isinstance(custom_eval, dict):
            mismatches["custom_eval"] = {
                "expected": "object with use_frame_steps=false and pad_frames=false",
                "actual": custom_eval,
            }
        else:
            if custom_eval.get("use_frame_steps") is not False:
                mismatches["custom_eval.use_frame_steps"] = {
                    "expected": False,
                    "actual": custom_eval.get("use_frame_steps"),
                }
            if custom_eval.get("pad_frames") is not False:
                mismatches["custom_eval.pad_frames"] = {
                    "expected": False,
                    "actual": custom_eval.get("pad_frames"),
                }
        attention = model_config.get("attn_implementation")
        if attention not in _PROFILES[self.profile]["model_info_attention"]:
            mismatches["attn_implementation"] = {
                "expected_profile": self.profile,
                "actual": attention,
            }
        checkpoint = model_config.get("checkpoint_path")
        if checkpoint not in {None, ""}:
            mismatches["checkpoint_path"] = {
                "expected": None,
                "actual": checkpoint,
            }
        if mismatches:
            raise RuntimeError(
                "the RynnValue server does not match the pinned official profile: "
                + json.dumps(mismatches, sort_keys=True, ensure_ascii=False)
            )
        self._health = health
        self._model_info = info

    def _multipart(self, traj: Trajectory) -> tuple[dict, dict]:
        frames = np.ascontiguousarray(traj.primary)
        metadata = _json_metadata(traj.meta)
        for key, value in {
            "robot_description": self.robot_description,
            "camera_description": self.camera_description,
            "data_source": self.data_source,
            "id": self.sample_id,
        }.items():
            if value is not None:
                metadata.setdefault(key, value)
        if not any(
            metadata.get(key)
            for key in ("robot_description", "camera_description", "data_source", "id")
        ):
            raise ValueError(
                "the public RynnValue checkpoint has processor.use_meta=true; Trajectory.meta "
                "or the model options must provide robot_description, camera_description, "
                "data_source or an id recognised by the official lookup; RewardBox does not invent "
                "robot/camera descriptions."
            )
        sample_id = metadata.get("id") or traj.source or "rewardbox"
        sample_id = str(sample_id)
        data_source = metadata.get("data_source")
        if data_source is not None and not isinstance(data_source, str):
            raise ValueError("Trajectory.meta.data_source must be a string or None")

        file_key = "sample_0_trajectory_frames"
        buffer = io.BytesIO()
        np.save(buffer, frames, allow_pickle=False)
        buffer.seek(0)
        files = {
            file_key: (
                f"{file_key}.npy",
                buffer,
                "application/octet-stream",
            )
        }
        sample = {
            "sample_type": "progress",
            "trajectory": {
                "frames": {"__numpy_file__": file_key},
                "frames_shape": list(frames.shape),
                "task": traj.instruction,
                "id": sample_id,
                "data_source": data_source,
                "metadata": metadata,
                "video_embeddings": None,
            },
        }
        data = {
            "sample_0": json.dumps(sample, ensure_ascii=False, allow_nan=False),
            # Explicitly select the official policy-ranking whole-trajectory path;
            # do not inherit start_server.sh's frame-step default.
            "use_frame_steps": "false",
        }
        return files, data

    def _score(self, traj: Trajectory) -> RewardOutput:
        self._ensure_server_identity()
        files, data = self._multipart(traj)
        try:
            response = self._client().post(
                f"{self.server_url}/evaluate_batch_npy",
                files=files,
                data=data,
                timeout=self.timeout_s,
            )
        except Exception as exc:
            raise RuntimeError(f"RynnValue /evaluate_batch_npy request failed: {exc}") from exc
        payload = self._response_json(response, endpoint="/evaluate_batch_npy")
        outputs = payload.get("outputs_progress")
        predictions = outputs.get("progress_pred") if isinstance(outputs, dict) else None
        if not isinstance(predictions, list) or len(predictions) != 1:
            raise RuntimeError(
                "the RynnValue service must return exactly one progress_pred for a single trajectory"
            )
        try:
            progress = np.asarray(predictions[0], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("RynnValue progress_pred is not a numeric sequence") from exc
        if progress.ndim != 1 or len(progress) != len(traj):
            raise RuntimeError(
                f"wrong RynnValue progress length/dimension: {progress.shape}, expected ({len(traj)},)"
            )
        if not np.all(np.isfinite(progress)):
            raise RuntimeError("RynnValue progress contains NaN/Inf")
        if np.any(progress < -512.0 - 1e-6) or np.any(progress > 1e-6):
            raise RuntimeError(
                "RynnValue absolute mode should return negative remaining-time seconds in [-512,0], "
                f"actual range [{progress.min()}, {progress.max()}]"
            )

        success = None
        success_values = outputs.get("success_probs") if isinstance(outputs, dict) else None
        if success_values is not None:
            if not isinstance(success_values, list) or len(success_values) != 1:
                raise RuntimeError("invalid RynnValue success_probs batch shape")
            success = np.asarray(success_values[0], dtype=np.float64)
            if success.ndim != 1 or len(success) != len(progress):
                raise RuntimeError("RynnValue success_probs length does not match progress")
            if not np.all(np.isfinite(success)) or np.any((success < 0) | (success > 1)):
                raise RuntimeError("RynnValue success_probs must be finite [0,1] probabilities")

        # The official baseline overwrites every score with -100 when its
        # generated Analysis says Match: No.  The HTTP response omits the parsed
        # Analysis, so this exact all-sentinel pattern is the strongest signal
        # available to the client.  Preserve the values and record the pattern;
        # never reinterpret -100 as ordinary calibrated remaining time.
        mismatch_sentinel_pattern = bool(np.all(progress == -100.0))

        pinned_weights = bool(
            self._server_provenance
            and self._server_provenance.get("weights", {}).get(
                "pinned_public_snapshot"
            )
        )
        return RewardOutput(
            progress=progress,
            frame_indices=traj.frame_indices,
            success=success,
            raw={"server_response": payload},
            model_name=self.spec.name,
            sampling=traj.sampling,
            meta={
                "n_backend_calls": 2,
                "n_http_calls": 1,
                "model_forwards": 1,
                "analysis_generations": 1,
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
                "official_profile": {
                    "name": self.profile,
                    "max_frames": 8,
                    "sampling": "np.rint(linspace(0,T-1,8)); first/last included",
                    "use_frame_steps": False,
                    "pad_frames": False,
                    "mode": "absolute",
                    "raw_semantics": (
                        "negative_remaining_time_seconds_or_all_-100_"
                        "analysis_match_no_sentinel"
                    ),
                    "stride": 2,
                    "num_frames": 8,
                    "effective_attention": _PROFILES[self.profile][
                        "effective_attention"
                    ],
                    "max_new_tokens": 128,
                    "do_sample": False,
                    "num_beams": 1,
                    "use_fuse": False,
                },
                "upstream_commit": _UPSTREAM_COMMIT,
                "weights_revision": self.expected_model["revision"],
                "official_mismatch_sentinel_pattern": mismatch_sentinel_pattern,
                "official_mismatch_sentinel_value": -100.0,
                "analysis_note": (
                    "official server generates Analysis for this path but its HTTP response "
                    "currently exposes only progress/success arrays; an all--100 curve is "
                    "the official Match: No sentinel pattern"
                ),
            },
        )

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
