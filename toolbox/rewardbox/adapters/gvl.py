"""GVL adapter for the pinned Robometer upstream implementation.

Official entry point:
    robometer.evals.baselines.gvl:GVL.compute_progress

The pinned upstream repository ships a GVL command only for
``confusion_matrix``.  That command fixes Gemini 2.5 Flash Lite and
``max_frames=8``.  It does *not* ship a reward-alignment or policy-ranking
command, so this adapter deliberately makes no Table-I / paper-equivalence
claim.  ``gvl`` preserves that frozen provider identity;
``gvl-gemini-3.5-flash-lite`` records Google's forced new-user migration as a
separate identity while retaining the same upstream GVL transport and parser.
``gvl-gemini-3.8-flash`` is a separate extension: it reuses the same
JPEG/shuffle/prompt/parser pipeline and replaces only the HTTP transport with
a non-streaming Gemini native generateContent request, sent to Google's
endpoint or to a Gemini-format gateway that the caller configures.
Its separately named ``-dense`` profile accepts the runner's sampled frame grid
(default 1 Hz, matching SOLE's current input grid), passing every sampled image
through that same pipeline in one globally shuffled request, with no fixed
frame-count truncation.

The shared upstream computation owns JPEG encoding, global-RNG shuffling,
prompt construction, JSON parsing, and restoring chronological order. Direct
Google profiles also retain its HTTP transport and retries. This wrapper
converts the returned sparse grid into :class:`RewardOutput` and records that
grid. A missing upstream prediction is never filled in or normalized.

The full command's sampler pads a short trajectory by repeating its last frame.
RewardOutput intentionally forbids duplicate frame indices, so this strict
eight-frame adapter accepts only the command's already-unpadded input case.
Short or long inputs are rejected before an API request, because their result
would not match the command's.
"""
from __future__ import annotations

import importlib
import math
import os
import re
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from ..core.base import BaseAdapter
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import RewardOutput, SamplingSpec, Trajectory
from ..upstream import UpstreamNotFoundError, upstream_candidates


_RECIPE_PATH = "reward_model_code/robometer/eval_commands/confusion_matrix.sh"
_UPSTREAM_COMMIT = "5b815254bf31ee1bea3753c3a2da9f9033736d9a"
_OFFICIAL_MODEL_NAME = "gemini-2.5-flash-lite"
_CURRENT_MODEL_NAME = "gemini-3.5-flash-lite"
_HTTP_PROFILE_NAME = "gvl-gemini-3.8-flash"
_DENSE_HTTP_PROFILE_NAME = "gvl-gemini-3.8-flash-dense"
_HTTP_MODEL_NAME = "gemini-3.8-flash"
_HTTP_BASE_URL = "https://generativelanguage.googleapis.com"
_HTTP_AUTH_HEADERS = ("x-goog-api-key", "bearer")
_HTTP_PROFILE_NAMES = frozenset({_HTTP_PROFILE_NAME, _DENSE_HTTP_PROFILE_NAME})
_PROFILE_MODELS = {
    "gvl": _OFFICIAL_MODEL_NAME,
    "gvl-gemini-3.5-flash-lite": _CURRENT_MODEL_NAME,
    _HTTP_PROFILE_NAME: _HTTP_MODEL_NAME,
    _DENSE_HTTP_PROFILE_NAME: _HTTP_MODEL_NAME,
}


def provider_model_for_profile(name: str) -> str:
    """Keep the routed model name bound to a separately registered identity."""
    try:
        return _PROFILE_MODELS[name]
    except KeyError:
        raise ValueError(f"unknown GVL provider profile: {name!r}") from None


def credential_env_for_profile(name: str) -> str:
    """Default credential variable; the HTTP profiles also accept ``api_key_env``."""
    if name not in _PROFILE_MODELS:
        raise ValueError(f"unknown GVL provider profile: {name!r}")
    return "GEMINI_API_KEY"


_REQ = Requirements(
    # ``GVL`` imports these modules directly.  It makes raw HTTPS requests;
    # google-generativeai is *not* used by this entry point.
    pip=("numpy>=1.21", "requests>=2.32.4", "opencv-python-headless>=4.8"),
    python=">=3.10",
    note=(
        "GVL has no local checkpoint: it sends JPEG/base64 images straight to the Gemini HTTP API. "
        "The strict path needs GEMINI_API_KEY (api_key can also be passed explicitly, but it is never written to result meta). "
        "gvl keeps the upstream 2.5 frozen recipe; gvl-gemini-3.5-flash-lite is the provider-forced migration identity. "
        "The upstream Robometer "
        "project as a whole declares Python ==3.10.*; this light GVL entry point itself only needs numpy/requests/cv2."
    ),
    install_hint=(
        "uv pip install --python .venv/bin/python requests opencv-python-headless",
        "set REWARDBOX_ROBOMETER_ROOT (or keep the vendored reward_model_code/robometer)",
    ),
)

SPEC = ModelSpec(
    name="gvl",
    cls="GVLAdapter",
    module=__name__,
    reward_mode=RewardMode.PROGRESS,
    signal_type=SignalType.GENERATED_TEXT,
    requires=_REQ,
    # The only shipped GVL recipe is the confusion-matrix command.  Its
    # ``max_frames=8`` is a hard cap in the official class, not an adapter
    # tuning parameter.
    sampling=SamplingSpec(num_frames=8, strategy="uniform"),
    score_range=(0.0, 1.0),
    higher_better=True,
    supports_pairwise=False,
    outputs_success=False,
    paper="https://arxiv.org/abs/2603.02115",
    weights="",
    upstream="robometer.evals.baselines.gvl:GVL.compute_progress",
    description=(
        "Robometer's API-based Gemini progress baseline. Strictly reuses the 8-frame sub-protocol "
        "expressible by its confusion-matrix GVL command (Gemini 2.5 Flash Lite); upstream released no "
        "reward-alignment/policy-ranking recipe, so Table I equivalence is not claimed."
    ),
    implementation_kind=ImplementationKind.OFFICIAL_ADAPTER,
)
CURRENT_SPEC = ModelSpec(
    name="gvl-gemini-3.5-flash-lite",
    cls="GVLAdapter",
    module=__name__,
    reward_mode=RewardMode.PROGRESS,
    signal_type=SignalType.GENERATED_TEXT,
    requires=_REQ,
    sampling=SamplingSpec(num_frames=8, strategy="uniform"),
    score_range=(0.0, 1.0),
    higher_better=True,
    supports_pairwise=False,
    outputs_success=False,
    paper="https://arxiv.org/abs/2603.02115",
    weights="",
    upstream="robometer.evals.baselines.gvl:GVL.compute_progress",
    description=(
        "Robometer GVL's official JPEG/shuffle/prompt/SSE/parser path, explicitly migrated to "
        "Gemini 3.5 Flash Lite. Google refuses the frozen 2.5 endpoint for new users; "
        "this identity does not pose as the upstream confusion-matrix recipe or a Table I reproduction."
    ),
    implementation_kind=ImplementationKind.OFFICIAL_ADAPTER,
)
HTTP_SPEC = ModelSpec(
    name=_HTTP_PROFILE_NAME,
    cls="GVLHttpAdapter",
    module=__name__,
    reward_mode=RewardMode.PROGRESS,
    signal_type=SignalType.GENERATED_TEXT,
    requires=Requirements(
        pip=_REQ.pip,
        python=_REQ.python,
        note=(
            "Calls gemini-3.8-flash with a non-streaming Gemini native generateContent request. "
            "Default: Google's endpoint with GEMINI_API_KEY; for a Gemini-format gateway set base_url, "
            "api_key_env and auth_header=bearer. "
            "Reuses the pinned Robometer 8-frame JPEG/shuffle/prompt/parser; "
            "the model and the non-streaming HTTP transport are explicit extensions; no real-endpoint or quality validation yet."
        ),
        install_hint=_REQ.install_hint,
    ),
    sampling=SamplingSpec(num_frames=8, strategy="uniform"),
    score_range=(0.0, 1.0),
    higher_better=True,
    supports_pairwise=False,
    outputs_success=False,
    paper="https://arxiv.org/abs/2411.04549",
    weights="",
    upstream="robometer.evals.baselines.gvl:GVL.compute_progress",
    description=(
        "GVL + Gemini 3.8 Flash in Gemini native format; "
        "keeps the existing 8-frame scoring flow and records the model/transport migration separately; no paper reproduction is claimed."
    ),
    implementation_kind=ImplementationKind.REWARDBOX_EXTENSION,
)
DENSE_HTTP_SPEC = replace(
    HTTP_SPEC,
    name=_DENSE_HTTP_PROFILE_NAME,
    cls="GVLDenseHttpAdapter",
    # Reuse the existing SOLE input grid. num_frames is ignored by fps sampling;
    # retaining its value also preserves exact SamplingSpec/key compatibility.
    sampling=SamplingSpec(strategy="fps", fps=1.0, include_last=False),
    requires=replace(
        HTTP_SPEC.requires,
        note=(
            "Gemini 3.8 Flash; credential GEMINI_API_KEY (or the variable named by api_key_env). "
            "Reuses the existing SOLE 1 Hz input grid by default; the runner may choose fps/interval/all; "
            "the frame count grows with duration and sampling settings, never truncated by num_frames. "
            "All images are shuffled globally within one logical request, reusing the original GVL initial reference, prompt, "
            "JPEG encoding, JSON parsing and un-shuffling, with no chunking, interpolation or fixed-frame-count truncation. "
            "The sampling density is a separate extension protocol; no real-endpoint or quality validation is claimed."
        ),
    ),
    description=(
        "GVL + Gemini 3.8 Flash; reuses the runner's sampling grid, aligned with SOLE 1 Hz by default. "
        "Keeps GVL's global random shuffle and the same prompt/parser; one request covers all sampled frames."
    ),
)
SPECS = [SPEC, CURRENT_SPEC, HTTP_SPEC, DENSE_HTTP_SPEC]


def validate_dense_sampling(sampling: SamplingSpec | None) -> None:
    """Dense GVL follows an external grid without a fixed frame-count budget."""
    if sampling is None or sampling.strategy not in {"fps", "interval", "all"}:
        raise ValueError("Dense GVL requires fps/interval/all sampling, not fixed-frame-count sampling")


def _repo_root() -> Path:
    """Return the pinned Robometer checkout without importing it at discovery."""
    explicit = os.environ.get("REWARDBOX_ROBOMETER_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if (root / "robometer" / "evals" / "baselines" / "gvl.py").exists():
            return root
        raise RuntimeError(
            f"REWARDBOX_ROBOMETER_ROOT={explicit!r} has no "
            "robometer/evals/baselines/gvl.py"
        )

    for candidate in upstream_candidates("robometer"):
        if (candidate / "robometer" / "evals" / "baselines" / "gvl.py").exists():
            return candidate
    raise UpstreamNotFoundError(
        "vendored Robometer GVL source not found. Set REWARDBOX_ROBOMETER_ROOT, "
        "or put the official repository at reward_model_code/robometer."
    )


def _upstream_gvl_class():
    """Load exactly the vendored GVL class, rejecting an accidental other checkout."""
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("robometer.evals.baselines.gvl")
    source = Path(module.__file__).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            "the loaded robometer package does not come from REWARDBOX_ROBOMETER_ROOT / the vendored checkout; "
            "use an isolated worker to avoid silently calling another GVL version."
        ) from exc
    return module.GVL


def _redact_api_secret(text: str, secret: str | None) -> str:
    """Remove Gemini credentials from upstream prints and exception strings."""
    value = str(text)
    if secret:
        value = value.replace(str(secret), "[REDACTED]")
    return re.sub(
        r"(?i)([?&]key=)[^&\s\"']+",
        r"\1[REDACTED]",
        value,
    )


class _RedactingWriter:
    """Forward text while filtering the query-string key used by upstream."""

    def __init__(self, target, secret: str | None):
        self._target = target
        self._secret = secret

    def write(self, value):
        return self._target.write(_redact_api_secret(value, self._secret))

    def flush(self):
        return self._target.flush()

    def __getattr__(self, name):
        return getattr(self._target, name)


class GVLAdapter(BaseAdapter):
    """Call the pinned upstream GVL object without altering its API protocol."""

    _fixed_frame_count = 8

    def __init__(
        self,
        spec: ModelSpec,
        api_key: str | None = None,
        **opts: Any,
    ):
        """Use the sole shipped Gemini GVL confusion-matrix recipe exactly.

        ``api_key=None`` intentionally delegates GEMINI_API_KEY lookup to
        upstream. The supplied value is private and is never copied to output
        metadata. The upstream source also has generic OpenAI/configurable
        branches, but exposing them here would be a non-command-equivalent
        protocol; add a separately evidenced adapter/profile if needed.
        """
        super().__init__(spec, **opts)
        self._api_key = api_key
        self.provider = "gemini"
        try:
            self.model_name = _PROFILE_MODELS[spec.name]
        except KeyError as exc:
            raise ValueError(f"unknown GVL provider profile: {spec.name!r}") from exc
        self._provider_migration = self.model_name != _OFFICIAL_MODEL_NAME
        self.offset = 0.5
        self.max_retries = 5
        self.base_delay = 1.0
        self.max_delay = 60.0
        self.max_frames = spec.sampling.num_frames
        if self._fixed_frame_count is not None and self.max_frames != self._fixed_frame_count:
            raise ValueError("the strict GVL confusion-matrix adapter only supports max_frames=8")
        self._client = None

    def client(self):
        if self._client is None:
            GVL = _upstream_gvl_class()
            self._client = GVL(
                api_key=self._api_key,
                max_frames=self.max_frames,
                offset=self.offset,
                model_name=self.model_name,
                provider=self.provider,
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                max_delay=self.max_delay,
            )
        return self._client

    def _returned_positions(self, client, predictions: np.ndarray) -> tuple[list[int], list[int]]:
        """Recover the official sparse output grid, with no guessed values.

        A malformed/partial model JSON response makes upstream omit the affected
        frame from its returned list.  We use its post-call ``frames_info`` to
        retain that omission rather than broadcasting or carrying a prior value.
        If a future upstream changes this private bookkeeping, fail loudly rather
        than silently mislabel predictions.
        """
        infos = getattr(client, "frames_info", None)
        if not isinstance(infos, list):
            raise RuntimeError("the official GVL object kept no frames_info; output frame numbers cannot be mapped safely")
        try:
            ordered = sorted(infos, key=lambda info: info["gt_index"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError("the official GVL frames_info lacks a usable gt_index") from exc
        if [info["gt_index"] for info in ordered] != list(range(1, len(ordered) + 1)):
            raise RuntimeError("the gt_index of the official GVL frames_info is no longer a contiguous sampling grid")

        # With max_frames equal to the supplied grid size, upstream's int(.5 + i)
        # branch extracts every supplied frame once. Thus gt_index
        # maps directly to the caller's 0-based, already-upstream-sampled grid.
        if len(ordered) != self.max_frames:
            raise RuntimeError(
                f"the frame count after official GVL JPEG encoding is not the expected {self.max_frames}; "
                "original frame numbers cannot be recovered safely"
            )
        returned = [info for info in ordered if info.get("model_output") is not None]
        if len(predictions) != len(returned):
            raise RuntimeError(
                f"official GVL returned {len(predictions)} progress values but frames_info maps to "
                f"{len(returned)}; refusing to guess the missing frames."
            )
        local_positions = [info["gt_index"] - 1 for info in returned]
        try:
            shuffled_indices = [int(info["shuffled_index"]) for info in returned]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("the official GVL frames_info lacks a usable shuffled_index") from exc
        return local_positions, shuffled_indices

    def _validate_trajectory(self, traj: Trajectory) -> None:
        if traj.sampling != self.spec.sampling:
            raise ValueError(
                "the strict GVL confusion-matrix protocol requires Trajectory.sampling to equal "
                "the SPEC's 8-frame uniform sampling exactly"
            )
        if len(traj) != self.max_frames:
            raise ValueError(
                "the strict GVL confusion-matrix protocol requires exactly 8 frames: the upstream sampler pads "
                "short trajectories by repeating the last frame, while RewardOutput frame_indices must be strictly increasing; "
                "to avoid faking command equivalence, short/long inputs are rejected before the API call."
            )

    def _prepare_client(self, traj: Trajectory):
        return self.client()

    def _score(self, traj: Trajectory) -> RewardOutput:
        self._validate_trajectory(traj)
        # np.ascontiguousarray is representation-only: upstream indexes the array
        # directly and requires [T,H,W,3] RGB uint8, already guaranteed by Trajectory.
        frames = np.ascontiguousarray(traj.primary)
        client = self._prepare_client(traj)
        secret = getattr(client, "api_key", None) or self._api_key
        try:
            # The pinned Gemini client places its key in the request URL and prints
            # RequestException strings while retrying.  Preserve its retry/protocol
            # behavior but never let that query parameter reach RewardBox logs.
            # Upstream also prints normal progress to stdout; route it to stderr so
            # ``rewardbox run gvl`` keeps stdout machine-readable JSON.
            with (
                redirect_stdout(_RedactingWriter(sys.stderr, secret)),
                redirect_stderr(_RedactingWriter(sys.stderr, secret)),
            ):
                official_progress = client.compute_progress(
                    frames, task_description=traj.instruction
                )
        except Exception as exc:
            message = _redact_api_secret(str(exc), secret)
            raise RuntimeError(f"official GVL API call failed: {message}") from None
        predictions = np.asarray(official_progress, dtype=np.float64)
        if predictions.ndim != 1 or len(predictions) == 0:
            raise RuntimeError("the official GVL entry point returned no non-empty progress usable for RewardOutput")
        if not np.all(np.isfinite(predictions)):
            raise RuntimeError("the official GVL entry point returned NaN/Inf progress")

        local_positions, shuffled_indices = self._returned_positions(client, predictions)
        parsed_frames = [
            {
                "gt_index": int(info["gt_index"]),
                "shuffled_index": int(info["shuffled_index"]),
                "model_output": info.get("model_output"),
            }
            for info in sorted(client.frames_info, key=lambda item: item["gt_index"])
        ]
        return self._out(
            predictions,
            traj,
            frame_indices=traj.frame_indices[np.asarray(local_positions, dtype=np.int64)],
            weights=None,
            raw={
                "official_progress": predictions.tolist(),
                "sampled_local_indices": list(range(self.max_frames)),
                "returned_local_indices": local_positions,
                "returned_shuffled_indices": shuffled_indices,
                # Keep the official parsed JSON (including frame descriptions) but
                # intentionally omit each frame's large base64 JPEG payload.
                "official_parsed_frames": parsed_frames,
            },
            meta={
                "adapter_calls_official": True,
                "protocol": (
                    "upstream_gvl_provider_migration_strict_8_frame"
                    if self._provider_migration else
                    "upstream_confusion_matrix_recipe_strict_8_frame"
                ),
                "recipe_evidence": _RECIPE_PATH,
                "paper_alignment_equivalence": (
                    "not asserted: provider model differs from the pinned upstream "
                    "confusion-matrix recipe"
                    if self._provider_migration else
                    "not asserted: pinned upstream ships no GVL reward_alignment "
                    "or policy_ranking command"
                ),
                "upstream_commit": _UPSTREAM_COMMIT,
                "provider": self.provider,
                "model_name": self.model_name,
                "upstream_recipe_model_name": _OFFICIAL_MODEL_NAME,
                "provider_model_migration": self._provider_migration,
                "remote_model_revision": None,
                "provider_managed_model_revision_pinned": False,
                "credential_source": (
                    "explicit_constructor" if self._api_key is not None
                    else "GEMINI_API_KEY_environment"
                ),
                "official_args": {
                    "max_frames": self.max_frames,
                    "offset": self.offset,
                    "max_retries": self.max_retries,
                    "base_delay": self.base_delay,
                    "max_delay": self.max_delay,
                },
                "normalized": False,
                "imputed_missing_predictions": False,
                "source_sampler_padding": "not needed: strict input has exactly 8 frames",
                "shuffle": "upstream random.shuffle (no seed parameter)",
                "n_backend_calls": 1,
            },
        )

    def close(self) -> None:
        self._client = None


class _NativeGeminiTransport:
    """Only the provider HTTP boundary; GVL still builds and parses its prompt.

    A non-streaming generateContent request with native Gemini contents/parts
    replaces upstream's SSE transport. The key travels in ``x-goog-api-key``
    (Google) or in a Bearer header (gateways); no generation settings, image
    encoding, or progress semantics are changed.
    """

    def __init__(self, adapter, api_key: str):
        self.adapter = adapter
        self.api_key = api_key
        self.http_attempts = 0
        self.response_metadata: dict[str, Any] = {}

    def __call__(self, parts: list[dict]) -> str:
        import requests

        adapter = self.adapter
        url = f"{adapter.base_url}/v1beta/models/{adapter.model_name}:generateContent"
        headers = {"Content-Type": "application/json"}
        if adapter.auth_header == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers[adapter.auth_header] = self.api_key
        # Keep the exact native parts produced by the pinned upstream GVL.
        body = {"contents": [{"parts": parts}]}
        self.http_attempts = 0
        self.response_metadata = {}
        for attempt in range(adapter.max_retries + 1):
            self.http_attempts += 1
            try:
                with requests.post(
                    url, headers=headers, json=body,
                    timeout=(10.0, adapter.timeout_s), allow_redirects=False,
                ) as response:
                    retryable = response.status_code == 429 or response.status_code >= 500
                    if not retryable or attempt == adapter.max_retries:
                        if not 200 <= response.status_code < 300:
                            try:
                                error = response.json().get("error", {})
                                detail = error.get("message", "") if isinstance(error, dict) else str(error)
                            except (ValueError, AttributeError):
                                detail = ""
                            detail = _redact_api_secret(str(detail), self.api_key)[:1000]
                            raise RuntimeError(f"Gemini API HTTP {response.status_code}: {detail}")
                        try:
                            payload = response.json()
                        except ValueError:
                            raise RuntimeError("Gemini API did not return Gemini native JSON") from None
                        break
            except requests.RequestException as exc:
                if attempt == adapter.max_retries:
                    message = _redact_api_secret(str(exc), self.api_key)
                    raise RuntimeError(f"Gemini API request failed: {message}") from None
            time.sleep(min(adapter.base_delay * (2**attempt), adapter.max_delay))

        if not isinstance(payload, dict):
            raise RuntimeError("Gemini API response must be a JSON object")
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
            raise RuntimeError("Gemini API response lacks candidates; no progress prediction produced")
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason")
        if finish_reason not in (None, "STOP"):
            raise RuntimeError(f"Gemini API did not finish normally: finishReason={finish_reason}")
        content = candidate.get("content")
        response_parts = content.get("parts", []) if isinstance(content, dict) else []
        if not isinstance(response_parts, list):
            raise RuntimeError("content.parts of the Gemini API response must be an array")
        # Gemini can return reasoning parts; only answer text belongs in GVL's parser.
        answer = "".join(
            part["text"] for part in response_parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
            and part.get("thought") is not True
        )
        if not answer.strip():
            raise RuntimeError("Gemini API response contains no non-empty answer text")
        usage = payload.get("usageMetadata")
        self.response_metadata = {
            "provider_reported_model_version": payload.get("modelVersion"),
            "provider_response_id": payload.get("responseId"),
            "provider_usage_metadata": {
                key: usage[key] for key in (
                    "promptTokenCount", "candidatesTokenCount", "totalTokenCount",
                    "cachedContentTokenCount", "thoughtsTokenCount",
                ) if key in usage
            } if isinstance(usage, dict) else {},
        }
        return answer


class GVLHttpAdapter(GVLAdapter):
    """GVL's existing eight-frame computation with a non-streaming native Gemini transport."""

    def __init__(
        self,
        spec: ModelSpec,
        api_key: str | None = None,
        base_url: str = _HTTP_BASE_URL,
        api_key_env: str = "GEMINI_API_KEY",
        auth_header: str = "x-goog-api-key",
        timeout_s: float = 180.0,
        **opts: Any,
    ):
        """``base_url``: API root that serves ``/v1beta/models/{model}:generateContent``,
        Google's by default or a Gemini-format gateway.  ``api_key_env``: name of the
        environment variable that holds the key.  ``auth_header``: ``x-goog-api-key``
        (Google) or ``bearer`` (most gateways)."""
        if spec.name not in _HTTP_PROFILE_NAMES:
            raise ValueError("GVLHttpAdapter needs one of its own registered profiles")
        if auth_header not in _HTTP_AUTH_HEADERS:
            raise ValueError(f"auth_header must be one of {list(_HTTP_AUTH_HEADERS)}")
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise ValueError("api_key_env must name an environment variable")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https" or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTPS address without credentials, query or fragment")
        root = base_url.rstrip("/")
        if root.endswith("/v1beta"):
            root = root[:-len("/v1beta")]
        if root.endswith("/v1"):
            raise ValueError("for Gemini native, base_url must be the site root or /v1beta, not /v1")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be a finite positive number")
        super().__init__(spec, api_key=api_key, **opts)
        self.base_url = root
        self.api_key_env = api_key_env
        self.auth_header = auth_header
        self.timeout_s = float(timeout_s)
        self._http_transport = None

    def client(self):
        if self._client is None:
            key = self._api_key if self._api_key is not None else os.environ.get(self.api_key_env)
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    f"no API key: export {self.api_key_env} (or name another variable with api_key_env=...)")
            GVL = _upstream_gvl_class()
            client = GVL(
                api_key=key,
                max_frames=self.max_frames,
                offset=self.offset,
                model_name=self.model_name,
                provider="gemini",
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                max_delay=self.max_delay,
            )
            transport = _NativeGeminiTransport(self, key)
            # Replace this instance's HTTP entry only, without modifying vendored
            # source or global requests functions used by other adapters/workers.
            client.stream_inference = transport
            self._http_transport = transport
            self._client = client
        return self._client

    def _score(self, traj: Trajectory) -> RewardOutput:
        out = super()._score(traj)
        out.meta.update({
            "protocol": "robometer_gvl_native_generate_content_strict_8_frame",
            "paper_alignment_equivalence": (
                f"not asserted: {self.model_name} with non-streaming native generateContent "
                "replaces the pinned upstream model and SSE transport"
            ),
            "rewardbox_extension": "model_and_native_http_transport",
            "api_protocol": "gemini_native",
            "api_method": "generateContent",
            "base_url": self.base_url,
            "auth_header": self.auth_header,
            "upstream_transport_reused": False,
            "generation_config": "provider defaults; no generationConfig override",
            "timeout_s": self.timeout_s,
            "credential_source": (
                "explicit_constructor" if self._api_key is not None
                else f"{self.api_key_env}_environment"
            ),
            "n_http_attempts": self._http_transport.http_attempts,
            **self._http_transport.response_metadata,
        })
        return out

    def close(self) -> None:
        super().close()
        self._http_transport = None


class GVLDenseHttpAdapter(GVLHttpAdapter):
    """Time-based sampling with the existing, globally shuffled GVL computation.

    Density is carried by Trajectory.sampling, never by hidden resampling inside
    the adapter. All sampled images share one initial anchor and one prompt.
    """

    _fixed_frame_count = None

    def _validate_trajectory(self, traj: Trajectory) -> None:
        sampling = traj.sampling
        validate_dense_sampling(sampling)
        if traj.fps is None or not math.isfinite(traj.fps) or traj.fps <= 0:
            raise ValueError("Dense GVL needs a valid source-video fps to recover the sampling timeline")
        if len(traj) == 0 or traj.frame_indices[0] != 0:
            raise ValueError("Dense GVL sampling must start at source frame 0 to keep the global initial reference")
        expected = sampling.indices(int(traj.frame_indices[-1]) + 1, source_fps=traj.fps)
        if not np.array_equal(traj.frame_indices, expected):
            raise ValueError("Dense GVL frame_indices do not match the declared sampling grid")

    def _prepare_client(self, traj: Trajectory):
        # Upstream must enter its keep-all branch for this already sampled grid.
        # Update even a cached client: consecutive videos may have different lengths.
        self.max_frames = len(traj)
        client = self.client()
        client.max_frames = self.max_frames
        return client

    def _score(self, traj: Trajectory) -> RewardOutput:
        out = super()._score(traj)
        out.raw.update({
            "sampled_frame_indices": traj.frame_indices.tolist(),
            "sampled_timestamps_s": (traj.frame_indices / traj.fps).tolist(),
            "returned_timestamps_s": (out.frame_indices / traj.fps).tolist(),
        })
        out.meta.update({
            "protocol": "gvl_time_sampled_globally_shuffled_v1",
            "paper_alignment_equivalence": (
                "not asserted: time-based frame density, model and native "
                "HTTP transport differ from the pinned eight-frame recipe"
            ),
            "rewardbox_extension": "time_based_sampling_model_and_native_http_transport",
            "frame_budget_policy": "time_based_no_fixed_frame_cap",
            "sampling_strategy": traj.sampling.strategy,
            "sampling_fps": traj.sampling.fps,
            "sampling_interval": traj.sampling.interval,
            "include_last": traj.sampling.include_last,
            "source_fps": traj.fps,
            "sampled_frame_count": len(traj),
            "initial_anchor_frame_index": 0,
            "source_sampler_padding": "none; all sampled frames are unique",
            "upstream_secondary_subsampling": False,
            "all_sampled_frames_in_one_request": True,
            "chunking": False,
            "chronological_metadata_sent_to_model": False,
        })
        return out
