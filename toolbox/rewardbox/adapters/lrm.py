"""Official Large-Reward-Models HTTP adapters.

Upstream (pinned):
  https://github.com/physical-superintelligence-lab/Large-Reward-Models
  commit 5845ab0f7c3c6a0bef8bd37bf7f07e9533f7ed2f

Large-Reward-Models intentionally runs its Qwen3-VL backend in a standalone
Flask process.  RewardBox keeps that boundary: this module loads the official
``vlm_reward_client.py`` and maps ``Trajectory`` frames to its public methods.
It never imports torch, transformers, PEFT, or the server in the RewardBox
process.

The four registered profiles mirror the upstream execution paths:

* ``lrm-contrastive``: first frame is the official zero baseline, then each
  adjacent pair produces one signed direction score in ``{-1, 0, +1}``;
* ``lrm-completion``: one binary completion score per frame;
* ``lrm-progress``: one raw progress score in ``[0, 1]`` per frame;
* ``lrm-tri``: the fixed upstream fusion.  Contrastive is mapped from
  ``[-1, 1]`` to ``[0, 1]`` and combined with progress/completion using the
  exact fixed thirds from ``config/tri.yaml``.  There is deliberately no
  runtime weight option and no independent tri checkpoint.

The published HF revision currently contains three full-model shard sets, not
PEFT ``adapter_config.json`` files.  A single-head official server supports
that layout.  The upstream one-backbone tri loader, however, requires three
PEFT adapters.  To prevent silently serving one model through all three
endpoints, a shared tri URL is accepted only when ``/health`` advertises all
three adapter names.  Three distinct dedicated full-model server URLs remain a
valid (more expensive) way to execute the same fixed fusion.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

from ..core.base import BaseAdapter
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import RewardOutput, SamplingSpec, Trajectory
from ..upstream import UpstreamNotFoundError, upstream_candidates

_UPSTREAM_COMMIT = "5845ab0f7c3c6a0bef8bd37bf7f07e9533f7ed2f"
_WEIGHTS_REPO = "USC-PSI-Lab/LRM-models"
_WEIGHTS_REVISION = "5f8da7f3934f5d658256d77552927a9f76ad1c0e"
_DEFAULT_SERVER_URL = "http://localhost:5002"
_HEADS = ("contrastive", "progress", "completion")
_TRI_WEIGHTS = (0.333333333333, 0.333333333333, 0.333333333334)

_REQ = Requirements(
    pip=("requests>=2",),
    python=">=3.10",
    note=(
        "RewardBox only loads the official HTTP client; the Qwen3-VL/PEFT server must run in a "
        "separate Python 3.10 environment per the pinned Large-Reward-Models vlm_reward/requirements.txt, "
        "so its torch>=2.4.1,<2.6 constraint stays out of the shared environment. Set "
        "REWARDBOX_LRM_ROOT, or put the official repository at "
        "reward_model_code/Large-Reward-Models."
    ),
    install_hint=(
        "uv pip install --python .venv/bin/python -e '.[lrm-client]'",
        "start the official server in a separate environment (docs/installation.md, section 5)",
    ),
)

_COMMON = dict(
    cls="LRMAdapter",
    module=__name__,
    signal_type=SignalType.GENERATED_TEXT,
    requires=_REQ,
    # Official eval/score_with_vlm.py scores every recorded frame by default.
    sampling=SamplingSpec(strategy="all"),
    higher_better=True,
    required_views=("primary",),
    supports_pairwise=False,
    paper="https://arxiv.org/abs/2603.16065",
    upstream=(
        "Large-Reward-Models/vlm_reward/vlm_reward_client.py:VLMRewardClient -> "
        "vlm_reward/vlm_reward_server.py"
    ),
    implementation_kind=ImplementationKind.OFFICIAL_ADAPTER,
)

SPECS = [
    ModelSpec(
        name="lrm-contrastive",
        reward_mode=RewardMode.PREFERENCE,
        score_range=(-1.0, 1.0),
        outputs_success=False,
        weights=f"{_WEIGHTS_REPO}@{_WEIGHTS_REVISION}/contrastive",
        description=(
            "LRM temporal contrastive head: 0 for the first frame, then adjacent-frame comparisons output "
            "-1/0/+1; inference through the official Flask server."
        ),
        **_COMMON,
    ),
    ModelSpec(
        name="lrm-completion",
        reward_mode=RewardMode.SUCCESS,
        score_range=(0.0, 1.0),
        outputs_success=True,
        weights=f"{_WEIGHTS_REPO}@{_WEIGHTS_REVISION}/completion",
        description=(
            "LRM binary completion head: native per-frame 0/1 completion verdicts; inference through the official "
            "Flask server."
        ),
        **_COMMON,
    ),
    ModelSpec(
        name="lrm-progress",
        reward_mode=RewardMode.PROGRESS,
        score_range=(0.0, 1.0),
        outputs_success=False,
        weights=f"{_WEIGHTS_REPO}@{_WEIGHTS_REVISION}/progress",
        description=(
            "LRM absolute progress head: native per-frame 0-1 progress; inference through the official "
            "Flask server."
        ),
        **_COMMON,
    ),
    ModelSpec(
        name="lrm-tri",
        reward_mode=RewardMode.PROGRESS,
        score_range=(0.0, 1.0),
        outputs_success=False,
        weights=(
            f"{_WEIGHTS_REPO}@{_WEIGHTS_REVISION}/"
            "{contrastive,progress,completion}; no independent tri weights"
        ),
        description=(
            "Fixed equal-weight fusion of the three LRM heads: contrastive is first mapped from [-1,1] to [0,1], "
            "then fused with progress/completion; no separate tri weights or tunable fusion coefficients."
        ),
        **_COMMON,
    ),
]


def _repo_root() -> Path:
    """Locate the pinned upstream checkout without importing server deps."""
    explicit = os.environ.get("REWARDBOX_LRM_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if (root / "vlm_reward" / "vlm_reward_client.py").is_file():
            return root
        raise RuntimeError(
            f"REWARDBOX_LRM_ROOT={explicit!r} has no "
            "vlm_reward/vlm_reward_client.py"
        )

    for candidate in upstream_candidates("Large-Reward-Models"):
        if (candidate / "vlm_reward" / "vlm_reward_client.py").is_file():
            return candidate
    raise UpstreamNotFoundError(
        "vendored Large-Reward-Models not found. Set REWARDBOX_LRM_ROOT, "
        "or put the official repository at reward_model_code/Large-Reward-Models."
    )


@lru_cache(maxsize=1)
def _upstream_client_module() -> ModuleType:
    """Load the official client source under a collision-free module name."""
    root = _repo_root()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0 or revision.stdout.strip() != _UPSTREAM_COMMIT:
        actual = revision.stdout.strip() or revision.stderr.strip() or "<unavailable>"
        raise RuntimeError(
            f"the Large-Reward-Models client checkout is not at the pinned commit: {actual}"
        )
    # core.fileMode=false: a chmod of the checkout (shared machines, filesystems
    # without mode bits) is not a source modification.
    client_diff = subprocess.run(
        [
            "git",
            "-c",
            "core.fileMode=false",
            "-C",
            str(root),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "vlm_reward/vlm_reward_client.py",
        ]
    )
    if client_diff.returncode == 1:
        raise RuntimeError("the official Large-Reward-Models client has tracked modifications; refusing to load")
    if client_diff.returncode != 0:
        raise RuntimeError("cannot verify the tracked state of the official Large-Reward-Models client")
    source = (root / "vlm_reward" / "vlm_reward_client.py").resolve()
    module_spec = importlib.util.spec_from_file_location(
        "_rewardbox_vendored_lrm_vlm_reward_client", source
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load the official Large-Reward-Models client: {source}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not hasattr(module, "VLMRewardClient"):
        raise RuntimeError("the official Large-Reward-Models client no longer exports VLMRewardClient")
    return module


def _upstream_client_class():
    return _upstream_client_module().VLMRewardClient


def _validate_url(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty HTTP(S) URL")
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
            f"{name} must be an HTTP(S) server root without credentials or path/query/fragment"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} has an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"the port of {name} must be in 1..65535")
    return value


def _fetch_health(server_url: str, timeout: float) -> dict:
    """Read the official public health endpoint without importing requests."""
    request = Request(f"{server_url}/health", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL validated
            status = getattr(response, "status", 200)
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"LRM server health check failed ({server_url}): {exc}") from exc
    if status != 200 or not isinstance(payload, dict):
        raise RuntimeError(
            f"invalid LRM server health response ({server_url}): HTTP {status}"
        )
    if not payload.get("model_loaded", False):
        raise RuntimeError(f"the LRM server has not loaded a model yet ({server_url})")
    adapters = payload.get("adapters", [])
    if not isinstance(adapters, list) or not all(isinstance(x, str) for x in adapters):
        raise RuntimeError(f"LRM server health.adapters is not a list of strings ({server_url})")
    return payload


def _finite_score(result: Any, *, mode: str, lo: float, hi: float) -> float:
    if not isinstance(result, dict):
        raise RuntimeError(f"the official LRM {mode} client returned a non-dict")
    if not result.get("success", False):
        raise RuntimeError(
            f"the official LRM {mode} client reported a failed request; refusing to treat its fallback 0 as model output"
        )
    value = result.get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"LRM {mode} response lacks a numeric score")
    value = float(value)
    if not math.isfinite(value) or not lo <= value <= hi:
        raise RuntimeError(f"LRM {mode} score={value!r} is outside [{lo}, {hi}]")
    return value


class LRMAdapter(BaseAdapter):
    """Thin adapter around the pinned official HTTP client."""

    def __init__(
        self,
        spec: ModelSpec,
        server_url: str | None = None,
        timeout: float = 60.0,
        validate_health: bool = True,
        include_initial_image: bool = False,
        include_goal_image: bool = False,
        contrastive_url: str | None = None,
        progress_url: str | None = None,
        completion_url: str | None = None,
        server_provenance: str | None = None,
        server_provenance_file: str | None = None,
        server_provenance_files: dict[str, str] | None = None,
        **opts: Any,
    ):
        super().__init__(spec, **opts)
        self.mode = spec.name.removeprefix("lrm-")
        if self.mode not in {*_HEADS, "tri"}:
            raise ValueError(f"cannot infer the LRM mode from the model spec: {spec.name!r}")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a finite number of seconds > 0")
        for name, value in (
            ("validate_health", validate_health),
            ("include_initial_image", include_initial_image),
            ("include_goal_image", include_goal_image),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")
        if self.mode not in {"progress", "tri"} and (
            include_initial_image or include_goal_image
        ):
            raise ValueError(
                "include_initial_image/include_goal_image only apply to lrm-progress/lrm-tri"
            )
        if self.mode != "tri" and any(
            value is not None
            for value in (contrastive_url, progress_url, completion_url)
        ):
            raise ValueError("the three per-head URL options only apply to lrm-tri")
        if server_provenance is not None and (
            not isinstance(server_provenance, str) or not server_provenance.strip()
        ):
            raise ValueError("server_provenance, when given, must be a non-empty string")
        if server_provenance_file is not None and (
            not isinstance(server_provenance_file, str)
            or not server_provenance_file.strip()
        ):
            raise ValueError("server_provenance_file, when given, must be a non-empty path string")
        provenance_sources = sum(
            value is not None
            for value in (
                server_provenance,
                server_provenance_file,
                server_provenance_files,
            )
        )
        if provenance_sources > 1:
            raise ValueError(
                "only one of server_provenance, server_provenance_file and "
                "server_provenance_files may be given"
            )
        if server_provenance_files is not None:
            if self.mode != "tri":
                raise ValueError("server_provenance_files only applies to lrm-tri")
            if not isinstance(server_provenance_files, dict) or set(
                server_provenance_files
            ) != set(_HEADS):
                raise ValueError(
                    "server_provenance_files must contain exactly "
                    "contrastive/progress/completion"
                )
            if not all(
                isinstance(value, str) and value.strip()
                for value in server_provenance_files.values()
            ):
                raise ValueError("the values of server_provenance_files must be non-empty path strings")

        default_url = (
            server_url
            or os.environ.get("REWARDBOX_LRM_SERVER_URL")
            or _DEFAULT_SERVER_URL
        )
        self.server_url = _validate_url("server_url", default_url)
        if self.mode == "tri":
            explicit_urls = {
                "contrastive": contrastive_url
                or os.environ.get("REWARDBOX_LRM_CONTRASTIVE_URL"),
                "progress": progress_url
                or os.environ.get("REWARDBOX_LRM_PROGRESS_URL"),
                "completion": completion_url
                or os.environ.get("REWARDBOX_LRM_COMPLETION_URL"),
            }
            self.server_urls = {
                head: _validate_url(f"{head}_url", url or self.server_url)
                for head, url in explicit_urls.items()
            }
        else:
            self.server_urls = {self.mode: self.server_url}

        self.timeout = float(timeout)
        self.validate_health = validate_health
        self.include_initial_image = include_initial_image
        self.include_goal_image = include_goal_image
        self.server_provenance = server_provenance
        self.server_provenance_file = (
            Path(server_provenance_file).expanduser().resolve()
            if server_provenance_file is not None
            else None
        )
        self.server_provenance_files = (
            {
                head: Path(path).expanduser().resolve()
                for head, path in server_provenance_files.items()
            }
            if server_provenance_files is not None
            else None
        )
        self._server_provenance_record = self._load_server_provenance()
        self._validate_wrapper_provenance()
        self._clients: dict[str, Any] = {}
        self._health: dict[str, dict] | None = None

    def _load_server_provenance(self) -> str | dict | None:
        if self.server_provenance_files is not None:
            return {
                head: self._read_server_provenance(path, label=head)
                for head, path in self.server_provenance_files.items()
            }
        if self.server_provenance_file is None:
            return self.server_provenance
        return self._read_server_provenance(
            self.server_provenance_file, label=self.mode
        )

    @staticmethod
    def _read_server_provenance(path: Path, *, label: str) -> dict:
        if not path.is_file():
            raise ValueError(
                f"{label} server provenance does not exist or is not a file: {path}"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{label} server provenance is not readable JSON: {path}: {exc}"
            ) from exc
        if not isinstance(value, dict) or not value:
            raise ValueError(
                f"{label} server provenance must have a non-empty JSON object at top level"
            )
        return value

    def _validate_wrapper_provenance(self) -> None:
        records: dict[str, Any]
        if self.server_provenance_files is not None:
            records = dict(self._server_provenance_record)
        else:
            records = {self.mode: self._server_provenance_record}
        for expected_mode, record in records.items():
            if not isinstance(record, dict) or record.get("schema") != (
                "rewardbox.lrm-server-provenance.v1"
            ):
                continue
            if record.get("mode") != expected_mode:
                raise ValueError(
                    f"{expected_mode} provenance actually declares mode={record.get('mode')!r}"
                )
            upstream = record.get("upstream", {})
            if not isinstance(upstream, dict) or upstream.get("commit") != _UPSTREAM_COMMIT:
                raise ValueError(
                    f"the LRM commit of the {expected_mode} provenance is not the pinned revision"
                )
            weights = record.get("weights", {})
            checkpoints = (
                weights.get("checkpoints", {}) if isinstance(weights, dict) else {}
            )
            required_heads = set(_HEADS) if expected_mode == "tri" else {expected_mode}
            if not isinstance(checkpoints, dict) or not required_heads <= set(checkpoints):
                raise ValueError(
                    f"{expected_mode} provenance lacks checkpoints={sorted(required_heads)}"
                )
            for head in required_heads:
                checkpoint = checkpoints[head]
                if not isinstance(checkpoint, dict):
                    raise ValueError(f"invalid {head} identity in the {expected_mode} provenance")
                if checkpoint.get("pinned_public_snapshot") and (
                    checkpoint.get("snapshot_revision") != _WEIGHTS_REVISION
                ):
                    raise ValueError(
                        f"the {head} snapshot revision of the {expected_mode} provenance drifted"
                    )

    def client(self, head: str):
        url = self.server_urls[head]
        if url not in self._clients:
            Client = _upstream_client_class()
            self._clients[url] = Client(server_url=url, timeout=self.timeout)
        return self._clients[url]

    def _ensure_health(self) -> None:
        if not self.validate_health or self._health is not None:
            return
        requested: dict[str, set[str]] = {}
        for head, url in self.server_urls.items():
            requested.setdefault(url, set()).add(head)
        health = {url: _fetch_health(url, self.timeout) for url in requested}
        for url, heads in requested.items():
            adapters = set(health[url].get("adapters", []))
            # An empty adapter list is valid only for one dedicated full-model
            # server.  Any shared URL must prove that endpoint selection really
            # activates all requested LoRA heads.
            if (adapters or len(heads) > 1) and not heads <= adapters:
                raise RuntimeError(
                    f"LRM server {url} was asked for heads={sorted(heads)} but /health only advertises "
                    f"adapters={sorted(adapters)}; refusing to silently reuse the wrong checkpoint"
                )
        self._health = health

    def _call(self, head: str, method: str, *args, **kwargs) -> dict:
        self._ensure_health()
        client = self.client(head)
        try:
            result = getattr(client, method)(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"official LRM {head} HTTP call failed ({self.server_urls[head]}): {exc}"
            ) from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"the official LRM {head} client returned a non-dict")
        return result

    def _progress_kwargs(self, traj: Trajectory) -> dict:
        if self.include_goal_image and traj.goal_image is None:
            raise ValueError("include_goal_image=true, but the Trajectory has no goal_image")
        return {
            "initial_image": traj.primary[0] if self.include_initial_image else None,
            "goal_image": traj.goal_image if self.include_goal_image else None,
        }

    def _score_contrastive(self, traj: Trajectory) -> tuple[np.ndarray, dict, int]:
        scores = [0.0]
        results: list[str | None] = ["first_frame"]
        responses = [""]
        for earlier, later in zip(traj.primary[:-1], traj.primary[1:]):
            result = self._call(
                "contrastive",
                "compute_comparison",
                earlier,
                later,
                traj.instruction,
            )
            score = _finite_score(result, mode="contrastive", lo=-1.0, hi=1.0)
            if score not in {-1.0, 0.0, 1.0}:
                raise RuntimeError(f"LRM contrastive returned a non-discrete score={score}")
            parsed = result.get("result")
            if parsed not in {"ImageA", "ImageB", None}:
                raise RuntimeError(f"invalid LRM contrastive result: {parsed!r}")
            expected = {"ImageA": -1.0, "ImageB": 1.0, None: 0.0}[parsed]
            if score != expected:
                raise RuntimeError(
                    f"LRM contrastive result={parsed!r} is inconsistent with score={score}"
                )
            scores.append(score)
            results.append(parsed)
            responses.append(str(result.get("response", "")))
        return np.asarray(scores), {
            "comparison_results": results,
            "responses": responses,
        }, max(0, len(traj) - 1)

    def _score_completion(self, traj: Trajectory) -> tuple[np.ndarray, dict, int]:
        scores = []
        completed_values = []
        responses = []
        for frame in traj.primary:
            result = self._call(
                "completion", "compute_completion", frame, traj.instruction
            )
            score = _finite_score(result, mode="completion", lo=0.0, hi=1.0)
            if score not in {0.0, 1.0}:
                raise RuntimeError(f"LRM completion returned a non-binary score={score}")
            completed = result.get("completed")
            if not isinstance(completed, bool):
                raise RuntimeError("LRM completion response lacks a bool completed")
            if score != float(completed):
                raise RuntimeError(
                    f"LRM completion completed={completed} is inconsistent with score={score}"
                )
            scores.append(score)
            completed_values.append(completed)
            responses.append(str(result.get("response", "")))
        return np.asarray(scores), {
            "completed": completed_values,
            "responses": responses,
        }, len(traj)

    def _score_progress(self, traj: Trajectory) -> tuple[np.ndarray, dict, int]:
        kwargs = self._progress_kwargs(traj)
        scores = []
        responses = []
        for frame in traj.primary:
            result = self._call(
                "progress",
                "compute_reward",
                frame,
                traj.instruction,
                reward_type="progress",
                **kwargs,
            )
            scores.append(_finite_score(result, mode="progress", lo=0.0, hi=1.0))
            responses.append(str(result.get("response", "")))
        return np.asarray(scores), {"responses": responses}, len(traj)

    def _score_tri(self, traj: Trajectory) -> tuple[np.ndarray, dict, int]:
        signed, contrastive_raw, contrastive_calls = self._score_contrastive(traj)
        contrastive_unit = np.zeros_like(signed)
        if len(signed) > 1:
            contrastive_unit[1:] = (signed[1:] + 1.0) / 2.0
        progress, progress_raw, progress_calls = self._score_progress(traj)
        completion, completion_raw, completion_calls = self._score_completion(traj)
        fused = (
            _TRI_WEIGHTS[0] * contrastive_unit
            + _TRI_WEIGHTS[1] * progress
            + _TRI_WEIGHTS[2] * completion
        )
        return fused, {
            "contrastive_signed": signed.tolist(),
            "contrastive_unit": contrastive_unit.tolist(),
            "progress": progress.tolist(),
            "completion": completion.tolist(),
            "component_responses": {
                "contrastive": contrastive_raw,
                "progress": progress_raw,
                "completion": completion_raw,
            },
        }, contrastive_calls + progress_calls + completion_calls

    def _score(self, traj: Trajectory) -> RewardOutput:
        if self.mode == "contrastive":
            progress, raw, calls = self._score_contrastive(traj)
            success = None
        elif self.mode == "completion":
            progress, raw, calls = self._score_completion(traj)
            success = progress.copy()
        elif self.mode == "progress":
            progress, raw, calls = self._score_progress(traj)
            success = None
        else:
            progress, raw, calls = self._score_tri(traj)
            success = None

        provenance = (
            self._server_provenance_record
            or "not supplied; remote checkpoint/runtime identity unverified"
        )
        return self._out(
            progress,
            traj,
            success=success,
            raw=raw,
            meta={
                "adapter_calls_official": True,
                "protocol": "official_lrm_http_open_loop",
                "profile": self.mode,
                "upstream_commit": _UPSTREAM_COMMIT,
                "weights_revision": _WEIGHTS_REVISION,
                "server_urls": dict(self.server_urls),
                "server_health": self._health,
                "server_checkpoint_provenance": provenance,
                "server_provenance_file": (
                    str(self.server_provenance_file)
                    if self.server_provenance_file is not None
                    else None
                ),
                "server_provenance_files": (
                    {
                        head: str(path)
                        for head, path in self.server_provenance_files.items()
                    }
                    if self.server_provenance_files is not None
                    else None
                ),
                "paper_equivalence": (
                    "not asserted by the adapter: the official HTTP route is used, "
                    "but equivalence additionally requires a reviewed server provenance "
                    "and real-checkpoint calibration"
                ),
                "official_args": {
                    "timeout_s": self.timeout,
                    "include_initial_image": self.include_initial_image,
                    "include_goal_image": self.include_goal_image,
                    "tri_weights": list(_TRI_WEIGHTS) if self.mode == "tri" else None,
                    "tri_contrastive_to_unit": self.mode == "tri",
                    "generation": {
                        "do_sample": False,
                        "temperature": None,
                        "top_p": None,
                        "max_new_tokens": {
                            "contrastive": 512,
                            "progress": 128,
                            "completion": 128,
                        },
                    },
                },
                "sampling_profile": (
                    "all supplied frames; adjacent pairs for contrastive; no interpolation"
                ),
                "published_weights_layout": (
                    "three full-model shard directories at the pinned HF revision; "
                    "the upstream shared-backbone tri loader instead requires PEFT adapters"
                ),
                "normalized": False,
                "n_transport_calls": calls,
                "n_backend_calls": calls,
            },
        )

    def close(self) -> None:
        # The official requests-based client has no close() method today.  Keep
        # future/test-double compatibility without reaching into its internals.
        for client in self._clients.values():
            close = getattr(client, "close", None)
            if callable(close):
                close()
        self._clients.clear()
        self._health = None
