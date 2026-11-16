"""Official SOLE-R1 reward-server adapter.

SOLE-R1's released inference entry point is a standalone ZMQ reward service,
not an importable ``model.score(video)`` function.  The official server owns
all model-facing behavior: composite-frame construction, the Qwen chat prompt,
per-timestep generation, previous-progress carry, answer parsing, and clipping.
The official client sends one episode as ``list[list[image]]`` through
``reward_client.utils.InferenceClient.call``.

RewardBox deliberately keeps that process boundary.  In particular, the
released server pins torch 2.9 / vLLM 0.12 / transformers 4.57.3 while the
shared RewardBox environment is locked to torch 2.8 / vLLM 0.11 /
transformers 4.57.1.  Importing the server into the toolbox process would both
violate the official deployment recipe and destabilize the other adapters.

The adapter therefore does only the normal boundary work:

* map ``primary`` to the official ``front_images`` field;
* map optional ``left_wrist`` to ``wrist_images`` (or use the documented
  external-only path when it is absent);
* call the vendored official ``InferenceClient``;
* validate and preserve the server's raw progress/reasoning response.

The model's released client samples video at 1 Hz by default.  ``SPEC`` mirrors
that with ``SamplingSpec(strategy="fps", fps=1, include_last=False)``; disabling
``include_last`` matters because the official stride sampler does not append a
final off-grid frame.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from ..core.base import BaseAdapter
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import RewardOutput, SamplingSpec, Trajectory
from ..upstream import UpstreamNotFoundError, upstream_candidates

_UPSTREAM_COMMIT = "2a1286b205577a38a2ef5d9d13efa76878485f87"
_AGENTLACE_COMMIT = "cf2c337c5e3694cdbfc14831b239bd657bc4894d"
_CHECKPOINT_REVISION = "4b322fbbff3a6b9815fc13b70803cdcb8884616b"
_DEFAULT_SERVICE = "rewards"
_DEFAULT_PORT = 8001

_REQ = Requirements(
    # InferenceClient lives in the official checkout.  Its module imports
    # matplotlib/Pillow/OpenCV at module load even though score() only uses the
    # Agentlace class, so every one of them is declared.
    pip=(
        f"agentlace @ git+https://github.com/youliangtan/agentlace.git@{_AGENTLACE_COMMIT}",
        "lz4>=4.3",
        "pyzmq>=25",
        "matplotlib>=3.7",
        "Pillow>=9",
        "opencv-python-headless>=4.7",
    ),
    python=">=3.10",
    note=(
        "This adapter is a thin client of the official SOLE-R1 reward_server; start the "
        "model service in its own Python 3.10/GPU environment as described in the upstream "
        "reward_server/README.md. The server's pinned torch 2.9, vLLM 0.12 and transformers "
        "4.57.3 are not merged into the shared RewardBox environment."
    ),
    install_hint=(
        "uv pip install --python .venv/bin/python -e '.[sole-r1-client]'",
        "uv pip install --python .venv/bin/python --no-deps "
        "-r locks/sole-r1-client-no-deps.txt",
        "start the official service in reward_server/.venv with scripts/run_sole_r1_server.py",
    ),
)

SPEC = ModelSpec(
    name="sole-r1-8b",
    cls="SoleR1Adapter",
    module=__name__,
    reward_mode=RewardMode.PROGRESS,
    signal_type=SignalType.GENERATED_TEXT,
    requires=_REQ,
    # reward_client defaults to target_fps=1.0 and uses a rounded integer
    # stride.  It does not force the final frame onto that grid.
    sampling=SamplingSpec(strategy="fps", fps=1.0, include_last=False),
    score_range=(-100.0, 100.0),
    higher_better=True,
    required_views=("primary",),
    optional_views=("left_wrist",),
    supports_pairwise=False,
    outputs_success=False,
    paper="https://arxiv.org/abs/2603.28730",
    weights="Philip-MIT/SOLE-R1-8B",
    upstream=(
        "sole-r1-model/reward_client/utils.py:InferenceClient.call -> "
        "sole-r1-model/reward_server/src/reward_server/main.py:callback"
    ),
    description=(
        "SOLE-R1 8B video-language reasoning reward model.  Calls the official "
        "standalone reward server and returns its raw per-timestep progress "
        "percentages plus reasoning traces; supports external-only or paired "
        "external/left-wrist views."
    ),
    implementation_kind=ImplementationKind.OFFICIAL_ADAPTER,
)
SPECS = [SPEC]


def _repo_root() -> Path:
    """Locate the vendored official checkout without importing client deps."""
    explicit = os.environ.get("REWARDBOX_SOLE_R1_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if (root / "reward_client" / "utils.py").is_file():
            return root
        raise RuntimeError(
            f"REWARDBOX_SOLE_R1_ROOT={explicit!r} has no reward_client/utils.py"
        )

    for candidate in upstream_candidates("sole-r1-model"):
        if (candidate / "reward_client" / "utils.py").is_file():
            return candidate
    raise UpstreamNotFoundError(
        "vendored SOLE-R1 source not found. Set REWARDBOX_SOLE_R1_ROOT, "
        "or put the official repository at reward_model_code/sole-r1-model."
    )


def _upstream_client_module() -> ModuleType:
    """Load exactly the vendored client module under a collision-free name."""
    source = (_repo_root() / "reward_client" / "utils.py").resolve()
    module_spec = importlib.util.spec_from_file_location(
        "_rewardbox_vendored_sole_r1_client_utils", source
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load the official SOLE-R1 client module: {source}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not hasattr(module, "InferenceClient"):
        raise RuntimeError("the official SOLE-R1 client no longer exports InferenceClient")
    return module


def _upstream_client_class():
    return _upstream_client_module().InferenceClient


def _single_episode_field(data: dict, key: str) -> Any:
    """Extract the sole episode while rejecting ambiguous batch responses."""
    if key not in data:
        raise RuntimeError(f"SOLE-R1 service response lacks data.{key}")
    batch = data[key]
    if isinstance(batch, np.ndarray):
        if batch.ndim < 1 or len(batch) != 1:
            raise RuntimeError(f"SOLE-R1 data.{key} is not a single-episode batch")
        return batch[0]
    if not isinstance(batch, (list, tuple)) or len(batch) != 1:
        raise RuntimeError(f"SOLE-R1 data.{key} is not a single-episode batch")
    return batch[0]


class SoleR1Adapter(BaseAdapter):
    """Thin client for the official standalone SOLE-R1 reward server."""

    def __init__(
        self,
        spec: ModelSpec,
        host: str | None = None,
        port: int = _DEFAULT_PORT,
        service_name: str = _DEFAULT_SERVICE,
        timeout_ms: int = 60 * 60 * 1000,
        wait_for_server: bool = False,
        from_zero: bool = False,
        temperature: float = 1.0,
        wrist_view: str = "left_wrist",
        server_provenance: str | None = None,
        server_provenance_file: str | None = None,
        **opts: Any,
    ):
        super().__init__(spec, **opts)
        resolved_host = host or os.environ.get("REWARDBOX_SOLE_R1_HOST", "localhost")
        if not isinstance(resolved_host, str) or not resolved_host.strip():
            raise ValueError("host must be a non-empty string")
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError("port must be an integer in 1..65535")
        if (not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool)
                or timeout_ms <= 0):
            raise ValueError("timeout_ms must be a positive integer")
        if not isinstance(service_name, str) or not service_name.strip():
            raise ValueError("service_name must be a non-empty string")
        if not isinstance(wait_for_server, bool) or not isinstance(from_zero, bool):
            raise ValueError("wait_for_server and from_zero must be bools")
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(float(temperature)) or temperature < 0):
            raise ValueError("temperature must be a finite number >= 0")
        if not isinstance(wrist_view, str) or not wrist_view.strip():
            raise ValueError("wrist_view must be a non-empty string")
        if server_provenance is not None and (
            not isinstance(server_provenance, str) or not server_provenance.strip()
        ):
            raise ValueError("server_provenance, when given, must be a non-empty string")
        if server_provenance_file is not None and (
            not isinstance(server_provenance_file, str)
            or not server_provenance_file.strip()
        ):
            raise ValueError("server_provenance_file, when given, must be a non-empty path string")
        if server_provenance is not None and server_provenance_file is not None:
            raise ValueError(
                "server_provenance and server_provenance_file cannot both be given"
            )

        self.host = resolved_host.strip()
        self.port = port
        self.service_name = service_name.strip()
        self.timeout_ms = timeout_ms
        self.wait_for_server = wait_for_server
        self.from_zero = from_zero
        self.temperature = float(temperature)
        self.wrist_view = wrist_view.strip()
        self.server_provenance = server_provenance
        self.server_provenance_file = (
            Path(server_provenance_file).expanduser().resolve()
            if server_provenance_file is not None else None
        )
        self._server_provenance_record = self._load_server_provenance()
        self._client = None

    def _load_server_provenance(self) -> str | dict | None:
        if self.server_provenance_file is None:
            return self.server_provenance
        path = self.server_provenance_file
        if not path.is_file():
            raise ValueError(f"server_provenance_file does not exist or is not a file: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"server_provenance_file is not readable JSON: {path}: {exc}"
            ) from exc
        if not isinstance(value, dict) or not value:
            raise ValueError("server_provenance_file must have a non-empty JSON object at top level")
        return value

    def client(self):
        if self._client is None:
            InferenceClient = _upstream_client_class()
            self._client = InferenceClient(
                server_ip=self.host,
                port_num=self.port,
                timeout_ms=self.timeout_ms,
                wait_for_server=self.wait_for_server,
            )
        return self._client

    def _payload(self, traj: Trajectory) -> tuple[dict, bool]:
        front = [np.ascontiguousarray(frame) for frame in traj.primary]
        wrist_frames = traj.views.get(self.wrist_view)
        external_only = wrist_frames is None
        wrist = (
            front
            if wrist_frames is None
            else [np.ascontiguousarray(frame) for frame in wrist_frames]
        )
        return {
            "task": traj.instruction,
            "front_images": [front],
            # The official server requires this key even for external-only
            # scoring; its own client duplicates front frames in that case.
            "wrist_images": [wrist],
            "temperature": self.temperature,
            "from_zero": self.from_zero,
            "external_only": external_only,
        }, external_only

    def _score(self, traj: Trajectory) -> RewardOutput:
        payload, external_only = self._payload(traj)
        try:
            response = self.client().call(self.service_name, payload)
        except Exception as exc:
            raise RuntimeError(
                f"official SOLE-R1 service call failed ({self.host}:{self.port}/"
                f"{self.service_name}): {exc}"
            ) from exc
        if response is None:
            raise RuntimeError("the official SOLE-R1 service returned no response (connection failed or timed out)")
        if not isinstance(response, dict):
            raise RuntimeError(
                f"the official SOLE-R1 service returned a non-dict: {type(response).__name__}"
            )
        if not response.get("success", False):
            message = response.get("message", "<no message>")
            raise RuntimeError(f"the official SOLE-R1 service returned an error: {message}")
        data = response.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("successful SOLE-R1 service response lacks a data dict")

        try:
            progress = np.asarray(
                _single_episode_field(data, "valid_answers"), dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("SOLE-R1 valid_answers cannot be parsed as float progress") from exc
        if progress.ndim != 1 or len(progress) != len(traj):
            raise RuntimeError(
                f"SOLE-R1 returned {progress.shape} progress; the official single-trajectory "
                f"protocol maps one to one onto the {len(traj)} input timesteps"
            )
        if not np.all(np.isfinite(progress)):
            raise RuntimeError("the official SOLE-R1 service returned NaN/Inf progress")
        if progress[0] != 0:
            raise RuntimeError("the official SOLE-R1 protocol's first-timestep baseline should be 0")
        if np.any((progress < self.spec.score_range[0])
                  | (progress > self.spec.score_range[1])):
            raise RuntimeError("SOLE-R1 progress is outside the official parser's [-100, 100] range")

        text_outputs = _single_episode_field(data, "text_outputs")
        text_inputs = _single_episode_field(data, "text_inputs")
        if not isinstance(text_outputs, (list, tuple)) or len(text_outputs) != len(traj) - 1:
            raise RuntimeError("SOLE-R1 text_outputs length does not match the T-1 generation steps")
        if not isinstance(text_inputs, (list, tuple)) or len(text_inputs) != len(traj) - 1:
            raise RuntimeError("SOLE-R1 text_inputs length does not match the T-1 generation steps")

        n_generations = max(0, len(traj) - 1)
        return self._out(
            progress,
            traj,
            weights=None,
            raw={
                "official_valid_answers": progress.tolist(),
                "text_outputs": list(text_outputs),
                "text_inputs": list(text_inputs),
            },
            meta={
                "adapter_calls_official": True,
                "protocol": "official_reward_server_zmq",
                "upstream_commit": _UPSTREAM_COMMIT,
                "service_endpoint": f"{self.host}:{self.port}",
                "service_name": self.service_name,
                "server_checkpoint_provenance": (
                    self._server_provenance_record
                    or "not supplied; remote identity unverified"
                ),
                "server_provenance_file": (
                    str(self.server_provenance_file)
                    if self.server_provenance_file is not None else None
                ),
                "paper_equivalence": (
                    "not asserted: transport response is validated, but checkpoint/runtime "
                    "identity requires explicit server_provenance"
                    if self._server_provenance_record is None
                    else "official server route; numeric paper equivalence not yet calibrated"
                ),
                "official_args": {
                    "temperature": self.temperature,
                    "from_zero": self.from_zero,
                    "external_only": external_only,
                },
                "front_view": "primary",
                "wrist_view": None if external_only else self.wrist_view,
                "ignored_views": sorted(
                    set(traj.views) - {"primary", self.wrist_view}
                ),
                "progress_scale": "raw integer percentage [-100, 100]",
                "normalized": False,
                "first_timestep": "official fixed zero baseline",
                "n_transport_calls": 1,
                "n_backend_calls": n_generations,
            },
        )

    def close(self) -> None:
        # The released InferenceClient exposes no close method.  Be compatible
        # with a future release (and simple test doubles) without reaching into
        # Agentlace's private socket fields.
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        self._client = None
