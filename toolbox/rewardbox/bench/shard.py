"""Single-model benchmark shard and its resume identity.

A ResultSet aggregates "one trajectory x many models"; a ModelShard aggregates
"one model x many trajectories".  Besides results, a shard binds the experiment
identity that would change those results: model, weights, pinned upstream
commit, RewardBox version, SamplingSpec, model options and key runtime package
versions.

Inputs are trusted fixed files (``INPUT_POLICY``).  Nothing is hashed: resume
is decided per manifest entry from its ID plus the manifest fields that reach the
model (video path(s) as written and instruction).  Editing one entry re-runs only
that entry; replacing a video file in place under the same path is not detected.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..core._atomic import atomic_write_json
from ..core.secrets import redact_sensitive_options
from ..core.types import RewardOutput

SHARD_SCHEMA_VERSION = 3
INPUT_POLICY = "trusted-input: resume by manifest entry; no content hashing"


class BenchmarkIdentityError(ValueError):
    """An existing shard was produced under a different experiment definition and
    cannot be resumed safely."""


def _package_versions() -> dict[str, str | None]:
    names = (
        "rewardbox", "torch", "transformers", "vllm", "unsloth",
        "flash-attn", "qwen-vl-utils", "evo-vlac", "ms-swift", "peft", "timm",
        # GVL's upstream API route depends on request/JPEG encoding behaviour.
        "sentence-transformers", "huggingface-hub", "einops", "requests",
        "opencv-python-headless",
        # SOLE-R1 crosses the official Agentlace/ZMQ client boundary.
        "agentlace", "lz4", "pyzmq",
    )
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _external_runtime_identity(model: str, model_opts: dict | None) -> dict:
    """Bind endpoint-selection state that is not already present in model_opts."""
    opts = dict(model_opts or {})

    def _path(value):
        return str(Path(value).expanduser().resolve()) if value else None

    provenance_file = _path(opts.get("server_provenance_file"))
    provenance_files = opts.get("server_provenance_files")
    if isinstance(provenance_files, dict):
        provenance_files = {head: _path(value) for head, value in sorted(provenance_files.items())}
    else:
        provenance_files = None
    if model == "sole-r1-8b":
        return {
            "host": opts.get("host") or os.environ.get("REWARDBOX_SOLE_R1_HOST", "localhost"),
            # Port/service/provenance are either explicit model_opts or adapter
            # defaults; a default change ships with a new RewardBox version.
            "server_provenance": opts.get("server_provenance"),
            "server_provenance_file": provenance_file,
        }
    # lrm-*-vllm runs in-process (rewardbox.adapters.lrm_vllm): no server to bind.
    if model.startswith("lrm-") and not model.endswith("-vllm"):
        base_url = (
            opts.get("server_url")
            or os.environ.get("REWARDBOX_LRM_SERVER_URL")
            or "http://localhost:5002"
        )
        mode = model.removeprefix("lrm-")
        if mode == "tri":
            server_urls = {
                "contrastive": opts.get("contrastive_url")
                or os.environ.get("REWARDBOX_LRM_CONTRASTIVE_URL")
                or base_url,
                "progress": opts.get("progress_url")
                or os.environ.get("REWARDBOX_LRM_PROGRESS_URL")
                or base_url,
                "completion": opts.get("completion_url")
                or os.environ.get("REWARDBOX_LRM_COMPLETION_URL")
                or base_url,
            }
        else:
            server_urls = {mode: base_url}
        return {
            "server_urls": server_urls,
            "server_provenance": opts.get("server_provenance"),
            "server_provenance_file": provenance_file,
            "server_provenance_files": provenance_files,
        }
    if model.startswith("rynnvalue-"):
        return {
            "server_url": (
                opts.get("server_url")
                or os.environ.get("REWARDBOX_RYNNVALUE_SERVER_URL")
                or "http://127.0.0.1:8001"
            ),
            "server_provenance_file": provenance_file,
            "allow_unverified_server": opts.get("allow_unverified_server", False),
        }
    if model == "procvlm-2b":
        return {
            "server_url": (
                opts.get("server_url")
                or os.environ.get("REWARDBOX_PROCVLM_SERVER_URL")
                or "http://127.0.0.1:8013"
            ),
            "server_provenance_file": provenance_file,
            "allow_unverified_server": opts.get("allow_unverified_server", False),
        }
    if model.startswith("robo-valuerl-"):
        return {
            "server_url": (
                opts.get("server_url")
                or os.environ.get("REWARDBOX_ROBOVALUERL_SERVER_URL")
                or "http://127.0.0.1:8012"
            ),
            "server_provenance_file": provenance_file,
            "allow_unverified_server": opts.get("allow_unverified_server", False),
        }
    return {}


def build_experiment_identity(model: str, sampling, model_opts: dict | None = None) -> dict:
    """Normalised identity that decides whether a shard may be resumed (JSON-serialisable).

    Model version = registry weights + upstream entry + pinned upstream commit(s) +
    RewardBox version; protocol = SamplingSpec + model options.  No file or source
    content is hashed.
    """
    from ..registry import get_spec
    from ..version import __version__
    from ..upstream import upstreams_for_model

    try:
        spec = get_spec(model)
        upstream = spec.upstream
        weights = spec.weights
    except Exception:
        # Lets tests / temporary third-party registries still build a shard; the real
        # create_reward call raises a clear error later.
        upstream = ""
        weights = ""
    identity = {
        "model": model,
        "input_policy": INPUT_POLICY,
        "rewardbox_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "sampling": sampling.to_dict(),
        # ``model_opts`` is embedded in every shard and RewardOutput benchmark block.
        # Keep protocol-affecting options, but never persist credentials accepted by
        # a programmatic adapter constructor.
        "model_opts": redact_sensitive_options(dict(model_opts or {})),
        "weights": weights,
        "upstream": upstream,
        "upstream_commits": {u.key: u.commit for u in upstreams_for_model(model)},
        "package_versions": _package_versions(),
    }
    external_runtime = redact_sensitive_options(
        _external_runtime_identity(model, model_opts)
    )
    if external_runtime:
        identity["external_runtime"] = external_runtime
    return identity


def worker_provenance() -> dict:
    """Record the execution environment without treating scheduling details such as
    GPU numbers as resume identity."""
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "package_versions": _package_versions(),
    }


def _grid(input_frame_indices) -> list[int] | None:
    if input_frame_indices is None:
        return None
    grid = [int(x) for x in input_frame_indices]
    if not grid:
        raise ValueError("input_frame_indices must not be empty")
    return grid


@dataclass
class ModelShard:
    model: str
    identity: dict = field(default_factory=dict)
    results: dict[str, dict] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    # trajectory_id -> {"entry": manifest fields that reach the model,
    #                   "input_frame_indices": sampled grid, or None if never decoded}
    inputs: dict[str, dict] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    fatal_error: str | None = None

    def has(self, trajectory_id: str) -> bool:
        """Status query only; production resume uses :meth:`has_current`."""
        return trajectory_id in self.results or trajectory_id in self.errors

    def has_current(
        self, trajectory_id: str, entry: dict, *, retry_errors: bool = False,
    ) -> bool:
        """Hit only when the stored manifest fields still match; a changed entry is
        discarded and re-run, every other entry is kept."""
        if not self.has(trajectory_id):
            return False
        if (self.inputs.get(trajectory_id) or {}).get("entry") != entry:
            self.discard(trajectory_id)
            return False
        if retry_errors and trajectory_id in self.errors:
            self.discard(trajectory_id)
            return False
        return True

    def discard(self, trajectory_id: str) -> None:
        self.results.pop(trajectory_id, None)
        self.errors.pop(trajectory_id, None)
        self.inputs.pop(trajectory_id, None)

    def add_result(
        self, trajectory_id: str, out: RewardOutput, entry: dict, input_frame_indices,
    ) -> None:
        if out.model_name != self.model:
            raise ValueError(
                f"shard model is {self.model!r}; cannot store a result of "
                f"model={out.model_name!r}")
        grid = _grid(input_frame_indices)
        if grid is None:
            raise ValueError("a benchmark result needs the sampled input_frame_indices")
        self.results[trajectory_id] = out.to_dict()
        self.errors.pop(trajectory_id, None)
        self.inputs[trajectory_id] = {"entry": entry, "input_frame_indices": grid}

    def add_error(self, trajectory_id: str, error: str, entry: dict,
                  input_frame_indices=None) -> None:
        error = str(error)
        if not error:
            raise ValueError("benchmark error must be a non-empty string")
        self.errors[trajectory_id] = error
        self.results.pop(trajectory_id, None)
        self.inputs[trajectory_id] = {"entry": entry,
                                      "input_frame_indices": _grid(input_frame_indices)}

    def to_dict(self) -> dict:
        overlap = set(self.results) & set(self.errors)
        if overlap:
            raise ValueError(f"shard has trajectory ids in both results and errors: {sorted(overlap)}")
        missing = (set(self.results) | set(self.errors)) - set(self.inputs)
        if missing:
            raise ValueError(f"shard records without input fields: {sorted(missing)}")
        return {
            "schema_version": SHARD_SCHEMA_VERSION,
            "model": self.model,
            "identity": self.identity,
            "results": dict(self.results),
            "errors": dict(self.errors),
            "inputs": dict(self.inputs),
            "provenance": self.provenance,
            "fatal_error": self.fatal_error,
        }

    def save(self, path: str | Path) -> None:
        atomic_write_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> ModelShard:
        path = Path(path)
        d = json.loads(path.read_text())
        if d.get("schema_version") != SHARD_SCHEMA_VERSION:
            raise BenchmarkIdentityError(
                f"{path} has shard schema_version={d.get('schema_version')!r}; only "
                f"{SHARD_SCHEMA_VERSION} is supported. Regenerate it with --no-resume")
        required = {
            "model", "identity", "results", "errors", "inputs", "provenance", "fatal_error",
        }
        missing = required - set(d)
        if missing:
            raise ValueError(f"{path} is missing required shard fields: {sorted(missing)}")
        shard = cls(
            model=d["model"],
            identity=dict(d["identity"]),
            results=dict(d["results"]),
            errors=dict(d["errors"]),
            inputs=dict(d["inputs"]),
            provenance=dict(d["provenance"]),
            fatal_error=d["fatal_error"],
        )
        # Reuse the object constructor to check each result's basic numeric contract.
        for trajectory_id, out_d in shard.results.items():
            out = RewardOutput.from_dict(out_d)
            if out.model_name != shard.model:
                raise ValueError(
                    f"{path}: {trajectory_id!r} has model={out.model_name!r}, "
                    f"shard model is {shard.model!r}")
            if (shard.inputs.get(trajectory_id) or {}).get("input_frame_indices") is None:
                raise ValueError(f"{path}: {trajectory_id!r} has no input_frame_indices")
        shard.to_dict()
        return shard

    @classmethod
    def load_or_new(
        cls, path: str | Path, model: str, identity: dict | None = None,
    ) -> ModelShard:
        p = Path(path)
        if p.exists():
            shard = cls.load(p)
            if shard.model != model:
                raise BenchmarkIdentityError(
                    f"{p} is the shard of model {shard.model!r}, not {model!r}; with "
                    "--resume the shard path and the model must correspond")
            if identity is not None and shard.identity != identity:
                changed = sorted(
                    k for k in set(shard.identity) | set(identity)
                    if shard.identity.get(k) != identity.get(k)
                )
                raise BenchmarkIdentityError(
                    f"{p} does not match this benchmark's identity (changed fields: "
                    f"{changed}); refusing to reuse stale results. Check the arguments, "
                    "or re-run explicitly with --no-resume")
            return shard
        return cls(model=model, identity=dict(identity or {}))
