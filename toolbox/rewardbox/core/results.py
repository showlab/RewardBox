"""ResultSet: the cross-model result container and its JSON schema v1.

Its job is to **refuse to merge results that are not comparable**.  This answers
the worst problem in reward-scope, whose ``save_json`` merges dictionaries
without checking that sampling/instruction/trajectory agree; that is how
``stackcubes1.json`` ends up declaring ``num_frames`` 10 while ``roboreward``
holds 20 values.  ResultSet rejects such merges in ``add()``, ``save()`` and
``load()``.

Comparability is decided from protocol fields only: instruction, video source,
``SamplingSpec.key()`` and the sampled input frame grid.  Inputs are trusted
fixed files; no frame or file content is hashed.

Two separate frame-grid concepts, do not mix them up:
  - ``input_frame_indices``: the sampled grid fed to **every** model (taken from
    the Trajectory; one per ResultSet, never changes).
  - each ``RewardOutput.frame_indices``: the subset of frames on which one model
    **actually reported a value** (TOPReward only at its prefix end points,
    Robo-Dopamine without the first frame, ...).
  The second is always a subset of the first; never treat one model's output
  grid as the shared input grid.

Crash safety: ``save()`` writes atomically (temp file + fsync + os.replace).
Concurrent writers to the same path are **not** supported; serialise writes or
use separate paths (see the benchmark worker design).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ._atomic import atomic_write_json
from .types import RewardOutput, SamplingSpec, Trajectory

SCHEMA_VERSION = 1

_REQUIRED_TOP_KEYS = ("schema_version", "video", "instruction", "sampling",
                      "created_at", "rewardbox_version", "environment",
                      "results", "errors")
_REQUIRED_SAMPLING_KEYS = ("spec", "key", "input_frame_indices")
_REQUIRED_RESULT_KEYS = (
    "model", "progress", "frame_indices", "normalize", "weights", "success",
    "raw", "cost", "meta",
)


class ResultSchemaError(ValueError):
    """The JSON does not satisfy the result-v1 schema (missing/unsupported
    version, missing required field, or a violated cross-field constraint)."""


class ResultConflictError(ValueError):
    """Results cannot be compared or merged (sampling/instruction/source/frame
    grid differ, or a same-name result exists and overwriting was not allowed)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rewardbox_version() -> str:
    try:
        from importlib.metadata import version
        return version("rewardbox")
    except Exception:
        return "unknown"


def _environment_info() -> dict:
    """python/torch/transformers/gpu.  Fields stay None when torch/transformers are
    not installed: ResultSet itself needs no heavy dependency, it only records them
    when they are already importable."""
    import platform
    info: dict = {"python": platform.python_version(), "torch": None,
                  "transformers": None, "gpu": None}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info


def _validate_payload(d: dict) -> None:
    """Required fields and version of result-v1.  Python-side object semantics
    are **not** checked here (the RewardOutput/SamplingSpec constructors do
    that); this only decides whether the JSON may be read as result-v1."""
    if not isinstance(d, dict):
        raise ResultSchemaError(f"result top level must be an object, got {type(d).__name__}")
    if "schema_version" not in d:
        raise ResultSchemaError("missing schema_version; cannot tell which result format this is")
    if d["schema_version"] != SCHEMA_VERSION:
        raise ResultSchemaError(
            f"unsupported schema_version={d['schema_version']!r} (this RewardBox only "
            f"supports {SCHEMA_VERSION}); refusing to guess an unknown version as v{SCHEMA_VERSION}")
    for k in _REQUIRED_TOP_KEYS:
        if k not in d:
            raise ResultSchemaError(f"missing required top-level field {k!r}")
    if not isinstance(d["video"], str):
        raise ResultSchemaError("video must be a string")
    if not isinstance(d["instruction"], str) or not d["instruction"].strip():
        raise ResultSchemaError("instruction must not be empty")
    sampling = d["sampling"]
    if not isinstance(sampling, dict):
        raise ResultSchemaError(f"sampling must be an object, got {type(sampling).__name__}")
    for k in _REQUIRED_SAMPLING_KEYS:
        if k not in sampling:
            extra = ("legacy results without input_frame_indices are not supported: "
                     "there is no way to check that each model's output grid is a subset "
                     "of the real input" if k == "input_frame_indices" else "")
            raise ResultSchemaError(f"sampling is missing field {k!r}. {extra}".rstrip())
    if not isinstance(d["results"], dict) or not isinstance(d["errors"], dict):
        raise ResultSchemaError("results and errors must be objects (dict)")
    input_idx = sampling["input_frame_indices"]
    if input_idx is None:
        # The input grid is only known once some model has actually read the video.
        if d["results"]:
            raise ResultSchemaError(
                "sampling.input_frame_indices may be null only when results is empty")
        input_idx = []
    else:
        if not isinstance(input_idx, list) or not input_idx:
            raise ResultSchemaError("sampling.input_frame_indices must be a non-empty array")
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in input_idx):
            raise ResultSchemaError(
                "sampling.input_frame_indices must contain only non-negative integers")
        if any(b <= a for a, b in zip(input_idx, input_idx[1:])):
            raise ResultSchemaError("sampling.input_frame_indices must be strictly increasing")
    overlap = set(d["results"]) & set(d["errors"])
    if overlap:
        raise ResultSchemaError(
            f"a model cannot appear in both results and errors: {sorted(overlap)}")
    for name, out_d in d["results"].items():
        if not isinstance(name, str) or not name:
            raise ResultSchemaError("model names in results must be non-empty strings")
        if not isinstance(out_d, dict):
            raise ResultSchemaError(f"results[{name!r}] must be an object")
        for k in _REQUIRED_RESULT_KEYS:
            if k not in out_d:
                raise ResultSchemaError(f"results[{name!r}] is missing field {k!r}")
        if out_d["model"] != name:
            raise ResultSchemaError(
                f"results key={name!r} but its content says model={out_d['model']!r}; they "
                f"must match (the file was edited by hand, or written under the wrong key)")
        progress = out_d["progress"]
        out_idx = out_d["frame_indices"]
        if not isinstance(progress, list) or not progress:
            raise ResultSchemaError(f"results[{name!r}].progress must be a non-empty array")
        if any(not isinstance(x, (int, float)) or isinstance(x, bool)
               or not np.isfinite(x) for x in progress):
            raise ResultSchemaError(
                f"results[{name!r}].progress must contain only finite numbers")
        if not isinstance(out_idx, list) or not out_idx:
            raise ResultSchemaError(f"results[{name!r}].frame_indices must be a non-empty array")
        if len(progress) != len(out_idx):
            raise ResultSchemaError(
                f"results[{name!r}] progress and frame_indices must have the same length")
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in out_idx):
            raise ResultSchemaError(
                f"results[{name!r}].frame_indices must contain only non-negative integers")
        if any(b <= a for a, b in zip(out_idx, out_idx[1:])):
            raise ResultSchemaError(
                f"results[{name!r}].frame_indices must be strictly increasing")
        missing = sorted(set(out_idx) - set(input_idx))
        if missing:
            raise ResultSchemaError(
                f"results[{name!r}].frame_indices contains frames outside the shared "
                f"input grid: {missing[:8]}")
        if out_d["normalize"] not in {"none", "minmax", "zscore", "rank"}:
            raise ResultSchemaError(
                f"results[{name!r}].normalize value is not supported: {out_d['normalize']!r}")
        for array_name in ("weights", "success"):
            values = out_d[array_name]
            if values is not None and (
                    not isinstance(values, list)
                    or len(values) != len(progress)
                    or any(not isinstance(x, (int, float)) or isinstance(x, bool)
                           or not np.isfinite(x) for x in values)):
                raise ResultSchemaError(
                    f"results[{name!r}].{array_name} must be null, or an array of finite "
                    "numbers as long as progress")
        if not isinstance(out_d["raw"], dict) or not isinstance(out_d["meta"], dict):
            raise ResultSchemaError(f"results[{name!r}].raw/meta must be objects")
        if out_d["cost"] is not None and not isinstance(out_d["cost"], dict):
            raise ResultSchemaError(f"results[{name!r}].cost must be an object or null")
    for name, error in d["errors"].items():
        if not isinstance(name, str) or not name:
            raise ResultSchemaError("model names in errors must be non-empty strings")
        if not isinstance(error, str) or not error:
            raise ResultSchemaError(f"errors[{name!r}] must be a non-empty string")


def _check_mergeable(existing: dict, rs: ResultSet, path: Path) -> None:
    """For ``save(merge=True)`` the existing file must describe the same trajectory."""
    mism = []
    if existing.get("instruction") != rs.instruction:
        mism.append(f"instruction: {existing.get('instruction')!r} != {rs.instruction!r}")
    if existing.get("video") != rs.source:
        mism.append(f"video: {existing.get('video')!r} != {rs.source!r}")
    existing_sampling = existing.get("sampling") or {}
    existing_key = existing_sampling.get("key")
    if existing_key != rs.sampling.key():
        mism.append(f"sampling.key: {existing_key!r} != {rs.sampling.key()!r}")
    existing_input_idx = existing_sampling.get("input_frame_indices")
    if (existing_input_idx is not None and rs.input_frame_indices is not None
            and list(existing_input_idx) != list(int(x) for x in rs.input_frame_indices)):
        mism.append("sampling.input_frame_indices differ: the two sides sampled different frames")
    if mism:
        raise ResultConflictError(
            f"refusing to merge into {path}: the existing file is not a result for the "
            "same trajectory\n  " + "\n  ".join(mism))


@dataclass
class ResultSet:
    """Results of several models on one trajectory (same video + same
    instruction + same SamplingSpec + same sampled frames).

    The recommended constructor is ``ResultSet.from_trajectory(traj)``: it locks
    instruction/sampling/source/input_frame_indices from the Trajectory, leaving
    no room for typos or omissions.  Direct ``ResultSet(...)`` construction is
    allowed (tests use it), but ``add()`` requires ``input_frame_indices``, the
    hard requirement that every formal result is traceable to its input grid.
    """

    instruction: str
    sampling: SamplingSpec
    source: str = ""
    input_frame_indices: np.ndarray | None = None
    results: dict[str, RewardOutput] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)

    def __post_init__(self):
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ResultSchemaError("instruction must be a non-empty string")
        if self.input_frame_indices is not None:
            self.input_frame_indices = np.asarray(self.input_frame_indices, dtype=np.int64)
            if self.input_frame_indices.ndim != 1 or len(self.input_frame_indices) == 0:
                raise ResultSchemaError("input_frame_indices must be a non-empty 1-D array")
            if np.any(self.input_frame_indices < 0):
                raise ResultSchemaError("input_frame_indices must not contain negative values")
            if (len(self.input_frame_indices) > 1
                    and np.any(np.diff(self.input_frame_indices) <= 0)):
                raise ResultSchemaError("input_frame_indices must be strictly increasing")

    @classmethod
    def from_trajectory(cls, traj: Trajectory) -> ResultSet:
        """The one "normal" constructor: lock every comparability field from an
        already sampled Trajectory."""
        if traj.sampling is None:
            raise ResultSchemaError(
                "Trajectory has no sampling, cannot build a ResultSet; construct the "
                "trajectory with Trajectory.from_video(..., sampling=...) or another "
                "standard entry point")
        return cls(instruction=traj.instruction, sampling=traj.sampling, source=traj.source,
                   input_frame_indices=np.asarray(traj.frame_indices, dtype=np.int64))

    # ---- comparability checks ----
    def _require_input_grid(self) -> None:
        if self.input_frame_indices is None:
            raise ResultSchemaError(
                "ResultSet has no input_frame_indices, cannot check comparability; build "
                "it with ResultSet.from_trajectory() or pass input_frame_indices=")

    def add(self, out: RewardOutput, *, overwrite: bool = False) -> None:
        """Add one model's result.

        Rejects (``ResultConflictError``, or ``ResultSchemaError`` for schema-level
        problems):
          - this ResultSet has no ``input_frame_indices``
          - ``out.sampling`` is missing or its ``.key()`` differs from this ResultSet
          - ``out.frame_indices`` is not a subset of the input grid
          - a same-name result exists and ``overwrite=True`` was not passed
        """
        self._require_input_grid()
        if out.sampling is None:
            raise ResultConflictError(
                f"RewardOutput of {out.model_name!r} has no sampling; cannot check comparability")
        if out.sampling.key() != self.sampling.key():
            raise ResultConflictError(
                f"sampling differs, refusing to add {out.model_name!r}: this ResultSet uses "
                f"{self.sampling.key()} ({self.sampling}), the output uses "
                f"{out.sampling.key()} ({out.sampling}). Results under different sampling "
                f"protocols are not comparable and cannot share a ResultSet.")
        missing = np.setdiff1d(
            out.frame_indices, self.input_frame_indices, assume_unique=True)
        if len(missing):
            raise ResultConflictError(
                f"output frames {missing[:8].tolist()} of {out.model_name!r} are not in the "
                f"shared input grid {self.input_frame_indices.tolist()}; a model may only "
                "report values on a subset of its actual input frames")
        if ((out.model_name in self.results or out.model_name in self.errors)
                and not overwrite):
            raise ResultConflictError(
                f"{out.model_name!r} already exists in this ResultSet (as a result or an "
                "error); pass overwrite=True to replace it")
        self.results[out.model_name] = out
        self.errors.pop(out.model_name, None)

    def add_error(self, model_name: str, error: str, *, overwrite: bool = False) -> None:
        """Record why a model failed without stopping the overall run."""
        model_name, error = str(model_name), str(error)
        if not model_name or not error:
            raise ResultSchemaError("model_name and error must both be non-empty strings")
        if model_name in self.results and not overwrite:
            raise ResultConflictError(
                f"{model_name!r} already has a successful result and cannot also record an "
                "error; pass overwrite=True to replace it with the failure")
        if overwrite:
            self.results.pop(model_name, None)
        self.errors[model_name] = error

    def common_grid(self) -> np.ndarray:
        """Union (sorted, unique) of every result's ``frame_indices``, **not**
        ``input_frame_indices``.  It is the reference grid of frames where at least
        one model reported a value, used for cross-model alignment and plots;
        ``input_frame_indices`` is the grid fed to all models.
        """
        if not self.results:
            return np.array([], dtype=np.int64)
        return np.unique(
            np.concatenate([r.frame_indices for r in self.results.values()]))

    def aligned(self) -> dict[str, np.ndarray]:
        """Interpolate every result's progress onto ``common_grid()``.

        Grid points outside the range covered by a model's own ``frame_indices`` are
        ``NaN``: no extrapolation and no constant fill that would invent values the
        model never reported (the same rule as the Robo-Dopamine adapter's "do not
        pad the first frame").  Points **inside** the range are linear
        interpolations, not native observations at that exact frame; serialisation
        and display layers must mark which points are native
        (``RewardOutput.frame_indices``) and which are interpolated.
        """
        grid = self.common_grid()
        out: dict[str, np.ndarray] = {}
        for name, r in self.results.items():
            vals = np.full(len(grid), np.nan)
            if len(grid):
                lo, hi = r.frame_indices[0], r.frame_indices[-1]
                mask = (grid >= lo) & (grid <= hi)
                if mask.any():
                    vals[mask] = np.interp(grid[mask], r.frame_indices, r.progress)
            out[name] = vals
        return out

    def to_dict(self) -> dict:
        if self.results:
            self._require_input_grid()
        d = {
            "schema_version": SCHEMA_VERSION,
            "video": self.source,
            "instruction": self.instruction,
            "sampling": {
                "spec": self.sampling.to_dict(),
                "key": self.sampling.key(),
                "input_frame_indices": (None if self.input_frame_indices is None
                                        else self.input_frame_indices.tolist()),
            },
            "created_at": self.created_at,
            "rewardbox_version": _rewardbox_version(),
            "environment": _environment_info(),
            "results": {name: out.to_dict() for name, out in self.results.items()},
            "errors": dict(self.errors),
        }
        _validate_payload(d)
        return d

    def save(self, path: str | Path, *, merge: bool = True,
             overwrite_models: bool = False) -> None:
        """Write JSON schema v1, atomically replacing the target file.

        With ``merge=True`` and an existing file: first check that its
        ``instruction`` / ``video`` / ``sampling.key`` / ``input_frame_indices``
        match this ResultSet and refuse otherwise (no blind dictionary merge, which
        is exactly reward-scope's bug); then merge ``results``/``errors``.
        **Same-name models are refused by default** (matching ``add()``); pass
        ``overwrite_models=True`` to replace them.  ``merge=False`` overwrites the
        file as a whole without any consistency check.

        The write is atomic: a temp file in the same directory, ``fsync``, then
        ``os.replace()``; the temp file is removed on error.  **Concurrent writers to
        the same path are not supported**; benchmark-time writers must serialise or
        use different output paths.
        """
        path = Path(path)
        payload = self.to_dict()
        if merge and path.exists():
            existing = json.loads(path.read_text())
            _validate_payload(existing)
            _check_mergeable(existing, self, path)
            existing_names = set(existing.get("results", {})) | set(existing.get("errors", {}))
            incoming_names = set(payload["results"]) | set(payload["errors"])
            dup = existing_names & incoming_names
            if dup and not overwrite_models:
                raise ResultConflictError(
                    f"merging would overwrite existing model results {sorted(dup)}; pass "
                    f"overwrite_models=True to replace them")
            payload["results"] = {**existing.get("results", {}), **payload["results"]}
            payload["errors"] = {**existing.get("errors", {}), **payload["errors"]}
            # A new success replaces an old failure and vice versa; keep the two
            # namespaces disjoint.
            for name in self.results:
                payload["errors"].pop(name, None)
            for name in self.errors:
                payload["results"].pop(name, None)
            if payload["sampling"]["input_frame_indices"] is None:
                payload["sampling"]["input_frame_indices"] = (
                    existing["sampling"]["input_frame_indices"])
            payload["created_at"] = existing.get("created_at", payload["created_at"])

        _validate_payload(payload)
        atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> ResultSet:
        """Read JSON schema v1.  Everything goes through ``add()``; no invariant is
        bypassed.  Unknown extra top-level fields are ignored."""
        d = json.loads(Path(path).read_text())
        _validate_payload(d)
        sampling_block = d["sampling"]
        sampling = SamplingSpec.from_dict(sampling_block["spec"])
        if sampling.key() != sampling_block["key"]:
            raise ResultSchemaError(
                f"sampling.key does not match sampling.spec: the file says "
                f"key={sampling_block['key']!r} but the spec gives {sampling.key()!r}; the "
                "file may have been edited by hand")
        input_idx = sampling_block["input_frame_indices"]
        if input_idx is not None:
            input_idx = np.asarray(input_idx, dtype=np.int64)

        rs = cls(instruction=d["instruction"], sampling=sampling, source=d.get("video", ""),
                 input_frame_indices=input_idx,
                 created_at=d.get("created_at", _now_iso()))
        for name, out_d in d.get("results", {}).items():
            out = RewardOutput.from_dict(out_d)
            out.sampling = sampling      # restore the top-level SamplingSpec (RewardOutput's
                                         # own to_dict/from_dict does not serialise sampling,
                                         # so it is not repeated in every model's result)
            rs.add(out)                  # the shared validation path, not rs.results directly
        for name, error in d.get("errors", {}).items():
            rs.add_error(name, error)
        return rs
