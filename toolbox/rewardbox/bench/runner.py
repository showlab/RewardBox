"""Benchmark entry point: models in the outer loop, trajectories in the inner loop.

```
for model in models:      # fixed order, written to model_order.json
    load model once
    for trajectory in manifest:
        run
    close model
```

Never reload the model per trajectory: copying reward-scope's ``compare_models``
pattern ("build and close every model") reloads an 8B model once per trajectory.

With ``isolate=True`` each model runs in its own subprocess (see ``worker.py``) so
global side effects such as unsloth/flash-attn cannot leak across models.  The
default ``isolate=False`` runs in-process: the path validated with real weights
(four models run back to back in one pytest process), faster and simpler; isolation
is an option for stronger guarantees, not a forced default.
``worker_pythons={model: executable}`` additionally lets each model use its own
Python/dependency environment and implies isolation for those models; the target
interpreter builds its own experiment identity first.

Each model has its own shard file (``ModelShard``, saved atomically).  A model that
fails as a whole (e.g. it cannot even load) only affects its own shard; with
``resume=True`` (default) entries already in the shard are skipped.

Inputs are trusted fixed files: resume compares manifest entries, each pending
video is decoded once at inference time, and ``aggregate`` reads only shards and
the manifest (no decoding, no hashing).

Only ``aggregate`` merges the per-model shards back into one ``ResultSet`` per
trajectory id, reusing every check of ``ResultSet.add()`` (sampling key, input
frame grid, ...) instead of re-inventing "refuse to merge incomparable results".
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ..core._atomic import atomic_write_json
from ..core.secrets import sensitive_option_paths
from .shard import (
    BenchmarkIdentityError,
    ModelShard,
    build_experiment_identity,
)
from .worker import run_model_over_manifest


def run_benchmark(models: list[str], manifest_path: str | Path, sampling,
                  output_dir: str | Path, *, isolate: bool = False,
                  resume: bool = True, model_opts: dict[str, dict] | None = None,
                  retry_errors: bool = False,
                  worker_pythons: dict[str, str] | None = None,
                  log=print) -> dict[str, ModelShard]:
    """The order of ``models`` is the run order: never re-sorted, and written to
    ``<output_dir>/model_order.json`` for later audit."""
    from ..data import Manifest

    model_opts = model_opts or {}
    for model, opts in model_opts.items():
        secret_paths = sensitive_option_paths(opts)
        if secret_paths:
            raise ValueError(
                f"benchmark model_opts of {model!r} contain credential fields {secret_paths}; "
                "isolated workers receive ordinary model_opts as process arguments, so "
                "credentials must come from the matching environment variable (GVL: "
                "GEMINI_API_KEY; Jev: JEV_API_KEY; or the variable named by api_key_env)"
            )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not models or len(set(models)) != len(models):
        raise ValueError("models must be non-empty and contain no duplicates")
    atomic_write_json(output_dir / "model_order.json", models)

    manifest_path = Path(manifest_path).resolve()
    manifest = Manifest.load(manifest_path)
    worker_pythons = worker_pythons or {}
    unknown_worker_models = set(worker_pythons) - set(models)
    if unknown_worker_models:
        raise ValueError(
            "worker_pythons names models that were not selected: "
            f"{sorted(unknown_worker_models)}"
        )

    shards: dict[str, ModelShard] = {}
    for model in models:
        shard_path = output_dir / f"{model}.json"
        opts = dict(model_opts.get(model, {}))
        worker_python = _resolve_worker_python(worker_pythons.get(model))
        use_isolated_worker = isolate or model in worker_pythons
        if worker_python == _resolve_worker_python(None):
            identity = build_experiment_identity(model, sampling, opts)
        else:
            identity = _query_worker_identity(
                worker_python, model, manifest_path, sampling, opts)
        if not resume and shard_path.exists():
            shard_path.unlink()
        if use_isolated_worker:
            shards[model] = _run_isolated(
                model, manifest, manifest_path, sampling, shard_path, identity,
                opts, retry_errors, log, worker_python=worker_python)
        else:
            shard = ModelShard.load_or_new(shard_path, model, identity)
            try:
                run_model_over_manifest(
                    model, manifest, sampling, shard, shard_path,
                    model_opts=opts, retry_errors=retry_errors, log=log)
            except Exception as e:
                # One model failing as a whole (missing dependency, load error) must not
                # take the others down: finished entries stay in its shard and the loop
                # moves on to the next model.
                if not shard.fatal_error:
                    shard.fatal_error = f"benchmark worker failed as a whole: {e}"
                    shard.save(shard_path)
                log(f"[{model}] load/run failed as a whole; skipping this model, the "
                    f"others are unaffected: {e}")
            shards[model] = shard
    return shards


def _resolve_worker_python(value: str | None) -> str:
    """Return an absolute executable path without dereferencing venv symlinks."""
    if value is None:
        return str(Path(sys.executable).absolute())
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.absolute()
    if not path.is_file():
        raise ValueError(f"worker Python does not exist or is not a file: {path}")
    if not os.access(path, os.X_OK):
        raise ValueError(f"worker Python is not executable: {path}")
    return str(path)


def _worker_base_cmd(python: str, model: str, manifest_path: Path, sampling,
                     model_opts: dict) -> list[str]:
    return [
        python, "-m", "rewardbox.bench.worker",
        "--model", model,
        "--manifest", str(manifest_path),
        "--sampling", json.dumps(sampling.to_dict()),
        "--model-opts", json.dumps(model_opts),
    ]


def _query_worker_identity(python: str, model: str, manifest_path: Path,
                           sampling, model_opts: dict) -> dict:
    """Ask the selected interpreter to describe its own dependency environment."""
    cmd = _worker_base_cmd(python, model, manifest_path, sampling, model_opts)
    cmd.append("--identity-only")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"the worker Python of {model!r} could not build the experiment identity "
            f"({python}): {proc.stderr.strip() or proc.stdout.strip() or 'no error output'}"
        )
    try:
        identity = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"the worker Python of {model!r} returned invalid identity JSON: "
            f"{proc.stdout[:500]!r}"
        ) from exc
    if not isinstance(identity, dict):
        raise RuntimeError(f"the worker identity of {model!r} is not a JSON object")
    return identity


def _run_isolated(model: str, manifest, manifest_path: Path, sampling,
                  shard_path: Path, identity: dict, model_opts: dict,
                  retry_errors: bool, log, *,
                  worker_python: str | None = None) -> ModelShard:
    worker_python = _resolve_worker_python(worker_python)
    cmd = _worker_base_cmd(
        worker_python, model, manifest_path, sampling, model_opts)
    cmd.extend(["--shard", str(shard_path)])
    if retry_errors:
        cmd.append("--retry-errors")
    log(f"[{model}] running in an isolated subprocess: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout_log = shard_path.with_suffix(".worker.stdout.log")
    stderr_log = shard_path.with_suffix(".worker.stderr.log")
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")
    for line in proc.stdout.splitlines():
        log(f"  {line}")
    if proc.returncode != 0:
        log(f"[{model}] worker subprocess exit code {proc.returncode}, stderr:\n{proc.stderr}")
    elif proc.stderr:
        log(f"[{model}] worker stderr/warnings saved to {stderr_log}")
    shard = ModelShard.load_or_new(shard_path, model, identity)
    if proc.returncode != 0 and not shard.fatal_error:
        message = (f"worker subprocess exit code {proc.returncode}: "
                   f"{proc.stderr.strip() or 'no stderr'}")
        shard.fatal_error = message
        for entry in manifest:
            if not shard.has(entry.id):
                shard.add_error(entry.id, message, manifest.entry_identity(entry))
        shard.save(shard_path)
    return shard


def aggregate(shards: dict[str, ModelShard], manifest_path: str | Path, sampling):
    """Merge the per-model shards into one ResultSet per trajectory id.  This is the
    only cross-model merge, and it fully reuses the checks of ResultSet.add().
    Reads only the manifest and the shards; no video is decoded.
    Returns {trajectory_id: ResultSet}."""
    from ..core.results import ResultSet
    from ..core.types import RewardOutput
    from ..data import Manifest

    manifest = Manifest.load(manifest_path)
    out: dict[str, ResultSet] = {}
    for entry in manifest:
        entry_identity = manifest.entry_identity(entry)
        grids: dict[str, tuple[int, ...]] = {}
        for model, shard in shards.items():
            if not shard.has(entry.id):
                raise BenchmarkIdentityError(
                    f"shard {model!r} has neither a result nor an error for entry "
                    f"{entry.id!r}; the benchmark is incomplete, refusing to produce an "
                    "aggregate that looks successful")
            record = shard.inputs.get(entry.id) or {}
            if record.get("entry") != entry_identity:
                raise BenchmarkIdentityError(
                    f"shard {model!r} scored entry {entry.id!r} with different manifest "
                    f"fields ({record.get('entry')!r} != {entry_identity!r}); refusing to "
                    "aggregate stale results")
            if record.get("input_frame_indices") is not None:
                grids[model] = tuple(record["input_frame_indices"])
        if len(set(grids.values())) > 1:
            raise BenchmarkIdentityError(
                f"models sampled different frame grids for entry {entry.id!r} "
                f"({ {m: list(g) for m, g in grids.items()} }); the video changed between "
                "model runs, re-run the affected models with --no-resume")
        grid = next(iter(grids.values()), None)
        rs = ResultSet(instruction=entry.instruction, sampling=sampling,
                       source=manifest.trajectory_source(entry),
                       input_frame_indices=None if grid is None else list(grid))
        for model, shard in shards.items():
            if entry.id in shard.results:
                reward_out = RewardOutput.from_dict(shard.results[entry.id])
                reward_out.sampling = sampling   # see the same note in ResultSet.load()
                rs.add(reward_out)
            else:
                rs.add_error(model, shard.errors[entry.id])
        out[entry.id] = rs
    return out
