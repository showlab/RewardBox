"""Single-model benchmark worker: load the model once, run a batch of trajectories,
update one shard.

``run_model_over_manifest`` can be called in-process by ``runner.py`` or run as a
separate subprocess (``python -m rewardbox.bench.worker ...``, isolated mode).  The
subprocess runs the **complete adapter** (real ``create_reward`` + ``score``), not a
proxied backend, so global side effects such as unsloth/flash-attn stay inside that
process and do not leak into the process running the next model.

``--identity-only`` lets the parent ask the target interpreter for its environment
identity first, so shards produced under a different Python/dependency environment
still resume correctly instead of the parent's versions standing in for the worker's.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .shard import (
    ModelShard,
    build_experiment_identity,
    worker_provenance,
)


def run_model_over_manifest(model: str, manifest, sampling, shard: ModelShard,
                            shard_path: str | Path, trajectory_ids=None,
                            model_opts: dict | None = None,
                            retry_errors: bool = False, log=print) -> ModelShard:
    """Load the model once and run the selected manifest entries (all by default).

    Entries already in the shard (success or failure) whose manifest fields are
    unchanged are skipped; this is the core of ``--resume``, decided from the
    manifest alone, so nothing is decoded to find the pending entries.  Each pending
    entry is decoded once, scored immediately, and the shard is saved atomically
    after every entry, so a killed process loses only the current entry.

    When resume leaves nothing to do the model is not even loaded: re-running a
    benchmark whose models are mostly finished does not pay a full model load
    (tens of seconds to minutes) just to confirm there is nothing left.
    """
    entries = list(manifest) if trajectory_ids is None else [
        manifest.by_id(tid) for tid in trajectory_ids]
    pending = [entry for entry in entries
               if not shard.has_current(entry.id, manifest.entry_identity(entry),
                                        retry_errors=retry_errors)]
    log(f"[{model}] {len(entries) - len(pending)} resumed by manifest entry, "
        f"{len(pending)} pending")
    if not pending:
        return shard

    import rewardbox

    shard.provenance = worker_provenance()
    shard.fatal_error = None
    shard.save(shard_path)
    try:
        rm = rewardbox.create_reward(model, **(model_opts or {}))
    except Exception as e:
        message = f"model load failed: {e}"
        shard.fatal_error = message
        for entry in pending:
            shard.add_error(entry.id, message, manifest.entry_identity(entry))
        shard.save(shard_path)
        raise
    try:
        for entry in pending:
            input_frame_indices = None
            try:
                traj = manifest.to_trajectory(entry, sampling)
                input_frame_indices = traj.frame_indices
                out = rm.score(traj)
                out.meta["benchmark"] = {
                    "identity": shard.identity,
                    "worker": shard.provenance,
                }
                shard.add_result(entry.id, out, manifest.entry_identity(entry),
                                 input_frame_indices)
                log(f"[{model}] {entry.id}: ok")
            except Exception as e:
                shard.add_error(entry.id, str(e), manifest.entry_identity(entry),
                                input_frame_indices)
                log(f"[{model}] {entry.id}: FAILED - {e}")
            shard.save(shard_path)
    finally:
        try:
            rm.close()
        except Exception as e:
            log(f"[{model}] WARNING: close() failed (finished results are kept): {e}")
    return shard


def main(argv: list[str] | None = None) -> int:
    from ..core.types import SamplingSpec
    from ..data import Manifest

    p = argparse.ArgumentParser(prog="python -m rewardbox.bench.worker")
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--sampling", required=True,
                   help="JSON string of SamplingSpec.to_dict()")
    p.add_argument("--shard")
    p.add_argument("--model-opts", default="{}",
                   help="JSON object of create_reward() arguments")
    p.add_argument(
        "--identity-only", action="store_true",
        help="print the benchmark identity built by this Python environment and exit "
             "without loading the model",
    )
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--trajectory-id", action="append", default=None, dest="trajectory_ids")
    args = p.parse_args(argv)

    sampling = SamplingSpec.from_dict(json.loads(args.sampling))
    model_opts = json.loads(args.model_opts)
    if not isinstance(model_opts, dict):
        p.error("--model-opts must parse to a JSON object")
    identity = build_experiment_identity(args.model, sampling, model_opts)
    if args.identity_only:
        print(json.dumps(identity, sort_keys=True))
        return 0
    if not args.shard:
        p.error("normal worker mode requires --shard")
    manifest = Manifest.load(args.manifest)
    shard = ModelShard.load_or_new(args.shard, args.model, identity)

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        run_model_over_manifest(args.model, manifest, sampling, shard, args.shard,
                                args.trajectory_ids, model_opts=model_opts,
                                retry_errors=args.retry_errors, log=log)
    except Exception as e:
        print(f"[{args.model}] FATAL: {e}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
