"""Run several models on the same manifest and print per-entry terminal values.

    python examples/compare_models.py datasets/local_example/manifest.jsonl \
        --model robometer-4b --model roboreward-4b --num-frames 8 -o runs/compare

Uses the resumable benchmark runner (one shard per model; rerun to resume).
Each model's values stay on its own raw scale; they are not directly comparable
across models without an explicit normalization.
"""
from __future__ import annotations

import argparse

from rewardbox import SamplingSpec
from rewardbox.bench import aggregate, run_benchmark


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--num-frames", type=int, required=True)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    sampling = SamplingSpec(num_frames=args.num_frames)   # one explicit protocol for all models
    shards = run_benchmark(args.model, args.manifest, sampling, args.output, log=print)
    for traj_id, rs in sorted(aggregate(shards, args.manifest, sampling).items()):
        cells = [f"{name}={out.progress[-1]:.3f}" for name, out in sorted(rs.results.items())]
        print(traj_id, *cells, *(f"{m}=ERROR" for m in sorted(rs.errors)))


if __name__ == "__main__":
    main()
