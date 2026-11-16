"""Score one video with one model, then print the curve.

    python examples/quickstart.py robot.mp4 "open the drawer" --model robometer-4b

Run `rewardbox doctor MODEL` first; it prints what is missing and how to fix it.
"""
from __future__ import annotations

import argparse

import rewardbox


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("instruction")
    ap.add_argument("--model", default="robometer-4b")
    args = ap.parse_args()

    spec = rewardbox.get_spec(args.model)
    print(f"{spec.name}: default sampling {spec.sampling.to_dict()}, "
          f"raw range {spec.score_range}, higher_better={spec.higher_better}")
    with rewardbox.create_reward(args.model) as rm:
        out = rm(args.video, args.instruction)
    for frame, value in zip(out.frame_indices.tolist(), out.progress.tolist(), strict=True):
        print(f"frame {frame:>6}  progress {value:.4f}")
    if out.success is not None:
        print("terminal success probability:", round(out.terminal_success, 4))
    print(f"latency {out.cost.latency_s:.2f}s over {out.cost.n_calls} backend call(s)")


if __name__ == "__main__":  # vLLM-based adapters spawn worker processes
    main()
