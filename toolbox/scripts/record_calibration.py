#!/usr/bin/env python3
"""Record one model's real-weight calibration output at a locked profile.

This is deliberately an explicit mutation command: an existing record is never
overwritten unless ``--replace`` is supplied.  It is intended for a default-change
recalibration, not for choosing a more favourable result after inspection.  Any
non-default adapter options are serialized into the record and replayed by the
calibration regression.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO = REPO_ROOT / "calibration/synthetic_pick.mp4"
INSTRUCTION = "move the red block onto the green target"


def _parse_model_opt(pair: str) -> tuple[str, object]:
    if "=" not in pair:
        raise ValueError(f"--model-opt needs the form KEY=JSON_VALUE, got {pair!r}")
    key, raw = pair.split("=", 1)
    if not key:
        raise ValueError(f"--model-opt key must not be empty: {pair!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def _environment() -> dict[str, str | None]:
    import torch
    import transformers

    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "gpu": gpu,
        "transformers": str(transformers.__version__),
    }


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path,
                        help="default: rewardbox/calibration/records/<model>.json")
    parser.add_argument("--replace", action="store_true",
                        help="explicitly replace a stale record after a locked-default change")
    parser.add_argument(
        "--model-opt", action="append", default=[], metavar="KEY=JSON_VALUE",
        help="adapter option to serialize and replay; may be repeated",
    )
    args = parser.parse_args()

    import rewardbox
    from rewardbox.core.types import Trajectory

    spec = rewardbox.get_spec(args.model)
    model_opts = dict(_parse_model_opt(pair) for pair in args.model_opt)
    if args.model == "sole-r1-8b":
        if model_opts.get("temperature") != 0.0:
            parser.error(
                "sole-r1-8b calibration must explicitly use "
                "--model-opt temperature=0.0; the upstream 1.0 default is stochastic"
            )
        if not ({"server_provenance", "server_provenance_file"} & model_opts.keys()):
            parser.error(
                "sole-r1-8b calibration requires explicit server provenance"
            )
    output_path = args.output or (REPO_ROOT / "rewardbox/calibration/records" / f"{args.model}.json")
    if output_path.exists() and not args.replace:
        raise FileExistsError(f"record exists: {output_path}; pass --replace only for a deliberate re-record")
    if not VIDEO.is_file():
        raise FileNotFoundError(VIDEO)

    trajectory = Trajectory.from_video(VIDEO, INSTRUCTION, spec.sampling)
    model = rewardbox.create_reward(args.model, **model_opts)
    try:
        result = model.score(trajectory)
    finally:
        model.close()
    payload = {
        "model": args.model,
        "video": str(VIDEO.relative_to(REPO_ROOT)),
        "instruction": INSTRUCTION,
        "sampling": spec.sampling.to_dict(),
        "model_opts": model_opts,
        "environment": _environment(),
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "output": result.to_dict(),
    }
    _atomic_write(output_path, payload)
    print(f"wrote {output_path}")
    print(f"progress={result.progress.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
