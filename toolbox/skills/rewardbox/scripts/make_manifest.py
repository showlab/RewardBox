#!/usr/bin/env python3
"""Build a RewardBox manifest (.jsonl) from a folder of videos.

    python make_manifest.py VIDEO_DIR --instruction "open the drawer" -o manifest.jsonl
    python make_manifest.py VIDEO_DIR --instructions tasks.json -o manifest.jsonl
    python make_manifest.py VIDEO_DIR --views primary=_front,left_wrist=_wrist ...

Paths are written relative to the manifest file, as `rewardbox bench` expects.
`--instructions` maps a video stem (or a prefix of it) to its instruction.
`--views` groups files that share a stem and differ by suffix into one
multi-view entry (e.g. ep1_front.mp4 + ep1_wrist.mp4). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_dir", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--instruction", help="one instruction for every video")
    group.add_argument("--instructions", type=Path, help="JSON {stem_or_prefix: instruction}")
    ap.add_argument("--views", help="multi-view grouping: view=suffix,view=suffix")
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args(argv)

    pattern = "**/*" if args.recursive else "*"
    files = sorted(p for p in args.video_dir.glob(pattern) if p.suffix.lower() in VIDEO_EXT)
    if not files:
        print(f"no videos under {args.video_dir}", file=sys.stderr)
        return 2
    table = json.loads(args.instructions.read_text()) if args.instructions else {}
    out_dir = args.output.resolve().parent

    def rel(p: Path) -> str:
        return os.path.relpath(p.resolve(), out_dir)

    def instruction_for(stem: str) -> str:
        if args.instruction:
            return args.instruction
        matches = [k for k in table if stem == k or stem.startswith(k)]
        if not matches:
            raise SystemExit(f"no instruction for {stem!r} in {args.instructions}")
        return table[max(matches, key=len)]

    rows = []
    if args.views:
        views = dict(item.split("=", 1) for item in args.views.split(","))
        groups: dict[str, dict[str, Path]] = {}
        for p in files:
            for view, suffix in views.items():
                if p.stem.endswith(suffix):
                    groups.setdefault(p.stem[: -len(suffix)], {})[view] = p
        for stem, vs in sorted(groups.items()):
            if set(vs) != set(views):
                print(f"skip {stem}: has views {sorted(vs)}, need {sorted(views)}", file=sys.stderr)
                continue
            rows.append({"id": stem, "video": {v: rel(p) for v, p in vs.items()},
                         "instruction": instruction_for(stem)})
    else:
        seen = set()
        for p in files:
            stem = p.relative_to(args.video_dir).with_suffix("").as_posix().replace("/", "__")
            if stem in seen:
                raise SystemExit(f"duplicate id {stem!r}")
            seen.add(stem)
            rows.append({"id": stem, "video": rel(p), "instruction": instruction_for(p.stem)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
