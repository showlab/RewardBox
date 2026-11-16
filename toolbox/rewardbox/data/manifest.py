"""Local dataset manifest: csv or jsonl, listing trajectory video paths + labels.

This minimal local format comes first; adapters for large datasets such as
LIBERO/OXE are later work and out of scope for this version (E1 covers only "small
local datasets").

Fields (at least):
  id           unique trajectory id; must not repeat within a manifest
  video        single view: video path string relative to the manifest's directory;
               multi-view: {view_name: relative path} dict (jsonl only; csv is flat
               and supports a single view)
  instruction  task instruction text
  split        dataset split (train/val/test, ...), default "test"
  success      success label (bool, optional)
  progress_gt  per-frame ground-truth progress (float sequence, optional)
  progress_gt_frame_indices
               source-video frame number of each progress_gt value; required with
               progress_gt.  When a trajectory is loaded the labels are linearly
               interpolated onto the actual sampled grid, never extrapolated
  pair_id      pair id (for ranking, e.g. a success/failure pair), optional
  group_id     group id (for ranking, e.g. all trajectories of one task), optional
  source       origin/provenance note, optional

Path resolution: video paths always resolve against the **manifest file's
directory**, not the current working directory, so a manifest copied/moved with its
directory structure intact keeps working.

Validation: duplicate ids, paths to missing files and missing required fields are
errors (records are never skipped silently, which would make the dataset look more
complete than it is).  Only file existence is checked; inputs are trusted and their
content is never hashed.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    video: str | dict[str, str]
    instruction: str
    split: str = "test"
    success: bool | None = None
    progress_gt: tuple[float, ...] | None = None
    progress_gt_frame_indices: tuple[int, ...] | None = None
    pair_id: str | None = None
    group_id: str | None = None
    source: str = ""

    def resolved_video(self, base_dir: Path) -> str | dict[str, str]:
        """Resolve video path(s) against base_dir (the manifest's directory)."""
        if isinstance(self.video, dict):
            return {k: str((base_dir / v).resolve()) for k, v in self.video.items()}
        return str((base_dir / self.video).resolve())


class Manifest:
    """A loaded and validated manifest."""

    def __init__(self, entries: list[ManifestEntry], base_dir: Path,
                 source_path: Path | None = None):
        self.entries = entries
        self.base_dir = base_dir
        # Isolated benchmark worker subprocesses reload the same manifest; keep the
        # original path, not only base_dir, so the runner never guesses a file name.
        self.source_path = source_path

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def by_split(self, split: str) -> list[ManifestEntry]:
        return [e for e in self.entries if e.split == split]

    def by_id(self, entry_id: str) -> ManifestEntry:
        for e in self.entries:
            if e.id == entry_id:
                return e
        raise KeyError(f"manifest has no id={entry_id!r}")

    @staticmethod
    def entry_identity(entry: ManifestEntry) -> dict:
        """The manifest fields that reach the model: benchmark resume compares these
        per entry, so editing one entry re-runs only that entry.  Paths are kept as
        written (moving the whole dataset directory does not invalidate results)."""
        video = dict(entry.video) if isinstance(entry.video, dict) else entry.video
        return {"video": video, "instruction": entry.instruction}

    def trajectory_source(self, entry: ManifestEntry) -> str:
        """``Trajectory.source`` that :meth:`to_trajectory` produces, without decoding."""
        video = entry.resolved_video(self.base_dir)
        if isinstance(video, dict):
            return json.dumps({k: str(v) for k, v in video.items()})
        return str(video)

    def to_trajectory(self, entry: ManifestEntry, sampling):
        """Build the Trajectory from the entry's video field (single or multi-view)
        and pass along the success/progress_gt ground-truth fields (they are
        Trajectory fields that most uses simply leave empty)."""
        import numpy as np

        from ..core.types import Trajectory

        video = entry.resolved_video(self.base_dir)
        meta = {
            "manifest_id": entry.id,
            "manifest_split": entry.split,
            "pair_id": entry.pair_id,
            "group_id": entry.group_id,
            "dataset_source": entry.source,
        }
        if isinstance(video, dict):
            traj = Trajectory.from_views(
                video, entry.instruction, sampling, success=entry.success, meta=meta)
        else:
            traj = Trajectory.from_video(
                video, entry.instruction, sampling, success=entry.success, meta=meta)
        if entry.progress_gt is not None:
            gt = np.asarray(entry.progress_gt, dtype=np.float64)
            gt_idx = np.asarray(entry.progress_gt_frame_indices, dtype=np.int64)
            outside = traj.frame_indices[
                (traj.frame_indices < gt_idx[0]) | (traj.frame_indices > gt_idx[-1])]
            if len(outside):
                raise ValueError(
                    f"progress_gt of entry {entry.id!r} only covers frames "
                    f"[{gt_idx[0]}, {gt_idx[-1]}], but the sampled grid includes frames "
                    f"outside it {outside[:8].tolist()}; refusing to invent labels by "
                    "constant extrapolation")
            traj.progress_gt = np.interp(traj.frame_indices, gt_idx, gt)
            traj.meta["progress_gt_source"] = {
                "frame_indices": gt_idx.tolist(),
                "values": gt.tolist(),
                "alignment": "linear_interpolation_no_extrapolation",
            }
        return traj

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"manifest does not exist: {path}")
        if path.suffix == ".csv":
            entries = _load_csv(path)
        elif path.suffix in (".jsonl", ".ndjson"):
            entries = _load_jsonl(path)
        else:
            raise ValueError(
                f"unsupported manifest format {path.suffix!r} (only .csv / .jsonl)")
        base_dir = path.resolve().parent
        _validate_entries(entries, base_dir)
        return cls(entries, base_dir, source_path=path.resolve())


def _validate_entries(entries: list[ManifestEntry], base_dir: Path) -> None:
    if not entries:
        raise ValueError("manifest is empty")
    seen_ids: set[str] = set()
    for e in entries:
        if not isinstance(e.id, str) or not e.id.strip():
            raise ValueError("manifest entry id must be a non-empty string")
        if not isinstance(e.instruction, str) or not e.instruction.strip():
            raise ValueError(f"instruction of entry {e.id!r} must be a non-empty string")
        if not isinstance(e.split, str) or not e.split.strip():
            raise ValueError(f"split of entry {e.id!r} must be a non-empty string")
        for field_name in ("pair_id", "group_id"):
            value = getattr(e, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"{field_name} of entry {e.id!r} must be a non-empty string or null")
        if not isinstance(e.source, str):
            raise ValueError(f"source of entry {e.id!r} must be a string")
        if e.success is not None and not isinstance(e.success, bool):
            raise ValueError(f"success of entry {e.id!r} must be a bool or null")
        if not isinstance(e.video, (str, dict)):
            raise ValueError(f"video of entry {e.id!r} must be a path string or a view dict")
        if isinstance(e.video, str) and not e.video.strip():
            raise ValueError(f"video path of entry {e.id!r} must not be empty")
        if isinstance(e.video, dict) and any(
                not isinstance(v, str) or not v.strip() for v in e.video.values()):
            raise ValueError(f"multi-view video paths of entry {e.id!r} must be non-empty strings")
        if e.progress_gt is None and e.progress_gt_frame_indices is not None:
            raise ValueError(
                f"entry {e.id!r} gives progress_gt_frame_indices without progress_gt")
        if e.progress_gt is not None:
            if e.progress_gt_frame_indices is None:
                raise ValueError(
                    f"entry {e.id!r} gives progress_gt, so progress_gt_frame_indices is "
                    "required too; the frames a label belongs to are never guessed")
            if not e.progress_gt:
                raise ValueError(f"progress_gt of entry {e.id!r} must not be empty")
            if len(e.progress_gt) != len(e.progress_gt_frame_indices):
                raise ValueError(
                    f"progress_gt and progress_gt_frame_indices of entry {e.id!r} differ "
                    "in length")
            if any(not isinstance(x, (int, float)) or isinstance(x, bool)
                   or not math.isfinite(x) for x in e.progress_gt):
                raise ValueError(f"progress_gt of entry {e.id!r} must not contain NaN/Inf")
            idx = e.progress_gt_frame_indices
            if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in idx):
                raise ValueError(
                    f"progress_gt_frame_indices of entry {e.id!r} must be non-negative integers")
            if any(b <= a for a, b in zip(idx, idx[1:])):
                raise ValueError(
                    f"progress_gt_frame_indices of entry {e.id!r} must be strictly increasing")
        if e.id in seen_ids:
            raise ValueError(f"manifest repeats id={e.id!r}")
        seen_ids.add(e.id)
        video = e.resolved_video(base_dir)
        if isinstance(video, dict) and (
                not video or any(not isinstance(k, str) or not k.strip()
                                 for k in video)):
            raise ValueError(f"multi-view video of entry {e.id!r} needs non-empty view names")
        paths = list(video.values()) if isinstance(video, dict) else [video]
        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"video file of entry {e.id!r} does not exist: {p}")


def _parse_bool(s: str | None) -> bool | None:
    s = (s or "").strip().lower()
    if s == "":
        return None
    if s in ("1", "true", "yes"):
        return True
    if s in ("0", "false", "no"):
        return False
    raise ValueError(
        f"cannot parse as a boolean: {s!r} (accepts true/false/1/0/yes/no, or empty)")


def _parse_progress_gt_csv(s: str | None) -> tuple[float, ...] | None:
    s = (s or "").strip()
    if not s:
        return None
    return tuple(float(x) for x in s.split(";"))


def _parse_gt_indices_csv(s: str | None) -> tuple[int, ...] | None:
    s = (s or "").strip()
    if not s:
        return None
    return tuple(int(x) for x in s.split(";"))


def _load_csv(path: Path) -> list[ManifestEntry]:
    """CSV is flat and supports a single view; progress_gt is a ';'-separated list."""
    entries = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "video", "instruction"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest CSV is missing required columns: {sorted(missing)}")
        for i, row in enumerate(reader):
            try:
                entries.append(ManifestEntry(
                    id=row["id"], video=row["video"], instruction=row["instruction"],
                    split=row.get("split") or "test",
                    success=_parse_bool(row.get("success")),
                    progress_gt=_parse_progress_gt_csv(row.get("progress_gt")),
                    progress_gt_frame_indices=_parse_gt_indices_csv(
                        row.get("progress_gt_frame_indices")),
                    pair_id=row.get("pair_id") or None,
                    group_id=row.get("group_id") or None,
                    source=row.get("source") or "",
                ))
            except ValueError as e:
                raise ValueError(f"{path}: line {i + 2}: {e}") from e
    return entries


def _load_jsonl(path: Path) -> list[ManifestEntry]:
    entries = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: line {i + 1} is not valid JSON: {e}") from e
        if not isinstance(d, dict):
            raise ValueError(f"{path}: line {i + 1} must be a JSON object")
        missing = {"id", "video", "instruction"} - set(d)
        if missing:
            raise ValueError(f"{path}: line {i + 1} is missing fields: {sorted(missing)}")
        entries.append(ManifestEntry(
            id=d["id"], video=d["video"], instruction=d["instruction"],
            split=d.get("split", "test"),
            success=d.get("success"),
            progress_gt=(tuple(d["progress_gt"]) if d.get("progress_gt") is not None
                         else None),
            progress_gt_frame_indices=(
                tuple(d["progress_gt_frame_indices"])
                if d.get("progress_gt_frame_indices") is not None else None),
            pair_id=d.get("pair_id"), group_id=d.get("group_id"),
            source=d.get("source", ""),
        ))
    return entries
