"""Local dataset manifest tests.  See rewardbox/data/manifest.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rewardbox.core.types import SamplingSpec
from rewardbox.data import Manifest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "local_example"


class TestLoadJSONL:
    def test_loads_example_fixture(self):
        m = Manifest.load(EXAMPLE_DIR / "manifest.jsonl")
        assert len(m) == 2
        ids = {e.id for e in m}
        assert ids == {"pick-001", "pick-002"}

    def test_fields_roundtrip(self):
        m = Manifest.load(EXAMPLE_DIR / "manifest.jsonl")
        e = m.by_id("pick-001")
        assert e.instruction == "move the red block onto the green target"
        assert e.split == "test"
        assert e.success is True
        assert e.progress_gt == (0.0, 0.25, 0.5, 0.75, 1.0)
        assert e.progress_gt_frame_indices == (0, 10, 20, 30, 39)
        assert e.pair_id == "pick-pair-1"
        assert e.group_id == "pick-task"
        assert e.source

    def test_paths_resolved_relative_to_manifest_dir_not_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)   # change cwd; the manifest paths must still work
        m = Manifest.load(EXAMPLE_DIR / "manifest.jsonl")
        e = m.by_id("pick-001")
        resolved = e.resolved_video(m.base_dir)
        assert Path(resolved).exists()
        assert Path(resolved).is_absolute()


class TestLoadCSV:
    def test_loads_example_fixture(self):
        m = Manifest.load(EXAMPLE_DIR / "manifest.csv")
        assert len(m) == 2

    def test_fields_parsed(self):
        m = Manifest.load(EXAMPLE_DIR / "manifest.csv")
        e = m.by_id("pick-002")
        assert e.success is False
        assert e.progress_gt == (0.0, 0.0, 0.0, 0.0, 0.0)
        assert e.progress_gt_frame_indices == (0, 10, 20, 30, 39)


class TestValidation:
    def test_duplicate_id_rejected(self, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"x")
        p = tmp_path / "manifest.jsonl"
        p.write_text(
            '{"id": "x", "video": "a.mp4", "instruction": "i"}\n'
            '{"id": "x", "video": "a.mp4", "instruction": "i2"}\n'
        )
        with pytest.raises(ValueError, match="repeats id"):
            Manifest.load(p)

    def test_missing_video_file_rejected(self, tmp_path):
        p = tmp_path / "manifest.jsonl"
        p.write_text('{"id": "x", "video": "does_not_exist.mp4", "instruction": "i"}\n')
        with pytest.raises(FileNotFoundError):
            Manifest.load(p)

    def test_missing_required_field_rejected_jsonl(self, tmp_path):
        p = tmp_path / "manifest.jsonl"
        p.write_text('{"id": "x", "video": "a.mp4"}\n')   # instruction missing
        with pytest.raises(ValueError, match="instruction"):
            Manifest.load(p)

    def test_missing_required_column_rejected_csv(self, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"x")
        p = tmp_path / "manifest.csv"
        p.write_text("id,video\nx,a.mp4\n")   # instruction column missing
        with pytest.raises(ValueError, match="instruction"):
            Manifest.load(p)

    def test_empty_manifest_rejected(self, tmp_path):
        p = tmp_path / "manifest.jsonl"
        p.write_text("")
        with pytest.raises(ValueError, match="manifest is empty"):
            Manifest.load(p)

    def test_unsupported_extension_rejected(self, tmp_path):
        p = tmp_path / "manifest.yaml"
        p.write_text("x: 1")
        with pytest.raises(ValueError, match="unsupported manifest format"):
            Manifest.load(p)

    def test_missing_manifest_file_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Manifest.load(tmp_path / "nope.jsonl")

    def test_invalid_bool_rejected_csv(self, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"x")
        p = tmp_path / "manifest.csv"
        p.write_text("id,video,instruction,success\nx,a.mp4,i,maybe\n")
        with pytest.raises(ValueError, match="cannot parse as a boolean"):
            Manifest.load(p)

    def test_progress_gt_requires_explicit_frame_grid(self, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"x")
        p = tmp_path / "manifest.jsonl"
        p.write_text(
            '{"id":"x","video":"a.mp4","instruction":"i","progress_gt":[0,1]}\n')
        with pytest.raises(ValueError, match="progress_gt_frame_indices"):
            Manifest.load(p)

    def test_invalid_jsonl_field_types_rejected(self, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"x")
        p = tmp_path / "manifest.jsonl"
        p.write_text(
            '{"id":"x","video":"a.mp4","instruction":"i","success":"yes"}\n')
        with pytest.raises(ValueError, match="success"):
            Manifest.load(p)

    def test_non_numeric_progress_gt_rejected_cleanly(self, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"x")
        p = tmp_path / "manifest.jsonl"
        p.write_text(
            '{"id":"x","video":"a.mp4","instruction":"i",'
            '"progress_gt":[0,"bad"],"progress_gt_frame_indices":[0,1]}\n')
        with pytest.raises(ValueError, match="progress_gt"):
            Manifest.load(p)


class TestMultiView:
    def test_jsonl_supports_multiview_dict(self, tmp_path):
        (tmp_path / "high.mp4").write_bytes(b"x")
        (tmp_path / "wrist.mp4").write_bytes(b"y")
        p = tmp_path / "manifest.jsonl"
        p.write_text(
            '{"id": "x", "video": {"primary": "high.mp4", "left_wrist": "wrist.mp4"}, '
            '"instruction": "i"}\n'
        )
        m = Manifest.load(p)
        e = m.by_id("x")
        resolved = e.resolved_video(m.base_dir)
        assert set(resolved) == {"primary", "left_wrist"}
        assert all(Path(v).exists() for v in resolved.values())


class TestBySplit:
    def test_filters_by_split(self, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"x")
        p = tmp_path / "manifest.jsonl"
        p.write_text(
            '{"id": "a", "video": "a.mp4", "instruction": "i", "split": "train"}\n'
            '{"id": "b", "video": "a.mp4", "instruction": "i", "split": "test"}\n'
        )
        m = Manifest.load(p)
        assert [e.id for e in m.by_split("train")] == ["a"]
        assert [e.id for e in m.by_split("test")] == ["b"]


class TestToTrajectory:
    def test_builds_valid_trajectory_with_ground_truth(self):
        m = Manifest.load(EXAMPLE_DIR / "manifest.jsonl")
        e = m.by_id("pick-001")
        traj = m.to_trajectory(e, SamplingSpec(num_frames=8))
        assert traj.instruction == e.instruction
        assert traj.success is True
        assert len(traj.progress_gt) == len(traj)
        expected = np.interp(
            traj.frame_indices, e.progress_gt_frame_indices, e.progress_gt)
        assert np.allclose(traj.progress_gt, expected)
        assert traj.meta["manifest_id"] == e.id
        assert traj.meta["pair_id"] == e.pair_id
        assert len(traj) <= 8

    def test_by_id_unknown_raises(self):
        m = Manifest.load(EXAMPLE_DIR / "manifest.jsonl")
        with pytest.raises(KeyError):
            m.by_id("nope")
