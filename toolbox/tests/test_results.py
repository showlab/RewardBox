"""ResultSet tests.  The core is "refuse to merge results that are not comparable"."""
from __future__ import annotations

import json

import numpy as np
import pytest

from rewardbox.core.results import (ResultConflictError, ResultSchemaError,
                                    ResultSet, SCHEMA_VERSION)
from rewardbox.core.types import RewardOutput, SamplingSpec, Trajectory


def _traj(sampling, instruction="pick", n=4, source="v.mp4"):
    frames = np.stack([np.full((8, 8, 3), min(255, i * 20), np.uint8) for i in range(n)])
    return Trajectory(frames=frames, instruction=instruction, sampling=sampling, source=source)


def _rs(sampling, instruction="pick", n=4, source="v.mp4"):
    return ResultSet.from_trajectory(_traj(sampling, instruction, n, source))


def _out(model_name, progress, frame_indices, sampling, **kw):
    return RewardOutput(progress=progress, frame_indices=frame_indices,
                        model_name=model_name, sampling=sampling, **kw)


def _valid_json(tmp_path, name="result.json"):
    """Write a valid result-v1 file; return (path, dict)."""
    s = SamplingSpec(num_frames=4)
    rs = _rs(s)
    rs.add(_out("a", [0.0, 1.0], [0, 3], s))
    p = tmp_path / name
    rs.save(p)
    return p, json.loads(p.read_text())


def _same_trajectory(rs, **kw):
    fields = dict(instruction=rs.instruction, sampling=rs.sampling, source=rs.source,
                  input_frame_indices=rs.input_frame_indices)
    fields.update(kw)
    return ResultSet(**fields)


class TestFromTrajectory:
    def test_locks_identity_from_trajectory(self):
        s = SamplingSpec(num_frames=4)
        traj = _traj(s, instruction="pick the cube")
        rs = ResultSet.from_trajectory(traj)
        assert rs.instruction == "pick the cube"
        assert rs.sampling.key() == s.key()
        assert rs.source == "v.mp4"
        assert rs.input_frame_indices.tolist() == traj.frame_indices.tolist()
        assert not hasattr(rs, "trajectory_hash")

    def test_rejects_trajectory_without_sampling(self):
        frames = np.zeros((4, 8, 8, 3), np.uint8)
        traj = Trajectory(frames=frames, instruction="x")   # sampling=None
        with pytest.raises(ResultSchemaError, match="has no sampling"):
            ResultSet.from_trajectory(traj)


class TestAdd:
    def test_basic_add(self):
        s = SamplingSpec(num_frames=4)
        rs = _rs(s)
        rs.add(_out("a", [0.0, 0.5, 1.0], [0, 1, 2], s))
        assert "a" in rs.results

    def test_rejects_different_sampling(self):
        """Core regression: reward-scope's stackcubes1.json declares num_frames=10
        while roboreward holds 20 values, because merging never checked sampling."""
        s10, s20 = SamplingSpec(num_frames=10), SamplingSpec(num_frames=20)
        rs = _rs(s10, n=10)
        rs.add(_out("a", [0.0, 1.0], [0, 9], s10))
        with pytest.raises(ResultConflictError, match="sampling differs"):
            rs.add(_out("b", np.zeros(20), np.arange(20), s20))

    def test_rejects_duplicate_without_overwrite(self):
        s = SamplingSpec(num_frames=4)
        rs = _rs(s)
        rs.add(_out("a", [0.0, 1.0], [0, 3], s))
        with pytest.raises(ResultConflictError, match="overwrite"):
            rs.add(_out("a", [0.2, 0.8], [0, 3], s))
        rs.add(_out("a", [0.2, 0.8], [0, 3], s), overwrite=True)
        assert rs.results["a"].progress.tolist() == [0.2, 0.8]

    def test_error_to_success_requires_overwrite(self):
        s = SamplingSpec(num_frames=4)
        rs = _rs(s)
        rs.add_error("a", "failed")
        out = _out("a", [0.0, 1.0], [0, 3], s)
        with pytest.raises(ResultConflictError, match="overwrite"):
            rs.add(out)
        rs.add(out, overwrite=True)
        assert "a" in rs.results and "a" not in rs.errors

    def test_rejects_missing_sampling(self):
        s = SamplingSpec(num_frames=4)
        rs = _rs(s)
        bad = RewardOutput(progress=[0.0], frame_indices=[0], model_name="a")
        with pytest.raises(ResultConflictError, match="has no sampling"):
            rs.add(bad)

    def test_rejects_add_without_input_grid(self):
        """A bare ResultSet without input_frame_indices refuses add()."""
        s = SamplingSpec(num_frames=4)
        rs = ResultSet(instruction="pick", sampling=s, source="v.mp4")
        with pytest.raises(ResultSchemaError, match="input_frame_indices"):
            rs.add(_out("a", [0.0], [0], s))

    def test_rejects_output_frame_outside_input_grid(self):
        s = SamplingSpec(num_frames=4)
        rs = _rs(s)
        with pytest.raises(ResultConflictError, match="not in the shared input grid"):
            rs.add(_out("a", [0.0, 1.0], [0, 999], s))

    @pytest.mark.parametrize("indices", ([], [-1, 1], [0, 0], [1, 0]))
    def test_rejects_invalid_input_grid(self, indices):
        s = SamplingSpec(num_frames=4)
        with pytest.raises(ResultSchemaError, match="input_frame_indices"):
            ResultSet(instruction="pick", sampling=s, input_frame_indices=indices)


class TestAlignment:
    def test_common_grid_is_union(self):
        s = SamplingSpec(num_frames=8)
        rs = _rs(s, n=8)
        rs.add(_out("a", [0.0, 1.0], [0, 6], s))
        rs.add(_out("b", [0.0, 0.5, 1.0], [0, 3, 6], s))
        assert rs.common_grid().tolist() == [0, 3, 6]

    def test_aligned_nan_outside_native_range(self):
        """No extrapolation: points a model never reported are NaN, not invented
        constant fills."""
        s = SamplingSpec(num_frames=8)
        rs = _rs(s, n=8)
        rs.add(_out("a", [0.0, 1.0], [0, 6], s))     # covers [0,6]
        rs.add(_out("b", [0.0, 1.0], [3, 6], s))     # covers only [3,6]
        aligned = rs.aligned()
        grid = rs.common_grid()
        assert grid.tolist() == [0, 3, 6]
        assert not np.isnan(aligned["a"]).any()
        assert np.isnan(aligned["b"][grid.tolist().index(0)])

    def test_empty_result_set(self):
        s = SamplingSpec(num_frames=4)
        rs = _rs(s)
        assert rs.common_grid().tolist() == []
        assert rs.aligned() == {}


class TestSchemaAndPersistence:
    def test_to_dict_has_required_keys(self):
        s = SamplingSpec(num_frames=4)
        rs = _rs(s)
        rs.add(_out("a", [0.0, 1.0], [0, 3], s))
        d = rs.to_dict()
        assert d["schema_version"] == SCHEMA_VERSION
        for k in ("schema_version", "video", "instruction", "sampling",
                  "created_at", "rewardbox_version", "environment",
                  "results", "errors"):
            assert k in d, f"missing schema field {k}"
        assert "trajectory_hash" not in d
        assert set(d["sampling"]) == {"spec", "key", "input_frame_indices"}
        assert d["sampling"]["key"] == s.key()
        assert d["video"] == "v.mp4"

    def test_errors_only_result_set_has_null_grid(self, tmp_path):
        """A model that never read the video leaves the input grid unknown."""
        s = SamplingSpec(num_frames=4)
        rs = ResultSet(instruction="pick", sampling=s, source="v.mp4")
        rs.add_error("a", "model load failed")
        p = tmp_path / "result.json"
        rs.save(p)
        d = json.loads(p.read_text())
        assert d["sampling"]["input_frame_indices"] is None
        assert ResultSet.load(p).errors == {"a": "model load failed"}

    def test_null_grid_with_results_rejected(self, tmp_path):
        p, d = _valid_json(tmp_path)
        d["sampling"]["input_frame_indices"] = None
        p.write_text(json.dumps(d))
        with pytest.raises(ResultSchemaError, match="null only when results is empty"):
            ResultSet.load(p)

    def test_legacy_trajectory_hash_field_is_ignored(self, tmp_path):
        p, d = _valid_json(tmp_path)
        d["trajectory_hash"] = "dd06c5055382a3ce6a4c8d15916b1c6d"
        p.write_text(json.dumps(d))
        assert set(ResultSet.load(p).results) == {"a"}

    def test_save_and_load_roundtrip(self, tmp_path):
        s = SamplingSpec(num_frames=4)
        rs = _rs(s)
        rs.add(_out("a", [0.0, 1.0], [0, 3], s))
        p = tmp_path / "result.json"
        rs.save(p)
        loaded = ResultSet.load(p)
        assert loaded.instruction == "pick"
        assert loaded.sampling.key() == s.key()
        assert np.allclose(loaded.results["a"].progress, [0.0, 1.0])
        assert loaded.input_frame_indices.tolist() == rs.input_frame_indices.tolist()

    def test_load_backfills_sampling_onto_outputs(self, tmp_path):
        """Review finding: out.sampling used to be None after load, breaking the
        object contract."""
        s = SamplingSpec(num_frames=4)
        rs = _rs(s)
        rs.add(_out("a", [0.0, 1.0], [0, 3], s))
        p = tmp_path / "result.json"
        rs.save(p)
        loaded = ResultSet.load(p)
        assert loaded.results["a"].sampling == loaded.sampling
        # A loaded output must pass a normal add() again (not only restore numbers).
        rs2 = _same_trajectory(loaded)
        rs2.add(loaded.results["a"])

    def test_save_merges_compatible_results(self, tmp_path):
        s = SamplingSpec(num_frames=4)
        rs1 = _rs(s)
        rs1.add(_out("a", [0.0, 1.0], [0, 3], s))
        p = tmp_path / "result.json"
        rs1.save(p)

        rs2 = _same_trajectory(rs1)
        rs2.add(_out("b", [0.1, 0.9], [0, 3], s))
        rs2.save(p, merge=True)

        merged = json.loads(p.read_text())
        assert set(merged["results"]) == {"a", "b"}

    def test_save_rejects_incompatible_existing_file(self, tmp_path):
        """Required: merging results of two SamplingSpecs raises."""
        s10, s20 = SamplingSpec(num_frames=10), SamplingSpec(num_frames=20)
        rs1 = _rs(s10, n=10)
        rs1.add(_out("a", np.zeros(10), np.arange(10), s10))
        p = tmp_path / "result.json"
        rs1.save(p)

        rs2 = _rs(s20, n=20)
        rs2.add(_out("b", np.zeros(20), np.arange(20), s20))
        with pytest.raises(ResultConflictError, match="refusing to merge"):
            rs2.save(p, merge=True)

    def test_save_rejects_different_instruction(self, tmp_path):
        s = SamplingSpec(num_frames=4)
        rs1 = _rs(s, instruction="pick the cube")
        rs1.add(_out("a", [0.0, 1.0], [0, 3], s))
        p = tmp_path / "result.json"
        rs1.save(p)

        rs2 = _rs(s, instruction="stack the cubes")
        rs2.add(_out("b", [0.0, 1.0], [0, 3], s))
        with pytest.raises(ResultConflictError, match="refusing to merge"):
            rs2.save(p, merge=True)

    def test_save_rejects_different_source(self, tmp_path):
        s = SamplingSpec(num_frames=4)
        rs1 = _rs(s, source="a.mp4")
        rs1.add(_out("a", [0.0, 1.0], [0, 3], s))
        p = tmp_path / "result.json"
        rs1.save(p)

        rs2 = _same_trajectory(rs1, source="b.mp4")
        rs2.add(_out("b", [0.0, 1.0], [0, 3], s))
        with pytest.raises(ResultConflictError, match="refusing to merge"):
            rs2.save(p, merge=True)

    def test_save_rejects_different_input_frame_indices(self, tmp_path):
        s = SamplingSpec(num_frames=4)
        rs1 = _rs(s)
        rs1.add(_out("a", [0.0, 1.0], [0, 3], s))
        p = tmp_path / "result.json"
        rs1.save(p)

        rs2 = _same_trajectory(rs1, input_frame_indices=np.array([0, 1, 2, 3, 4]))
        rs2.add(_out("b", [0.0, 1.0], [0, 4], s))
        with pytest.raises(ResultConflictError, match="refusing to merge"):
            rs2.save(p, merge=True)

    def test_save_rejects_duplicate_model_without_overwrite_models(self, tmp_path):
        s = SamplingSpec(num_frames=4)
        rs1 = _rs(s)
        rs1.add(_out("a", [0.0, 1.0], [0, 3], s))
        p = tmp_path / "result.json"
        rs1.save(p)

        rs2 = _same_trajectory(rs1)
        rs2.add(_out("a", [0.9, 0.1], [0, 3], s))
        with pytest.raises(ResultConflictError, match="overwrite_models"):
            rs2.save(p, merge=True)

    def test_save_overwrite_models_replaces_same_name(self, tmp_path):
        s = SamplingSpec(num_frames=4)
        rs1 = _rs(s)
        rs1.add(_out("a", [0.0, 1.0], [0, 3], s))
        p = tmp_path / "result.json"
        rs1.save(p)

        rs2 = _same_trajectory(rs1)
        rs2.add(_out("a", [0.9, 0.1], [0, 3], s))
        rs2.save(p, merge=True, overwrite_models=True)

        merged = json.loads(p.read_text())
        assert merged["results"]["a"]["progress"] == [0.9, 0.1]

    def test_save_no_merge_ignores_existing(self, tmp_path):
        s10, s20 = SamplingSpec(num_frames=10), SamplingSpec(num_frames=20)
        rs1 = _rs(s10, n=10)
        rs1.add(_out("a", np.zeros(10), np.arange(10), s10))
        p = tmp_path / "result.json"
        rs1.save(p)

        rs2 = _rs(s20, n=20)
        rs2.add(_out("b", np.zeros(20), np.arange(20), s20))
        rs2.save(p, merge=False)   # no merge: overwrite as a whole, no error

        reloaded = json.loads(p.read_text())
        assert set(reloaded["results"]) == {"b"}


class TestAtomicSave:
    def test_save_produces_no_leftover_tmp_files(self, tmp_path):
        s = SamplingSpec(num_frames=4)
        rs = _rs(s)
        rs.add(_out("a", [0.0, 1.0], [0, 3], s))
        p = tmp_path / "result.json"
        rs.save(p)
        assert list(tmp_path.glob(".*.tmp")) == []
        assert p.exists()

    def test_old_file_intact_after_write_failure(self, tmp_path, monkeypatch):
        s = SamplingSpec(num_frames=4)
        rs = _rs(s)
        rs.add(_out("a", [0.0, 1.0], [0, 3], s))
        p = tmp_path / "result.json"
        rs.save(p)
        original = p.read_text()

        import rewardbox.core.results as results_mod

        def boom(*a, **kw):
            raise OSError("simulated disk failure")

        monkeypatch.setattr(results_mod.json, "dump", boom)

        rs2 = _rs(s)
        rs2.add(_out("b", [0.2, 0.8], [0, 3], s))
        with pytest.raises(OSError):
            rs2.save(p, merge=False)

        assert p.read_text() == original, "the old file must survive a failed write"
        assert list(tmp_path.glob(".*.tmp")) == [], "temp files must be cleaned up"


class TestLoadValidation:
    def test_missing_schema_version_rejected(self, tmp_path):
        p, d = _valid_json(tmp_path)
        del d["schema_version"]
        p.write_text(json.dumps(d))
        with pytest.raises(ResultSchemaError, match="schema_version"):
            ResultSet.load(p)

    def test_schema_version_zero_rejected(self, tmp_path):
        p, d = _valid_json(tmp_path)
        d["schema_version"] = 0
        p.write_text(json.dumps(d))
        with pytest.raises(ResultSchemaError, match="unsupported schema_version"):
            ResultSet.load(p)

    def test_schema_version_future_rejected(self, tmp_path):
        p, d = _valid_json(tmp_path)
        d["schema_version"] = 2
        p.write_text(json.dumps(d))
        with pytest.raises(ResultSchemaError, match="unsupported schema_version"):
            ResultSet.load(p)

    def test_missing_input_frame_indices_rejected(self, tmp_path):
        p, d = _valid_json(tmp_path)
        del d["sampling"]["input_frame_indices"]
        p.write_text(json.dumps(d))
        with pytest.raises(ResultSchemaError, match="input_frame_indices"):
            ResultSet.load(p)

    def test_empty_input_frame_indices_rejected(self, tmp_path):
        p, d = _valid_json(tmp_path)
        d["sampling"]["input_frame_indices"] = []
        p.write_text(json.dumps(d))
        with pytest.raises(ResultSchemaError, match="input_frame_indices"):
            ResultSet.load(p)

    def test_result_key_model_mismatch_rejected(self, tmp_path):
        p, d = _valid_json(tmp_path)
        d["results"]["a"]["model"] = "b"
        p.write_text(json.dumps(d))
        with pytest.raises(ResultSchemaError, match="must match"):
            ResultSet.load(p)

    def test_reversed_frame_indices_rejected(self, tmp_path):
        p, d = _valid_json(tmp_path)
        d["results"]["a"]["frame_indices"] = [3, 0]
        d["results"]["a"]["progress"] = [0.0, 1.0]
        p.write_text(json.dumps(d))
        with pytest.raises(ValueError, match="strictly increasing"):
            ResultSet.load(p)

    def test_missing_required_result_key_rejected(self, tmp_path):
        p, d = _valid_json(tmp_path)
        del d["results"]["a"]["raw"]
        p.write_text(json.dumps(d))
        with pytest.raises(ResultSchemaError, match="raw"):
            ResultSet.load(p)

    @pytest.mark.parametrize("field,value", (
        ("progress", ["not-a-number", 1.0]),
        ("frame_indices", [0, "3"]),
        ("weights", [0.0]),
    ))
    def test_schema_invalid_result_arrays_rejected(self, tmp_path, field, value):
        p, d = _valid_json(tmp_path)
        d["results"]["a"][field] = value
        p.write_text(json.dumps(d))
        with pytest.raises(ResultSchemaError):
            ResultSet.load(p)

    def test_sampling_key_spec_mismatch_rejected(self, tmp_path):
        p, d = _valid_json(tmp_path)
        d["sampling"]["key"] = "uniform:n=99:fps=-:interval=-:resize=-:last=1"
        p.write_text(json.dumps(d))
        with pytest.raises(ResultSchemaError, match="does not match"):
            ResultSet.load(p)

    def test_output_frame_outside_input_grid_rejected(self, tmp_path):
        p, d = _valid_json(tmp_path)
        d["results"]["a"]["frame_indices"] = [0, 999]
        p.write_text(json.dumps(d))
        with pytest.raises(ResultSchemaError, match="outside the shared input grid"):
            ResultSet.load(p)

    def test_result_error_overlap_rejected(self, tmp_path):
        p, d = _valid_json(tmp_path)
        d["errors"]["a"] = "also failed"
        p.write_text(json.dumps(d))
        with pytest.raises(ResultSchemaError, match="both results and errors"):
            ResultSet.load(p)


class TestErrors:
    def test_add_error_recorded_and_persisted(self, tmp_path):
        s = SamplingSpec(num_frames=4)
        rs = _rs(s)
        rs.add(_out("a", [0.0, 1.0], [0, 3], s))
        rs.add_error("b", "CUDA out of memory")
        p = tmp_path / "result.json"
        rs.save(p)
        d = json.loads(p.read_text())
        assert d["errors"] == {"b": "CUDA out of memory"}

    def test_success_and_error_are_mutually_exclusive(self):
        s = SamplingSpec(num_frames=4)
        rs = _rs(s)
        rs.add(_out("a", [0.0, 1.0], [0, 3], s))
        with pytest.raises(ResultConflictError, match="cannot also record an error"):
            rs.add_error("a", "late failure")
        rs.add_error("a", "late failure", overwrite=True)
        assert "a" not in rs.results
        assert rs.errors == {"a": "late failure"}


class TestJSONSchemaCrossCheck:
    """Optional: when jsonschema happens to be installed, cross-check that
    _validate_payload() and result-v1.schema.json agree.  Not a hard dependency."""

    def test_to_dict_validates_against_public_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        from rewardbox.schemas import load_result_v1_schema

        s = SamplingSpec(num_frames=4)
        rs = _rs(s)
        rs.add(_out("a", [0.0, 1.0], [0, 3], s))
        jsonschema.validate(rs.to_dict(), load_result_v1_schema())
