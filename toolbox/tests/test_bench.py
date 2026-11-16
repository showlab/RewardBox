"""Benchmark tests.

``isolate=False`` (in-process, the default path) is tested end to end with a fake
adapter.  The orchestration of ``isolate=True`` subprocess isolation (command
construction, non-zero exit handling) is tested with mocks; proving that real
models in separate subprocesses really do not contaminate each other needs real
weights/GPUs and is a manual check, not a resident CPU unit test (the same reason
the GPU tests in test_calibration.py carry their own marker).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from rewardbox.bench.shard import INPUT_POLICY, ModelShard
from rewardbox.core.types import RewardOutput, SamplingSpec
from rewardbox.data import Manifest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "local_example"
EXAMPLE_MANIFEST = EXAMPLE_DIR / "manifest.jsonl"
ENTRY = {"video": "a.mp4", "instruction": "pick"}


@pytest.fixture
def fake_model(monkeypatch):
    import rewardbox.registry as registry
    from tests.fixtures.fake_adapter import FAKE_SPEC, FakeAdapter

    FakeAdapter.last_instance = None
    manifest = dict(registry.load_manifest())
    manifest[FAKE_SPEC.name] = FAKE_SPEC
    monkeypatch.setattr(registry, "_CACHE", manifest)
    yield FAKE_SPEC.name


@pytest.fixture
def count_decodes(monkeypatch):
    """Count Manifest.to_trajectory calls (each one decodes the entry's video)."""
    calls: list[str] = []
    real = Manifest.to_trajectory

    def counting(self, entry, sampling):
        calls.append(entry.id)
        return real(self, entry, sampling)

    monkeypatch.setattr(Manifest, "to_trajectory", counting)
    return calls


@pytest.fixture
def no_hashing(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("content hashing is not allowed on the benchmark path")

    for name in ("sha1", "sha256", "md5", "blake2b"):
        monkeypatch.setattr(hashlib, name, forbidden)


def _copy_manifest(dest, edit=None):
    """Copy the example manifest into a new directory (videos linked) so a test can
    edit entries."""
    dest.mkdir()
    for video in ("pick_success.mp4", "pick_fail.mp4"):
        (dest / video).symlink_to(EXAMPLE_DIR / video)
    rows = [json.loads(line) for line in EXAMPLE_MANIFEST.read_text().splitlines() if line]
    if edit:
        edit(rows)
    path = dest / "manifest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# ---------------------------------------------------------------- ModelShard
class TestModelShard:
    def test_add_result_and_error(self):
        shard = ModelShard(model="m")
        out = RewardOutput(progress=[0.0, 1.0], frame_indices=[0, 1], model_name="m")
        shard.add_result("t1", out, ENTRY, [0, 1])
        shard.add_error("t2", "boom", {"video": "b.mp4", "instruction": "pick"})
        assert shard.has("t1") and shard.has("t2")
        assert not shard.has("t3")
        assert shard.has_current("t1", ENTRY)
        assert not shard.has_current("t1", {**ENTRY, "instruction": "place"})
        assert "t1" not in shard.results and "t1" not in shard.inputs

    def test_result_requires_input_grid(self):
        shard = ModelShard(model="m")
        out = RewardOutput(progress=[0.0], frame_indices=[0], model_name="m")
        with pytest.raises(ValueError, match="input_frame_indices"):
            shard.add_result("t1", out, ENTRY, None)

    def test_retry_errors_discards_only_errors(self):
        shard = ModelShard(model="m")
        shard.add_result("t1", RewardOutput(progress=[0.0], frame_indices=[0], model_name="m"),
                         ENTRY, [0])
        shard.add_error("t2", "boom", ENTRY)
        assert shard.has_current("t1", ENTRY, retry_errors=True)
        assert not shard.has_current("t2", ENTRY, retry_errors=True)

    def test_save_load_roundtrip(self, tmp_path):
        shard = ModelShard(model="m", identity={"sampling": "fixed"})
        shard.add_result(
            "t1", RewardOutput(progress=[0.5], frame_indices=[0], model_name="m"),
            ENTRY, [0, 3])
        p = tmp_path / "m.json"
        shard.save(p)
        loaded = ModelShard.load(p)
        assert loaded.model == "m"
        assert loaded.results["t1"]["progress"] == [0.5]
        assert loaded.identity == {"sampling": "fixed"}
        assert loaded.inputs == {"t1": {"entry": ENTRY, "input_frame_indices": [0, 3]}}
        assert "trajectory_hashes" not in json.loads(p.read_text())

    def test_load_or_new_creates_empty_when_missing(self, tmp_path):
        shard = ModelShard.load_or_new(tmp_path / "nope.json", "m")
        assert shard.model == "m" and shard.results == {} and shard.errors == {}

    def test_load_or_new_rejects_model_mismatch(self, tmp_path):
        p = tmp_path / "m.json"
        ModelShard(model="m").save(p)
        with pytest.raises(ValueError, match="not 'other-model'"):
            ModelShard.load_or_new(p, "other-model")

    def test_load_or_new_rejects_changed_identity(self, tmp_path):
        from rewardbox.bench.shard import BenchmarkIdentityError

        p = tmp_path / "m.json"
        ModelShard(model="m", identity={"sampling": "a"}).save(p)
        with pytest.raises(BenchmarkIdentityError, match="--no-resume"):
            ModelShard.load_or_new(p, "m", {"sampling": "b"})

    @pytest.mark.parametrize("version", (2, 99))
    def test_load_rejects_other_schema_versions(self, tmp_path, version):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"schema_version": version, "model": "m",
                                 "results": {}, "errors": {}}))
        with pytest.raises(ValueError, match="schema_version"):
            ModelShard.load(p)


# ---------------------------------------------------------------- experiment identity
class TestExperimentIdentity:
    def test_identity_is_protocol_and_version_only(self, no_hashing):
        from rewardbox import __version__
        from rewardbox.bench.shard import build_experiment_identity
        from rewardbox.upstream import get_upstream

        identity = build_experiment_identity("robometer-4b", SamplingSpec(num_frames=8), {})
        assert identity["input_policy"] == INPUT_POLICY
        assert identity["rewardbox_version"] == __version__
        assert identity["weights"] == "aliangdw/Robometer-4B"
        assert identity["upstream_commits"] == {"robometer": get_upstream("robometer").commit}
        assert not any("sha" in key or "digest" in key for key in identity)

    def test_gvl_runtime_dependencies_are_recorded(self):
        from rewardbox.bench.shard import _package_versions

        assert {
            "sentence-transformers", "huggingface-hub", "einops", "requests",
            "opencv-python-headless", "agentlace", "pyzmq",
        } <= set(_package_versions())

    def test_identity_redacts_nested_credentials_without_hiding_normal_options(self):
        from rewardbox.bench.shard import build_experiment_identity

        secret = "must-not-reach-a-shard"
        opts = {
            "api_key": secret,
            "nested": {"Authorization": f"Bearer {secret}"},
            "max_tokens": 128,
            "tokenizer": "official",
        }
        identity = build_experiment_identity(
            "gvl", SamplingSpec(num_frames=8, strategy="uniform"), opts)

        serialized = json.dumps(identity)
        assert secret not in serialized
        assert identity["model_opts"]["api_key"] == "<redacted>"
        assert identity["model_opts"]["nested"]["Authorization"] == "<redacted>"
        assert identity["model_opts"]["max_tokens"] == 128
        assert identity["model_opts"]["tokenizer"] == "official"
        assert opts["api_key"] == secret  # caller-owned options were not mutated

    def test_sole_r1_identity_binds_environment_selected_endpoint(self, monkeypatch):
        from rewardbox.bench.shard import build_experiment_identity

        monkeypatch.setenv("REWARDBOX_SOLE_R1_HOST", "server-a")
        first = build_experiment_identity("sole-r1-8b", SamplingSpec(num_frames=8), {})
        monkeypatch.setenv("REWARDBOX_SOLE_R1_HOST", "server-b")
        second = build_experiment_identity("sole-r1-8b", SamplingSpec(num_frames=8), {})

        assert first["external_runtime"]["host"] == "server-a"
        assert second["external_runtime"]["host"] == "server-b"
        assert first != second

    def test_provenance_file_is_bound_by_path_not_content(self, tmp_path, no_hashing):
        from rewardbox.bench.shard import build_experiment_identity

        provenance = tmp_path / "server.json"
        provenance.write_text('{"revision":"a"}', encoding="utf-8")
        opts = {"server_provenance_file": str(provenance)}
        first = build_experiment_identity("sole-r1-8b", SamplingSpec(num_frames=8), opts)
        provenance.write_text('{"revision":"b"}', encoding="utf-8")
        second = build_experiment_identity("sole-r1-8b", SamplingSpec(num_frames=8), opts)

        assert first["external_runtime"]["server_provenance_file"] == str(provenance.resolve())
        assert first == second

    def test_lrm_identity_binds_endpoint_environment(self, monkeypatch):
        from rewardbox.bench.shard import build_experiment_identity

        monkeypatch.setenv("REWARDBOX_LRM_SERVER_URL", "http://server-a:5002")
        first = build_experiment_identity("lrm-progress", SamplingSpec(strategy="all"), {})
        monkeypatch.setenv("REWARDBOX_LRM_SERVER_URL", "http://server-b:5002")
        second = build_experiment_identity("lrm-progress", SamplingSpec(strategy="all"), {})
        assert first["external_runtime"]["server_urls"] == {
            "progress": "http://server-a:5002"
        }
        assert first != second

    def test_lrm_tri_identity_binds_each_server_and_provenance_path(self, tmp_path):
        from rewardbox.bench.shard import build_experiment_identity

        provenance_files = {}
        for head in ("contrastive", "progress", "completion"):
            path = tmp_path / f"{head}.json"
            path.write_text(f'{{"head":"{head}"}}', encoding="utf-8")
            provenance_files[head] = str(path)
        opts = {
            "contrastive_url": "http://c:5002",
            "progress_url": "http://p:5002",
            "completion_url": "http://d:5002",
            "server_provenance_files": provenance_files,
        }
        identity = build_experiment_identity("lrm-tri", SamplingSpec(strategy="all"), opts)
        runtime = identity["external_runtime"]
        assert runtime["server_urls"] == {
            "contrastive": "http://c:5002",
            "progress": "http://p:5002",
            "completion": "http://d:5002",
        }
        assert runtime["server_provenance_files"]["progress"] == str(
            Path(provenance_files["progress"]).resolve())

    @pytest.mark.parametrize("model,env,default", (
        ("rynnvalue-8b", "REWARDBOX_RYNNVALUE_SERVER_URL", "http://127.0.0.1:8001"),
        ("procvlm-2b", "REWARDBOX_PROCVLM_SERVER_URL", "http://127.0.0.1:8013"),
        ("robo-valuerl-value", "REWARDBOX_ROBOVALUERL_SERVER_URL", "http://127.0.0.1:8012"),
    ))
    def test_server_models_bind_endpoint(self, monkeypatch, model, env, default):
        from rewardbox.bench.shard import build_experiment_identity

        monkeypatch.delenv(env, raising=False)
        assert build_experiment_identity(model, SamplingSpec(strategy="all"), {})[
            "external_runtime"]["server_url"] == default
        monkeypatch.setenv(env, "http://server-a:9000")
        assert build_experiment_identity(model, SamplingSpec(strategy="all"), {})[
            "external_runtime"]["server_url"] == "http://server-a:9000"

    @pytest.mark.parametrize("model", (
        "lrm-progress-vllm", "lrm-completion-vllm", "lrm-contrastive-vllm"))
    def test_in_process_vllm_models_bind_no_server(self, monkeypatch, model):
        """The vLLM re-implementations never contact an LRM server, so a server
        variable in the environment must not change (or block) their resume identity."""
        from rewardbox.bench.shard import build_experiment_identity

        monkeypatch.delenv("REWARDBOX_LRM_SERVER_URL", raising=False)
        first = build_experiment_identity(model, SamplingSpec(strategy="all"), {})
        monkeypatch.setenv("REWARDBOX_LRM_SERVER_URL", "http://server-a:9000")
        second = build_experiment_identity(model, SamplingSpec(strategy="all"), {})
        assert "external_runtime" not in first
        assert first == second


# ---------------------------------------------------------------- in-process run
class TestRunModelOverManifest:
    def test_runs_all_entries_decoding_each_once(self, fake_model, tmp_path, count_decodes):
        from rewardbox.bench.worker import run_model_over_manifest

        manifest = Manifest.load(EXAMPLE_MANIFEST)
        shard = ModelShard(model=fake_model)
        run_model_over_manifest(fake_model, manifest, SamplingSpec(num_frames=4),
                                shard, tmp_path / "shard.json", log=lambda m: None)
        assert set(shard.results) == {"pick-001", "pick-002"}
        assert shard.errors == {}
        assert sorted(count_decodes) == ["pick-001", "pick-002"]   # no pre-scan
        assert shard.inputs["pick-001"]["entry"] == manifest.entry_identity(
            manifest.by_id("pick-001"))
        assert shard.inputs["pick-001"]["input_frame_indices"] == [0, 13, 26, 39]

    def test_resume_skips_existing_without_decoding_them(self, fake_model, tmp_path,
                                                         count_decodes):
        from rewardbox.bench.worker import run_model_over_manifest
        from tests.fixtures.fake_adapter import FakeAdapter

        manifest = Manifest.load(EXAMPLE_MANIFEST)
        shard = ModelShard(model=fake_model)
        sampling = SamplingSpec(num_frames=4)
        existing = RewardOutput(progress=[0.0], frame_indices=[0],
                                model_name=fake_model, sampling=sampling)
        shard.add_result("pick-001", existing,
                         manifest.entry_identity(manifest.by_id("pick-001")), [0, 13, 26, 39])

        seen = []
        orig_score = FakeAdapter._score

        def tracking_score(self, traj):
            seen.append(traj.meta["manifest_id"])
            return orig_score(self, traj)

        with patch.object(FakeAdapter, "_score", tracking_score):
            run_model_over_manifest(fake_model, manifest, sampling,
                                    shard, tmp_path / "shard.json", log=lambda m: None)
        assert shard.results["pick-001"]["progress"] == [0.0]   # not overwritten by a re-run
        assert "pick-002" in shard.results
        assert seen == ["pick-002"]            # only the entry that did not resume ran
        assert count_decodes == ["pick-002"]   # the resumed entry was never decoded

    def test_one_entry_failure_does_not_stop_others(self, fake_model, monkeypatch, tmp_path):
        from rewardbox.bench.worker import run_model_over_manifest
        from tests.fixtures.fake_adapter import FakeAdapter

        manifest = Manifest.load(EXAMPLE_MANIFEST)
        shard = ModelShard(model=fake_model)

        real_score = FakeAdapter._score

        def flaky_score(self, traj):
            if "green target" in traj.instruction and not getattr(self, "_failed_once", False):
                self._failed_once = True
                raise RuntimeError("simulated failure of the first trajectory")
            return real_score(self, traj)

        monkeypatch.setattr(FakeAdapter, "_score", flaky_score)
        run_model_over_manifest(fake_model, manifest, SamplingSpec(num_frames=4),
                                shard, tmp_path / "shard.json", log=lambda m: None)
        assert len(shard.errors) == 1
        assert len(shard.results) == 1
        failed = next(iter(shard.errors))
        assert shard.inputs[failed]["input_frame_indices"] == [0, 13, 26, 39]


# ---------------------------------------------------------------- run_benchmark (in-process)
class TestRunBenchmarkInProcess:
    def test_rejects_credentials_before_creating_benchmark_artifacts(self, tmp_path):
        from rewardbox.bench.runner import run_benchmark

        out = tmp_path / "must-not-exist"
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            run_benchmark(
                ["gvl"], EXAMPLE_MANIFEST,
                SamplingSpec(num_frames=8, strategy="uniform"), out,
                model_opts={"gvl": {"api_key": "do-not-put-me-in-argv"}},
            )
        assert not out.exists()

    def test_writes_model_order_and_shards(self, fake_model, tmp_path):
        from rewardbox.bench.runner import run_benchmark

        out = tmp_path / "bench_out"
        shards = run_benchmark([fake_model], EXAMPLE_MANIFEST, SamplingSpec(num_frames=4),
                               out, isolate=False, log=lambda m: None)
        assert json.loads((out / "model_order.json").read_text()) == [fake_model]
        assert (out / f"{fake_model}.json").exists()
        assert set(shards[fake_model].results) == {"pick-001", "pick-002"}

    def test_model_failure_does_not_affect_other_models(self, fake_model, tmp_path):
        """One model failing to load/run as a whole must not take down the others."""
        from rewardbox.bench.runner import run_benchmark

        out = tmp_path / "bench_out"
        shards = run_benchmark(["not-a-real-model", fake_model], EXAMPLE_MANIFEST,
                               SamplingSpec(num_frames=4), out, isolate=False,
                               log=lambda m: None)
        assert set(shards[fake_model].results) == {"pick-001", "pick-002"}
        failed = shards["not-a-real-model"]
        assert set(failed.errors) == {"pick-001", "pick-002"}
        assert failed.fatal_error
        assert failed.inputs["pick-001"]["input_frame_indices"] is None

    def test_resume_across_runs_decodes_nothing(self, fake_model, tmp_path, count_decodes):
        from rewardbox.bench.runner import run_benchmark

        out = tmp_path / "bench_out"
        run_benchmark([fake_model], EXAMPLE_MANIFEST, SamplingSpec(num_frames=4),
                      out, isolate=False, log=lambda m: None)
        assert len(count_decodes) == 2
        count_decodes.clear()

        # Second run (resume=True by default): both finished entries are skipped.
        run_benchmark([fake_model], EXAMPLE_MANIFEST, SamplingSpec(num_frames=4),
                      out, isolate=False, resume=True, log=lambda m: None)
        loaded = ModelShard.load(out / f"{fake_model}.json")
        assert set(loaded.results) == {"pick-001", "pick-002"}
        assert count_decodes == []

    def test_editing_one_entry_reruns_only_that_entry(self, fake_model, tmp_path,
                                                      count_decodes):
        from rewardbox.bench.runner import run_benchmark

        out = tmp_path / "bench_out"
        first = _copy_manifest(tmp_path / "v1")
        run_benchmark([fake_model], first, SamplingSpec(num_frames=4), out,
                      isolate=False, log=lambda m: None)
        count_decodes.clear()

        def edit(rows):
            rows[1]["instruction"] = "push the red block off the table"

        second = _copy_manifest(tmp_path / "v2", edit)
        shards = run_benchmark([fake_model], second, SamplingSpec(num_frames=4), out,
                               isolate=False, log=lambda m: None)
        assert count_decodes == ["pick-002"]
        assert shards[fake_model].inputs["pick-002"]["entry"]["instruction"] == (
            "push the red block off the table")

    def test_benchmark_and_aggregate_compute_no_hash(self, fake_model, tmp_path, no_hashing):
        from rewardbox.bench.runner import aggregate, run_benchmark

        out = tmp_path / "bench_out"
        shards = run_benchmark([fake_model], EXAMPLE_MANIFEST, SamplingSpec(num_frames=4),
                               out, isolate=False, log=lambda m: None)
        assert set(aggregate(shards, EXAMPLE_MANIFEST, SamplingSpec(num_frames=4))) == {
            "pick-001", "pick-002"}


# ---------------------------------------------------------------- isolated-mode orchestration (mocked subprocess)
class TestIsolatedOrchestration:
    def test_builds_correct_worker_command(self, tmp_path):
        from rewardbox.bench.runner import run_benchmark

        out = tmp_path / "bench_out"
        captured = {}

        class FakeCompleted:
            returncode = 0
            stdout = "[fake] pick-001: ok\n[fake] pick-002: ok\n"
            stderr = ""

        def fake_run(cmd, capture_output, text):
            captured["cmd"] = cmd
            from rewardbox.bench.shard import build_experiment_identity

            shard_path = Path(cmd[cmd.index("--shard") + 1])
            manifest_path = Path(cmd[cmd.index("--manifest") + 1])
            sampling = SamplingSpec.from_dict(
                json.loads(cmd[cmd.index("--sampling") + 1]))
            model_opts = json.loads(cmd[cmd.index("--model-opts") + 1])
            identity = build_experiment_identity("fake-model", sampling, model_opts)
            manifest = Manifest.load(manifest_path)
            shard = ModelShard(model="fake-model", identity=identity)
            for entry in manifest:
                reward_out = RewardOutput(progress=[0.0], frame_indices=[0],
                                          model_name="fake-model")
                shard.add_result(entry.id, reward_out, manifest.entry_identity(entry),
                                 [0, 13, 26, 39])
            shard.save(shard_path)
            return FakeCompleted()

        with patch("subprocess.run", side_effect=fake_run):
            shards = run_benchmark(["fake-model"], EXAMPLE_MANIFEST,
                                   SamplingSpec(num_frames=4), out, isolate=True,
                                   log=lambda m: None)

        cmd = captured["cmd"]
        assert cmd[:4] == [cmd[0], "-m", "rewardbox.bench.worker", "--model"]
        assert "--manifest" in cmd and "--sampling" in cmd and "--shard" in cmd
        assert set(shards["fake-model"].results) == {"pick-001", "pick-002"}
        assert (out / "fake-model.worker.stdout.log").exists()
        assert (out / "fake-model.worker.stderr.log").exists()

    def test_nonzero_exit_does_not_raise_or_decode(self, tmp_path, count_decodes):
        from rewardbox.bench.runner import run_benchmark

        out = tmp_path / "bench_out"

        class FakeFailed:
            returncode = 1
            stdout = ""
            stderr = "boom"

        with patch("subprocess.run", return_value=FakeFailed()):
            shards = run_benchmark(["fake-model"], EXAMPLE_MANIFEST,
                                   SamplingSpec(num_frames=4), out, isolate=True,
                                   log=lambda m: None)
        assert shards["fake-model"].results == {}
        assert set(shards["fake-model"].errors) == {"pick-001", "pick-002"}
        assert shards["fake-model"].fatal_error
        assert count_decodes == []

    def test_model_specific_python_implies_isolation(self, tmp_path):
        from rewardbox.bench.runner import run_benchmark
        from rewardbox.bench.shard import build_experiment_identity

        out = tmp_path / "bench_out"
        alternate_python = tmp_path / "vlac-python"
        alternate_python.write_text("#!/bin/sh\nexit 99\n")
        alternate_python.chmod(0o755)
        captured = {}
        identity = build_experiment_identity("fake-model", SamplingSpec(num_frames=4), {})

        class FakeCompleted:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, capture_output, text):
            captured["cmd"] = cmd
            shard_path = Path(cmd[cmd.index("--shard") + 1])
            manifest = Manifest.load(EXAMPLE_MANIFEST)
            shard = ModelShard(model="fake-model", identity=identity)
            for entry in manifest:
                result = RewardOutput(progress=[0.0], frame_indices=[0],
                                      model_name="fake-model")
                shard.add_result(entry.id, result, manifest.entry_identity(entry),
                                 [0, 13, 26, 39])
            shard.save(shard_path)
            return FakeCompleted()

        with patch(
            "rewardbox.bench.runner._query_worker_identity",
            return_value=identity,
        ), patch("subprocess.run", side_effect=fake_run):
            shards = run_benchmark(
                ["fake-model"], EXAMPLE_MANIFEST, SamplingSpec(num_frames=4), out,
                worker_pythons={"fake-model": str(alternate_python)},
                log=lambda m: None,
            )

        assert captured["cmd"][0] == str(alternate_python.resolve())
        assert set(shards["fake-model"].results) == {"pick-001", "pick-002"}

    def test_unknown_worker_python_model_is_rejected(self, tmp_path):
        from rewardbox.bench.runner import run_benchmark

        with pytest.raises(ValueError, match="not selected"):
            run_benchmark(
                ["fake-model"], EXAMPLE_MANIFEST, SamplingSpec(num_frames=4),
                tmp_path / "out",
                worker_pythons={"vlac-2b": sys.executable},
                log=lambda m: None,
            )

    def test_worker_python_preserves_virtualenv_symlink(self, tmp_path):
        from rewardbox.bench.runner import _resolve_worker_python

        venv_entry = tmp_path / "python"
        venv_entry.symlink_to(sys.executable)
        assert _resolve_worker_python(str(venv_entry)) == str(venv_entry.absolute())


def test_worker_identity_only_needs_no_shard(capsys):
    from rewardbox.bench.worker import main

    rc = main([
        "--model", "not-a-real-model",
        "--manifest", str(EXAMPLE_MANIFEST),
        "--sampling", json.dumps(SamplingSpec(num_frames=4).to_dict()),
        "--identity-only",
    ])
    assert rc == 0
    identity = json.loads(capsys.readouterr().out)
    assert identity["python_version"]
    assert identity["platform"]
    assert identity["input_policy"] == INPUT_POLICY


# ---------------------------------------------------------------- aggregate
class TestAggregate:
    def test_merges_shards_into_result_sets_without_decoding(self, fake_model, tmp_path,
                                                            count_decodes):
        from rewardbox.bench.runner import aggregate, run_benchmark

        out = tmp_path / "bench_out"
        shards = run_benchmark([fake_model], EXAMPLE_MANIFEST, SamplingSpec(num_frames=4),
                               out, isolate=False, log=lambda m: None)
        count_decodes.clear()
        result_sets = aggregate(shards, EXAMPLE_MANIFEST, SamplingSpec(num_frames=4))
        assert count_decodes == []
        assert set(result_sets) == {"pick-001", "pick-002"}
        rs = result_sets["pick-001"]
        assert fake_model in rs.results
        assert rs.results[fake_model].sampling == rs.sampling   # backfilled
        assert rs.input_frame_indices.tolist() == [0, 13, 26, 39]
        manifest = Manifest.load(EXAMPLE_MANIFEST)
        traj = manifest.to_trajectory(manifest.by_id("pick-001"), SamplingSpec(num_frames=4))
        assert rs.source == traj.source == manifest.trajectory_source(manifest.by_id("pick-001"))

    def test_aggregated_result_set_passes_schema_and_saves(self, fake_model, tmp_path):
        from rewardbox.bench.runner import aggregate, run_benchmark

        out = tmp_path / "bench_out"
        shards = run_benchmark([fake_model], EXAMPLE_MANIFEST, SamplingSpec(num_frames=4),
                               out, isolate=False, log=lambda m: None)
        result_sets = aggregate(shards, EXAMPLE_MANIFEST, SamplingSpec(num_frames=4))
        for traj_id, rs in result_sets.items():
            rs.save(out / f"trajectory_{traj_id}.json")   # passing = schema validated

    def test_all_models_failed_entry_aggregates_with_null_grid(self, fake_model, tmp_path):
        from rewardbox.bench.runner import aggregate, run_benchmark

        out = tmp_path / "bench_out"
        shards = run_benchmark(["not-a-real-model"], EXAMPLE_MANIFEST,
                               SamplingSpec(num_frames=4), out, isolate=False,
                               log=lambda m: None)
        rs = aggregate(shards, EXAMPLE_MANIFEST, SamplingSpec(num_frames=4))["pick-001"]
        assert rs.input_frame_indices is None and set(rs.errors) == {"not-a-real-model"}
        rs.save(out / "trajectory_pick-001.json")

    def test_rejects_stale_entry(self, fake_model, tmp_path):
        from rewardbox.bench.runner import aggregate, run_benchmark
        from rewardbox.bench.shard import BenchmarkIdentityError

        out = tmp_path / "bench_out"
        shards = run_benchmark([fake_model], EXAMPLE_MANIFEST, SamplingSpec(num_frames=4),
                               out, isolate=False, log=lambda m: None)
        shards[fake_model].inputs["pick-001"]["entry"]["instruction"] = "something else"
        with pytest.raises(BenchmarkIdentityError, match="stale"):
            aggregate(shards, EXAMPLE_MANIFEST, SamplingSpec(num_frames=4))

    def test_rejects_models_that_saw_different_grids(self, fake_model, tmp_path):
        from rewardbox.bench.runner import aggregate
        from rewardbox.bench.shard import BenchmarkIdentityError

        manifest = Manifest.load(EXAMPLE_MANIFEST)
        shards = {}
        for model, grid in (("m1", [0, 13, 26, 39]), ("m2", [0, 10, 20, 39])):
            shard = ModelShard(model=model)
            for entry in manifest:
                shard.add_result(entry.id, RewardOutput(progress=[0.0], frame_indices=[0],
                                                        model_name=model),
                                 manifest.entry_identity(entry), grid)
            shards[model] = shard
        with pytest.raises(BenchmarkIdentityError, match="different frame grids"):
            aggregate(shards, EXAMPLE_MANIFEST, SamplingSpec(num_frames=4))
