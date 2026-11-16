"""CLI tests.

Key principle (C7/C5): the fake adapter replaces only the "model execution" boundary; the
path from argument parsing to create_reward() is never monkeypatched, because the tests
must prove that the real argv -> argparse -> create_reward(**opts) chain drops no argument.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rewardbox.cli.main import EXIT_OK, EXIT_RUNTIME_ERROR, EXIT_USAGE_ERROR, main

VIDEO = str(Path(__file__).resolve().parent.parent / "calibration" / "synthetic_pick.mp4")


@pytest.fixture
def fake_model(monkeypatch):
    """Register test-fake-adapter in the real registry (next to the real models) and go
    through create_reward()'s real parsing/construction path; only cls itself is a light fake."""
    import rewardbox.registry as registry
    from tests.fixtures.fake_adapter import FAKE_SPEC, FakeAdapter

    FakeAdapter.last_instance = None
    manifest = dict(registry.load_manifest())
    manifest[FAKE_SPEC.name] = FAKE_SPEC
    monkeypatch.setattr(registry, "_CACHE", manifest)
    yield FAKE_SPEC.name


# ---------------------------------------------------------------- the entry point itself
def test_module_help():
    r = subprocess.run([sys.executable, "-m", "rewardbox.cli", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "list" in r.stdout and "run" in r.stdout


def test_installed_console_script_help():
    venv_bin = Path(sys.executable).parent
    script = venv_bin / "rewardbox"
    if not script.exists():
        pytest.skip("rewardbox console script not installed (uv pip install -e . was not run)")
    r = subprocess.run([str(script), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "rewardbox" in r.stdout


def test_no_command_is_usage_error():
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == EXIT_USAGE_ERROR


# ---------------------------------------------------------------- list
def test_list_does_not_load_heavy_deps():
    code = (
        "import sys\n"
        "from rewardbox.cli.main import main\n"
        "main(['list'])\n"
        "leaked = {m for m in ('torch','transformers','vllm','unsloth','swift','evo_vlac') "
        "if m in sys.modules}\n"
        "assert not leaked, leaked\n"
        "print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok" in r.stdout


def test_list_json_parseable(capsys):
    rc = main(["list", "--json"])
    assert rc == EXIT_OK
    rows = json.loads(capsys.readouterr().out)
    names = {r["name"] for r in rows}
    assert {
        "topreward", "robometer-4b", "robodopamine-8b",
        "roboreward-4b", "gvl", "gvl-gemini-3.5-flash-lite", "sole-r1-8b",
        "lrm-contrastive", "lrm-completion", "lrm-progress", "lrm-tri",
        "procvlm-2b",
        "vlac-2b", "vlac-8b",
    } <= names
    assert "rewind-rbm1m" not in names
    for r in rows:
        assert set(r) >= {"name", "reward_mode", "signal_type",
                          "implementation_kind", "verified", "available"}


def test_list_filter(capsys):
    rc = main(["list", "--filter", "robo*", "--json"])
    assert rc == EXIT_OK
    names = {r["name"] for r in json.loads(capsys.readouterr().out)}
    assert names and all(n.startswith("robo") for n in names)


# ---------------------------------------------------------------- info
def test_info_json_parseable(capsys):
    rc = main(["info", "topreward", "--json"])
    assert rc == EXIT_OK
    info = json.loads(capsys.readouterr().out)
    assert info["name"] == "topreward"
    assert info["implementation_kind"] == "official_adapter"
    assert set(info["verified"]) == {"score", "compare"}


def test_info_unknown_model_returns_usage_error(capsys):
    rc = main(["info", "topreward1"])
    assert rc == EXIT_USAGE_ERROR
    err = capsys.readouterr().err
    assert "Did you mean" in err
    assert "topreward" in err


# ---------------------------------------------------------------- doctor
def test_doctor_all_models_runs(capsys):
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc in (EXIT_OK, EXIT_RUNTIME_ERROR)
    assert "topreward" in out


def test_doctor_unknown_model_is_usage_error():
    """doctor on a non-existent model goes through the shared "unknown model" branch, not a bare KeyError."""
    rc = main(["doctor", "nope-not-a-model"])
    assert rc == EXIT_USAGE_ERROR


@pytest.mark.upstream("robometer")
def test_doctor_recognizes_separately_named_gvl_provider_profile(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    rc = main(["doctor", "gvl-gemini-3.5-flash-lite"])

    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert "GVL pinned Robometer repository" in out
    assert "GEMINI_API_KEY is not set" in out


@pytest.mark.upstream("robometer")
@pytest.mark.parametrize("model", ["gvl-gemini-3.8-flash", "gvl-gemini-3.8-flash-dense"])
def test_doctor_reports_the_gemini_credential_without_showing_it(monkeypatch, capsys, model):
    monkeypatch.setenv("GEMINI_API_KEY", "do-not-display-this-key")

    rc = main(["doctor", model])

    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert "GEMINI_API_KEY is set" in out
    assert "do-not-display-this-key" not in out


# ---------------------------------------------------------------- run: arguments must really reach create_reward/Trajectory
class TestRunParameterPassthrough:
    def test_rejects_api_keys_in_model_opts_without_echoing_the_value(self, capsys):
        secret = "must-not-enter-shell-artifacts"
        rc = main([
            "run", "gvl", VIDEO, "--instruction", "pick",
            "--model-opt", f"api_key={secret}",
        ])

        captured = capsys.readouterr()
        assert rc == EXIT_USAGE_ERROR
        assert "GEMINI_API_KEY" in captured.err
        assert secret not in captured.out + captured.err

    def test_sampling_args_reach_trajectory(self, fake_model, tmp_path):
        """Regression: sampling arguments parsed by the CLI must really build
        the matching Trajectory instead of being dropped half way."""
        from tests.fixtures.fake_adapter import FakeAdapter

        rc = main(["run", fake_model, VIDEO, "--instruction", "pick",
                  "--num-frames", "5", "--strategy", "uniform"])
        assert rc == EXIT_OK
        traj = FakeAdapter.last_instance.last_traj
        assert traj is not None
        assert traj.sampling.num_frames == 5
        assert traj.sampling.strategy == "uniform"
        assert len(traj) <= 5

    def test_model_opt_reaches_create_reward(self, fake_model):
        """Regression: --model-opt must reach create_reward(**opts) -> the adapter
        constructor unchanged, never parsed and silently dropped."""
        from tests.fixtures.fake_adapter import FakeAdapter

        rc = main(["run", fake_model, VIDEO, "--instruction", "pick",
                  "--model-opt", "custom_flag=42", "--model-opt", "note=hello"])
        assert rc == EXIT_OK
        assert FakeAdapter.last_instance.received_opts == {"custom_flag": 42, "note": "hello"}

    def test_model_opt_json_types(self, fake_model):
        from tests.fixtures.fake_adapter import FakeAdapter

        rc = main(["run", fake_model, VIDEO, "--instruction", "pick",
                  "--model-opt", "flag=true", "--model-opt", "ratio=0.5",
                  "--model-opt", "items=[1, 2, 3]"])
        assert rc == EXIT_OK
        got = FakeAdapter.last_instance.received_opts
        assert got == {"flag": True, "ratio": 0.5, "items": [1, 2, 3]}

    def test_unknown_model_opt_is_usage_or_runtime_error_not_silently_ignored(
            self, fake_model, capsys):
        rc = main([
            "run", fake_model, VIDEO, "--instruction", "pick",
            "--model-opt", "definitely_a_typo=123",
        ])
        assert rc == EXIT_USAGE_ERROR
        assert "does not support model options" in capsys.readouterr().err

    def test_resize_only_preserves_model_default_sampling(self, fake_model):
        from tests.fixtures.fake_adapter import FakeAdapter

        rc = main([
            "run", fake_model, VIDEO, "--instruction", "pick",
            "--resize", "32", "24",
        ])
        assert rc == EXIT_OK
        sampling = FakeAdapter.last_instance.last_traj.sampling
        assert sampling.num_frames == 4
        assert sampling.strategy == "uniform"
        assert sampling.resize == (32, 24)

    def test_fps_option_infers_fps_strategy(self, fake_model):
        from tests.fixtures.fake_adapter import FakeAdapter

        rc = main([
            "run", fake_model, VIDEO, "--instruction", "pick", "--fps", "2",
        ])
        assert rc == EXIT_OK
        sampling = FakeAdapter.last_instance.last_traj.sampling
        assert sampling.strategy == "fps"
        assert sampling.fps == 2

    def test_malformed_model_opt_is_usage_error(self, fake_model):
        rc = main(["run", fake_model, VIDEO, "--instruction", "pick",
                  "--model-opt", "no-equals-sign"])
        assert rc == EXIT_USAGE_ERROR

    def test_unknown_model_is_usage_error(self):
        rc = main(["run", "not-a-real-model", VIDEO, "--instruction", "pick"])
        assert rc == EXIT_USAGE_ERROR


class TestRunLifecycle:
    def test_close_called_on_success(self, fake_model, tmp_path):
        from tests.fixtures.fake_adapter import FakeAdapter

        rc = main(["run", fake_model, VIDEO, "--instruction", "pick"])
        assert rc == EXIT_OK
        assert FakeAdapter.last_instance.closed is True

    def test_close_called_on_failure(self, fake_model):
        from tests.fixtures.fake_adapter import FakeAdapter

        rc = main(["run", fake_model, VIDEO, "--instruction", "pick",
                  "--model-opt", "fail=true"])
        assert rc == EXIT_RUNTIME_ERROR
        assert FakeAdapter.last_instance.closed is True

    def test_failure_records_error_in_saved_resultset(self, fake_model, tmp_path):
        out_path = tmp_path / "result.json"
        rc = main(["run", fake_model, VIDEO, "--instruction", "pick",
                  "--model-opt", "fail=true", "-o", str(out_path)])
        assert rc == EXIT_RUNTIME_ERROR
        d = json.loads(out_path.read_text())
        assert fake_model in d["errors"]
        assert fake_model not in d["results"]


class TestRunOutput:
    def test_output_json_passes_result_v1_schema(self, fake_model, tmp_path):
        from rewardbox.core.results import ResultSet

        out_path = tmp_path / "result.json"
        rc = main(["run", fake_model, VIDEO, "--instruction", "pick",
                  "--num-frames", "4", "-o", str(out_path)])
        assert rc == EXIT_OK
        loaded = ResultSet.load(out_path)   # full schema validation; not raising means it passed
        assert fake_model in loaded.results
        assert loaded.results[fake_model].cost.n_calls == 1

    def test_stdout_output_when_no_path_given(self, fake_model, capsys):
        rc = main(["run", fake_model, VIDEO, "--instruction", "pick"])
        assert rc == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["schema_version"] == 1
        assert fake_model in payload["results"]

    def test_repeated_output_is_reported_not_uncaught(self, fake_model, tmp_path, capsys):
        out_path = tmp_path / "result.json"
        args = [
            "run", fake_model, VIDEO, "--instruction", "pick", "-o", str(out_path),
        ]
        assert main(args) == EXIT_OK
        assert main(args) == EXIT_RUNTIME_ERROR
        assert "save failed" in capsys.readouterr().err
        assert main(args + ["--overwrite-models"]) == EXIT_OK


# ---------------------------------------------------------------- bench (Stage E: real implementation)
EXAMPLE_MANIFEST = str(Path(__file__).resolve().parent.parent / "datasets"
                       / "local_example" / "manifest.jsonl")


class TestBench:
    def test_bench_missing_required_args_is_usage_error(self):
        with pytest.raises(SystemExit) as e:
            main(["bench"])
        assert e.value.code == EXIT_USAGE_ERROR

    def test_bench_requires_explicit_sampling(self, fake_model, tmp_path, capsys):
        rc = main(["bench", "--manifest", EXAMPLE_MANIFEST, "--model", fake_model,
                  "-o", str(tmp_path / "out")])
        assert rc == EXIT_USAGE_ERROR
        assert "sampling" in capsys.readouterr().err

    def test_bench_resize_alone_is_not_a_sampling_protocol(
            self, fake_model, tmp_path, capsys):
        rc = main([
            "bench", "--manifest", EXAMPLE_MANIFEST, "--model", fake_model,
            "--resize", "32", "24", "-o", str(tmp_path / "out"),
        ])
        assert rc == EXIT_USAGE_ERROR
        assert "sampling protocol" in capsys.readouterr().err

    def test_bench_unknown_model_is_usage_error(self, tmp_path):
        rc = main(["bench", "--manifest", EXAMPLE_MANIFEST, "--model", "not-a-model",
                  "--num-frames", "4", "-o", str(tmp_path / "out")])
        assert rc == EXIT_USAGE_ERROR

    def test_worker_python_must_target_a_selected_model(
            self, fake_model, tmp_path, capsys):
        rc = main([
            "bench", "--manifest", EXAMPLE_MANIFEST,
            "--model", fake_model, "--num-frames", "4",
            "--worker-python", "/different-model=/tmp/python",
            "-o", str(tmp_path / "out"),
        ])
        assert rc == EXIT_USAGE_ERROR
        assert "not selected with --model" in capsys.readouterr().err

    def test_bench_model_opt_reaches_selected_adapter(
            self, fake_model, tmp_path):
        from tests.fixtures.fake_adapter import FakeAdapter

        rc = main([
            "bench", "--manifest", EXAMPLE_MANIFEST,
            "--model", fake_model, "--num-frames", "4",
            "--model-opt", f"{fake_model}.custom_flag=42",
            "-o", str(tmp_path / "out"),
        ])
        assert rc == EXIT_OK
        assert FakeAdapter.last_instance.received_opts == {"custom_flag": 42}

    def test_bench_model_opt_rejects_unknown_adapter_option(
            self, fake_model, tmp_path, capsys):
        rc = main([
            "bench", "--manifest", EXAMPLE_MANIFEST,
            "--model", fake_model, "--num-frames", "4",
            "--model-opt", f"{fake_model}.definitely_a_typo=42",
            "-o", str(tmp_path / "out"),
        ])
        assert rc == EXIT_USAGE_ERROR
        assert "does not support model options" in capsys.readouterr().err

    def test_bench_runs_model_outer_loop_and_aggregates(self, fake_model, tmp_path, capsys):
        out = tmp_path / "out"
        rc = main(["bench", "--manifest", EXAMPLE_MANIFEST, "--model", fake_model,
                  "--num-frames", "4", "-o", str(out)])
        assert rc == EXIT_OK
        assert (out / f"{fake_model}.json").exists()
        assert (out / "trajectory_pick-001.json").exists()
        assert (out / "trajectory_pick-002.json").exists()
        assert "2 ok" in capsys.readouterr().out

    def test_bench_rerun_into_same_output_dir_does_not_fail(self, fake_model, tmp_path):
        """Regression: running a second time after a finished run (the typical --resume case)
        used to hit ResultSet's own "refuse to overwrite same-name models" rule during
        aggregation; the aggregate output is a complete view regenerated from the shards and
        must not use merge semantics."""
        out = tmp_path / "out"
        args = ["bench", "--manifest", EXAMPLE_MANIFEST, "--model", fake_model,
                "--num-frames", "4", "-o", str(out)]
        assert main(args) == EXIT_OK
        assert main(args) == EXIT_OK   # the second run must not raise ResultConflictError

    def test_changed_sampling_refuses_stale_resume(self, fake_model, tmp_path, capsys):
        out = tmp_path / "out"
        common = [
            "bench", "--manifest", EXAMPLE_MANIFEST, "--model", fake_model,
            "-o", str(out),
        ]
        assert main(common + ["--num-frames", "4"]) == EXIT_OK
        assert main(common + ["--num-frames", "5"]) == EXIT_USAGE_ERROR
        assert "--no-resume" in capsys.readouterr().err

    def test_model_load_failure_returns_nonzero_and_records_every_entry(
            self, fake_model, tmp_path, monkeypatch):
        import rewardbox
        from rewardbox.bench.shard import ModelShard

        def fail_create(name, **opts):
            raise RuntimeError("simulated load failure")

        monkeypatch.setattr(rewardbox, "create_reward", fail_create)
        out = tmp_path / "out"
        rc = main([
            "bench", "--manifest", EXAMPLE_MANIFEST, "--model", fake_model,
            "--num-frames", "4", "-o", str(out),
        ])
        assert rc == EXIT_RUNTIME_ERROR
        shard = ModelShard.load(out / f"{fake_model}.json")
        assert set(shard.errors) == {"pick-001", "pick-002"}
        assert shard.fatal_error


# ---------------------------------------------------------------- doctor: checkouts come from the pinned table
def test_doctor_vlac_cut_names_its_own_checkout(monkeypatch, capsys):
    """vlac-cut loads VLAC-cut, not VLAC; the printed fix must point at the right repository."""
    from rewardbox import upstream

    monkeypatch.setattr(upstream, "find_upstream", lambda key: None)
    rc = main(["doctor", "vlac-cut"])
    out = capsys.readouterr().out
    assert rc == EXIT_RUNTIME_ERROR
    assert "VLAC-cut official repository: not found -- FAIL" in out
    assert "rewardbox fetch-upstream vlac-cut" in out and "REWARDBOX_VLAC_CUT_ROOT" in out
    assert "VLAC official repository" not in out and "REWARDBOX_VLAC_ROOT" not in out


@pytest.mark.parametrize("model, dirname, key", [
    ("procvlm-2b", "ProcVLM", "procvlm"),
    ("robo-valuerl-value", "Robo-ValueRL", "robovaluerl"),
])
def test_doctor_reports_server_side_checkouts_without_failing_the_client(
        monkeypatch, capsys, model, dirname, key):
    from rewardbox import upstream

    monkeypatch.setattr(upstream, "find_upstream", lambda key: None)
    main(["doctor", model])
    out = capsys.readouterr().out
    assert f"{dirname} official repository: not found here" in out
    assert f"rewardbox fetch-upstream {key}" in out
    assert f"{dirname} official repository: not found -- FAIL" not in out


def test_doctor_reports_a_found_checkout_once(monkeypatch, tmp_path, capsys):
    from rewardbox import upstream

    monkeypatch.setattr(upstream, "find_upstream", lambda key: tmp_path / key)
    main(["doctor", "primo-r1-7b"])
    out = capsys.readouterr().out
    assert out.count("PRIMO-R1 official repository:") == 1
    assert str(tmp_path / "primo") in out
