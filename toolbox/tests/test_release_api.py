"""Release-facing API: pyiqa-style call, context manager, version, upstream table."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import rewardbox
from rewardbox import upstream as up
from rewardbox.cli.main import main
from rewardbox.core.types import SamplingSpec, Trajectory

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "datasets" / "local_example" / "pick_success.mp4"


@pytest.fixture
def fake(monkeypatch):
    from rewardbox import registry

    from tests.fixtures.fake_adapter import FAKE_SPEC, FakeAdapter

    manifest = dict(registry.load_manifest())
    manifest[FAKE_SPEC.name] = FAKE_SPEC
    monkeypatch.setattr(registry, "_CACHE", manifest)
    FakeAdapter.last_instance = None
    return FAKE_SPEC.name


def _installed_dist_is_this_tree():
    from importlib.metadata import PackageNotFoundError, distribution
    from urllib.parse import unquote, urlparse

    try:
        dist = distribution("rewardbox")
    except PackageNotFoundError:
        return None
    here = Path(rewardbox.__file__).resolve().parent
    direct = dist.read_text("direct_url.json")
    if direct and json.loads(direct).get("dir_info", {}).get("editable"):
        src = Path(unquote(urlparse(json.loads(direct)["url"]).path)).resolve()
        return dist if here.parent == src else None
    located = Path(dist.locate_file("rewardbox/__init__.py")).resolve().parent
    return dist if located == here else None


def test_version_is_single_sourced():
    assert rewardbox.__version__ == "0.1.0"
    dist = _installed_dist_is_this_tree()
    if dist is None:  # plain source tree, or another rewardbox install shadows metadata
        pytest.skip("installed rewardbox metadata does not belong to this tree")
    assert dist.version == rewardbox.__version__


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert rewardbox.__version__ in capsys.readouterr().out


def test_registry_matches_paper_roster():
    names = rewardbox.list_models()
    protocol_profiles = [n for n in names if n.startswith(("vlm-", "jev-"))]
    assert sorted(protocol_profiles) == [
        "jev-1.14-pair", "jev-1.14-sequence", "vlm-pair", "vlm-sequence"]
    assert len(names) - len(protocol_profiles) == 35      # the 14 reward-model methods
    assert not [n for n in names if "argon" in n or n.startswith("gemini-er")]
    assert "rewind-rbm1m" not in names


def test_call_with_video_path_uses_model_default_sampling(fake):
    rm = rewardbox.create_reward(fake)
    out = rm(VIDEO, "move the red block onto the green target")
    traj = rm.last_traj
    assert traj.sampling == rm.spec.sampling          # 4-frame uniform default
    assert len(out.progress) == len(traj) == 4
    assert out.frame_indices.tolist() == traj.frame_indices.tolist()
    assert out.frame_indices[-1] == 39                 # include_last on a 40-frame clip


def test_call_accepts_explicit_sampling_and_str_path(fake):
    rm = rewardbox.create_reward(fake)
    out = rm(str(VIDEO), "task", sampling=SamplingSpec(num_frames=6))
    assert len(out.progress) == 6


def test_call_with_raw_array_and_time_based_sampling(fake):
    frames = np.zeros((90, 8, 8, 3), dtype=np.uint8)
    rm = rewardbox.create_reward(fake)
    out = rm(frames, "task", sampling=SamplingSpec(strategy="fps", fps=1.0), fps=30.0)
    assert out.frame_indices.tolist() == [0, 30, 60, 89]


def test_call_with_trajectory_rejects_extra_arguments(fake):
    rm = rewardbox.create_reward(fake)
    traj = Trajectory.from_video(VIDEO, "task", SamplingSpec(num_frames=3))
    assert len(rm(traj).progress) == 3
    with pytest.raises(TypeError, match="already carries"):
        rm(traj, "another instruction")


def test_call_requires_instruction(fake):
    rm = rewardbox.create_reward(fake)
    with pytest.raises(TypeError, match="instruction is required"):
        rm(VIDEO)


def test_call_with_view_dict(fake):
    rm = rewardbox.create_reward(fake)
    out = rm({"primary": VIDEO, "left_wrist": VIDEO}, "task")
    assert set(rm.last_traj.views) == {"primary", "left_wrist"}
    assert len(out.progress) == 4


def test_context_manager_closes(fake):
    with rewardbox.create_reward(fake) as rm:
        rm(VIDEO, "task")
    assert rm.closed


# ---------------------------------------------------------------- upstream table
def test_every_checkout_model_is_registered_and_mapped():
    names = rewardbox.list_models()
    for u in up.UPSTREAMS:
        assert len(u.commit) == 40 and u.url.startswith("https://github.com/")
        assert any(u.serves(n) for n in names), u.key
    # models that load official source code must know where it comes from
    needs_source = [n for n in names
                    if not n.startswith(("roboreward-", "flora-", "vlm-", "jev-"))]
    assert all(up.upstreams_for_model(n) for n in needs_source), [
        n for n in needs_source if not up.upstreams_for_model(n)]
    assert up.upstreams_for_model("roboreward-4b") == []


def test_notice_cites_every_pinned_commit():
    text = (ROOT / "NOTICE.md").read_text(encoding="utf-8")
    for u in up.UPSTREAMS:
        assert u.commit in text, u.key


def test_shared_dir_is_searched_first(monkeypatch, tmp_path):
    root = tmp_path / "VLAC-cut"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "run_example.py").write_text("")
    monkeypatch.delenv("REWARDBOX_VLAC_CUT_ROOT", raising=False)
    monkeypatch.setenv(up.UPSTREAM_DIR_ENV, str(tmp_path))
    assert up.upstream_candidates("VLAC-cut")[0] == root
    assert up.find_upstream("vlac-cut") == root.resolve()

    from rewardbox.adapters.vlac_cut import _repo_root
    assert _repo_root() == root


def test_missing_checkout_raises_dedicated_error(monkeypatch, tmp_path):
    monkeypatch.setenv(up.UPSTREAM_DIR_ENV, str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("REWARDBOX_VLAC_CUT_ROOT", raising=False)
    monkeypatch.setattr(up, "upstream_candidates", lambda d: [tmp_path / d])
    import rewardbox.adapters.vlac_cut as vc
    monkeypatch.setattr(vc, "upstream_candidates", lambda d: [tmp_path / d])
    with pytest.raises(up.UpstreamNotFoundError):
        vc._repo_root()
    assert issubclass(up.UpstreamNotFoundError, RuntimeError)


def test_fetch_dry_run_prints_pinned_commands(tmp_path, capsys):
    assert main(["fetch-upstream", "robometer-4b", "--dest", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "git clone --filter=blob:none --no-checkout https://github.com/robometer/robometer.git" in out
    assert "5b815254bf31ee1bea3753c3a2da9f9033736d9a" in out
    assert not (tmp_path / "robometer").exists()


def test_fetch_list_json(capsys):
    assert main(["fetch-upstream", "--list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["key"] for r in rows} == {u.key for u in up.UPSTREAMS}


def test_fetch_unknown_target_is_usage_error(capsys):
    assert main(["fetch-upstream", "no-such-model"]) == 2


def test_fetch_refuses_wrong_existing_checkout(tmp_path):
    target = tmp_path / "TOPReward"
    target.mkdir()
    subprocess.run(["git", "init", "-q", str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "-c", "user.email=a@b", "-c", "user.name=t",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    with pytest.raises(RuntimeError, match="pinned commit"):
        up.fetch("topreward", tmp_path, log=lambda *_: None)


def test_public_schema_lists_every_sampling_strategy():
    from rewardbox.schemas import load_result_v1_schema

    enum = set(load_result_v1_schema()["properties"]["sampling"]["properties"]["spec"]
               ["properties"]["strategy"]["enum"])
    accepted = set()
    for strategy in ("uniform", "uniform_floor", "all", "fps", "interval", "random", "prefix"):
        kw = {"fps": 1.0} if strategy == "fps" else {"interval": 2} if strategy == "interval" else {}
        try:
            SamplingSpec(strategy=strategy, **kw)
        except ValueError:
            continue
        accepted.add(strategy)
    assert enum == accepted


def test_fetch_topreward_name_gets_both_key_and_model_checkouts(tmp_path, capsys):
    # ``topreward`` is the TOPReward repo key *and* a model that runs Robometer's baseline.
    assert up.upstreams_for_model("topreward") == [up.get_upstream("robometer")]
    assert main(["fetch-upstream", "topreward", "--dest", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "TOPReward/TOPReward.git" in out and "robometer/robometer.git" in out


def test_robometer_puts_checkout_on_path_before_importing(monkeypatch):
    # Fresh process, robometer not pip-installed: _ensure_loaded (which calls
    # ensure_importable) must run before any ``import robometer.*``.
    from rewardbox.adapters.robometer import RobometerAdapter

    class Reached(Exception):
        pass

    def fake_loaded():
        raise Reached

    rm = RobometerAdapter(rewardbox.get_spec("robometer-4b"))
    monkeypatch.setattr(rm, "_ensure_loaded", fake_loaded)
    with pytest.raises(Reached):
        rm._forward_progress(None)
    with pytest.raises(Reached):
        rm._compare(None, None)


def test_topreward_uniform_matches_robometer_eval_sampler():
    # robometer/data/datasets/helpers.py::linspace_subsample_frames @5b815254, which the
    # Table-I recipe (DataConfig.max_frames=64) applies before TopReward is called.
    def robometer_sampler(total, n):
        if total <= n:
            return list(range(total))
        idx = np.rint(np.linspace(0, total - 1, n)).astype(int).tolist()
        idx[0], idx[-1] = 0, total - 1
        return idx

    spec = rewardbox.get_spec("topreward").sampling
    for total in (10, 64, 65, 100, 150, 1000):
        assert spec.indices(total).tolist() == robometer_sampler(total, 64), total
