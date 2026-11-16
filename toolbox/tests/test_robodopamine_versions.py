"""Keep checkpoint generations distinct through registry, inference and provenance."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from rewardbox import create_reward, get_spec, list_models
from rewardbox.adapters.robodopamine import MODEL_NAMES
from rewardbox.bench.shard import build_experiment_identity
from rewardbox.upstream import get_upstream
from rewardbox.cli.main import main
from rewardbox.core.types import SamplingSpec, Trajectory


PROFILES = [
    ("robodopamine-v1-3b", "Robo-Dopamine-GRM-3B", "v1"),
    ("robodopamine-v1-8b", "Robo-Dopamine-GRM-8B", "v1"),
    ("robodopamine-v2-4b", "Robo-Dopamine-GRM-2.0-4B-Preview", "v2"),
    ("robodopamine-v2-8b", "Robo-Dopamine-GRM-2.0-8B-Preview", "v2"),
    ("robodopamine-8b", "Robo-Dopamine-GRM-2.0-8B-Preview", "v2"),
]


@pytest.mark.parametrize("name,checkpoint,version", PROFILES)
def test_checkpoint_routing_and_output_identity(name, checkpoint, version, monkeypatch, tmp_path):
    calls = []
    module = types.ModuleType("examples.inference")

    def llm(**kwargs):
        calls.append(kwargs)

    class GRM:
        def __init__(self, model_path):
            module.LLM(model=model_path, gpu_memory_utilization=0.9)

        def run_pipeline(self, *, out_root, eval_mode, frame_interval, goal_image, **kwargs):
            assert frame_interval == 10
            assert Path(goal_image).is_file()
            path = Path(out_root)
            path.mkdir()
            (path / "sample.json").write_text(json.dumps([{"id": "step-bf-af_000010"}]))
            value = {"forward": 0.2, "incremental": 0.4, "backward": 0.6}[eval_mode]
            (path / "pred_vllm.json").write_text(json.dumps([
                {"progress": value, "hop": 0.1, "pred": "<score>+10%</score>"}]))
            return str(path)

    module.LLM = llm
    module.GRMInference = GRM
    monkeypatch.setitem(sys.modules, "examples.inference", module)
    monkeypatch.setattr("rewardbox.adapters.robodopamine._repo_root", lambda: tmp_path)
    assert set(MODEL_NAMES) <= set(list_models(filter="robodopamine*"))
    spec = get_spec(name)
    assert spec.weights == f"tanhuajie2001/{checkpoint}"
    adapter = create_reward(name, check_deps=False)
    frames = np.zeros((11, 8, 8, 3), dtype=np.uint8)
    traj = Trajectory(frames=frames, instruction="pick", sampling=spec.sampling,
                      goal_image=frames[-1])
    result = adapter.score(traj)
    assert calls == [{"model": spec.weights, "gpu_memory_utilization": 0.7}]
    assert result.model_name == name
    assert result.meta["model_version"] == version
    assert result.meta["model_path"] == spec.weights
    assert result.meta["canonical_model_name"] == (
        "robodopamine-v2-8b" if name == "robodopamine-8b" else name)
    assert result.meta["goal_image_source"] == "provided"
    assert result.frame_indices.tolist() == [10]
    assert result.progress.tolist() == pytest.approx([0.4])
    assert module.LLM is llm


@pytest.mark.parametrize("name,checkpoint,version", PROFILES)
def test_versions_keep_guards_and_allow_local_checkpoints(name, checkpoint, version):
    adapter = create_reward(name, check_deps=False, model_path="/local/checkpoint")
    assert adapter.model_path == "/local/checkpoint"
    traj = Trajectory(frames=np.zeros((2, 8, 8, 3), dtype=np.uint8), instruction="pick",
                      sampling=SamplingSpec(num_frames=2))
    with pytest.raises(ValueError, match=name):
        adapter._score(traj)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_versioned_upstream_identity_and_doctor(name, monkeypatch, tmp_path, capsys):
    identity = build_experiment_identity(name, SamplingSpec(strategy="interval", interval=4))
    assert identity["upstream_commits"] == {
        "robodopamine": get_upstream("robodopamine").commit}
    monkeypatch.setenv("REWARDBOX_ROBODOPAMINE_ROOT", str(tmp_path))
    main(["doctor", name])
    assert "Robo-Dopamine official repository: not found" in capsys.readouterr().out
