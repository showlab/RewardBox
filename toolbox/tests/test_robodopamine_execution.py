"""Prepared execution guards preserve task text and native default settings."""
import re
import sys
import types

import pytest
import rewardbox
from rewardbox.adapters.robodopamine_execution import bounded_run_pipeline


def test_bounded_directory_function_keeps_full_task_and_original_module_globals():
    class GRM:
        def run_pipeline(self, *, task):
            slug = re.sub(r'[^A-Za-z0-9_]', '_', task)
            return {"directory": slug, "prompt_task": task, "parsed": re.findall("task", task)}

    grm = GRM()
    original_regex = grm.run_pipeline.__func__.__globals__["re"]
    task = "long task " * 40
    wrapped = bounded_run_pipeline(grm)
    actual = wrapped(task=task)
    assert len(actual["directory"]) <= 96
    assert actual["prompt_task"] == task
    assert actual["parsed"] == ["task"] * 40
    assert grm.run_pipeline.__func__.__globals__["re"] is original_regex
    assert len(grm.run_pipeline(task=task)["directory"]) > 255
    assert wrapped(task="short task")["directory"] == "short_task"


def test_explicit_context_override_only_replaces_native_limit(monkeypatch):
    module = types.ModuleType("examples.inference")
    calls = []

    def llm(**kwargs):
        calls.append(kwargs)
        return object()

    class GRM:
        def __init__(self, model_path):
            self.model = module.LLM(model=model_path, max_model_len=8192, temperature="unchanged")

    module.LLM, module.GRMInference = llm, GRM
    monkeypatch.setitem(sys.modules, "examples.inference", module)
    adapter = rewardbox.create_reward("robodopamine-v1-8b", check_deps=False,
                                      max_model_len=16384, safe_run_names=True, batch_size=4)
    adapter.grm()
    assert calls[0]["max_model_len"] == 16384
    assert calls[0]["temperature"] == "unchanged"
    assert module.LLM is llm


@pytest.mark.parametrize("options", [{"max_model_len": 4096}, {"max_model_len": True},
                                     {"safe_run_names": "yes"}])
def test_invalid_guard_options_are_rejected(options):
    with pytest.raises(ValueError):
        rewardbox.create_reward("robodopamine-v1-8b", check_deps=False, **options)


@pytest.mark.parametrize("copy_fallback", [False, True])
def test_replicated_primary_writes_png_once_and_preserves_all_camera_pixels(
        tmp_path, monkeypatch, copy_fallback):
    import numpy as np
    from PIL import Image
    from rewardbox.core.types import Trajectory
    from rewardbox.core import video
    from rewardbox.adapters import robodopamine

    frames = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    trajectory = Trajectory(frames={"primary": frames}, instruction="move")
    original = video.write_png_dir
    writes = []

    def write(values, path, **kwargs):
        writes.append(path)
        return original(values, path, **kwargs)

    if copy_fallback:
        def unsupported(*args):
            raise OSError("hard links unsupported")
        monkeypatch.setattr(robodopamine.os, "link", unsupported)
    monkeypatch.setattr(video, "write_png_dir", write)
    adapter = rewardbox.create_reward("robodopamine-v1-8b", check_deps=False)
    paths = adapter._write_cams(trajectory, tmp_path)
    assert len(writes) == 1
    assert len(set(paths.values())) == 3
    for path in paths.values():
        for index, expected in enumerate(frames):
            actual = np.asarray(Image.open(f"{path}/frame_{index:06d}.png"))
            np.testing.assert_array_equal(actual, expected)


def test_real_three_views_keep_their_independent_pixels(tmp_path):
    import numpy as np
    from PIL import Image
    from rewardbox.core.types import Trajectory
    from rewardbox.adapters.robodopamine import VIEW_TO_CAM

    views = {name: np.full((1, 4, 4, 3), value, dtype=np.uint8)
             for name, value in [("primary", 10), ("left_wrist", 20), ("right_wrist", 30)]}
    adapter = rewardbox.create_reward("robodopamine-v1-8b", check_deps=False)
    paths = adapter._write_cams(Trajectory(frames=views, instruction="move"), tmp_path)
    for name, frames in views.items():
        actual = np.asarray(Image.open(f"{paths[VIEW_TO_CAM[name]]}/frame_000000.png"))
        np.testing.assert_array_equal(actual, frames[0])
