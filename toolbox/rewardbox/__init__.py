"""RewardBox: one interface for robot reward models.

Quick start (pyiqa-style)::

    import rewardbox

    rewardbox.list_models()                      # names, no heavy imports
    rm = rewardbox.create_reward("robometer-4b")  # checks deps before loading
    out = rm("robot.mp4", "open the drawer")      # model's own default sampling
    out.progress, out.frame_indices               # raw-scale curve on source frames
"""
from .core.results import ResultSet
from .core.spec import DependencyError
from .core.types import RewardOutput, SamplingSpec, Trajectory
from .registry import (create_reward, get_model_info, get_spec, list_models,
                       load_manifest)
from .version import __version__

__all__ = ["create_reward", "list_models", "get_model_info", "get_spec",
           "load_manifest", "SamplingSpec", "Trajectory", "RewardOutput",
           "ResultSet", "DependencyError", "__version__"]
