"""Fake adapter for CLI tests only.

It replaces only the "model execution" boundary (the heavy official inference call) and
leaves the real argument parsing -> create_reward() path alone; the CLI tests exist to
prove that this real path drops no argument (``--backend`` was once parsed and then
dropped on the spot, a real bug class found in review).
"""
from __future__ import annotations

import numpy as np

from rewardbox.core.base import BaseAdapter
from rewardbox.core.enums import ImplementationKind, RewardMode, SignalType
from rewardbox.core.spec import ModelSpec, Requirements
from rewardbox.core.types import SamplingSpec


class FakeAdapter(BaseAdapter):
    """Records the constructor arguments and the Trajectory it received, so tests can assert
    that the actual arguments arrived here."""

    last_instance: "FakeAdapter | None" = None

    def __init__(self, spec, check_deps: bool = True, fail: bool = False,
                 custom_flag=None, note=None, flag=None, ratio=None, items=None,
                 **opts):
        super().__init__(spec, check_deps=check_deps, **opts)
        explicit = {
            "custom_flag": custom_flag, "note": note, "flag": flag,
            "ratio": ratio, "items": items,
        }
        self.received_opts = {k: v for k, v in explicit.items() if v is not None}
        self.received_opts.update(opts)
        self.fail = fail
        self.closed = False
        self.last_traj = None
        FakeAdapter.last_instance = self

    def _score(self, traj):
        self.last_traj = traj
        if self.fail:
            raise RuntimeError("FakeAdapter failing on purpose (test only)")
        n = len(traj)
        return self._out(np.linspace(0.0, 1.0, n), traj)

    def close(self):
        self.closed = True


FAKE_SPEC = ModelSpec(
    name="test-fake-adapter", cls="FakeAdapter", module=__name__,
    reward_mode=RewardMode.PROGRESS, signal_type=SignalType.VALUE_HEAD,
    requires=Requirements(), model_opts={},
    sampling=SamplingSpec(num_frames=4, strategy="uniform"),
    upstream="tests.fixtures.fake_adapter:FakeAdapter (test only, not a real model)",
    implementation_kind=ImplementationKind.REWARDBOX_EXTENSION,
)
