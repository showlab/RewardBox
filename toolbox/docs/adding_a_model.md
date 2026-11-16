# Adding a model

An adapter does four things and nothing else:

1. convert a `Trajectory` into the input the official API expects,
2. call the official entry point,
3. convert the official output into a `RewardOutput` (raw scale),
4. record every conversion in `RewardOutput.meta`.

Anything beyond that (a different frame protocol, prefix curves from a
terminal-only model, a new backend) is an extension: register it under a
separate name with `implementation_kind=REWARDBOX_EXTENSION` so results never
mix with the official recipe.

## Template

```python
# rewardbox/adapters/mymodel.py
from __future__ import annotations

import numpy as np

from ..core.base import BaseAdapter
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import RewardOutput, SamplingSpec, Trajectory

SPEC = ModelSpec(
    name="mymodel-7b",
    cls="MyModelAdapter",
    module=__name__,
    reward_mode=RewardMode.PROGRESS,
    signal_type=SignalType.GENERATED_TEXT,
    requires=Requirements(pip=("torch>=2.8", "transformers>=4.57,<4.58"), vram_gb=20.0,
                          install_hint=("pip install -e '.[models]'",)),
    sampling=SamplingSpec(num_frames=16, strategy="uniform"),  # = the upstream recipe
    score_range=(0.0, 1.0),
    weights="org/MyModel-7B@<immutable revision>",
    upstream="mymodel.inference:Scorer.score",                 # official entry point
    paper="https://arxiv.org/abs/....",
    implementation_kind=ImplementationKind.OFFICIAL_ADAPTER,
)


class MyModelAdapter(BaseAdapter):
    def __init__(self, spec, check_deps: bool = True, model_path: str | None = None,
                 device: str | None = None):
        super().__init__(spec, check_deps=check_deps)
        self.model_path = model_path or spec.weights
        self.device = device
        self._scorer = None

    def _load(self):
        if self._scorer is None:
            from mymodel.inference import Scorer      # heavy import stays inside the method
            self._scorer = Scorer(self.model_path, device=self.device)
        return self._scorer

    def _score(self, traj: Trajectory) -> RewardOutput:
        values = self._load().score(list(traj.primary), traj.instruction)   # official call
        return self._out(np.asarray(values, dtype=float), traj,
                         meta={"protocol": "official uniform-16", "n_backend_calls": 1})

    def close(self):
        self._scorer = None
```

Then add `"rewardbox.adapters.mymodel"` to `_ADAPTER_MODULES` in
`rewardbox/registry.py`. Only explicit constructor parameters are accepted as
model options (`create_reward(name, **opts)` rejects anything else), so list
every option the user may set.

If the method loads source code from a git repository rather than a pip
package, add an `Upstream(...)` row to `rewardbox/upstream.py` (key, directory,
URL, **full commit hash**, served model names, a marker file, an optional
`REWARDBOX_*_ROOT` override) and resolve it with `upstream_candidates()` /
raise `UpstreamNotFoundError` when it is missing; `rewardbox fetch-upstream`
and the test suite then handle it automatically.

## Checklist

- Module top level imports no torch / transformers / vllm (`rewardbox list` must
  stay light; `tests/test_core.py` enforces it).
- The default `SamplingSpec` is the upstream protocol; frame counts are never
  hard-coded elsewhere.
- Output is the native scale; no clipping, normalization or interpolation
  unless the official code does it.
- `frame_indices` are the source frames the values belong to (a subset of the
  input grid is fine).
- Weights and upstream code are pinned to immutable revisions.
- Offline tests: fake only the model call and run the real upstream pre/post
  processing (see `tests/test_vlac_cut_adapter.py`). Mark tests that need a
  checkout with `@pytest.mark.upstream("key")`.
- Real-weight evidence: `scripts/record_calibration.py` writes a calibration
  record that `tests/test_calibration.py` replays on GPU.
- Add the adapter module to `MODULES` in `scripts/gen_model_docs.py` (method family,
  runtime class, environment, validation document) and a row to `docs/validation.md`,
  then regenerate the docs:
  `python scripts/gen_model_docs.py`. Without the `MODULES` row the generator stops
  with a `KeyError`.
- Update the expected roster in `tests/test_release_api.py`
  (`test_registry_matches_paper_roster`; a model without a source checkout also joins
  the `needs_source` exemption in `test_every_checkout_model_is_registered_and_mapped`)
  and the configuration counts in the READMEs, `AGENTS.md`, `llms.txt` and the skill.
