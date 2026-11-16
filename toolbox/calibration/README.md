# Calibration

A calibration record stores what a model returned with its real weights, and a
regression test replays it. This is how an adapter is checked against the
official code: correct shapes and finite values do not show that the output is
the one the official code produces.

## Contents

- `synthetic_pick.mp4`: synthetic fixture (40 frames, a red block moves towards a
  green target). It is generated, so no dataset license applies.
- `synthetic_pick_fail.mp4`: the same scene (320x240, same background, block
  colours and positions) where the red block jitters in place and never reaches
  the target. It is the "failure" side of the `compare()` check.

The records are package data, in `rewardbox/calibration/records/`:

- `<model>.json`: the model's `score()` output on `synthetic_pick.mp4`, with the
  environment it was recorded in.
- `compare/<model>.json`: the model's `compare()` output (success against
  failure video), see
  `tests/test_calibration.py::test_compare_matches_recorded_output`.
- `official_examples/<model>.json` (+ `compare/<model>.json`): the same checks on
  example images shipped with the upstream repository, for `vlac-2b` and
  `vlac-8b`. The synthetic videos are far from what `vlac-2b` was trained on (its
  `score()` signal is close to zero and `compare()` does not prefer the
  successful video), so `DIRECTION_ASSERTION_EXEMPT` in `tests/test_calibration.py`
  skips the direction assertion on the fixture for this model and the direction
  is asserted on the official images. `vlac-8b` gets the direction right on the
  fixture and runs on the official images as well. These records need the VLAC
  checkout (the images are not copied here); without it the tests skip.
- `server_provenance/<model>.json`: what the model server reported when a
  server-based record was taken.

## Record format

A record holds what is needed to reproduce it: `model`, `video`, `instruction`,
`sampling` (including its key), `environment` (python, torch, transformers, gpu)
and `output`, plus `model_opts` when the recording used non-default adapter
options.

If a fixture or the sampling logic is changed on purpose, the regression test
fails and the record has to be recorded again
(`scripts/record_calibration.py --replace`).

## Running the regression

```bash
HF_HOME=$PWD/.hf \
PYTHONPATH=$PWD/reward_model_code/robometer:$PWD/reward_model_code/TOPReward \
REWARDBOX_ROBODOPAMINE_ROOT=$PWD/reward_model_code/Robo-Dopamine \
  .venv/bin/python -m pytest tests/test_calibration.py -q -m calibration
```

VLAC uses its own environment (`ms-swift==3.3` does not go into the shared
`.venv`):

```bash
export HF_HOME=$PWD/.hf REWARDBOX_VLAC_ROOT=$PWD/reward_model_code/VLAC CUDA_VISIBLE_DEVICES=<id>
.venv-vlac/bin/python -m rewardbox.cli doctor vlac-2b --deep
.venv-vlac/bin/python -m pytest tests/test_calibration.py -q -m calibration -k vlac
```

SOLE-R1 needs its reward server. Start it as described in
[docs/installation.md](../docs/installation.md), then enable the test (it is off
by default so that an ordinary calibration run does not wait for a server):

```bash
REWARDBOX_SOLE_R1_CALIBRATION=1 \
  .venv/bin/python -m pytest -q \
  'tests/test_calibration.py::test_matches_recorded_output[sole-r1-8b]'
```

The SOLE-R1 record uses the adapter's 1 Hz external-view protocol with
`temperature=0.0`; the regression test replays the recorded `model_opts`. One
repeat differed by one point, so the tolerance is `atol=1.0`.

## Adding a model

Record with `scripts/record_calibration.py` once the model runs with real
weights, and put the tolerance in `tests/test_calibration.py::TOL` with a reason
(discrete outputs must match exactly; continuous outputs get a tolerance for
bf16 jitter). Non-default adapter options are passed as `--model-opt
KEY=JSON_VALUE`; they go into the record and are replayed by the regression. A
model that samples randomly needs a reproducible profile (for example a fixed
seed or temperature 0) before a record is useful. Add the result to
[docs/validation.md](../docs/validation.md).
