# AGENTS.md: evaluating on RRM-Bench-1K

This directory is a benchmark (data + official evaluator), not a code base to
modify. Read `README.md` for the protocol; this file is the short version.
Commands assume this directory is the working directory.

## What is here

- `inputs/`: the manifests (`single_view.jsonl`, `multiview.jsonl`), the labels and
  the 1 FPS grid. `tools/`: the evaluator and its tests. `episodes/*/*/source/`:
  frame maps.
- `inputs/events.jsonl`: the event marks behind the labels (clip, time, English
  description). For reading the labels only: no metric uses it, and it is never
  part of a model's input.
- **The videos (`episodes/*/*/videos/*.mp4`) are not in git.** They are in the Hugging Face dataset Ethanliang99/RRM-Bench-1K; from the repository root, download them with `huggingface-cli download Ethanliang99/RRM-Bench-1K --repo-type dataset --include "episodes/*" --local-dir benchmark`. Scoring saved predictions and `tools/test_evaluate.py` work
  without the videos; running a model does not (`video file of entry ... does not
  exist`). Put the `episodes/` tree here so the manifest paths resolve.
- `rewardbox` is the package in `../toolbox` (`pip install -e ../toolbox`).

## Rules

- Never edit `inputs/` or `episodes/`; they are the release.
- Use the instruction in `inputs/single_view.jsonl` verbatim. Do not add outcome
  hints, GT, view labels or category information to a model's input.
- Score only with `tools/evaluate.py`. Report `n` and `missing` with every number,
  say whether `--mode interpolate` (†) was used, and name the exact model
  configuration and sampling.
- Compare against the paper's tables, always together with the always-rising and time-only
  references. Per-suite numbers come
  with the always-rising Macro-F1 on the same transitions; (A) Action–Effect is
  scored on its effect-showing videos.
- Aggregate or resample by execution (`group_id`), never treat the 1,117 videos as
  independent executions.
- The ‡ multiview rows and the § LRM-8B (Contrastive) row of the paper's Table II
  use inputs that are not in this release (README, "Evaluators of the paper");
  do not present a run on `inputs/multiview.jsonl` as a reproduction of them.

## Run a RewardBox model

```bash
rewardbox info MODEL --json            # default sampling, views, score range
rewardbox doctor MODEL                 # fix deps / checkout / server first
rewardbox bench --manifest inputs/single_view.jsonl --model MODEL \
    --strategy fps --fps 1 --no-include-last -o runs/MODEL      # exact 1 FPS grid
python tools/evaluate.py --predictions runs/MODEL/MODEL.json --name MODEL -o runs/MODEL/eval.json
```

General VLMs and Jev use the `vlm-sequence` / `vlm-pair` / `jev-1.14-*` profiles with the
same two commands (`--model-opt vlm-sequence.model=ID`); Pair shards are scored as
per-transition judgements, without ranking metrics.

If the model must keep its native protocol (fixed frame count, prefixes), run
bench with that protocol and evaluate with `--mode interpolate`; the result is
marked † and is not a native 1 FPS prediction. Multi-view models use
`inputs/multiview.jsonl` and `--cohort multiview`. The configuration behind each
row of the paper's Table II is listed in the README.

## Run any other model

Write `{"id", "frame_indices", "progress"}` (a curve on source frames) or
`{"id", "directions"}` (one -1/0/+1/null per 1 FPS transition) per line, then
`python tools/evaluate.py --predictions FILE.jsonl --name NAME`. Ids, frame
counts and the grid are in `inputs/ground_truth.json` and `inputs/eval_grid_1fps.json`.

## Checks

```bash
python tools/test_evaluate.py               # evaluator unit tests (synthetic, seconds)
```
