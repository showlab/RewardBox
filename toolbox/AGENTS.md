# AGENTS.md: using RewardBox from a coding agent

RewardBox (`import rewardbox`, CLI `rewardbox`) runs robot reward models on
robot videos through one interface. It wraps each method's **official** code;
it does not reimplement models. 35 configurations from 14 methods are
registered, plus 4 protocol profiles that query general VLMs and Jev
(`vlm-sequence`, `vlm-pair`, `jev-1.14-*`; `docs/guides/general_vlm_and_jev.md`).
The live registry is the source of truth, not this file:
`rewardbox list --json`, `rewardbox info NAME --json`.

## Where you are

This directory (`toolbox/`, the one with `pyproject.toml`) is the package root.
Every command and relative path below assumes it is the working directory. The
benchmark lives next to it in `../benchmark/` and has its own `AGENTS.md`.

```bash
pip install -e .                  # once; Python >= 3.10 (the tested shared env is 3.11)
rewardbox list                    # if the entry point is not on PATH: python -m rewardbox.cli list
```

The core install needs no GPU and no torch. Model weights, the pinned upstream
checkouts (`reward_model_code/`), virtual environments and results are not in git.

## Rules

1. **Create models only through the registry**: `rewardbox.create_reward(name, **opts)`.
   Never rebuild a model's prompt, preprocessing or decoding by hand.
2. **Keep the model's default sampling** (`spec.sampling`, its upstream recipe)
   unless the user asks for a protocol. When comparing models on the same videos,
   either keep each model's native protocol or pass one explicit `SamplingSpec`
   to all, and say which.
3. **Report raw outputs with their unit.** `out.progress` is the native scale
   (`score_range` in `rewardbox info`: fraction, percent, `-512..0` seconds remaining,
   signed `-100..100`). If you normalize (`out.normalized("minmax")`), say so.
   Check `higher_better` before ranking.
4. **Run `rewardbox doctor NAME` before loading a model** and apply the printed fix.
   It takes a few seconds and loads no model (it imports torch only to probe the GPU).
5. **Use interpreters by absolute path** (`.venv/bin/python`, `.venv-vlac/bin/python`).
   Do not `source` activate scripts. Do not pip-install a model's stack into a
   different model's environment; isolated models have their own venv for a reason.
6. **Credentials come from environment variables only** (`GEMINI_API_KEY`, `JEV_API_KEY`;
   `vlm-*`, `jev-*` and `gvl-gemini-3.8-flash*` also read the variable named by `api_key_env`).
   Never put keys in `--model-opt`, code, logs or result files.
7. **Share GPUs politely.** Check `nvidia-smi` first; never kill processes you did not
   start. vLLM's `gpu_memory_utilization` is a fraction of the *whole* card.
8. **Do not edit upstream checkouts** under `reward_model_code/` or
   `$REWARDBOX_UPSTREAM_DIR`. They are pinned; the fetch tool refuses dirty trees.
9. **State validation precisely.** A model in `rewardbox/calibration/records/` reproduces
   a recorded run of the official code, not a paper's numbers; `docs/validation.md` says what
   was checked for each method. Name the exact configuration (`robodopamine-v2-8b`, not
   "Robo-Dopamine") and sampling in reports.
10. **Do not overwrite results.** `ResultSet.save` refuses to replace a model already in
    the file unless `--overwrite-models`; `bench` resumes instead of rerunning.

## Pick a model

| need | start with | notes |
|---|---|---|
| dense progress curve, one GPU | `robometer-4b` | 8 uniform frames, one forward pass; also per-frame `success`; supports `compare()` |
| progress without a local GPU (GVL, hosted Gemini API) | `gvl-gemini-3.5-flash-lite` | needs `GEMINI_API_KEY`; the frozen `gvl` profile is no longer served to new keys |
| multi-view input | `robodopamine-v2-8b` (optional wrists), `robo-valuerl-value` (requires 3 views), `flora-rewarder-27b` | pass `{"primary": ..., "left_wrist": ..., "right_wrist": ...}` |
| pairwise preference | `robometer-4b`, `vlac-2b`/`vlac-8b` (`supports_pairwise`) | `rm.compare(a, b) > 0` means a is better |
| terminal success / completion | `lrm-completion` (server), `robometer-4b` (`out.success`) | |
| cheapest smoke test | `robometer-4b` or `roboreward-4b` | in-process, ~14-20 GB VRAM |
| a general VLM (GPT / Claude / Gemini / local OpenAI-compatible server) or Jev as evaluator | `vlm-sequence`, `vlm-pair`, `jev-1.14-sequence`, `jev-1.14-pair` | RewardBox protocols, not reward models; `model=...` is required for `vlm-*`; Pair results are per-transition judgements in `raw.transition_directions`, not a curve |

For quality numbers on real robot data, see the RRM-Bench-1K tables of the
paper; do not infer quality from model size.
`docs/models.json` lists every configuration's runtime, default sampling,
views, score range, checkouts and validation document.

Runtime classes (`docs/models.json` → `runtime`):
- `in-process` / `in-process vLLM`: runs inside the current Python (shared py3.11 env).
- `isolated worker`: needs its own venv (VLAC, VLAC-Cut, PRIMO-R1, Flora-Rewarder).
  From Python, run your script with that venv's interpreter; for batches use
  `rewardbox bench --worker-python MODEL=/abs/path/to/python`.
- `model server`: start the authors' server first (SOLE-R1, LRM, RynnValue, ProcVLM,
  Robo-ValueRL), then pass `server_url` / `host` and `server_provenance_file`.
- `hosted API`: GVL, general-VLM and Jev profiles; needs an API key in the environment.

## Recipes

### 1. Inspect

```bash
rewardbox list --json                    # [{name, reward_mode, signal_type, implementation_kind, verified, available}]
rewardbox info robometer-4b --json       # full spec: sampling, required_views, score_range, requires, missing
rewardbox fetch-upstream --list --json   # pinned source checkouts and whether each is found
```

`available: false` means a pip requirement is missing in *this* interpreter (an
isolated-worker model is normally unavailable in the shared env). `available: true`
says nothing about the source checkout, a model server or an API key; `doctor` checks
the checkout and reports whether the key variable is set.

### 2. Make a model runnable

```bash
rewardbox doctor robometer-4b            # exit 0 = ready; otherwise read the FAIL lines
rewardbox fetch-upstream robometer-4b    # pinned official source, if doctor says the checkout is missing
```

Fix order: Python deps (doctor prints the `uv pip install ...` line) → upstream
checkout (`fetch-upstream`) → server or API key → GPU. Full per-runtime recipes:
`docs/installation.md`.

### 3. Score one video

```python
import rewardbox

with rewardbox.create_reward("robometer-4b") as rm:      # deps checked before heavy imports
    out = rm("/abs/path/robot.mp4", "put the cup on the coaster")
print(out.frame_indices.tolist())   # source-video frame of each value
print(out.progress.tolist())        # raw scale; see rm.spec.score_range / higher_better
print(out.cost.latency_s, out.meta.get("upstream"))
```

```bash
rewardbox run robometer-4b /abs/path/robot.mp4 --instruction "put the cup on the coaster" -o out.json
```

`rm(...)` input forms: video path, frame directory, `{view: path}` dict,
`[T,H,W,3]` uint8 RGB array of all frames (add `fps=30.0` for time-based
sampling), or a `Trajectory` (then no other arguments). Extra keywords
(`goal_image=`, `reference_images=`) go to `Trajectory`. Model options go to
`create_reward(name, **opts)`; unknown options raise `TypeError` listing the
allowed ones.

### 4. Score many videos (resumable)

Write a manifest (`.jsonl`; paths are relative to the manifest file):

```json
{"id": "ep001", "video": "videos/ep001.mp4", "instruction": "open the drawer", "success": true}
{"id": "ep002", "video": {"primary": "v/ep002_front.mp4", "left_wrist": "v/ep002_wrist.mp4"}, "instruction": "open the drawer"}
```

Optional fields: `split`, `success`, `progress_gt` + `progress_gt_frame_indices`,
`pair_id`, `group_id`, `source`. Then:

```bash
rewardbox bench --manifest data.jsonl --model robometer-4b --model roboreward-4b \
    --num-frames 8 -o runs/exp1               # sampling must be explicit for bench
rewardbox bench ... -o runs/exp1              # rerun = resume; finished entries are skipped
rewardbox bench ... --retry-errors            # also retry entries that failed before
rewardbox bench ... --isolate                 # one subprocess per model
rewardbox bench ... --worker-python vlac-2b=/abs/.venv-vlac/bin/python
rewardbox bench ... --model-opt robometer-4b.device=cuda:1   # MODEL.KEY=JSON_OR_STRING
```

Outputs: `runs/exp1/<model>.json` (one shard per model: results, errors,
experiment identity), `runs/exp1/trajectory_<id>.json` (all models for one
entry, schema `rewardbox/schemas/result-v1.schema.json`), worker logs for isolated
runs. Resume compares each manifest entry's id, video path(s) as written and instruction;
file contents are not hashed. Editing one entry re-runs only that entry, and a video replaced
in place under the same path is not detected. A resume with a different sampling, model
options, weights, pinned upstream commit, RewardBox version or package versions raises
`BenchmarkIdentityError` instead of mixing incomparable results; use a new `-o`.

### 5. Metrics

```python
from rewardbox import metrics
metrics.voc(out.progress)                          # Spearman of the curve vs. time order
metrics.srcc(pred, gt); metrics.plcc(pred, gt); metrics.krcc(pred, gt)
metrics.success_auroc(terminal_scores, success_labels)
metrics.success_f1(terminal_scores, success_labels, threshold=0.5)
metrics.paired_accuracy(preferred_scores, nonpreferred_scores)   # ties count 0.5
```

These are generic metrics. RRM-Bench-1K's official numbers come from the
benchmark's own evaluator (recipe 6), whose definitions differ in detail
(for example VOC there is a per-trajectory Pearson r on successful clips).

### 6. Evaluate on RRM-Bench-1K

`../benchmark/` holds the RRM-Bench-1K manifests, labels and the official evaluator;
its own `AGENTS.md` has the rules. The videos are not in git (they are in the Hugging Face dataset Ethanliang99/RRM-Bench-1K; from the repository root, `huggingface-cli download Ethanliang99/RRM-Bench-1K --repo-type dataset --include "episodes/*" --local-dir benchmark`): without them
`bench` stops with `video file of entry ... does not exist`, while the evaluator
and its tests work on saved predictions. The protocol scores 1 FPS transitions,
which is RewardBox `SamplingSpec(strategy="fps", fps=1.0, include_last=False)`:

```bash
B=../benchmark
rewardbox bench --manifest $B/inputs/single_view.jsonl --model robometer-4b \
    --strategy fps --fps 1 --no-include-last -o runs/robometer-4b
python $B/tools/evaluate.py --predictions runs/robometer-4b/robometer-4b.json \
    --name robometer-4b -o runs/robometer-4b/eval.json
```

A model that must keep its native protocol (fixed frame count, prefixes): run
`bench` with that protocol and add `--mode interpolate` (reported as †). Compare
with the paper's tables, always next to the always-rising (Macro-F1 0.474) and
time-only (VOC 1.000) references, and report `n`/`missing`.
Benchmark VOC is a per-trajectory Pearson r on successful videos, unlike
`rewardbox.metrics.voc` (Spearman).

### 7. Add a model

1. `rewardbox/adapters/<name>.py`: subclass `BaseAdapter`, implement
   `_score(self, traj) -> RewardOutput` (and `_compare` for pairwise). No torch /
   transformers / vllm imports at module top level.
2. Declare a module-level `SPEC` (or `SPECS`) `ModelSpec` with requirements, default
   `SamplingSpec` = the upstream recipe, views, `score_range`, `upstream` entry point,
   and `implementation_kind`.
3. Register the module in `_ADAPTER_MODULES` (`rewardbox/registry.py`); if it needs a
   source checkout, add an `Upstream(...)` row in `rewardbox/upstream.py`.
4. Add offline tests (fake the model call, run the real upstream pre/post-processing).
5. Tell the docs and the roster about it, or step 6 fails: add the module to `MODULES`
   in `scripts/gen_model_docs.py` (family, runtime, environment, validation document) and a
   row to `docs/validation.md`;
   update the expected roster in `tests/test_release_api.py`
   (`test_registry_matches_paper_roster`, and the `needs_source` exemption in
   `test_every_checkout_model_is_registered_and_mapped` if the model needs no checkout);
   update the configuration counts and the model-zoo table in `../README.md`, and the
   counts in `README.md`, this file, `llms.txt` and `skills/rewardbox/SKILL.md`.
6. Run `python scripts/gen_model_docs.py` and `pytest -m "not gpu and not calibration"`.

Full template and checklist: `docs/adding_a_model.md`.

### 8. Test

```bash
pytest -m "not gpu and not calibration"            # offline; upstream-dependent tests skip if checkouts are absent
rewardbox fetch-upstream all
REWARDBOX_REQUIRE_UPSTREAM=1 pytest -m "not gpu and not calibration"   # nothing may skip for a missing checkout
pytest tests/test_calibration.py -m calibration    # real weights + GPU; replays calibration records
```

### 9. Change RewardBox itself

| path | what lives there |
|---|---|
| `rewardbox/core/types.py` | `SamplingSpec`, `Trajectory`, `RewardOutput` (the I/O contract) |
| `rewardbox/core/base.py`, `core/spec.py` | `BaseAdapter`; `ModelSpec`, `Requirements` and the dependency check |
| `rewardbox/core/video.py` | frame decoding, sampling and image encoding |
| `rewardbox/core/results.py`, `schemas/` | `ResultSet` and the result-v1 JSON schema |
| `rewardbox/registry.py`, `upstream.py` | model registry and factory; pinned upstream checkouts and `fetch-upstream` |
| `rewardbox/adapters/` | one module per method (`SPEC` / `SPECS` + adapter class) |
| `rewardbox/backends/` | shared vLLM backend and the VLAC worker process |
| `rewardbox/bench/` | `runner.py` (loop, isolation), `worker.py` (one model), `shard.py` (resume identity) |
| `rewardbox/data/manifest.py`, `metrics/`, `cli/main.py` | manifests, metrics, the `rewardbox` command |
| `rewardbox/calibration/records/`, `prompts/` | frozen real-weight records; packaged prompt texts |
| `scripts/` | model-server launchers (`run_*_server.py`), `gen_model_docs.py`, `smoke_upstream.py` |
| `tests/` | offline tests (`tests/fixtures/fake_adapter.py` is the fake model) |
| `docs/` | `models.md` / `models.json` (generated), `installation.md`, `validation.md`, `guides/` |

Invariants that tests or users rely on:

- No `torch` / `transformers` / `vllm` import at module top level anywhere under
  `rewardbox/`; `cv2` is imported only in `core/video.py`.
- Model options are explicit constructor parameters; `**opts` does not make an
  option accepted (`registry.validate_model_opts`).
- Whatever can change a result must be part of the resume identity
  (`bench/shard.py`: sampling, options, weights, upstream commit, endpoints).
- Credentials never enter options, argv, logs or result files (`core/secrets.py`).
- Outputs stay on the native scale; nothing is clipped, normalized, interpolated or
  filled in unless the official code does it.
- Files are English only; `ruff` settings are in `pyproject.toml` (line length 100).
- After a registry or spec change run `python scripts/gen_model_docs.py`; CI-style
  check: `python scripts/gen_model_docs.py --check`.

## Contracts

`Trajectory` (input): `frames` `[T,H,W,3]` uint8 RGB or `{view: array}` (all views
same `T`); `instruction` (non-empty); `frame_indices` (strictly increasing source
frames); `fps`; optional `goal_image` `[H,W,3]`, `reference_images`
`{name: [K,H,W,3]}`, `success`, `progress_gt`, `sampling`. Build with
`Trajectory.from_video / from_views / from_frames_dir / from_array`.

`RewardOutput` (output): `progress [T']` raw, `frame_indices [T']` (a subset of the
input grid: prefix models report only some points), optional `success [T']` in
[0,1], `weights`, `raw` (upstream payload), `cost` (`latency_s`, `n_calls`,
`peak_vram_gb`), `meta` (protocol, upstream entry, weights, conversions).
Helpers: `normalized(method)`, `resample_to(frames)` (refuses to extrapolate
unless `extrapolate="constant"`), `to_dict()` / `from_dict()`.

## Troubleshooting

| message (fragment) | meaning | fix |
|---|---|---|
| `DependencyError: ... cannot create 'X': dependencies not satisfied` | pip requirement missing in this interpreter | run the printed `Fix:` lines, or use the model's own venv |
| `UpstreamNotFoundError` / `official ... checkout not found` / `... repository not found` | pinned source missing | `rewardbox fetch-upstream X` or set the printed `REWARDBOX_*_ROOT` |
| `unknown model 'X'. Did you mean: ...` | unknown name | use a suggested name; `rewardbox list` |
| `'X' does not support model options [...]; available: [...]` | misspelled model option | use one of the listed options |
| `X needs views [...], the trajectory only has [...]` | model needs more camera views | pass a `{view: path}` dict |
| `--model-opt of 'X' contains credential fields` | key on the command line | export `GEMINI_API_KEY` / `JEV_API_KEY` (or the `api_key_env` variable) |
| `X needs the API model id: model=...` | `vlm-sequence` / `vlm-pair` without a model | `--model-opt model=ID` (`bench`: `--model-opt vlm-sequence.model=ID`) |
| `answered in the wrong format` | a general VLM ignored the output format | recorded as that entry's error; `--retry-errors` asks again, nothing is repaired |
| `bench needs an explicit sampling protocol` | `bench` without sampling | add `--num-frames N` or `--strategy fps --fps 1` etc. |
| `BenchmarkIdentityError` | resume with a different protocol | new output dir, or rerun with the original settings |
| `ResultConflictError` | merging incomparable results / same model twice | separate output file, or `--overwrite-models` if intended |
| RynnValue `allow_unverified_server=true` | no server provenance | pass `server_provenance_file=` from the launcher |
| `gvl` HTTP 404 / model unavailable | frozen Gemini 2.5 profile | use `gvl-gemini-3.5-flash-lite` or a `gvl-gemini-3.8-flash` profile |
| `git-lfs filter-process: git-lfs: not found` | global git LFS config without git-lfs | `fetch-upstream` already works around it; for manual git use `-c filter.lfs.process= -c filter.lfs.required=false` |

## Environment variables

| variable | purpose |
|---|---|
| `HF_HOME` | Hugging Face cache for weights |
| `REWARDBOX_UPSTREAM_DIR` | shared directory of pinned checkouts (searched first) |
| `REWARDBOX_<NAME>_ROOT` | one checkout override (`ROBOMETER`, `ROBODOPAMINE`, `SOLE_R1`, `LRM`, `RYNNVALUE`, `PROCVLM`, `VLAC`, `VLAC_CUT`, `PRIMO`) |
| `REWARDBOX_SOLE_R1_HOST`, `REWARDBOX_LRM_SERVER_URL`, `REWARDBOX_LRM_{PROGRESS,COMPLETION,CONTRASTIVE}_URL`, `REWARDBOX_RYNNVALUE_SERVER_URL`, `REWARDBOX_PROCVLM_SERVER_URL`, `REWARDBOX_ROBOVALUERL_SERVER_URL` | model-server endpoints |
| `GEMINI_API_KEY`, `JEV_API_KEY` | GVL and Jev credentials (`jev-*` also need `base_url`) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` | default credentials of `vlm-*` for `api=openai / anthropic / gemini`; another variable via `api_key_env` |
| `REWARDBOX_CALIBRATION_DIR` | alternative calibration-record directory |
| `REWARDBOX_REQUIRE_UPSTREAM=1` | tests: fail instead of skip when a checkout is missing |
| `REWARDBOX_REQUIRE_MODELS=1` | calibration tests: fail instead of skip when a model cannot be created |
