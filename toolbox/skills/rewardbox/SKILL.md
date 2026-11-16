---
name: rewardbox
description: Score robot manipulation videos with robot reward models (task progress curves, success probability, pairwise preference) through the RewardBox toolbox, and benchmark or compare those models (including on RRM-Bench-1K). Use when the user wants a reward/progress/value estimate for a robot video, wants to run Robometer, RoboReward, TOPReward, Robo-Dopamine, GVL, VLAC, VLAC-Cut, PRIMO-R1, Flora-Rewarder, SOLE-R1, LRM, RynnValue, ProcVLM or Robo-ValueRL, wants to test a general VLM (GPT, Claude, Gemini) or Jev as a robot progress evaluator, or wants to set up, debug or extend RewardBox.
---

# RewardBox

RewardBox (CLI `rewardbox`) runs 35 configurations of 14 robot reward-model
methods through their **official** code behind one API, plus 4 protocol profiles
for general VLMs and Jev (`vlm-sequence`, `vlm-pair`, `jev-1.14-*`).

The package is the `toolbox/` directory of https://github.com/showlab/RewardBox
(`git clone`, then `pip install -e .` inside `toolbox/`); the benchmark is
`benchmark/` in the same repository. Paths below such as `docs/...` and `AGENTS.md`
are relative to `toolbox/`. If `rewardbox` is not on PATH, find the clone first
(`python -c "import rewardbox; print(rewardbox.__file__)"`).
The registry is the source of truth: query it instead of relying on memory.

## Workflow

1. **Find the model and its contract**
   ```bash
   rewardbox list --json
   rewardbox info MODEL --json   # default sampling, required views, score_range, higher_better, missing deps
   ```
   Default choice for a dense progress curve on one GPU: `robometer-4b`.
   No GPU: `gvl-gemini-3.5-flash-lite` with `GEMINI_API_KEY` set. A general VLM or
   Jev as the evaluator: `vlm-sequence` / `vlm-pair` (`--model-opt model=ID`) or
   `jev-1.14-*`; read `docs/guides/general_vlm_and_jev.md` first.
2. **Make it runnable**: `rewardbox doctor MODEL`, then apply the printed fix:
   install deps, `rewardbox fetch-upstream MODEL` for the pinned official source,
   start the model server for server-based methods, or use the model's isolated venv.
3. **Score**
   ```python
   import rewardbox
   with rewardbox.create_reward("robometer-4b") as rm:
       out = rm("/abs/robot.mp4", "open the drawer")    # model's upstream sampling
   out.progress, out.frame_indices, out.success          # raw native scale
   ```
   or `rewardbox run MODEL VIDEO --instruction "..." -o out.json`.
4. **Batch**: write a manifest (`scripts/make_manifest.py` in this skill builds one
   from a folder of videos), then
   `rewardbox bench --manifest m.jsonl --model MODEL --num-frames N -o runs/x`
   (resumable; `--worker-python MODEL=PY` for isolated-venv models).
5. **Benchmark on RRM-Bench-1K**: `rewardbox bench --manifest BENCH/inputs/single_view.jsonl
   --model MODEL --strategy fps --fps 1 --no-include-last -o runs/MODEL`, then
   `python BENCH/tools/evaluate.py --predictions runs/MODEL/MODEL.json` (see BENCH/AGENTS.md).
   BENCH is `../benchmark`; its videos are not in git (ask the user where they are).
6. **Report** the exact configuration name, sampling and the raw unit
   (`score_range`); say explicitly if you normalized.

## Rules

- Never reimplement a model's prompt/preprocessing; always go through `create_reward`.
- Keep `spec.sampling` unless the user specifies a protocol; `bench` requires an explicit one.
- API keys only via environment variables, never `--model-opt`.
- Run interpreters by absolute path; do not install one model's stack into another's venv.
- Check `nvidia-smi` before loading; never stop other users' GPU jobs.
- A calibration record is not a paper reproduction; do not claim one.

## When something fails

`rewardbox doctor MODEL` first. Error fragments:
`dependencies not satisfied` → missing deps (follow the `Fix:` lines); `repository not found` /
`UpstreamNotFoundError` → `rewardbox fetch-upstream MODEL`; `does not support model options` →
misspelled option (allowed list is printed); `needs views` → pass
`{"primary": ..., "left_wrist": ...}`; `contains credential fields` → use env vars.
The repository's `AGENTS.md` has the full troubleshooting table, the I/O
contract and recipes for adding a model.
