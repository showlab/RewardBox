# AGENTS.md

RewardBox runs robot reward and progress-estimation models on robot videos
through one interface, the `rewardbox` package in `toolbox/`. That is the main
part of this repository. `benchmark/` is a companion directory: the evaluator of
RRM-Bench-1K with the manifests and labels it reads. Read this file first, then
the manual of the part you work in.

## Layout

| path | what | manual |
|---|---|---|
| `toolbox/` | the `rewardbox` Python package: adapters for 14 methods, CLI, metrics, benchmark runner, tests, docs | [`toolbox/AGENTS.md`](toolbox/AGENTS.md) |
| `benchmark/` | companion: the RRM-Bench-1K evaluator, with manifests and task-state labels for reference | [`benchmark/AGENTS.md`](benchmark/AGENTS.md) |
| `README.md` | project overview for people | |

## Which manual

| the user wants to | read |
|---|---|
| score a video, compare models, run a dataset, compute metrics | `toolbox/AGENTS.md`, recipes 1-5 |
| evaluate a model on RRM-Bench-1K | `benchmark/AGENTS.md`, then `toolbox/AGENTS.md` recipe 6 |
| evaluate a general VLM (GPT, Claude, Gemini, a local server) or Jev | `toolbox/docs/guides/general_vlm_and_jev.md` |
| install a model's environment, server or source checkout | `toolbox/docs/installation.md` |
| add a model | `toolbox/AGENTS.md` recipe 7, `toolbox/docs/adding_a_model.md` |
| fix or change the package | `toolbox/AGENTS.md` recipes 8-9 |

## Setup

```bash
cd toolbox
pip install -e .          # Python >= 3.10; core only, no GPU and no torch
rewardbox list            # or: python -m rewardbox.cli list
```

Commands in `toolbox/AGENTS.md` assume the working directory `toolbox/`; commands
in `benchmark/AGENTS.md` assume `benchmark/`.

## Not in this repository

- **Benchmark videos** (`benchmark/episodes/**/videos/*.mp4`, about 3.9 GB): the
  benchmark data will be hosted on Hugging Face and is not published yet; ask the
  user where a local copy is. Labels, manifests and frame maps are here, so the
  evaluator and its tests run without the videos; running a model on the
  benchmark does not.
- **Upstream source checkouts**: fetched at pinned commits by
  `rewardbox fetch-upstream` into `toolbox/reward_model_code/`.
- **Model weights, virtual environments, results**: downloaded or created locally.

## Rules that hold everywhere

1. Create models only through `rewardbox.create_reward(name, **opts)`; never rebuild
   a model's prompt, preprocessing or decoding by hand.
2. Report raw outputs with their unit and the exact configuration name and sampling.
3. API keys come from environment variables only; never put one in an option, a
   file, a log or a commit.
4. Do not edit `benchmark/inputs/`, `benchmark/episodes/` or an upstream checkout.
5. Check `nvidia-smi` before loading a model; never stop a process you did not start.
6. Files are written in English.
7. Before finishing a change: `cd toolbox && pytest -m "not gpu and not calibration"`
   and `python scripts/gen_model_docs.py --check`; after touching the evaluator,
   `python benchmark/tools/test_evaluate.py`.
