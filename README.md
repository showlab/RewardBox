<div align="center">

# RewardBox

#### **One interface for robot reward models.**

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://ethanliang99.github.io/awesome-robot-reward-model)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-RRM--Bench--1K-yellow)](https://huggingface.co/datasets/Ethanliang99/RRM-Bench-1K)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
[![Citation](https://img.shields.io/badge/Citation-bibtex-green)](#bookmark_tabs-citation)

📦 [Model Zoo](toolbox/docs/models.md) | 🛠 [Installation](toolbox/docs/installation.md) | 💬 [General VLMs & Jev](toolbox/docs/guides/general_vlm_and_jev.md) | ➕ [Add a Model](toolbox/docs/adding_a_model.md) | 🤖 [Agent Guide](AGENTS.md) | 📈 [RRM-Bench-1K](benchmark/README.md)

</div>

Give RewardBox a robot video and a task instruction, and get back the reward that a published model assigns.

RewardBox is the toolbox (RRM-Hub) of the survey *Robotic Reward Models in the Foundation-Model Era: A Survey and Capability Benchmark*. It runs the official implementation of 14 reward-model methods through one Python and command-line interface, and it evaluates any of them on the paper's benchmark, RRM-Bench-1K.

```python
import rewardbox

with rewardbox.create_reward("robometer-4b") as rm:
    out = rm("robot.mp4", "open the drawer")

out.progress        # one value per sampled frame, in the model's native unit
out.frame_indices   # the source-video frame of each value
```

## :robot: Supported Models

| Method | Family | Native output | Configurations | Runs as |
|---|---|---|---|---|
| [Robometer](https://arxiv.org/abs/2603.02115) | Hybrid | Progress in [0, 1] and success per frame; preference between two videos | `robometer-4b`, `robometer-4b-libero` | in-process |
| [RoboReward](https://arxiv.org/abs/2601.00675) | Outcome | Episode grade from 1 to 5 | `roboreward-4b`, `roboreward-8b` | in-process |
| [TOPReward](https://arxiv.org/abs/2602.19313) | Absolute Progress | Log-probability of the answer token | `topreward`, `topreward-qwen2.5-7b` | in-process |
| [Robo-Dopamine](https://arxiv.org/abs/2512.23703) | Relative | Signed progress between two states, chained over the video | `robodopamine-8b`, `robodopamine-v1-3b`, `robodopamine-v1-8b`, `robodopamine-v2-4b`, `robodopamine-v2-8b` | in-process (vLLM) |
| [GVL](https://arxiv.org/abs/2411.04549) | Absolute Progress | Completion per frame, from shuffled frames | `gvl`, `gvl-gemini-3.5-flash-lite`, `gvl-gemini-3.8-flash`, `gvl-gemini-3.8-flash-dense` | hosted API |
| [VLAC](https://arxiv.org/abs/2509.15937) | Hybrid | Cumulative signed progress | `vlac-2b`, `vlac-8b` | isolated environment |
| [VLAC-Cut](https://arxiv.org/abs/2607.09776) | Hybrid | Signed progress at keypoints, with cut points | `vlac-cut`, `vlac-cut-vllm` | isolated environment or in-process (vLLM) |
| [PRIMO-R1](https://arxiv.org/abs/2603.15600) | Hybrid | Progress in percent | `primo-r1-7b`, `primo-r1-7b-prefix` | isolated environment |
| [Flora-Rewarder](https://huggingface.co/nvidia/Flora-Rewarder) | Hybrid | Episode grade from 1 to 5 | `flora-rewarder-27b`, `flora-rewarder-27b-prefix` | isolated environment |
| [SOLE-R1](https://arxiv.org/abs/2603.28730) | Absolute Progress | Progress in percent, with a rationale | `sole-r1-8b` | official model server |
| [LRM](https://arxiv.org/abs/2603.16065) | Hybrid | Progress, completion, or signed change, one interface each | `lrm-completion`, `lrm-completion-vllm`, `lrm-contrastive`, `lrm-contrastive-vllm`, `lrm-progress`, `lrm-progress-vllm`, `lrm-tri` | official model server or in-process (vLLM) |
| [RynnValue](https://arxiv.org/abs/2608.09853) | Temporal Value | Negative remaining time in seconds | `rynnvalue-4b`, `rynnvalue-8b` | official model server |
| [ProcVLM](https://arxiv.org/abs/2605.08774) | Absolute Progress | Progress in percent | `procvlm-2b` | official model server |
| [Robo-ValueRL](https://arxiv.org/abs/2607.09866) | Temporal Value | Value from the remaining steps, three views | `robo-valuerl-value` | official model server |

Family is the method's semantic family in the paper: the question its native output answers. The [model zoo](toolbox/docs/models.md) gives the default sampling, camera views, score range, and pinned source of every configuration, and `rewardbox list` shows the same data.

General VLMs and Jev 1.14 are not reward models, but the paper evaluates them with two fixed query protocols, which RewardBox also provides: `vlm-sequence` and `vlm-pair` for any OpenAI-, Anthropic-, or Gemini-format API, and `jev-1.14-sequence` and `jev-1.14-pair` ([guide](toolbox/docs/guides/general_vlm_and_jev.md)).

Seven configurations replay frozen real-weight calibration records in the test suite; [validation](toolbox/docs/validation.md) lists what was checked against the official code for each method.

## :open_book: How It Works

- **Official code.** An adapter turns a video into the input of the authors' own entry point at a pinned commit and calls it. Nothing is reimplemented.
- **Native outputs.** `progress` keeps the model's own unit: a fraction, a percentage, seconds remaining, or a signed score. Rescaling is the caller's choice.
- **Explicit sampling.** Each model defaults to its upstream frame protocol, and a comparison across models states the protocol it used.
- **Light core.** Listing models, computing metrics, and running a benchmark need no GPU and no torch. Heavy dependencies load only when a model is created.

## :package: Installation

```bash
git clone https://github.com/showlab/RewardBox.git
cd RewardBox/toolbox

# core: registry, CLI, metrics, benchmark runner (no GPU, no torch)
pip install -e .

# in-process models: Robometer, RoboReward, TOPReward, Robo-Dopamine
pip install -e ".[models]"
```

Python 3.10 or newer; the tested shared environment is Python 3.11 on Linux
x86_64 with CUDA 12.8. A model may also need the pinned official source it calls
or the authors' model server. `rewardbox doctor MODEL` reports what is missing
and prints the fix:

```bash
rewardbox doctor robometer-4b            # dependencies, source checkout, GPU
rewardbox fetch-upstream robometer-4b    # clone the pinned official source
```

Reproducible lock files, isolated environments (VLAC, VLAC-Cut, PRIMO-R1,
Flora-Rewarder) and model servers (SOLE-R1, LRM, RynnValue, ProcVLM,
Robo-ValueRL) are covered in [Installation](toolbox/docs/installation.md).

## :zap: Usage

### Command line

```bash
# list all configurations, or the details of one
rewardbox list
rewardbox info robometer-4b

# score one video with the model's default sampling
rewardbox run robometer-4b robot.mp4 --instruction "open the drawer" -o result.json

# score a dataset with several models (resumable)
rewardbox bench --manifest data.jsonl --model robometer-4b --model topreward \
    --num-frames 8 -o runs/exp1
```

### Python

```python
import rewardbox

# list models, no heavy import
print(rewardbox.list_models())

# create a model; dependencies are checked before anything heavy is loaded
with rewardbox.create_reward("robometer-4b") as rm:
    out = rm("robot.mp4", "open the drawer")     # the model's own default sampling

print(out.progress)        # np.ndarray [T], raw native scale (see rm.spec.score_range)
print(out.frame_indices)   # source-video frame of each value
print(out.success)         # per-frame success probability, for models that output it
```

Choose the sampling, pass several camera views, or compare two videos:

```python
from rewardbox import SamplingSpec, Trajectory

with rewardbox.create_reward("robometer-4b") as rm:
    # an explicit frame protocol instead of the model default
    out = rm("robot.mp4", "stack the cubes", sampling=SamplingSpec(num_frames=16))

    # multi-view models take a {view: path} dict
    out = rm({"primary": "front.mp4", "left_wrist": "wrist.mp4"}, "fold the towel")

    # pairwise models: > 0 means the first trajectory is better
    a = Trajectory.from_video("success.mp4", "stack the cubes", rm.spec.sampling)
    b = Trajectory.from_video("failure.mp4", "stack the cubes", rm.spec.sampling)
    preference = rm.compare(a, b)
```

Besides a video file, `rm(...)` accepts a directory of frames, a `[T,H,W,3]`
uint8 array or a `Trajectory`.

### Score a dataset

Write one JSON line per video (paths are relative to the manifest file):

```json
{"id": "ep001", "video": "videos/ep001.mp4", "instruction": "open the drawer", "success": true}
{"id": "ep002", "video": {"primary": "v/ep002_front.mp4", "left_wrist": "v/ep002_wrist.mp4"}, "instruction": "open the drawer"}
```

```bash
rewardbox bench --manifest data.jsonl --model robometer-4b --num-frames 8 -o runs/exp1
```

Running the same command again resumes; `runs/exp1/robometer-4b.json` holds the
results of that model for every video.

### Metrics

```python
from rewardbox import metrics

metrics.voc(out.progress)                                    # value-order correlation
metrics.success_auroc(terminal_scores, success_labels)
metrics.paired_accuracy(preferred_scores, nonpreferred_scores)
```

## :1st_place_medal: Evaluate on RRM-Bench-1K

Every registered model can be evaluated on RRM-Bench-1K, the benchmark of the
paper. It tests whether an evaluator follows the *task state* of a real robot
execution rather than elapsed time: 1,117 real-robot videos across 52 tasks with task-state
progress labels, scored on the direction of 1 FPS transitions (Rising / Falling
F1) and on trajectory ranking (VOC, Kendall τ_a). Four contrast suites locate
where an evaluator fails: (A) Action–Effect, (B) Semantic/Ordered Constraint,
(C) Persistence/Reversal and (D) Visibility.

> [!NOTE]
> The benchmark videos are on Hugging Face:
> [Ethanliang99/RRM-Bench-1K](https://huggingface.co/datasets/Ethanliang99/RRM-Bench-1K). [`benchmark/`](benchmark) in this
> repository holds the official evaluator and, for reference, the manifests and
> labels it reads; download the videos into it with
> `huggingface-cli download Ethanliang99/RRM-Bench-1K --repo-type dataset --include "episodes/*" --local-dir benchmark`.

```bash
B=../benchmark      # from toolbox/
rewardbox bench --manifest $B/inputs/single_view.jsonl --model robometer-4b \
    --strategy fps --fps 1 --no-include-last -o runs/robometer-4b
python $B/tools/evaluate.py --predictions runs/robometer-4b/robometer-4b.json \
    --name robometer-4b -o runs/robometer-4b/eval.json
```

The evaluator also scores a predictions file written by any other code, without
RewardBox. That format, the protocol, the metric definitions, the suites and the
configuration behind each row of the paper's tables are in the
[benchmark README](benchmark/README.md); general VLMs and Jev are covered in
their [guide](toolbox/docs/guides/general_vlm_and_jev.md).

## :electric_plug: Add a Model

Subclass `rewardbox.core.base.BaseAdapter`, declare a `ModelSpec` and register
the module:

```python
class MyModelAdapter(BaseAdapter):
    def _score(self, traj):                       # Trajectory -> RewardOutput
        values = self._load().score(list(traj.primary), traj.instruction)   # official call
        return self._out(values, traj, meta={"protocol": "official uniform-16"})
```

The full template, the checklist and the test conventions are in
[Add a Model](toolbox/docs/adding_a_model.md).

## :computer: Use with Coding Agents

[`AGENTS.md`](AGENTS.md) is the entry point for coding agents (rules, recipes,
the I/O contract and troubleshooting); `CLAUDE.md` loads the same file.
[`toolbox/skills/rewardbox/`](toolbox/skills/rewardbox/SKILL.md) is a drop-in
Agent Skill and [`toolbox/llms.txt`](toolbox/llms.txt) indexes the documentation.

## :hammer_and_wrench: Development

```bash
cd toolbox
pip install -e ".[dev]"
pytest -m "not gpu and not calibration"       # offline tests; no GPU, no weights
python scripts/gen_model_docs.py --check      # model table in sync with the registry
python ../benchmark/tools/test_evaluate.py    # evaluator tests
```

## :scroll: License

The RewardBox code is released under the [Apache License 2.0](LICENSE).
The wrapped projects and their weights keep their own licenses, several of them
noncommercial or missing; read [NOTICE.md](toolbox/NOTICE.md) before
redistributing anything. The RRM-Bench-1K data follows RoboChallenge's license,
[CC BY-NC-SA 4.0](benchmark/LICENSE).

## :bookmark_tabs: Citation

```bibtex
@article{liang2026rrm,
  title   = {Robotic Reward Models in the Foundation-Model Era: A Survey and Capability Benchmark},
  author  = {Liang, Guoqiang and Lin, Yiqi and Zhou, Shangchen and Zeng, Wenzheng and
             Chen, Yanzhe and Bai, Zechen and Shou, Mike Zheng},
  year    = {2026},
  note    = {Project page: https://ethanliang99.github.io/awesome-robot-reward-model}
}
```

Please also cite the methods you run; their papers are linked in the model zoo.

## :heart: Acknowledgement

The interface is modeled on [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch).
RewardBox calls the official code of
[Robometer](https://github.com/robometer/robometer),
[TOPReward](https://github.com/TOPReward/TOPReward),
[Robo-Dopamine](https://github.com/FlagOpen/Robo-Dopamine),
[VLAC](https://github.com/InternRobotics/VLAC),
[VLAC-Cut](https://github.com/InternRobotics/VLAC-cut),
[PRIMO-R1](https://github.com/10-OASIS-01/PRIMO-R1),
[SOLE-R1](https://github.com/Philip-MIT/sole-r1-model),
[LRM](https://github.com/physical-superintelligence-lab/Large-Reward-Models),
[RynnValue](https://github.com/alibaba-damo-academy/RynnValue),
[ProcVLM](https://github.com/RUCKBReasoning/ProcVLM) and
[Robo-ValueRL](https://github.com/Open-X-Humanoid/Robo-ValueRL), and the released
weights of [RoboReward](https://huggingface.co/teetone/RoboReward-4B) and
[Flora-Rewarder](https://huggingface.co/nvidia/Flora-Rewarder).
All RRM-Bench-1K videos come from [RoboChallenge](https://robochallenge.ai): they
record policy executions on its Table30 and Table30 V2 benchmarks.
We thank the authors and the RoboChallenge team for releasing their work.

## :e-mail: Contact

Questions and bug reports: please open an issue in this repository.
