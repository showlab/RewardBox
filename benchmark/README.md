# RRM-Bench-1K

A benchmark for **robot reward / progress models on real robot executions**,
labelled with how far the *task state* has progressed, not how much time has
passed. It is released together with the [RewardBox](https://github.com/showlab/RewardBox) toolbox (RRM-Hub).

- **1,117 single-view videos** of real-robot policy executions across 52 tasks
  on five robot embodiments: ARX-5, UR5 and Franka (single-arm), Cobot Magic
  ALOHA and DOS-W1 (dual-arm). The executions are closed-loop runs of learned
  policies recorded on RoboChallenge Table30 and Table30 V2. Clips with three
  synchronized cameras also come as three-camera inputs.
- **Task-state labels**: a per-clip progress curve compiled from human event marks
  over each task's rubric. Credit rises as effects are achieved and
  falls when progress is lost: a failed attempt, a dropped or misplaced object, or
  an achieved state that is undone.
- **Direction metrics** on the non-flat 1 FPS transitions of the task-state
  curves, plus trajectory-level ranking metrics.
- **Four contrast suites** that probe where evaluators fail: (A) Action–Effect,
  (B) Semantic/Ordered Constraint, (C) Persistence/Reversal and (D) Visibility
  (each reduced-evidence video is paired with a sufficient view of the same clip,
  under identical labels), with success and failure controls.

## Terms

The paper describes the data as tasks, executions, videos and scope
variants; the files add the clip and keep "episode" in field names.

- An *execution* is one attempt of a policy at a task (`episode_uid`, `group_id`).
- A *clip* is an execution, or a scope variant of it, with all its camera views.
  Its `condition` is `main` for the whole execution (trimmed, sometimes sped up)
  and `unit_split`, `unit_window` or `object_split` for a *scope variant*, which
  keeps one unit, a window of units or one object of the execution under a
  narrower instruction.
- A *video* is one camera view of a clip.

All clips of one execution share a `group_id`; aggregate or resample by
execution, not by video.

## Suites

Suites overlap (a video can be in several). The slugs are the `categories`
values of the label files.

| slug | paper | what it tests |
|---|---|---|
| `action_effect` | (A) Action–Effect | the motion looks like success but the effect is absent |
| `semantic_ordered` | (B) Semantic/Ordered Constraint | right object, right destination, right order |
| `persistence_reversal` | (C) Persistence/Reversal | achieved states that are later lost or undone |
| `occlusion_viewpoint` | (D) Visibility | the camera gives partial or insufficient evidence of the task state |
| `control` | controls | clean successes and obvious failures |
| `no_task_view` | stress condition | a camera that shows no task evidence |

`action_effect` is assigned per execution: a clip belongs to it when the robot
completes a success-like motion whose effect is absent and one camera shows that
absence. The suite is scored on the videos of its clips whose camera shows the
effect (`action_effect_scored`); its other videos give partial or insufficient
evidence and are scored under `occlusion_viewpoint`.

`occlusion_viewpoint` contains every view labelled `P` (partial evidence) or `X`
(insufficient evidence; excluding no-task `X` views), plus the `V` views of clips
whose suite is visibility-abstention. Each such video has a `result_case_id`:
another camera of the same clip labelled `V` (sufficient evidence) with the
identical GT curve. A clip with no single camera that shows everything has
`result_view: "none"`.

## Files

```
inputs/
  single_view.jsonl            model inputs: id, video (relative), instruction, split (= condition), group_id
  multiview.jsonl              three-camera inputs, one per clip (video = {primary, left_wrist, right_wrist})
  ground_truth.json            official labels per single-view id (fields below)
  multiview_ground_truth.json  official labels per multiview id
  events.jsonl                 event marks behind the labels: clip, time, English description (fields below)
  eval_grid_1fps.json          the 1 FPS evaluation frames of every id
episodes/<task>/<clip>/
  videos/*.mp4                 one file per camera (30 fps)
  source/frame_map.csv         output frame -> source frame in the original recording, per camera
tools/
  evaluate.py                  official evaluator (numpy + scipy)
  test_evaluate.py             evaluator tests on the released labels (no media, no models)
metadata.json                  counts, suites, sources
```

The videos are in the Hugging Face dataset
[Ethanliang99/RRM-Bench-1K](https://huggingface.co/datasets/Ethanliang99/RRM-Bench-1K), which holds this directory together with
its `episodes/` tree; the RewardBox git repository does not include them. In a
RewardBox checkout, download them into this directory so that the manifest paths
resolve:

```bash
huggingface-cli download Ethanliang99/RRM-Bench-1K --repo-type dataset \
    --include "episodes/*" --local-dir benchmark    # from the repository root
```

The labels, manifests, frame maps and tools kept here are enough to read the
format and to score saved predictions.

`inputs/*.jsonl` are RewardBox manifests: `rewardbox bench --manifest
inputs/single_view.jsonl ...` runs any registered model on them directly (paths
resolve relative to the manifest file). RewardBox is the package in
[`../toolbox`](../toolbox).

### `ground_truth.json` (one entry per video id)

| field | meaning |
|---|---|
| `curve` | the label: `[{time, value}]`, output-video seconds → task progress in [0, 1]; linear between knots |
| `gt_fraction`, `gt_total`, `max_score` | endpoint of the curve and the rubric points behind it |
| `quality` | endpoint grade: `successful` (endpoint 1, full fulfillment in the paper), `suboptimal` (between, partial), `failure` (0, zero); used by the ranking metrics |
| `categories` | suite slugs above; select suites by this field |
| `human_suite` | the finer annotation-time suite of the execution, kept for reference. It predates the per-execution assignment of (A) Action–Effect |
| `action_effect_scored` | (A) Action–Effect is scored on this video |
| `view`, `camera`, `view_label` | camera file, `fixed`/`wrist`, `V`/`P`/`X` (sufficient / partial / insufficient evidence for judging the task) |
| `main_camera` | the one camera per whole-execution (`main`) clip used by the main-clip ranking scope |
| `no_task_view` | the camera shows no task-relevant motion |
| `result_view`, `result_case_id` | the paired `V` camera of the same clip (identical GT), or `none` |
| `clip_id`, `episode_uid`, `group_id`, `task`, `condition` | identity and grouping |
| `frames`, `fps` | video length in frames and frame rate (30) |
| `label`, `gt_policy` | `state_v2`; the policy name keeps a historical `_draft` suffix, the label is final |
| `frame_map_path` | the clip's frame map (relative to this directory) |

### `events.jsonl` (one line per clip event)

The event marks from which the curves were compiled, in reduced form: when each
event happens and what happens, in English. They are for reading the labels.
No metric uses them, and they must not be given to a model as input.

| field | meaning |
|---|---|
| `clip_id` | the clip, as in `ground_truth.json`; all cameras of a clip share its events |
| `event_id` | id of the mark. One physical event keeps its id in every scope variant that contains it |
| `time_sec` | `[start, end]` in the clip's video seconds, the clock of `curve`; the two are equal for an instant mark |
| `description_en` | what happens, at most 12 words |
| `source` | `annotator`: rendered from the annotator's sentence; `frames`: the mark had no sentence, so the text was written from the task unit and the frames |

Reading notes:

- Every clip has events except two zero-progress clips.
- **Times.** A completion (a pick, a place, another achieved step) is timed when
  its outcome is confirmed, which can be 1–2 s after the object leaves the table
  or the gripper, and not at that instant. A failed attempt is timed at its closest
  approach to success. Events are not always knots of the curve: marks that carry
  no credit (the start of an attempt, a reach) only shape it.
- **Ordinals** ("second plate", "fifth item") count objects in the order they are
  handled over the whole execution. They include an object that was dropped or was
  already in place, and a scope variant keeps the ordinal of its execution.
- **Positions** (left / right, near / far) are given from the robot's side.
- **How the text was made.** The marks and their sentences are the annotators'.
  The English descriptions were written from those sentences and the video
  frames, one execution at a time, and every description was checked against the
  frames. Where the frames show something other than the annotator's sentence,
  the description follows the frames. The
  descriptions are not labels: the curves were compiled before the English was
  written and were not changed by it.

The rest of the per-clip annotation records (task units, event types and the
compile log) is not part of this release, and neither is the action-history
reference. The paper's analyses that read them (fall causes, scope consistency,
the comparison with the action-history reference) cannot be recomputed from
these files.

## Evaluation protocol

**Grid.** Every video is evaluated at source frames `0, 30, 60, … < frames`
(1 FPS, no extra endpoint; `inputs/eval_grid_1fps.json`). This equals RewardBox
`SamplingSpec(strategy="fps", fps=1.0, include_last=False)`.

**Direction (Rising / Falling).** The GT direction of each adjacent grid
transition is the sign of the difference of the GT curve at the two frames
(|Δ| ≤ 1e-9 is flat and excluded). A curve prediction's direction is the sign of
its value difference on the same frames; a predicted tie counts as wrong. An
evaluator that compares observation pairs may instead judge each transition
directly. All
transitions are pooled; report Rising and Falling precision/recall/F1,
**Macro-F1** (mean of the two F1) and accuracy. Transitions without a prediction
are excluded from P/R/F1 and reported as `missing`. Because most transitions
rise, an always-rising predictor already reaches a moderate Macro-F1; the falling
class is where evaluators separate. Curves whose native points do not cover the grid can
be read with `--mode interpolate` (linear, inside the native range only); such
rows are marked † and are not native 1 FPS predictions.

**By suite and paired evidence.** The same metrics per suite and for the two
control groups (`control_clean_success`, `control_obvious_failure`), each next to
the always-rising Macro-F1 on the same transitions. (A) Action–Effect uses its
effect-showing videos; the other suites use every member video. For (D)
Visibility, each P/X video is also compared with its paired V view on identical GT
transitions, with a 95% bootstrap CI over episodes (2,000 resamples, seed
20261018, episodes in `ground_truth.json` order).

**Ranking.** On the native output points (no interpolation):
VOC = mean over successful-endpoint videos of the Pearson r between time and
predicted value (undefined → 0); Kendall τ_a = class-balanced ranking of the last
predicted value, failure < suboptimal < successful, within each task (ties 0),
averaged over tasks; Succ–Fail Diff = mean last value of successful minus failure
videos within each task, averaged over tasks (raw units; a fraction for
percent/fraction outputs; an extra of this evaluator, not reported in the paper).
Reported on all single views and on one main camera per main clip. A time-only
reference reaches a perfect VOC:
VOC on successful trajectories measures temporal order, not task state.

## Evaluate a model

With RewardBox (any registered configuration):

```bash
# on the 1 FPS grid itself (exact mode), e.g. a model that accepts any frame grid
rewardbox bench --manifest inputs/single_view.jsonl --model MODEL \
    --strategy fps --fps 1 --no-include-last -o runs/MODEL
# or with the model's own protocol (see `rewardbox info MODEL`), then evaluate with --mode interpolate
python tools/evaluate.py --predictions runs/MODEL/MODEL.json --name MODEL -o runs/MODEL/eval.json
```

A general VLM or Jev: RewardBox's `vlm-sequence`, `vlm-pair`, `jev-1.14-sequence`
and `jev-1.14-pair` profiles run on the same manifests ([guide](../toolbox/docs/guides/general_vlm_and_jev.md)). The evaluator reads a Pair shard as
per-transition judgements automatically.

Your own model: write one JSON line per video id and run the evaluator.

```json
{"id": "table30_v2_2945_..._main__videos_front", "frame_indices": [0, 30, 60], "progress": [0.0, 0.1, 0.4]}
{"id": "table30_v2_2945_..._main__videos_left", "directions": [1, 1, -1, 0]}
```

```bash
python tools/evaluate.py --predictions my_model.jsonl --name my-model [--mode interpolate] \
    [--value-scale 0.01] [--cohort multiview] -o my_model.eval.json
```

`frame_indices` are source frames of the video (0-based); `progress` is on any
monotone-meaningful scale (direction and Kendall are scale-free; VOC is Pearson;
pass `--value-scale` only to express Succ–Fail Diff as a fraction).
`directions` has one entry per grid transition (`len(grid) - 1`), each `-1`, `0`,
`+1` or `null` (missing). Use the instruction from `inputs/single_view.jsonl`
unchanged (task scope only, no hints about the outcome).

## Evaluators of the paper

The rows of the paper's Table II and the RewardBox configuration behind each.
Family is the method's semantic family in the paper; † rows are evaluated with
`--mode interpolate`, the others on the 1 FPS grid itself. Table S2 of the paper
gives the input of one query and the native output of each row.

| Table II row | configuration | family |
|---|---|---|
| ProcVLM-2B† | `procvlm-2b` | Absolute Progress |
| VLAC-2B†, VLAC-8B† | `vlac-2b`, `vlac-8b` | Hybrid |
| Robo-Dopamine-GRM-3B, -8B | `robodopamine-v1-3b`, `robodopamine-v1-8b` | Relative |
| Robo-Dopamine-GRM-2.0-4B-Preview, -8B-Preview | `robodopamine-v2-4b`, `robodopamine-v2-8b` | Relative |
| Robo-ValueRL‡ | `robo-valuerl-value` | Temporal Value |
| Robometer-4B† | `robometer-4b` | Hybrid |
| RoboReward-4B† | `roboreward-4b` | Outcome |
| RynnValue-4B†, RynnValue-8B† | `rynnvalue-4b`, `rynnvalue-8b` | Temporal Value |
| PRIMO-R1-7B† | `primo-r1-7b-prefix` | Hybrid |
| LRM-8B (Contrastive)§ | `lrm-contrastive` | Hybrid |
| LRM-8B (Progress) | `lrm-progress` | Hybrid |
| SOLE-R1-8B | `sole-r1-8b` | Absolute Progress |
| TOPReward (Qwen3-VL-8B)† | `topreward` | Absolute Progress |
| Flora-Rewarder-27B | `flora-rewarder-27b-prefix` | Hybrid |
| VLAC-Cut-30B-A3B | `vlac-cut` | Hybrid |
| GVL (Gemini 3.8 Flash) | `gvl-gemini-3.8-flash-dense` | Absolute Progress |
| general-purpose VLMs, Pair and Sequence | `vlm-pair`, `vlm-sequence` | |
| Jev 1.14, Pair and Sequence | `jev-1.14-pair`, `jev-1.14-sequence` | |

Two kinds of rows use inputs that this release does not include:

- **‡ Multiview rows** (Robo-ValueRL and the "Multiview" rows of Robo-Dopamine,
  SOLE-R1 and Flora-Rewarder). The paper places each video's camera in the
  primary slot and the other cameras of its execution in the remaining slots,
  for every video whose execution has more than one camera.
  `inputs/multiview.jsonl` holds the clips with three synchronized cameras, one input per clip, which the paper uses for the
  three-view queries of the general-purpose VLMs and Jev.
- **§ LRM-8B (Contrastive)** is scored in the paper on adjacent source frames,
  not on the 1 FPS grid that `tools/evaluate.py` scores.

## Results

Results of the evaluators studied with this benchmark are reported in the paper
and on the [project page](https://ethanliang99.github.io/awesome-robot-reward-model).
Read every number next to the two analytic references that need no model: an
always-rising predictor, which never predicts a fall, and normalized elapsed time,
which reaches a perfect VOC but does not rank endpoints. Coverage can differ between evaluators (`n` and
`missing`); never rank rows on different supports without reading them.

`tools/evaluate.py` reproduces the paper's direction, ranking and per-suite tables
from the authors' saved model outputs. It does not compute the paper's
endpoint-selection and false-credit shares.

## Limitations

- Labels follow one platform (RoboChallenge Table30 and Table30 V2, five robot
  embodiments) and its tasks; endpoint grades are GT-endpoint proxies, not
  independent success labels.
- View labels (`V`/`P`/`X`) were not all confirmed video by video; part of them
  were assigned per camera group.
- Event times come from human marks snapped to output frames; transitions inside
  a marked interval follow the compiled linear ramp.
- Exposure of evaluated models to RoboChallenge data during training was not
  checked and cannot be excluded.
- Media are trusted fixed files; content hashes were not recomputed for this release.

## License

The data (videos, frame maps, labels, and manifests) follows RoboChallenge's
license, [CC BY-NC-SA 4.0](LICENSE): non-commercial use, with attribution, and
derivatives shared under the same terms. The code in `tools/` is released under
the Apache License 2.0, like the rest of RewardBox.

## Acknowledgement

All videos in RRM-Bench-1K come from [RoboChallenge](https://robochallenge.ai).
They record policy executions on its Table30 and Table30 V2 real-robot
benchmarks. We thank the RoboChallenge team for building the platform and for
making its runs public.

## Citation

```bibtex
@article{liang2026rrm,
  title   = {Robotic Reward Models in the Foundation-Model Era},
  author  = {Liang, Guoqiang and Lin, Yiqi and Zhou, Shangchen and
             Zeng, Wenzheng and Chen, Yanzhe and Bai, Zechen and
             Shou, Mike Zheng},
  journal = {arXiv preprint arXiv:2611.20741},
  year    = {2026},
  url     = {https://arxiv.org/abs/2611.20741}
}
```

Please also cite RoboChallenge, the source of the videos:

```bibtex
@article{yakefu2025robochallenge,
  title   = {RoboChallenge: Large-scale Real-robot Evaluation of Embodied Policies},
  author  = {Yakefu, Adina and Xie, Bin and Xu, Chongyang and others},
  journal = {arXiv preprint arXiv:2510.17950},
  year    = {2025}
}
```
