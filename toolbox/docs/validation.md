# Validation status

What was checked for each adapter, and with what result. The question this page
answers is whether an adapter returns what the official code returns for the
same input. It is not a quality ranking of the methods.

## Levels

| level | meaning | where it lives |
|---|---|---|
| offline tests | the model call is faked; the real upstream pre- and post-processing runs | `tests/`, `pytest -m "not gpu and not calibration"` |
| upstream smoke | the official entry points exist at the pinned commit with the signatures the adapter assumes | `scripts/smoke_upstream.py` |
| real-weight run | the adapter ran end to end with the pinned weights; the outcome is described below | this page |
| calibration record | a real-weight output is stored and replayed as a regression test | `rewardbox/calibration/records/`, `pytest -m calibration` |

Every registered configuration has offline tests. `rewardbox list` shows
`verified=score` or `verified=score+compare` for the configurations that have a
calibration record.

Scope: during development some methods were also compared with numbers published
in papers or shipped with datasets. Those comparisons depend on evaluation
subsets and scripts that are not part of this release, and they are not reported
here. A calibration record reproduces a recorded run of the official code; it is
not a reproduction of a paper table. Results on RRM-Bench-1K are in the paper.

GPU runs used NVIDIA H200 NVL cards.

## Per method

| method | configurations | real-weight check | result | date |
|---|---|---|---|---|
| Robometer | `robometer-4b` | calibration record, `score()` and `compare()` on the synthetic fixture | replayed within tolerance; `compare(success, fail)` = 1.0, reversed = −0.99 | 2026-07 |
| | `robometer-4b-libero` | none | | |
| RoboReward | `roboreward-4b` | calibration record | replayed exactly | 2026-07 |
| | `roboreward-8b` | none | | |
| TOPReward | `topreward` | calibration record at the defaults of the Robometer baseline (Qwen3-VL-8B, 64-frame cap, 15 prefixes) | replayed within tolerance | 2026-07-30 |
| | `topreward-qwen2.5-7b` | none | | |
| Robo-Dopamine | `robodopamine-8b` | calibration record | replayed within tolerance | 2026-07 |
| | `robodopamine-v1-3b`, `-v1-8b`, `-v2-4b`, `-v2-8b` | first 21 frames of the official demo, three views and the provided goal image | ran end to end | 2026-10-11 |
| GVL | `gvl` | request to the Gemini 2.5 Flash Lite endpoint of the upstream recipe | HTTP 404: the provider no longer serves this model to new keys | 2026-08-23 |
| | `gvl-gemini-3.5-flash-lite` | one 8-frame request | 8 of 8 frames parsed | 2026-08-23 |
| | `gvl-gemini-3.8-flash-dense` | six videos through a Gemini-format gateway (`auth_header=bearer`), 250 sampled frames in total, the longest request 110 frames | all 250 frames returned | 2026-10-15 |
| | `gvl-gemini-3.8-flash` | none | | |
| VLAC | `vlac-2b`, `vlac-8b` | calibration records for `score()` and `compare()`, on the synthetic fixture and on the example images of the official repository | replayed within tolerance; see the note on `vlac-2b` | 2026-07-29 |
| VLAC-Cut | `vlac-cut` | the three examples bundled with the official repository, adapter against `scripts/run_example.py` in the same environment | 3 of 3 responses byte-identical | 2026-10-06 |
| | `vlac-cut-vllm` | two short clips against `vlac-cut` | responses differ; largest curve difference 20 percentage points | 2026-10-14 |
| PRIMO-R1 | `primo-r1-7b`, `-prefix` | model inputs for two videos against the official Decord / OpenCV preparation, on CPU | prompt, pixels, FPS, input ids, masks and grids identical | 2026-10-15 |
| Flora-Rewarder | `flora-rewarder-27b`, `-prefix` | 16 clips against the unmodified `batch_utils/terminal_reward.py` | score and generated text agree on 16 of 16; see the note | 2026-10-26 |
| SOLE-R1 | `sole-r1-8b` | calibration record through the official reward server (1 Hz, external view, `temperature=0.0`) | replayed within ±1 point | 2026-08-03 |
| LRM | `lrm-contrastive`, `-progress`, `-completion`, `-tri` | three official single-head servers with the public weights; adapter against the official client | 72 of 72 scores and raw responses equal | 2026-08-17 |
| | `lrm-completion-vllm` | 32 frames against the official server | 32 of 32 scores equal | 2026-10-14 |
| | `lrm-progress-vllm`, `lrm-contrastive-vllm` | two clips against the official server | inputs identical; outputs differ at the level described in the note | 2026-10-15 |
| RynnValue | `rynnvalue-4b`, `rynnvalue-8b` | the example video of the official repository, three runs across two server starts | bit-identical 8-point outputs for each model | 2026-08-20 |
| ProcVLM | `procvlm-2b` | the bundled close-oven demo (250 frames), adapter twice and the official entry point once | 250 of 250 parsed in each run; see the note | 2026-08-24 |
| Robo-ValueRL | `robo-valuerl-value` | two episodes of the released dataset (866 and 1,753 frames), each scored twice | bit-identical between repeats | 2026-08-23 |
| General VLM, Jev | `vlm-sequence`, `vlm-pair`, `jev-1.14-*` | none; see [guides/general_vlm_and_jev.md](guides/general_vlm_and_jev.md) | | |

## Notes

**Greedy decoding and bf16 near-ties.** Several models decide a token between
candidates whose logits differ by less than one bf16 step. A different attention
kernel, torch build or batch size then flips some tokens, although the inputs
are identical. This is why a few comparisons below are reported as a rate of
agreement and not as exact equality.

**VLAC.** The defaults (`num_frames=64`, `rich=True`, `skip=1`, `batch_size=5`,
no reference trajectory) are those of the VLAC baseline in the Robometer
repository. The synthetic fixture is far from the training distribution of
`vlac-2b`: its per-frame signal there is close to zero and `compare()` does not
prefer the successful video, so the direction assertion on the fixture skips
this model (`DIRECTION_ASSERTION_EXEMPT` in `tests/test_calibration.py`). On the
example images of the official repository both models give monotone values and
the expected `compare()` direction; those records are in
`rewardbox/calibration/records/official_examples/`.

**VLAC-Cut.** The adapter adds no difference of its own to the official script.
Against the reference outputs stored in the official repository, one of the
three examples is byte-identical; the other two diverge at a near-tie in the
decimal digit of a keypoint time and differ from there on (curve MAE 2.9 and 5.2
percentage points). The environment that produced the reference outputs is not
documented upstream. `backend="transformers"` fails under the transformers
version the official environment recipe pins and is not usable; the default
`backend="swift"` is the one validated. `vlac-cut-vllm` is an opt-in backend
that has not reached output parity with `vlac-cut`.

**Flora-Rewarder.** Decoded frames are bit-identical to the official PyAV path on
8 clips and the rendered prompt is identical. The score probabilities differ in
the third decimal (largest difference in progress 0.008), which is the same
spread two runs of the unmodified official script show. The adapter is therefore
not bit-exact, and neither is the official script.

**SOLE-R1.** The official client and server default to `temperature=1.0`, so
outputs are stochastic; the calibration record sets `temperature=0.0`. The 19
example episodes of the official repository were also regenerated three times
with the official RewardGen entry point: the sampling grid matched in every
episode, and the distance between two of our runs (mean MAE 8.1) is about the
same as the distance to the released outputs (mean MAE 8.9).

**LRM.** The public weights are three full models. The official one-backbone
`tri` loader expects PEFT adapters, so `lrm-tri` calls three single-head servers
and applies the fixed fusion of `config/tri.yaml`. For the vLLM progress and
contrastive backends the
prompt token ids, image grids and pixel values are identical to the official
server's. On two clips (239 frames, 237 adjacent pairs) the vLLM progress head
disagrees with the official server on 13 to 23 frames and the contrastive head
on 52 to 58 pairs. Re-running the official code with a different attention
kernel or torch build disagrees with itself on 22 to 31 frames and 51 to 56
pairs. The vLLM backends are deterministic for a fixed engine configuration but
not across batch sizes, which is why engine options are part of the resume
identity.

**ProcVLM.** Of the 250 per-frame records, 246 are identical between the two
adapter runs and 248 between the official entry point and the adapter (progress
MAE 0.07). The vLLM path is not bit-deterministic at temperature 0.

**Robo-ValueRL.** The adapter returns the `remain_time` output of the released
history critic, decoded and interpolated with the official functions. The
checkpoint also contains value and return-to-go heads that the released model
code does not build; they are not used. The server builds `Pi0Config` from the
fields of the pinned `all_value_function_training` entry, because importing
`openpi.training.config` loads a data file that is not in the repository.

**RynnValue.** The public checkpoints need embodiment and camera metadata
(`use_meta=true`). The adapter takes them from `Trajectory.meta` or from model
options and stops before the request when they are missing. The default profile
is the released RBM-EVAL-OOD command (`pred_slot_isolated_eager` attention,
`use_frame_steps=false`); the launcher's legacy eager attention is the separate
profile `paper-eager`.

**GVL.** Upstream publishes one GVL command (the confusion-matrix recipe: 8
uniform frames, Gemini 2.5 Flash Lite). `gvl` keeps that recipe.
`gvl-gemini-3.5-flash-lite` and the `gvl-gemini-3.8-flash` profiles change the provider
model and are registered under their own names. Provider models are not pinned
by a revision; each result stores the model version the provider reports.

## Environments

| environment | Python | key packages | lock or input file |
|---|---|---|---|
| shared (`.venv`) | 3.11.15 | torch 2.8.0, transformers 4.57.1, vLLM 0.11.0, flash-attn 2.8.3 | `locks/shared-py311-linux-x86_64.lock` |
| VLAC (`.venv-vlac`) | 3.10.12 | torch 2.8.0, transformers 4.51.3, ms-swift 3.3.0, sentencepiece 0.1.99 | `locks/vlac-py310-requirements.in` |
| VLAC-Cut (`.venv-vlac-cut`) | 3.11.15 | torch 2.6.0, transformers 4.57.6, ms-swift 3.12.4 | `locks/vlac-cut-py311-requirements.in` |
| PRIMO-R1 (`.venv-primo`) | 3.11 | torch 2.5.1, vLLM 0.7.2, PRIMO's vendored transformers 4.49.0.dev0 | `locks/primo-py311-linux-x86_64.lock` |
| Flora-Rewarder (`.venv-flora`) | 3.12.13 | vLLM 0.24.0, torch 2.11.0, transformers 5.8.1 | `locks/flora-py312-linux-x86_64.lock` |
| SOLE-R1 server | 3.10.20 | torch 2.9.0, vLLM 0.12.0, transformers 4.57.3 | `locks/sole-r1-server-py310-linux-x86_64.lock` |
| LRM server | 3.10.20 | torch 2.4.1+cu124, transformers 4.57.1, accelerate 1.12.0 | `locks/lrm-server-py310-linux-x86_64.lock` |
| RynnValue server | 3.10.20 | torch 2.8.0, transformers 4.57.6 | upstream `robometer/uv.lock` |
| ProcVLM server | 3.10.20 | torch 2.10.0, vLLM 0.18.1, transformers 4.57.3 | upstream `uv.lock` |
| Robo-ValueRL server | 3.11.15 | torch 2.7.1, transformers 4.53.2 with the official replacement files | upstream `robo_valuerl/requirements.txt` |

Setup commands are in [installation.md](installation.md), together with the
places where these environments differ from what upstream declares.
