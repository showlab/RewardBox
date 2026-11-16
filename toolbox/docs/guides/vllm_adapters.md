# RewardBox vLLM adapters

These opt-in profiles use the normal `create_reward(...).score(Trajectory)` interface:

| Profile | Input and native output | Submission |
| --- | --- | --- |
| `vlac-cut-vllm` | Native 2 Hz video input; signed, unclamped percentage curve | One video request per score |
| `lrm-completion-vllm` | Each supplied frame; binary 0/1 completion | Bounded batches of frame requests |
| `lrm-progress-vllm` | Each supplied frame; native progress in [0, 1] | Bounded batches of frame requests |
| `lrm-contrastive-vllm` | Adjacent supplied frames; first-frame zero, then -1/0/+1 | Bounded batches of pair requests |

The original `vlac-cut`, `lrm-completion`, `lrm-progress`, `lrm-contrastive`, and
`lrm-tri` profiles retain their original backends. No default is switched. There
is no new vLLM tri profile: the published tri weights are three dedicated full
models, and selecting one checkpoint for three heads would be incorrect.

The vLLM profiles are labeled `rewardbox_extension`. They keep the native
prompts, sampling and parsing and replace the official execution backend. None
of them has a calibration record; how close each one is to the official backend
is in [validation.md](../validation.md).

## API

```python
import rewardbox

rm = rewardbox.create_reward(
    "lrm-completion-vllm",
    batch_size=32,
    gpu_memory_utilization=0.7,
    max_model_len=8192,
)
try:
    output = rm.score(trajectory)
finally:
    rm.close()
```

`trajectory` supplies RGB frames and the task instruction through the normal
RewardBox interface. Select the assigned GPU with `CUDA_VISIBLE_DEVICES` before
starting Python. Model construction is lazy: creating the adapter does not
load a processor, model, CUDA runtime or vLLM engine. `close()` shuts down only
the engine created by this adapter.

Both adapter families share these options:

| Option | LRM default | VLAC-Cut default |
| --- | --- | --- |
| `batch_size` | 32 | 1 |
| `gpu_memory_utilization` | 0.7 | 0.7 |
| `max_model_len` | 8192 | 32768 |
| `max_num_batched_tokens` | 8192 | 8192 |
| `tensor_parallel_size` | 1 | 1 |
| `dtype` | bfloat16 | bfloat16 |
| `seed` | 0 | 0 |
| `enforce_eager` | false | false |

All requests use greedy generation. LRM keeps the original token limits:
completion/progress 128, contrastive 512. VLAC-Cut defaults to 1024, matching its
existing adapter. Context overflow fails instead of dropping frames. Chunked
prefill is enabled to bound scheduler/initialization memory.

The memory fraction refers to total device memory, including model and KV
cache, rather than a fraction of currently free memory. Lower it on a GPU that
other processes use.

For a short VLAC-Cut test run, `skip_mm_profiling=True` can skip vLLM's
large synthetic video during startup. This is disabled by default and requires
an explicit `max_video_frames` bound. The adapter rejects larger inputs,
including native odd-frame padding, instead of dropping frames. The caller
must leave room for the real vision encoder's activations; the KV allocation
remains within the configured engine budget.

`model_path` accepts an explicit local checkpoint directory. The defaults
resolve the pinned local snapshots, including RewardBox's `.hf/hub` cache;
these adapters do not download weights. LRM `model_path`
must point to the selected head directory, not the repository's three-head root.

## Native semantics

LRM submits independent requests in supplied frame order. It does not combine
multiple frames into one completion/progress prompt. Contrastive pairs remain
`(frame[i-1], frame[i])`; batching never crosses trajectory boundaries. Progress
supports the original `include_initial_image` and `include_goal_image` options,
and keeps the original goal/initial/current image order. Native fallback
parsing is retained; raw responses and generation finish reasons are saved.

VLAC-Cut inherits the existing adapter's official selection, JPEG quality 95
boundary, prompt builder, keypoint parser, and both official curve alignments.
Its private copy of `qwen-vl-utils` uses Swift's native visual budgets without
changing global environment settings. Already selected frames are passed with
`do_sample_frames=False` and `do_resize=False`; the backend adds no resampling.
The model's time axis stays at 2 Hz and uses the existing time-scale conversion.

For an already sampled 2 Hz trajectory:

```python
rm = rewardbox.create_reward(
    "vlac-cut-vllm",
    frame_selection="as_given",
    sample_hz=2.0,
    curve_alignment="vpb_index",
    gpu_memory_utilization=0.7,
    vision_preprocess_python=".venv-vlac-cut/bin/python",
)
```

For full dense source frames with `Trajectory.fps`, keep
`frame_selection="official"` to reuse the official 2 Hz selection.

`vision_preprocess_python` runs the official frame preparation in a persistent
CPU worker using the specified interpreter, with `CUDA_VISIBLE_DEVICES=''`.
It separates native Torch/torchvision resize behavior from vLLM's Torch runtime.
This matters for strict comparison: the local Torch 2.6 and 2.8 resize paths
differed by one gray level in a small number of pixels on one test clip.
If this option is omitted, preprocessing runs in the vLLM process and
metadata explicitly records that choice. The CPU worker is created lazily,
does not load model weights, and closes with its adapter.

## Results

Output metadata records checkpoint paths/revisions, actual runtime versions,
engine options, greedy generation, native reference protocol, token counts,
`n_inference_requests`, and `n_backend_calls`. The last field counts calls to
`LLM.generate` (batches), not individual frame requests. An LRM trajectory with
65 frames and `batch_size=32` has 65 requests and 3 backend calls.

Multimodal requests carry per-request UUIDs, so vLLM does not hash pixels and
does not reuse its multimodal cache between requests. For a benchmark, use
`rewardbox bench` with a fresh output directory for each model profile; a shard
written by the official HTTP or Swift backend cannot be resumed under a vLLM
profile.

## Validation

Run the offline tests without exposing GPUs:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q \
  tests/test_vllm_adapters.py tests/test_lrm_adapter.py tests/test_vlac_cut_adapter.py
```

The tests compare LRM messages against recorded execution of the official
server functions, verify batching and frame/pair order, and compare VLAC-Cut CPU
preprocessing with Swift's native constants, including odd-frame padding and
large-frame resize behavior. Heavy runtimes remain lazy during registry
discovery and adapter creation.

Measured with real weights (2026-10-14 and 2026-10-15, on GPUs shared with other
jobs, so the timings are indicative):

| profile | agreement with the official backend | speed |
|---|---|---|
| `lrm-completion-vllm` | 32 of 32 frame scores equal | about 8 times the official serial Transformers server at batch 8; batch 8 about 3.8 times batch 1 |
| `lrm-progress-vllm`, `lrm-contrastive-vllm` | identical inputs; outputs differ on some frames, as much as the official code differs from itself across attention kernels or torch builds | progress: about 30 frames/s against about 2; contrastive: 6 to 12 pairs/s against 0.1 to 0.2 |
| `vlac-cut-vllm` | not reached: both responses differ, largest curve difference 20 percentage points | about 8 times `vlac-cut` |

Details and counts are in [validation.md](../validation.md). `vlac-cut-vllm`
reproduces the token ids, video grids and bf16 vision inputs of the official
path on the tested clip and still generates a different response; the cause is
not known. Use `vlac-cut` when the official numbers matter.
