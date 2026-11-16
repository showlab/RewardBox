# General VLMs and Jev as progress evaluators

RewardBox ships four profiles that turn a general model into a progress
evaluator. They are RewardBox protocols (`implementation_kind:
rewardbox_extension`), not official reward models: the prompts in
`rewardbox/prompts/vlm_*_v1.txt` were written for the RRM-Bench-1K study,
following the way RoboPulse++ evaluates general VLMs (Sequence and Pair).

| profile | what is sent | what comes back |
|---|---|---|
| `vlm-sequence` | one request per video: every sampled observation in order | `{"progress": [...]}`, one value in [0, 100] per observation |
| `vlm-pair` | one request per adjacent pair of sampled observations (BEFORE, AFTER) | `<score>+1</score>` or `<score>-1</score>` |
| `jev-1.14-sequence` | one request per video, one five-level question per observation | option probabilities; progress = expected level × 25 |
| `jev-1.14-pair` | one request per adjacent pair, a rising / falling choice | option probabilities; the more probable option, tie = 0 |

All four sample at 1 FPS without an extra last frame by default
(`SamplingSpec(strategy="fps", fps=1.0, include_last=False)`), which is the
RRM-Bench-1K grid, and accept the optional views `left_wrist` and `right_wrist`.
Images of one observation are sent as simultaneous views in the order primary,
left_wrist, right_wrist. The model sees the task instruction, the observation
order and the images, nothing else.

## Setup

```bash
pip install -e ".[general-vlm]"        # requests + the core dependencies; no GPU, no checkout
rewardbox info vlm-sequence
```

The API key comes from an environment variable and is never accepted on the
command line or written to results:

| `api` | endpoint | default key variable | default `auth_header` |
|---|---|---|---|
| `openai` (default) | `{base_url}/chat/completions`, default `https://api.openai.com/v1` | `OPENAI_API_KEY` | `bearer` |
| `anthropic` | `{base_url}/v1/messages`, default `https://api.anthropic.com` | `ANTHROPIC_API_KEY` | `x-api-key` |
| `gemini` | `{base_url}/v1beta/models/{model}:generateContent`, default Google | `GEMINI_API_KEY` | `x-goog-api-key` |
| Jev | `{base_url}/v1/systemone`, no default: `base_url` is required | `JEV_API_KEY` | Bearer |

`openai` also covers OpenAI-compatible gateways and a local vLLM server
(`base_url=http://localhost:8000/v1`). For a gateway, set `base_url`,
`api_key_env` (the name of the variable that holds its key) and usually
`auth_header=bearer`. The Jev profiles ship without an endpoint: set `base_url`
to the API root of your own Jev access.

## Run

One video:

```bash
export OPENAI_API_KEY=...
rewardbox run vlm-sequence robot.mp4 --instruction "put the cup on the coaster" \
    --model-opt model=MODEL_ID -o out.json
```

```python
import rewardbox

with rewardbox.create_reward("vlm-sequence", api="anthropic", model="MODEL_ID",
                             extra_body={"output_config": {"effort": "medium"}}) as rm:
    out = rm("robot.mp4", "put the cup on the coaster")
out.progress        # 0-100, one value per sampled observation
out.frame_indices   # their source frames
```

RRM-Bench-1K (`B` is the benchmark directory). `bench` options are written
`PROFILE.OPTION=VALUE`; use one output directory per API model, because the
shard is named after the profile:

```bash
# Sequence
rewardbox bench --manifest $B/inputs/single_view.jsonl --model vlm-sequence \
    --strategy fps --fps 1 --no-include-last \
    --model-opt vlm-sequence.model=MODEL_ID -o runs/MODEL_ID-sequence
python $B/tools/evaluate.py --predictions runs/MODEL_ID-sequence/vlm-sequence.json \
    --name "MODEL_ID (Seq)" --value-scale 0.01 -o runs/MODEL_ID-sequence/eval.json

# Pair
rewardbox bench --manifest $B/inputs/single_view.jsonl --model vlm-pair \
    --strategy fps --fps 1 --no-include-last \
    --model-opt vlm-pair.model=MODEL_ID --model-opt vlm-pair.max_concurrency=8 \
    -o runs/MODEL_ID-pair
python $B/tools/evaluate.py --predictions runs/MODEL_ID-pair/vlm-pair.json --name "MODEL_ID (Pair)"

# Jev (model and image recipe are fixed; the endpoint is yours)
export JEV_API_KEY=...
rewardbox bench --manifest $B/inputs/single_view.jsonl --model jev-1.14-sequence \
    --strategy fps --fps 1 --no-include-last \
    --model-opt jev-1.14-sequence.base_url=https://YOUR_JEV_ENDPOINT -o runs/jev-sequence

# three-camera inputs
rewardbox bench --manifest $B/inputs/multiview.jsonl --model vlm-sequence ... -o runs/MODEL_ID-mv
python $B/tools/evaluate.py --predictions runs/MODEL_ID-mv/vlm-sequence.json --cohort multiview
```

Running the command again resumes: finished videos are skipped, and
`--retry-errors` asks again for the videos that failed. Try a handful of
manifest lines first; a full single-view run is 1,117 requests for Sequence and
19,151 for Pair, per model.

## Options

`vlm-sequence` and `vlm-pair` (`rewardbox info NAME` lists the same):

| option | default | meaning |
|---|---|---|
| `model` | required | the provider's model id |
| `api` | `openai` | wire format: `openai`, `anthropic`, `gemini` |
| `base_url` | provider | API root; a gateway or a local server |
| `api_key_env` | per `api` | name of the environment variable holding the key |
| `auth_header` | per `api` | `bearer`, `x-api-key` or `x-goog-api-key` |
| `extra_body` | `{}` | extra top-level request fields (JSON), e.g. reasoning effort or temperature |
| `max_tokens` | 16000 | sent by the `anthropic` format only, which requires it |
| `timeout_s`, `max_retries` | 600, 3 | per request; HTTP 408/409/429/5xx and network errors are retried |
| `image_format`, `jpeg_quality`, `image_max_side` | `png`, 90, none | image payload; nothing is shrunk unless you ask |
| `max_concurrency` (`vlm-pair` only) | 4 | pair requests of one video in flight at once |

Sampling parameters and reasoning effort stay at the provider default unless
set in `extra_body`, for example
`--model-opt 'vlm-sequence.extra_body={"output_config":{"effort":"medium"}}'`
(Anthropic) or `'{"generationConfig":{"thinkingConfig":{"thinkingLevel":"medium"}}}'`
(Gemini). Everything in the options, except the key, is stored in the result and
is part of the resume identity.

`jev-1.14-sequence` and `jev-1.14-pair` take `base_url` (required), `api_key_env`
(default `JEV_API_KEY`),
`timeout_s`, `max_retries` and `max_concurrency` (Pair). The model
(`jev-1.14.0`) and the image recipe (75% prescale, 280×280 black letterbox,
JPEG quality 70) are fixed.

## Reading the outputs

- **Sequence**: `progress` is the curve. An answer in the wrong format, a
  truncated answer or a refusal fails that video with the reason in the shard's
  `errors`; values are never repaired or filled in. One Markdown code fence
  around the whole JSON is accepted and recorded in `meta.markdown_fence_removed`.
- **Pair**: the answers are in `raw.transition_directions`, one per adjacent
  pair: `+1`, `-1`, `0` (a Jev tie) or `null` (an answer in the wrong format,
  counted as missing). `progress` is `0` for the first observation and then
  those judgements, with `null` written as `0`; it is not a progress curve, so
  do not compute VOC or rankings from it. `tools/evaluate.py` recognises Pair
  shards (`meta.output_kind == "transition_directions"`) and scores the
  judgements directly, without ranking metrics. A failed request (after
  retries) fails the whole video, so `--retry-errors` asks all of its pairs again.
- Requests that exceed a provider's limit (long three-camera videos at native
  resolution) fail with the provider's message. Use `image_format=jpeg` and
  `image_max_side` and report that setting; nothing is truncated, split or
  tiled automatically.

## Relation to the paper's rows

These profiles use the same prompts, 1 FPS grid, request layout, answer
parsing and evaluator as the study. A run through them is a new API run, not a
reproduction of the paper's rows: state the date and the settings when numbers
are compared with the paper. One input case differs:

- **Jev on long three-camera inputs.** A few three-camera clips exceeded the service's input limit in the study and were scored from a
  three-view mosaic. That fallback is not implemented; those clips fail here.

Provider-side model versions are not pinned by a model id; record the date and
the `reported_model` stored in each result.

## Validation status

Offline tests (`tests/test_general_vlm_jev_adapters.py`) cover the three wire
formats, the request layout, answer parsing, retries, credential handling, the
Pair layout and the benchmark path with a faked HTTP layer. The request text
and the Jev request bodies were compared with the study's request builders on
benchmark clips (identical, including the JPEG bytes). No request to a real
provider endpoint has been made with these adapters.
