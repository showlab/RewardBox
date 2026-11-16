# GVL on Gemini 3.8 Flash

Two GVL profiles call model `gemini-3.8-flash` with a non-streaming request in
Gemini's native format. You bring the API access: Google's Gemini API with your
own key by default, or a gateway of yours that serves the same format.

```text
POST {base_url}/v1beta/models/gemini-3.8-flash:generateContent
x-goog-api-key: <GEMINI_API_KEY>
Content-Type: application/json

{"contents": [{"parts": [...]}]}
```

| profile | frames sent |
|---|---|
| `gvl-gemini-3.8-flash` | 8 uniform frames, as in the upstream GVL recipe |
| `gvl-gemini-3.8-flash-dense` | every frame of the sampling grid, 1 FPS by default |

Both are `rewardbox_extension` profiles: the model, the transport and (for the
dense profile) the number of frames differ from the upstream recipe, which is
`gvl`.

## Sampling of the dense profile

The dense profile scores the whole grid the caller samples. Its default is
`strategy=fps`, `fps=1.0`, `include_last=false`, the grid RRM-Bench-1K uses.
`fps`, `interval` and `all` sampling are accepted; `num_frames` has no effect
with these strategies. The adapter does not sample a second time: every sampled
image goes into one request, shuffled once, next to the initial-scene anchor of
the GVL prompt. A 3,271-frame video at 30 FPS gives 110 images at 1 FPS or 219
at 2 FPS, plus the anchor.

Source timestamps, frame indices and file names are stored in the output
metadata and are not added to the prompt. The two profiles write separate
shards: an 8-frame result cannot be resumed as a dense run, and a dense run
with a different sampling needs a new output directory.

## Run

```bash
rewardbox doctor gvl-gemini-3.8-flash-dense --deep

read -rsp 'GEMINI_API_KEY: ' GEMINI_API_KEY
export GEMINI_API_KEY
rewardbox run gvl-gemini-3.8-flash-dense \
  /absolute/path/to/video.mp4 --instruction 'Task description' \
  -o /absolute/path/to/output.json

rewardbox bench --manifest /absolute/path/to/manifest.jsonl \
  --model gvl-gemini-3.8-flash-dense \
  --strategy fps --fps 1 --no-include-last -o /absolute/path/to/output-dir
unset GEMINI_API_KEY
```

`doctor` sends no generation request. Try a few manifest entries before a full
run.

## Options: your own endpoint

| option | default | meaning |
|---|---|---|
| `base_url` | `https://generativelanguage.googleapis.com` | API root; a trailing `/v1beta` is accepted, the OpenAI-style `/v1` path is not |
| `api_key_env` | `GEMINI_API_KEY` | name of the environment variable that holds your key |
| `auth_header` | `x-goog-api-key` | `bearer` sends `Authorization: Bearer <key>`, which most gateways expect |
| `timeout_s` | 180 | seconds per request |

A gateway that serves the Gemini native format:

```bash
export MY_GATEWAY_KEY=...
rewardbox run gvl-gemini-3.8-flash-dense video.mp4 --instruction 'Task description' \
  --model-opt base_url=https://YOUR_GATEWAY --model-opt api_key_env=MY_GATEWAY_KEY \
  --model-opt auth_header=bearer -o out.json
```

With `bench`, write the options as `gvl-gemini-3.8-flash-dense.base_url=...`.
The adapter reads only the variable named by `api_key_env`; no other key is used
as a fallback. Python callers may pass `api_key`; the CLI and `bench` do not
accept credentials as options. HTTP redirects are not followed, and the adapter
never switches to another endpoint or model on its own.

## What stays upstream and what changes

The pinned Robometer `GVL.compute_progress` still does frame extraction, JPEG
encoding, the shuffle, the initial-frame anchor, prompt construction, JSON
parsing, the percentage-to-progress conversion and the chronological remapping.
For the dense profile, upstream's `max_frames` is set to the size of the grid
for every video so that upstream does not subsample again. Short videos keep
their frames without padding. Frames the model does not return stay missing;
no value is filled in.

The transport differs from upstream: a non-streaming `generateContent` request
replaces upstream's SSE request. `contents` and `parts` are sent
unchanged, and no temperature, thinking, schema or token-limit setting is
added. Truncated or blocked responses raise an error. HTTP 429, 5xx and network
failures are retried a bounded number of times; other 4xx errors fail at once.
A request that exceeds the provider's limits fails; images are not dropped,
chunked or tiled.

Each output records the endpoint, the requested model, the API method,
the number of HTTP attempts, the model version and response id the provider
reports, token counts, and the sampled and returned frame indices with their
timestamps. The reported model version is what the endpoint states; it is not
pinned by a revision.

## Validation

Offline tests cover the request format with both authentication headers, the
unmodified upstream JPEG/shuffle/parser path, sparse output mapping, retries,
response errors, credential handling, CLI discovery, 110- and 219-image
requests and reuse of one client across videos of different length. On
2026-10-15 the dense profile scored six videos through a live Gemini-format
gateway with Bearer authentication: all 250 sampled frames came back, the
longest request had 110 frames, and the six requests took nine HTTP attempts
including retries. The default Google endpoint with `x-goog-api-key` has
offline tests only. Neither profile has a calibration record.
