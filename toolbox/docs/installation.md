# Installation and runtimes

RewardBox separates **what a model needs** into three things, all reported by
`rewardbox doctor NAME`:

1. Python dependencies (in the environment that executes the model),
2. the pinned official source the adapter calls (`rewardbox fetch-upstream NAME`),
3. for some methods, the authors' model server running in its own environment.

Run every interpreter by absolute path (`.venv/bin/python ...`) instead of
`source .venv/bin/activate`: activation leaks across tools and an agent's
shell does not keep it between commands.

## 1. Core (no GPU)

```bash
git clone https://github.com/showlab/RewardBox.git && cd RewardBox/toolbox
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/rewardbox list
```

This is enough for the registry, the CLI, metrics, manifests and benchmark
orchestration. Weights are downloaded from the Hugging Face Hub on first use;
set `HF_HOME` to choose the cache.

## 2. Pinned upstream sources

```bash
rewardbox fetch-upstream --list              # table: key, commit, models, found/missing
rewardbox fetch-upstream robometer-4b        # by model name ...
rewardbox fetch-upstream robodopamine lrm    # ... or upstream key
rewardbox fetch-upstream all --dest /data/rewardbox-upstream
export REWARDBOX_UPSTREAM_DIR=/data/rewardbox-upstream
```

Clones are blobless, skip Git LFS media, and are verified (pinned commit,
clean tree, marker file). An existing directory is never modified.
Search order for a checkout: the adapter's `REWARDBOX_<NAME>_ROOT` override,
`$REWARDBOX_UPSTREAM_DIR/<dir>`, `<source clone>/reward_model_code/<dir>`,
`~/.cache/rewardbox/upstream/<dir>`.

## 3. Shared in-process environment (Python 3.11)

TOPReward, RoboReward, Robometer, Robo-Dopamine, GVL and the `*-vllm` backends
run inside the RewardBox process. The exact verified environment
(torch 2.8.0, transformers 4.57.1, vLLM 0.11.0, CUDA 12.8) is a lock file:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --override locks/shared-overrides.txt \
    -r locks/shared-py311-linux-x86_64.lock          # --override is required
uv pip install --python .venv/bin/python --no-deps -e .
# TOPReward's official client hard-codes flash_attention_2:
uv pip install --python .venv/bin/python \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
rewardbox fetch-upstream topreward robometer robodopamine
```

`pip install -e ".[models]"` gives coarse version ranges instead of the lock;
use it for exploration, not for reproducing numbers. Importing
`robometer.utils.setup_utils` triggers Unsloth's global monkeypatching of
torch/transformers; `rewardbox bench --isolate` runs each model in its own
subprocess to contain such side effects.

### Hosted API (GVL)

```bash
read -rsp 'GEMINI_API_KEY: ' GEMINI_API_KEY; export GEMINI_API_KEY   # every gvl profile
```

Credentials are read from the environment only; the CLI and `bench` reject
`--model-opt api_key=...` so keys never land in shell history, worker argv or
result files. The frozen `gvl` profile keeps the upstream Gemini 2.5 Flash Lite
recipe, which the provider no longer serves to new keys; use
`gvl-gemini-3.5-flash-lite` or a `gvl-gemini-3.8-flash` profile to actually run GVL.
The `gvl-gemini-3.8-flash*` profiles can also call a Gemini-format gateway of
your own through `base_url`, `api_key_env` and `auth_header`
([guide](guides/gvl_gemini_http.md)).

### Hosted API (general VLMs and Jev)

```bash
pip install -e ".[general-vlm]"
export OPENAI_API_KEY=...        # vlm-sequence / vlm-pair with api=openai (default)
export ANTHROPIC_API_KEY=...     # api=anthropic
export GEMINI_API_KEY=...        # api=gemini
export JEV_API_KEY=...           # jev-1.14-sequence / jev-1.14-pair; also pass base_url
```

`jev-*` have no default endpoint: pass `base_url`, the API root of your own Jev
access. `vlm-*` need the model id (`--model-opt model=ID`); a gateway or a local
OpenAI-compatible server is selected with `base_url`, `api_key_env` and
`auth_header`. Protocols, options and evaluation:
[guides/general_vlm_and_jev.md](guides/general_vlm_and_jev.md).

## 4. Isolated workers

These official stacks conflict with the shared environment, so they get their
own interpreter with RewardBox installed `--no-deps`. Run Python scripts with
that interpreter, or let `rewardbox bench` dispatch to it with
`--worker-python MODEL=PATH`.

| model | environment | setup |
|---|---|---|
| `vlac-2b`, `vlac-8b` | `.venv-vlac` (Python 3.10, transformers 4.51.3, ms-swift 3.3) | `uv venv --python 3.10 .venv-vlac`; `uv pip install --python .venv-vlac/bin/python -r locks/vlac-py310-requirements.in`; flash-attn 2.8.3 cp310 wheel; `-e .`; `--no-deps -e reward_model_code/VLAC` |
| `vlac-cut` | `.venv-vlac-cut` (Python 3.11, torch 2.6, transformers 4.57.6, ms-swift 3.12.4) | `uv venv --python 3.11 .venv-vlac-cut`; `-r locks/vlac-cut-py311-requirements.in`; `--no-deps -e .`; weights `InternRobotics/VLAC-Cut@2a809cb` (~62 GB bf16) |
| `primo-r1-7b[-prefix]` | `.venv-primo` (Python 3.11, vLLM 0.7.2, PRIMO's vendored Transformers) | `rewardbox fetch-upstream primo`; `-r locks/primo-py311-requirements.in` (installs `./reward_model_code/PRIMO-R1/...` editable); `--no-deps -e .` |
| `flora-rewarder-27b[-prefix]` | `.venv-flora` (Python 3.12, vLLM 0.24, transformers 5.8) | `uv venv --python 3.12 .venv-flora`; `-r locks/flora-py312-requirements.in`; `--no-deps -e .`; weights `nvidia/Flora-Rewarder@031299e` (noncommercial license) |

```bash
rewardbox bench --manifest data.jsonl --model vlac-2b --num-frames 64 -o runs/ \
    --worker-python vlac-2b=.venv-vlac/bin/python
```

The `*-vllm` variants (`vlac-cut-vllm`, `lrm-*-vllm`) are opt-in RewardBox vLLM
backends that run in the shared environment; they are separate model names
because their numerics are not bit-identical to the official runtime (see
[guides/vllm_adapters.md](guides/vllm_adapters.md) and [validation.md](validation.md)).

## 5. Official model servers

RewardBox only sends requests; the authors' inference stack runs as a server
in its own pinned environment. Each launcher verifies the upstream commit, the
weight revision and the runtime before serving, and writes a provenance JSON
that you pass back to the client (`server_provenance_file`). Without it,
depending on the adapter, results are marked as having an unverified remote
identity (SOLE-R1, LRM) or the client refuses to score (RynnValue).

| model | launcher | client endpoint |
|---|---|---|
| `sole-r1-8b` | `scripts/run_sole_r1_server.py` (creates the upstream Python 3.10 runtime from `locks/sole-r1-server-py310-linux-x86_64.lock`) | `--model-opt host=...` or `REWARDBOX_SOLE_R1_HOST` (port 8001) |
| `lrm-progress` / `-completion` / `-contrastive` | `scripts/run_lrm_server.py --mode progress --server-python <py3.10 env> --weights-root <snapshot>` | `server_url` or `REWARDBOX_LRM_SERVER_URL` |
| `lrm-tri` | three single-head servers (the public weights are full models, not PEFT adapters) | `contrastive_url`, `progress_url`, `completion_url` |
| `rynnvalue-4b`, `rynnvalue-8b` | `scripts/run_rynnvalue_server.py --server-python reward_model_code/RynnValue/robometer/.venv/bin/python --model-size 4b --model-path <snapshot>` | `server_url` or `REWARDBOX_RYNNVALUE_SERVER_URL` |
| `procvlm-2b` | `reward_model_code/ProcVLM/.venv/bin/python scripts/run_procvlm_server.py --model-path <snapshot>` | `server_url` or `REWARDBOX_PROCVLM_SERVER_URL` (port 8013) |
| `robo-valuerl-value` | `<robovaluerl venv>/bin/python scripts/run_robovaluerl_server.py --upstream-root ... --provenance-file ...` | `server_url` or `REWARDBOX_ROBOVALUERL_SERVER_URL` (port 8012) |

Example (LRM progress head):

```bash
uv pip install --python .venv/bin/python -e '.[lrm-client]'
rewardbox fetch-upstream lrm
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_lrm_server.py --mode progress \
  --server-python reward_model_code/Large-Reward-Models/vlm_reward/.venv/bin/python \
  --weights-root "$HF_HOME/hub/models--USC-PSI-Lab--LRM-models/snapshots/5f8da7f3934f5d658256d77552927a9f76ad1c0e" \
  --host 127.0.0.1 --port 5102 --provenance-file lrm-progress.provenance.json
rewardbox run lrm-progress robot.mp4 --instruction "put the object on the plate" \
  --model-opt server_url=http://127.0.0.1:5102 \
  --model-opt server_provenance_file=lrm-progress.provenance.json -o lrm.json
```

### Server environments

Each server runs in its own environment; none of these packages go into the
shared `.venv`. Fetch the checkout first (`rewardbox fetch-upstream NAME`) and
download the weights at the revision listed in [models.md](models.md).

| server | environment |
|---|---|
| SOLE-R1 | created by `scripts/run_sole_r1_server.py` (Python 3.10 with development headers, managed by `uv`). The client side needs `uv pip install --python .venv/bin/python -e '.[sole-r1-client]'`, then `uv pip install --python .venv/bin/python --no-deps -r locks/sole-r1-client-no-deps.txt` |
| LRM | `uv venv --python 3.10 reward_model_code/Large-Reward-Models/vlm_reward/.venv`, install torch 2.4.1 and torchvision 0.19.1 from the PyTorch cu124 index, then `uv pip install --python <that venv>/bin/python -r locks/lrm-server-py310-linux-x86_64.lock` |
| RynnValue | `uv sync --frozen --extra robometer --project reward_model_code/RynnValue/robometer`. The server module imports Unsloth, which needs a visible GPU at import |
| ProcVLM | `uv sync --project reward_model_code/ProcVLM --python 3.10 --frozen` |
| Robo-ValueRL | see below |

Robo-ValueRL follows the official `install_simple.sh` inside a separate venv
(`R` is the checkout, `reward_model_code/Robo-ValueRL`):

```bash
uv venv --python 3.11 .venv-robovaluerl
PY=.venv-robovaluerl/bin/python
uv pip install --python $PY -r $R/robo_valuerl/requirements.txt
uv pip install --python $PY --no-deps -e $R/lerobot_2_1 \
    -e $R/openpi/packages/openpi-client -e $R/openpi
uv pip install --python $PY pyserial==3.5 albumentations==2.0.8 matplotlib==3.11.1
# the official replacement files, into this venv only
cp -r $R/openpi/src/openpi/models_pytorch/transformers_replace/* \
    "$($PY -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')/"
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_robovaluerl_server.py --upstream-root $R \
    --checkpoint-dir <snapshot>/value_estimator \
    --provenance-file robovaluerl.provenance.json --port 8012
```

What was run with each server and the package versions of the tested
environments: [validation.md](validation.md).

## Differences from the upstream environments

The shared environment hosts several official code bases at once, so a few
versions differ from what one of them declares.

| package | upstream declares | used | reason |
|---|---|---|---|
| Python | Robometer `pyproject`: 3.10 | 3.11 | TOPReward needs 3.11 (`StrEnum`); Robometer's `setup.py` allows `>=3.10` |
| transformers | Robo-Dopamine: 4.57.0 | 4.57.1 | TOPReward pins 4.57.1 |
| xformers | Robometer: 0.0.32.post2 | 0.0.32.post1 | pinned by vLLM 0.11.0 |
| torchvision | lerobot 0.4.3: `<0.23` | 0.23.0 | the same override TOPReward's own `pyproject` uses |
| tensorflow, tensorflow-datasets, `protobuf<3.21` | Robometer | not installed | only needed for dataset upload; `protobuf<3.21` conflicts with vLLM |
| opencv | VLAC: `opencv-python` | `opencv-python-headless` | one OpenCV build per environment |
| accelerate | LRM server: not declared | 1.12.0 (server environment) | the server passes `device_map`, which needs it |
| flash-attn | ProcVLM model card: separate install | not installed | vLLM 0.18.1 uses its built-in FlashAttention backend for the default profile |

Things to know when environments or processes are shared:

- **flash-attn is a manual step.** TOPReward's `QwenClient` and VLAC's
  `init_model` request FlashAttention explicitly; without the package the model
  load raises `ImportError`. It is installed from the prebuilt wheel matching
  torch, CUDA and Python, outside the lock file.
- **flash-attn changes Robometer's attention backend.** Robometer's setup code
  switches to `flash_attention_2` whenever `flash_attn` imports. The
  `robometer-4b` calibration record was taken with flash-attn installed, and the
  adapter stores the effective backend in `raw.attn_implementation`.
- **Unsloth patches torch and transformers globally.** Importing
  `robometer.utils.setup_utils` loads Unsloth, which affects every model in the
  same process. `rewardbox bench --isolate` runs each model in its own
  subprocess.
- **Robo-Dopamine sets `CUDA_VISIBLE_DEVICES=0` on import.** The adapter
  restores the caller's value before the model is created.
- **VLAC pins `sentencepiece==0.1.99`.** With 0.2.x the InternLM2 tokenizer
  fails to load and `AutoTokenizer.from_pretrained` returns `False`, which
  surfaces later as `'bool' object has no attribute 'model_info'`.
- **VLAC pins torch 2.8.0.** An unconstrained resolve picks a torch version for
  which no prebuilt flash-attn wheel exists.
- **Flora-Rewarder needs a C++ host compiler** for FlashInfer's JIT at engine
  start. If the default `gcc` has no matching `g++`, set `CC`, `CXX` and
  `NVCC_PREPEND_FLAGS='-ccbin <g++>'`.

## GPU memory

`vram_gb` in `rewardbox info` is a static estimate and is not probed.
vLLM-based adapters accept `gpu_memory_utilization`, a fraction of the whole
card rather than of the free memory, so check what else is running on the
card first. `rewardbox doctor` reports how many GPUs are visible.
