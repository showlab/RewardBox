# rewardbox

The Python package of [RewardBox](../README.md): one interface for robot reward
and progress-estimation models. It wraps the official implementation of 14
methods (35 configurations), plus 4 protocol profiles for general VLMs and Jev.
The project overview, the model zoo and the benchmark are described in the
[main README](../README.md).

## Install

```bash
pip install -e .                 # core: registry, CLI, metrics, benchmark runner (no torch)
pip install -e ".[models]"       # in-process models: Robometer, RoboReward, TOPReward, Robo-Dopamine
pip install -e ".[general-vlm]"  # general VLMs and Jev through HTTP APIs
rewardbox list                   # 39 configurations
```

`rewardbox doctor MODEL` reports missing dependencies, source checkouts and GPU
state and prints the fix. Lock files, isolated environments and model servers:
[docs/installation.md](docs/installation.md).

## Use

```python
import rewardbox

with rewardbox.create_reward("robometer-4b") as rm:
    out = rm("robot.mp4", "open the drawer")   # the model's own default sampling

out.progress        # np.ndarray [T], raw native scale
out.frame_indices   # source-video frame of each value
```

```bash
rewardbox run robometer-4b robot.mp4 --instruction "open the drawer" -o result.json
rewardbox bench --manifest data.jsonl --model robometer-4b --num-frames 8 -o runs/exp1
```

Exit codes: `0` success, `2` usage error (unknown model, bad option), `1`
runtime failure. Machine-readable output goes to stdout, diagnostics to stderr.

## Documentation

| | |
|---|---|
| [docs/models.md](docs/models.md) | every configuration: runtime, default sampling, views, score range, checkouts, calibration ([JSON](docs/models.json)) |
| [docs/installation.md](docs/installation.md) | shared environment, isolated workers, model servers, API keys |
| [docs/guides/general_vlm_and_jev.md](docs/guides/general_vlm_and_jev.md) | general VLMs and Jev as evaluators (Sequence / Pair) |
| [docs/guides/](docs/guides) | vLLM backends, GVL on Gemini 3.8 Flash |
| [docs/validation.md](docs/validation.md) | what was checked for each adapter with real weights, and the tested environments |
| [docs/adding_a_model.md](docs/adding_a_model.md) | adapter template and checklist |
| [AGENTS.md](AGENTS.md) | manual for coding agents: rules, recipes, repository map, troubleshooting |
| [NOTICE.md](NOTICE.md) | licenses of the upstream code and weights |

## Develop

```bash
pip install -e ".[dev]"
pytest -m "not gpu and not calibration"     # offline tests; upstream-dependent ones skip
rewardbox fetch-upstream all && REWARDBOX_REQUIRE_UPSTREAM=1 pytest -m "not gpu and not calibration"
python scripts/gen_model_docs.py --check    # docs/models.* in sync with the registry
```

## License

Apache License 2.0 ([LICENSE](LICENSE)). Upstream code and model weights keep
their own licenses; see [NOTICE.md](NOTICE.md). Citation: [CITATION.cff](CITATION.cff)
and the [main README](../README.md#bookmark_tabs-citation).
