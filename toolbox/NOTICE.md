# NOTICE

RewardBox calls the official implementations listed below and contains no copy
of their source code: adapters import it at run time from a checkout that
`rewardbox fetch-upstream` clones at a pinned commit into `reward_model_code/`
(ignored by git). Weights are downloaded from the Hugging Face Hub at run time.
Each project's own license applies to its code and to its weights.

License statements below describe the pinned commit or revision. Where a
project states no license, this file says so.

---

## Upstream projects that RewardBox calls

### TOPReward
- Repository: https://github.com/TOPReward/TOPReward, commit `4877a0ee5098cbec18485125466631b1dcc4a573`
- License: **MIT**. The copyright line reads
  `Copyright (c) 2024 Paweł Budzianowski, Gracjan Góral, Emilia Wiśnios (OpenGVL)`:
  the code base derives from **OpenGVL**, which should be credited together
  with TOPReward.
- Paper: https://arxiv.org/pdf/2602.19313

### Robo-Dopamine
- Repository: https://github.com/FlagOpen/Robo-Dopamine, commit `2c714abca66329f8e2113587226bc9556726f6c2`
- License: **Apache License 2.0**
- Weights: `tanhuajie2001/Robo-Dopamine-GRM-3B`, `-GRM-8B`, `-GRM-2.0-4B-Preview`,
  `-GRM-2.0-8B-Preview`
- Paper: https://arxiv.org/pdf/2512.23703 (CVPR 2026)

### PRIMO-R1
- Repository: https://github.com/10-OASIS-01/PRIMO-R1, commit `f746e1f83af001337dabffafa948129353ac962e`
- License: **not stated** (no root `LICENSE` file). The projects it vendors
  (`transformers`, `qwen-vl-utils`) keep their own license files.
- RewardBox loads its `src/primo_prompts.py` and its vendored `qwen-vl-utils`
  vision preprocessing; the customized Transformers build is installed in a
  separate `.venv-primo`.
- Weights: `LeonOverload/PRIMO-R1-7B`, revision `4d452de3f721ec20e311e3a2dff309df7e022a69`
- Paper: https://arxiv.org/abs/2603.15600

### Flora-Rewarder (NVIDIA)
- Model repository: `nvidia/Flora-Rewarder`, revision
  `031299e46e452fc0cedb4e0975eb9fa95fc37783` (full fine-tune of Qwen3.5-27B).
  The official inference code ships in the same repository (`batch_utils/`);
  RewardBox imports `core.py` and `engine.py` from the downloaded snapshot.
- License: **NVIDIA OneWay Noncommercial License (22 March 2022)**: noncommercial
  research use only. The base model `Qwen/Qwen3.5-27B` is Apache-2.0. Third-party
  dependencies are listed in `batch_utils/THIRD_PARTY_NOTICES.md`.

### Robometer (also hosts the GVL and TOPReward baselines RewardBox calls)
- Repository: https://github.com/robometer/robometer, commit `5b815254bf31ee1bea3753c3a2da9f9033736d9a`
- License: **not stated** (no `LICENSE` file; the `setup.py` classifier names MIT).
- Weights: `aliangdw/Robometer-4B`, `aliangdw/Robometer-4B-LIBERO`
- Paper: https://arxiv.org/abs/2603.02115

### GVL
- Called through the GVL baseline in the Robometer checkout above (same license
  status). GVL has no local checkpoint: frames are sent to the Gemini API
  (`GEMINI_API_KEY`) or to a Gemini-format gateway that the user configures,
  under the provider's terms of service.
- Paper: https://arxiv.org/abs/2411.04549

### RoboReward
- There is no official inference code, only the Hugging Face model card.
  `rewardbox/adapters/roboreward.py` implements inference from the model card;
  the prompt text is taken verbatim from the card and stored in
  `rewardbox/prompts/roboreward.txt`.
- Weights license: **CC-BY-4.0**
- Model card: https://huggingface.co/teetone/RoboReward-4B
- Paper: https://arxiv.org/abs/2601.00675

### VLAC
- Repository: https://github.com/InternRobotics/VLAC, commit `dc57e57169f6ab823c397698ce8a629951d89774`
- Code license: **MIT** (root `LICENSE` and README). The upstream
  `pyproject.toml` says `Apache-2.0`.
- Weights: `InternRobotics/VLAC`, `InternRobotics/VLAC-8b`, license
  **CC BY-NC-SA 4.0** (noncommercial; separate from the code license).
- Paper: https://arxiv.org/abs/2509.15937

### VLAC-Cut
- Repository: https://github.com/InternRobotics/VLAC-cut, commit `354df56ff177018f4ed3849996b3b1fc9cadedc8`
- Code license: **MIT** (`Copyright (c) Shanghai AI Lab, VLAC Team`). The upstream
  README adds that model weights, benchmark videos and third-party source data
  may be subject to additional terms.
- Weights: `InternRobotics/VLAC-Cut`, revision `2a809cbb10d63c2e5b1b3f046426c554f437bdb5`;
  the model card says `license: other`.
- Paper: https://arxiv.org/abs/2607.09776

### SOLE-R1
- Repository: https://github.com/Philip-MIT/sole-r1-model, commit `2a1286b205577a38a2ef5d9d13efa76878485f87`
- License: the README says **MIT**; the commit has no `LICENSE` file.
- The adapter calls the official `reward_client/utils.py:InferenceClient` from
  the checkout; the model server runs in the upstream environment.
- Weights: `Philip-MIT/SOLE-R1-8B`, revision `4b322fbbff3a6b9815fc13b70803cdcb8884616b`
- Paper: https://arxiv.org/abs/2603.28730

### RewardGen
- Repository: https://github.com/Philip-MIT/rewardgen, commit `5c9ef6d155e2733cbcc9081d3eab65e254ad6ab1`
- The official entry point that generated the example outputs published with
  SOLE-R1. The adapter does not need it; it was used to validate the adapter
  ([docs/validation.md](docs/validation.md)).
- License: **not stated** (no `LICENSE` file and no license field or statement).

### Large-Reward-Models (LRM)
- Repository: https://github.com/physical-superintelligence-lab/Large-Reward-Models,
  commit `5845ab0f7c3c6a0bef8bd37bf7f07e9533f7ed2f`
- Code license: **not stated** (no `LICENSE` file and no statement in the README).
- Weights: `USC-PSI-Lab/LRM-models`, revision `5f8da7f3934f5d658256d77552927a9f76ad1c0e`,
  license **Apache-2.0**.
- Paper: https://arxiv.org/abs/2603.16065

### RynnValue
- Repository: https://github.com/alibaba-damo-academy/RynnValue, commit `10e0d333f5f3811d0d130587e50f1faf48da49e5`
- Code license: the README says `rynn_value/`, `rynn_infer/` and `tools/` are
  released under **Apache License 2.0**; the commit has no root `LICENSE`.
  `robometer/LICENSE` is MIT and `pi-rl/LICENSE` is Apache License 2.0.
- Weights: `Alibaba-DAMO-Academy/RynnValue-4B` revision `3f73b5d2b5e53b21f248c8791004dde6a8cf2b92`
  and `Alibaba-DAMO-Academy/RynnValue-8B` revision `8738c5e4ce4418ea0266e9fbeffae0eb9bbb230e`,
  license **Apache-2.0**.
- Paper: https://arxiv.org/abs/2608.09853

### Robo-ValueRL
- Repository: https://github.com/Open-X-Humanoid/Robo-ValueRL, commit `1618bcb01212eb747a5d2659228b7fb5db87a563`
- Code license: **not stated** at the repository root. The embedded
  `openpi/LICENSE` and `lerobot_2_1/LICENSE` cover those sub-projects.
- Weights: `X-Humanoid/Robo-ValueRL` revision `b44f33eaa5892233174d596cc7d8a7e685967eef`,
  license **MIT** (model card).
- Paper: https://arxiv.org/abs/2607.09866

### ProcVLM
- Repository: https://github.com/RUCKBReasoning/ProcVLM, commit `377523a31f05bab9c0db5ac8b9edfa7b7f03968a`
- Code license: **not stated** (no project-level `LICENSE`; some training files
  carry Apache-2.0 headers from the code they derive from).
- Weights: `ce-amtic/ProcVLM-2B` revision `a7f5c18c6128a9c2a478a62ecc356f19f0e0bcf5`,
  license **CC BY 4.0** (model card).
- Paper: https://arxiv.org/abs/2605.08774

---

## Design reference

The registry and factory design (a default-configuration table, a registry and
a `create_metric`-style factory) follows
[pyiqa (IQA-PyTorch)](https://github.com/chaofengc/IQA-PyTorch). No code is
taken from it.

---

## Weights and data

Model weights are not distributed with this repository; they are downloaded
from the Hugging Face Hub at run time and remain under their original licenses.
