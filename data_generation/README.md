# Data Generation

Preference-dataset pipeline for Applied Project 2 (RLHF).

This module produces the labeled trajectory pairs that downstream DPO and
PPO-RLHF components train on. It handles policy training, trajectory rollout and Bradley-Terry preference labeling end-to-end.

## Layout

```
data_generation/
├── config.py               # main entry point: envs, budgets, K, seeds
├── train_policies.py       # trains pi_1 (expert) and saves pi_2 (mid) checkpoint
├── generate_preferences.py # rolls out trajectory pairs, applies labels
├── utils.py                # rollout, BT probability, JSON/CSV writers
├── requirements.txt
├── README.md
└── outputs/                # created at runtime
    ├── policies/           # <env>_expert.zip, <env>_mid.zip
    ├── preferences/        # <env>_K<size>_s<seed>.{json,csv}
    └── logs/               # SB3 eval logs (wandb handles training curves)
```

## Usage

```bash
cd RL-project
pip install -r requirements.txt
wandb login

# 1. Train pi_1 and capture pi_2 checkpoint
cd RL-project/data_generation
python train_policies.py

# 2. Generate preference datasets. By default: 3 envs x 3 sizes x 5 seeds = 45 datasets.
python generate_preferences.py
```

Subset flags are supported:

- `python train_policies.py --envs CartPole-v1` trains only a subset.
- `python generate_preferences.py --sizes 50 200` restricts dataset sizes.
- `python generate_preferences.py --seeds 1 2` restricts to those seeds

To run training without a W&B account, set `WANDB_MODE=disabled`.

## Environments

- **CartPole-v1** — discrete actions, dense reward. Trained with PPO. Ensures
  DPO and PPO-RLHF are exercised in the discrete regime.
- **Pendulum-v1** — continuous actions, dense (cost-based) reward. Trained
  with PPO. Continuous-control baseline with a smooth reward signal.
- **MountainCarContinuous-v0** — continuous actions, sparse reward (+100 only
  on reaching the flag, otherwise a small action penalty). Trained with SAC
  using gSDE because standard SAC reliably gets trapped in a "do-nothing" local
  optimum on this env. Provides a sparse-reward stress test for the same
  RLHF pipeline.

Additional environments can be added by appending an `EnvConfig` entry.

## Dataset format

Each dataset is written as a JSON file with this schema:

```jsonc
{
  "env_id": "CartPole-v1",           // gymnasium environment id
  "K": 200,                          // number of (tau_1, tau_2) pairs
  "seed": 3088,                      // derived seed actually used for rollouts
  "base_seed": 1,                    // sweep-slot identifier (also in filename)
  "policies": {                      // checkpoint stems under outputs/policies/
    "pi1": "CartPole-v1_expert",
    "pi2": "CartPole-v1_mid"
  },
  "stats": {                         // aggregate statistics for quick inspection
    "mean_R_tau1": ..., "std_R_tau1": ...,
    "mean_R_tau2": ..., "std_R_tau2": ...,
    "fraction_tau1_preferred": ...
  },
  "pairs": [                         // length K
    {
      "tau1": {"states": [[...]], "actions": [...], "rewards": [...],
               "return": float, "length": int},
      "tau2": {"states": [[...]], "actions": [...], "rewards": [...],
               "return": float, "length": int},
      "p_tau1_preferred": 0.87,      // Bradley-Terry probability
      "preferred": 0                 // 0 = tau1 preferred, 1 = tau2 preferred
    },
    ...
  ]
}
```

A companion `*.csv` stores one row per pair with returns, lengths and labels. Handy for quick EDA without loading the full JSON.

## Loading a dataset

Minimal example that opens one JSON and walks every field:

```python
import json
import numpy as np
from pathlib import Path

path = Path("data_generation/outputs/preferences/CartPole-v1_K200_s1.json")
with path.open() as f:
    data = json.load(f)

# --- top-level metadata ---
env_id     = data["env_id"]       # e.g. "CartPole-v1"
K          = data["K"]            # number of pairs (= len(data["pairs"]))
seed       = data["seed"]         # derived rollout seed
base_seed  = data["base_seed"]    # sweep-slot seed (matches filename)
policies   = data["policies"]     # {"pi1": "<stem>", "pi2": "<stem>"}
stats      = data["stats"]        # aggregate stats dict

# --- per-pair access ---
for pair in data["pairs"]:
    tau1 = pair["tau1"]
    tau2 = pair["tau2"]
    p    = pair["p_tau1_preferred"]   # float in (0, 1)
    y    = pair["preferred"]          # int: 0 -> tau1 preferred, 1 -> tau2

    # Each trajectory dict contains:
    states  = np.asarray(tau1["states"])    # shape (T, obs_dim), float32
    actions = np.asarray(tau1["actions"])   # shape (T,) for discrete,
                                            #        (T, act_dim) for continuous
    rewards = np.asarray(tau1["rewards"])   # shape (T,), per-step rewards
    R       = tau1["return"]                # float, sum of rewards
    T       = tau1["length"]                # int, episode length
```

### Loading all datasets in the sweep

```python
import json
from pathlib import Path

PREF_DIR = Path("data_generation/outputs/preferences")

# iterate every (env, K, seed) slot
for json_path in sorted(PREF_DIR.glob("*.json")):
    with json_path.open() as f:
        d = json.load(f)
    print(json_path.name, d["env_id"], d["K"], d["base_seed"],
          len(d["pairs"]))

# or filter by axis, e.g. all K=200 datasets across seeds for CartPole:
for json_path in sorted(PREF_DIR.glob("CartPole-v1_K200_s*.json")):
    ...
```

### CSV companion for quick EDA

```python
import pandas as pd
df = pd.read_csv("data_generation/outputs/preferences/CartPole-v1_K200_s1.csv")
df["preferred"].value_counts()          # class balance
df[["R_tau1", "R_tau2"]].describe()     # return distributions
```

## Monitoring

Training curves (reward, loss, entropy, value-function error, evaluation mean
return) are logged to Weights & Biases via SB3's `WandbCallback`. Each
environment produces its own run named `<env_id>-<algo>-seed<N>`. Multiple
runs can be overlaid on the W&B project page to compare seeds or environments.