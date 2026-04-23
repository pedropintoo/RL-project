# Data Generation

Preference-dataset pipeline for Applied Project 2 (RLHF).

This module produces the labeled trajectory pairs that downstream DPO and
PPO-RLHF components train on. It handles policy training, trajectory rollout
and Bradley-Terry preference labeling end-to-end.

## Layout

```
data_generation/
├── config.py               # single source of truth: envs, budgets, K, seeds
├── train_policies.py       # trains pi_1 (expert) and saves pi_2 (mid) checkpoint
├── generate_preferences.py # rolls out trajectory pairs, applies Bradley-Terry labels
├── utils.py                # rollout, BT probability, JSON/CSV writers
├── requirements.txt
├── README.md
└── outputs/                # created at runtime
    ├── policies/           # <env>_expert.zip, <env>_mid.zip
    ├── preferences/        # <env>_K<size>_s<seed>.{json,csv}
    └── logs/               # SB3 eval logs (Weights & Biases handles training curves)
```

## Usage

```bash
cd RL-project/data_generation
pip install -r requirements.txt

# First-time setup: authenticate with Weights & Biases (used for training curves)
wandb login

# 1. Train pi_1 and capture pi_2 checkpoint (~1-3 min on Apple Silicon, longer on Intel).
python train_policies.py

# 2. Generate preference datasets. By default this sweeps
#    2 envs x 3 sizes x 5 seeds = 30 datasets.
python generate_preferences.py
```

Subset flags are supported:

- `python train_policies.py --envs CartPole-v1` trains only a subset.
- `python generate_preferences.py --sizes 50 200` restricts dataset sizes.
- `python generate_preferences.py --seeds 1 2` restricts to those seeds
  (default: 5 consecutive seeds starting at `config.ROLLOUT_SEED`, i.e. 1-5).

To run training without a W&B account, set `WANDB_MODE=disabled` (no logging) or
`WANDB_MODE=offline` (log locally, sync later with `wandb sync`).

## Environments

- **CartPole-v1** — discrete actions. Ensures DPO and PPO-RLHF are exercised in
  the discrete regime.
- **Pendulum-v1** — continuous actions. Ensures both methods are exercised in
  the continuous regime.

Both are in the course's allowed environment list and train quickly with PPO.
Additional environments can be added by editing [config.py](config.py).

## Design notes

### 1. Capturing pi_2 during pi_1 training

Rather than training two policies independently, a single PPO run is used per
environment. A `MidCheckpointCallback` is attached as a child of SB3's
`EvalCallback` (see [train_policies.py](train_policies.py)). Every `eval_freq`
steps the eval callback records the mean evaluation return; the child callback
then fires, and the first time that return crosses a halfway threshold between
random and expert performance, the current weights are dumped to
`<env>_mid.zip`. Training then continues until pi_1 converges and is saved as
`<env>_expert.zip`.

Benefits of this single-trajectory approach:

- pi_2 and pi_1 lie on the **same learning trajectory**, so pi_2 is genuinely a
  less-trained version of pi_1 rather than a different algorithm or seed — this
  matches the project specification.
- Total training compute is halved.

The mid-return threshold is `random + mid_fraction * (expert - random)`. The
"random" anchor is taken as the smaller of (measured random rollout, config
default) so that a lucky random-policy evaluation cannot inflate the target.
This matters especially on Pendulum, where rewards are always negative and
"halfway" is otherwise ambiguous.

### 2. Rollouts

[utils.py](utils.py) records `(state, action, reward)` at every step. Rewards
are retained in the output even though DPO itself ignores them: they are cheap
to store and are needed by the PPO-RLHF consumer when debugging the reward
model. Actions are converted via `np.asarray(action).tolist()` so that both
discrete integer actions (CartPole) and continuous action vectors (Pendulum)
serialize cleanly to JSON.

Rollouts for preference generation are stochastic (`deterministic=False` in
[generate_preferences.py](generate_preferences.py)). This is load-bearing: two
rollouts of pi_1 must differ, otherwise every tau_1 in a dataset would be
identical and DPO would see no intra-policy variance.

### 3. Bradley-Terry label

The specification gives `p = exp(R_1) / (exp(R_1) + exp(R_2))`. Implementing
this naively overflows: CartPole returns reach 500, and `exp(500)` is not
representable in float64. The algebraically equivalent form
`p = sigmoid(R_1 - R_2)` is numerically stable and is used in
[utils.py](utils.py). The return difference is additionally clipped at +/-50
so that extremely lopsided pairs do not silently saturate to exactly 0 or 1.
The binary preference label is then a Bernoulli(p) sample.

### 4. Dataset layout

Each `(env, K, seed)` triple produces one JSON file (full trajectories) and one
CSV file (one line per pair, for quick EDA). With the defaults (2 envs, 3 sizes,
5 seeds) this is **30 JSON files + 30 CSV files**. Files are named
`<env>_K<size>_s<base_seed>.{json,csv}` so the sweep axes are readable from the
filename alone.

Each `K` is generated with its own derived seed rather than by taking prefixes
of a larger dataset, so that different sizes under one base seed are not
trivially nested. The derived seed is `base_seed + hash((env_id, K)) mod 10000`
and is stored as `"seed"` inside the JSON, alongside the original `"base_seed"`.

Sanity checks: `frac_tau1_preferred` should sit near 1 on CartPole (large
expert-vs-mid return gap), and closer to 0.7-0.9 on Pendulum (smaller gap).

## Dataset format

Each dataset is written as a JSON file with this schema:

```jsonc
{
  "env_id": "CartPole-v1",          // gymnasium environment id
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

A companion `*.csv` stores one row per pair with returns, lengths, and labels —
handy for quick EDA without loading the full JSON.

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

## Troubleshooting

- **pi_2 is never saved.** The mid-return target was too high for the budget.
  Either raise `total_timesteps` or lower `mid_fraction` in
  [config.py](config.py). The sanity-check block at the end of
  `train_one_environment` prints a clear `NOT FOUND` warning when this happens.
- **Pendulum looks slow to learn.** Pendulum is a harder control problem than
  CartPole and is configured for 300k steps vs. CartPole's 100k. SAC converges
  faster than PPO on Pendulum; change `algo="PPO"` to `algo="SAC"` in
  [config.py](config.py) to use it.
- **`frac_tau1_preferred` close to 0.5.** Either pi_1 and pi_2 are too close in
  performance (raise `mid_fraction` toward 0.3-0.4, i.e. save pi_2 earlier), or
  the rollout seed is producing unusually similar trajectories (try a different
  `ROLLOUT_SEED`).
