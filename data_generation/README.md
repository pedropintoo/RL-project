Notes - To Do

Files created
data_generation/config.py — single source of truth
data_generation/utils.py — rollout + Bradley-Terry + I/O
data_generation/train_policies.py — trains π₁, captures π₂ mid-run
data_generation/generate_preferences.py — builds the preference datasets
data_generation/requirements.txt, data_generation/README.md
How to run

cd RL-project/data_generation
pip install -r requirements.txt
python train_policies.py         # trains π₁ and saves π₂ checkpoint
python generate_preferences.py   # produces JSON + CSV for K ∈ {50, 200, 1000}
The logic, step by step
1. Environment choice. CartPole-v1 (discrete) + Pendulum-v1 (continuous) covers both action-space regimes that Persons 2 and 3 need to handle. Both are in the allowed list and train quickly with PPO.

2. Getting π₂ "for free" during π₁ training. The interesting trick is in train_policies.py:33. Training two policies independently would work but is wasteful — instead I attach a MidCheckpointCallback as a child of SB3's EvalCallback. Every eval_freq steps the eval callback measures mean return; my child callback fires right after and, the first time the eval return crosses the halfway point between random and expert, it dumps weights to <env>_mid.zip. Training then continues until the expert converges. One run, two checkpoints, guaranteed to lie on the same learning trajectory.

The threshold is random + 0.5 · (expert − random). I anchor "random" on the smaller of (measured random rollout, config default) so a lucky random seed can't push the threshold too high — see train_policies.py:97. This matters for Pendulum especially, where "half the max reward" is ambiguous because rewards are always negative.

3. Rollouts. utils.py:18 records (state, action, reward) at every step. I deliberately keep rewards in the output even though DPO ignores them — they're cheap to store and Person 3 will want per-step rewards when debugging the reward model. Actions are converted via np.asarray(action).tolist() so both discrete ints and continuous vectors serialize cleanly.

4. Bradley-Terry label. The spec gives p = exp(R₁) / (exp(R₁) + exp(R₂)). Implementing that naively blows up: CartPole returns reach 500, and exp(500) overflows float64. The identity p = σ(R₁ − R₂) where σ is the sigmoid is algebraically equivalent and numerically stable — see utils.py:44. I also clip the difference at ±50 so that extremely lopsided pairs don't silently saturate to exactly 0 or 1. The label is then a Bernoulli(p) sample.

5. Dataset layout. One (env, K) → one JSON file + one CSV summary. JSON holds full trajectories so DPO/PPO-RLHF can train directly; CSV is a one-line-per-pair summary for eyeballing (check that frac_tau1_preferred ≈ 1 on CartPole, and closer to 0.7–0.9 on Pendulum where the expert/mid gap is smaller). I generate each K with its own seed rather than taking prefixes — this makes the seeds explicit and independent, which matters when Persons 2 and 3 vary dataset size as an axis.

Things worth knowing before running
If π₂ never saves: the mid-return target was too high. Either raise total_timesteps or lower mid_fraction in config.py. The sanity-check block at the end of train_one_environment prints a clear warning.
Pendulum needs more steps (300k) than CartPole (100k) — it's a harder control problem. SAC would converge faster than PPO on Pendulum; swap algo="PPO" → algo="SAC" in config if you want that.
Stochastic rollouts (deterministic=False in generate_preferences.py:58) matter: two rollouts of π₁ must differ, otherwise every τ₁ in a dataset is identical and DPO sees no intra-policy variance.


# Data Generation

Preference-dataset pipeline for Applied Project 2 (RLHF).

## Layout

```
data_generation/
├── config.py               # envs, training budget, dataset sizes K, seeds
├── train_policies.py       # trains pi_1 (expert) and saves pi_2 (mid) checkpoint
├── generate_preferences.py # rolls out pairs, applies Bradley-Terry labels
├── utils.py                # rollout, BT probability, JSON/CSV writers
├── requirements.txt
└── outputs/                # created at runtime
    ├── policies/           # <env>_expert.zip, <env>_mid.zip
    ├── preferences/        # <env>_K<size>.{json,csv}
    └── logs/               # SB3 tensorboard + eval logs
```

## Environments used

- **CartPole-v1** — discrete actions (ensures DPO/PPO-RLHF are tested in the discrete regime).
- **Pendulum-v1** — continuous actions (ensures they also cover the continuous regime).

Edit [config.py](config.py) to add more.

## Usage

```bash
pip install -r requirements.txt

# 1. Train pi_1 and capture pi_2 checkpoint (≈ 15–30 min on CPU).
python train_policies.py

# 2. Generate preference datasets for all K in config.DATASET_SIZES.
python generate_preferences.py
```

Subset flags are available, e.g. `python train_policies.py --envs CartPole-v1`
or `python generate_preferences.py --sizes 50 200`.

## Dataset format

Each dataset is written as a JSON file with this schema:

```jsonc
{
  "env_id": "CartPole-v1",
  "K": 200,
  "seed": 12345,
  "policies": {"pi1": "CartPole-v1_expert", "pi2": "CartPole-v1_mid"},
  "stats": { "mean_R_tau1": ..., "mean_R_tau2": ..., ... },
  "pairs": [
    {
      "tau1": {"states": [[...]], "actions": [...], "rewards": [...],
               "return": float, "length": int},
      "tau2": {"states": [[...]], "actions": [...], "rewards": [...],
               "return": float, "length": int},
      "p_tau1_preferred": 0.87,
      "preferred": 0    // 0 = tau1 preferred, 1 = tau2 preferred
    },
    ...
  ]
}
```

A companion `*.csv` stores one row per pair with returns, lengths, and labels —
handy for quick EDA without loading the full JSON.

## Consumers

- **Person 2 (DPO)** needs `pairs[i].tau_{1,2}.{states,actions}` and
  `pairs[i].preferred` — the rewards are not used.
- **Person 3 (PPO-RLHF)** trains a reward model `r_phi` on the same labels,
  then discards the labels and runs PPO using `r_phi` for rollout rewards.
