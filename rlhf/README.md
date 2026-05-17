# Reinforcement Learning from Human Feedback (RLHF)

This directory contains the full PPO-RLHF pipeline: training a reward model from
synthetic preference data and using it to fine-tune a policy via RLHF.

The pipeline measures how performance and computational cost scale with the number
of preference pairs (K ∈ {50, 200, 1000}) across three environments:
CartPole-v1, Pendulum-v1, and MountainCarContinuous-v0.

## Repository Structure

```text
rlhf/
├── rm_config.py             # Per-environment reward model hyperparameters (epochs, lr)
├── config_rlhf.py           # KL penalty coefficient BETA (reads RLHF_BETA env var)
├── reward_model.py          # 2-layer MLP scoring (state, action) → scalar reward
├── rlhf_env.py              # Gym wrapper: replaces env reward with RM output + KL penalty
├── train_reward_model.py    # Phase 1: train RMs using Bradley-Terry loss
├── train_ppo_rlhf.py        # Phase 2: fine-tune policy against frozen RM (with early stopping)
├── evaluate_results.py      # Phase 3: evaluate policies, save mean/std per (env, K)
├── plot_results.py          # Phase 4: generate scaling plots from evaluation JSON
├── aggregate_efficiency.py  # Merge timing files → efficiency_results.json
├── run_efficiency_experiment.py  # Single entry point: runs all five stages in sequence
├── run_beta_ablation.py     # Sweeps BETA values (run after the main pipeline)
└── outputs/
    ├── reward_models/            # Phase 1 weights (.pth)
    ├── ppo_rlhf_results/beta*/  # Phase 2 policy checkpoints (.zip)
    ├── logs/beta*/              # TensorBoard event files
    ├── evaluation_results/beta*/ # Evaluation JSONs (mean/std)
    ├── efficiency_results/       # Timing JSONs + efficiency_results.json
    └── plots/
        ├── beta*/                # Scaling plots (.png)
        └── rm_convergence/       # Diagnostic loss curves used to tune Phase 1 (see README there)
```

## Reproducing the Experiments

The entire pipeline (45 models: 3 envs × 3 K-sizes × 5 seeds) is run with a single command:

```bash
cd rlhf
python run_efficiency_experiment.py
```

This runs the five stages below in order. Each stage must complete before the next starts.

### Stage 1 — Train Reward Models

Reads preference `.json` files from `data_generation/outputs/preferences/` and trains
one reward model per (env, K, seed) using Bradley-Terry loss with gradient clipping.
Per-environment hyperparameters (epochs, lr) are defined in `rm_config.py` and were
chosen by inspecting loss curves in `outputs/plots/rm_convergence/` (see the README there).

```bash
python train_reward_model.py
```

Outputs: `outputs/reward_models/{env}_K{K}_seed{seed}_reward_model.pth`
Timing:  `outputs/efficiency_results/rm_timing.json`

### Stage 2 — Fine-tune Policies

Loads each reward model and fine-tunes a copy of the mid-performing policy against it.
Uses SB3's `EvalCallback` + `StopTrainingOnNoModelImprovement` for early stopping:
the policy is evaluated in the raw environment every `cfg.eval_freq` timesteps; training
stops once `PLATEAU_WINDOW=12` consecutive evaluations show no improvement. The best
checkpoint (highest mean return) is saved, not the last weights.

```bash
python train_ppo_rlhf.py
```

Outputs: `outputs/ppo_rlhf_results/beta{BETA}/{env}_K{K}_seed{seed}.zip`
Timing:  `outputs/efficiency_results/ppo_timing.json`

### Stage 3 — Aggregate Timing

Merges `rm_timing.json` and `ppo_timing.json` into a single summary with mean ± std
over 5 seeds for wall-clock time (both phases) and gradient steps (Phase 2).
Phase 1 gradient steps are computed analytically (see below); Phase 2 gradient steps
are recorded at runtime because early stopping makes them variable.

```bash
python aggregate_efficiency.py
```

Output: `outputs/efficiency_results/efficiency_results.json`

### Stage 4 — Evaluate Policies

Evaluates expert baselines, mid-policy baselines, and all fine-tuned policies over
50 deterministic episodes. Saves mean and standard deviation per (env, K).

```bash
python evaluate_results.py
```

Output: `outputs/evaluation_results/beta{BETA}/results_{env}.json`

### Stage 5 — Plot Results

Reads the evaluation JSON and generates scaling plots (true environment return vs. K
on a log scale, with ± std bands).

```bash
python plot_results.py
```

Output: `outputs/plots/beta{BETA}/{env}_scaling_plot.png`

---

## Methodology

**Independent seeds:** 5 separate reward models are trained on 5 isolated datasets per
(env, K). 5 separate policies are then fine-tuned against those specific reward models.
Final metrics are mean ± std over these 5 independent end-to-end runs.

**Starting point:** all PPO/SAC agents initialise from the mid-performing policy (not
random) to reduce variance and ensure a controlled starting point.

**KL penalty:** the RLHF environment wrapper replaces the true environment reward with
`r_hat(s, a) - BETA * KL(pi || pi_ref)`. BETA is set in `config_rlhf.py` (default 0.1)
and can be overridden via the `RLHF_BETA` environment variable.

---

## Gradient Steps

### Phase 1 — Reward Model training

Batch size is 1 (one gradient step per preference pair per epoch). Epochs are fixed
per environment (defined in `rm_config.py`):

| Environment              | epochs |
|--------------------------|--------|
| CartPole-v1              | 5      |
| Pendulum-v1              | 10     |
| MountainCarContinuous-v0 | 25     |

```text
Phase 1 grad steps = K × epochs
```

| Environment              | K=50 | K=200 | K=1000 |
|--------------------------|------|-------|--------|
| CartPole-v1              | 250  | 1,000 | 5,000  |
| Pendulum-v1              | 500  | 2,000 | 10,000 |
| MountainCarContinuous-v0 | 1,250| 5,000 | 25,000 |

### Phase 2 — Policy fine-tuning

Phase 2 grad steps are variable because early stopping terminates training as soon as
the policy plateaus. They are measured at runtime from `model.num_timesteps` and
reported in `efficiency_results.json` as mean ± std over 5 seeds.

For reference, the analytical formula for a full-budget run (no early stopping):

**PPO** (CartPole-v1, Pendulum-v1):

```text
grad steps = floor(tune_budget / n_steps) × floor(n_steps / batch_size) × n_epochs
           = floor(tune_budget / 2048) × 32 × 10
```

| Environment | tune_budget | Max grad steps (no early stop) |
|-------------|-------------|--------------------------------|
| CartPole-v1 | 50,000      | ~7,680                         |
| Pendulum-v1 | 150,000     | ~23,360                        |

**SAC** (MountainCarContinuous-v0, train_freq=32, gradient_steps=32):

```text
grad steps = floor(tune_budget / train_freq) × gradient_steps = tune_budget
```

| Environment              | tune_budget | Max grad steps (no early stop) |
|--------------------------|-------------|--------------------------------|
| MountainCarContinuous-v0 | 25,000      | ~25,000                        |

---

## Beta Ablation Study

To study how the KL penalty coefficient affects performance, run the ablation after
the main pipeline has completed (reward models in `outputs/reward_models/` are reused):

```bash
python run_beta_ablation.py
python plot_beta_ablation.py
```

Results are written to `outputs/ppo_rlhf_results/beta*/`,
`outputs/evaluation_results/beta*/`, and `outputs/plots/ablation_comparisons/`.
