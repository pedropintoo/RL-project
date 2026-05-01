"""
Central configuration for the RLHF data-generation pipeline.

All experiment-level knobs live here. To add a new environment,
tune training length or change the dataset sizes K, only this
file needs to be modified.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
POLICY_DIR = OUTPUT_DIR / "policies"
PREFERENCE_DIR = OUTPUT_DIR / "preferences"
LOG_DIR = OUTPUT_DIR / "logs"

for d in (POLICY_DIR, PREFERENCE_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-environment configuration
# ---------------------------------------------------------------------------
@dataclass
class EnvConfig:
    env_id: str                 # gymnasium id, e.g. "CartPole-v1"
    algo: str                   # "PPO" or "SAC" — SB3 algorithm class name
    total_timesteps: int        # budget for training pi_1 (the expert)
    eval_freq: int              # env steps between evaluations (used to catch pi_2)
    n_eval_episodes: int = 10   # episodes averaged at each eval
    max_episode_steps: int = 500
    # Reference returns, used to determine a mid-performing policy:
    #   pi_2 target return = random_return + mid_fraction * (expert_return - random_return)
    # No need for exact numbers, these rough anchors are enough. The training
    # script will print the true random baseline as sanity-check.
    random_return: float = 0.0
    expert_return: float = 0.0
    mid_fraction: float = 0.5
    # Extra kwargs forwarded to the SB3 algorithm constructor. Used for
    # envs that need non-default hyperparameters to learn (e.g. gSDE for
    # MountainCarContinuous-v0).
    algo_kwargs: dict = field(default_factory=dict)


ENVIRONMENTS: List[EnvConfig] = [
    # Discrete action space, +1 reward per step, max return = 500.
    EnvConfig(
        env_id="CartPole-v1",
        algo="PPO",
        total_timesteps=100_000,
        eval_freq=2_000,
        max_episode_steps=500,
        random_return=22.0,       # typical random-policy return
        expert_return=500.0,      # environment cap
        mid_fraction=0.5,
    ),
    # Continuous action space, cost-based reward (always <= 0).
    # "Expert" is ~ -150, random is ~ -1200. Half-way ≈ -675.
    EnvConfig(
        env_id="Pendulum-v1",
        algo="PPO",
        total_timesteps=300_000,
        eval_freq=5_000,
        max_episode_steps=200,
        random_return=-1200.0,
        expert_return=-150.0,
        mid_fraction=0.5,
    ),
    # Continuous 1-D action, sparse reward: +100 on reaching the goal,
    # otherwise -0.1 * action^2 per step. Random ≈ -35 (action penalty),
    # a "do-nothing" policy ≈ 0 (no penalty, never reaches the goal), a
    # solved policy ≈ +90. Vanilla SAC reliably gets stuck at the
    # do-nothing local optimum on this env, so we use the SB3 RL-Zoo
    # recipe: gSDE for state-dependent exploration, larger ent_coef,
    # high gamma, and frequent gradient steps.
    EnvConfig(
        env_id="MountainCarContinuous-v0",
        algo="SAC",
        total_timesteps=50_000,
        eval_freq=1_000,
        max_episode_steps=999,
        random_return=-35.0,
        expert_return=90.0,
        mid_fraction=0.5,
        algo_kwargs=dict(
            learning_rate=3e-4,
            buffer_size=50_000,
            batch_size=512,
            ent_coef=0.1,
            train_freq=32,
            gradient_steps=32,
            gamma=0.9999,
            tau=0.01,
            learning_starts=0,
            use_sde=True,
            policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64]),
        ),
    ),
]


# ---------------------------------------------------------------------------
# Preference-dataset configuration
# ---------------------------------------------------------------------------
# K = number of (tau_1, tau_2) pairs produced. Several sizes are generated, to
# study how DPO / PPO-RLHF scales with dataset size.
DATASET_SIZES: List[int] = [50, 200, 1000]

# Base seed used when rolling out trajectories. The default sweep in
# generate_preferences.py generates seeds ROLLOUT_SEED, ROLLOUT_SEED+1, ...
ROLLOUT_SEED: int = 1

# Training seeds, e.g. three separate seeds produce three expert checkpoints;
# this allows to generate more diverse preference data if wanted. Default is
# a single training seed.
TRAIN_SEED: int = 0
