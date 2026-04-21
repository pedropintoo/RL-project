"""
Central configuration for the RLHF data-generation pipeline.

All experiment-level knobs live here so that `train_policies.py` and
`generate_preferences.py` stay short and reproducible. If you want to add
an environment, tune training length, or change the dataset sizes K,
this is the only file you need to touch.
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
    # Reference returns (used to decide what "mid-performing" means):
    #   pi_2 target return = random_return + mid_fraction * (expert_return - random_return)
    # We do NOT need exact numbers; these are rough anchors. The training
    # script will also print the true random baseline so you can sanity-check.
    random_return: float = 0.0
    expert_return: float = 0.0
    mid_fraction: float = 0.5


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
]


# ---------------------------------------------------------------------------
# Preference-dataset configuration
# ---------------------------------------------------------------------------
# K = number of (tau_1, tau_2) pairs produced. We generate several sizes so
# that Persons 2 and 3 can study how DPO / PPO-RLHF scale with dataset size.
DATASET_SIZES: List[int] = [50, 200, 1000]

# Seeds used when rolling out trajectories. We keep them separate from the
# seeds that Persons 2/3 will use for training their algorithms — the goal
# here is simply reproducibility of the preference data itself.
ROLLOUT_SEED: int = 12345

# Training seeds — three separate seeds produce three expert checkpoints;
# this lets you generate more diverse preference data if you want. For
# simplicity we default to a single training seed.
TRAIN_SEED: int = 0
