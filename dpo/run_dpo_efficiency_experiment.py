"""
run_dpo_efficiency_experiment.py
---------------------------------
Runs the DPO scaling experiment with wall-clock timing instrumentation and
saves per-seed summary.json files to outputs/dpo_efficiency_results/.

Results are isolated from the canonical outputs/dpo_scaling/ artifacts used
by the notebook, so re-running this script never overwrites prior results.

Usage
-----
    cd dpo
    python run_dpo_efficiency_experiment.py
    python run_dpo_efficiency_experiment.py --env MountainCarContinuous-v0
"""

import argparse
import sys
from pathlib import Path

import torch

DPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DPO_DIR))

from dpo_experiments import run_dpo_scaling_experiment
from aggregate_dpo_efficiency import aggregate

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
SEEDS = [1, 2, 3, 4, 5]
DATASET_SIZES = [50, 200, 1000]
N_EVAL_EPISODES = 100

PREFERENCE_DIR = DPO_DIR.parent / "data_generation" / "outputs" / "preferences"
POLICY_DIR = DPO_DIR.parent / "data_generation" / "outputs" / "policies"
OUTPUT_DIR = DPO_DIR / "outputs" / "dpo_efficiency_results"

DEVICE = torch.device("cpu")

# Hyperparameters copied from notebook cell (RUN_CONFIG).
RUN_CONFIG = {
    "CartPole-v1": {
        "default": {
            "lr": 1e-2,
            "beta": 0.1,
            "kl_coef": 0.1,
            "batch_size": 1,
            "n_epochs": 300,
            "plateau_window": 12,
            "early_stop": True,
            "eval_every": 5,
        },
        50:   {"batch_size": 2},
        200:  {"batch_size": 8},
        1000: {"batch_size": 32},
    },
    "Pendulum-v1": {
        "default": {
            "lr": 1e-3,
            "beta": 0.3,
            "kl_coef": 0.2,
            "batch_size": 1,
            "n_epochs": 300,
            "plateau_window": 12,
            "early_stop": True,
            "eval_every": 5,
        },
        50:   {"batch_size": 2},
        200:  {"batch_size": 8},
        1000: {"batch_size": 32},
    },
    "MountainCarContinuous-v0": {
        "algo": "SAC",
        "default": {
            "lr": 1e-3,
            "beta": 0.4,
            "kl_coef": 0.4,
            "batch_size": 1,
            "n_epochs": 300,
            "plateau_window": 12,
            "early_stop": True,
            "eval_every": 5,
        },
        50:  {"batch_size": 2},
        200: {"batch_size": 8},
        1000: {"batch_size": 32},
    },
}


def run(envs=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = envs if envs else list(RUN_CONFIG.keys())
    for env_id in targets:
        if env_id not in RUN_CONFIG:
            raise ValueError(f"Unknown env '{env_id}'. Choose from: {list(RUN_CONFIG.keys())}")
        print(f"\n{'=' * 60}")
        print(f"  DPO efficiency experiment: {env_id}")
        print(f"{'=' * 60}\n")
        run_dpo_scaling_experiment(
            env_id=env_id,
            dataset_sizes=DATASET_SIZES,
            seeds=SEEDS,
            preference_dir=PREFERENCE_DIR,
            policy_dir=POLICY_DIR,
            output_dir=OUTPUT_DIR,
            device=DEVICE,
            n_eval_episodes=N_EVAL_EPISODES,
            run_config=RUN_CONFIG,
        )

    print("\n  Aggregating results...")
    aggregate()
    print("\n  DPO efficiency experiment complete.")
    print(f"    Results: {OUTPUT_DIR / 'dpo_efficiency_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", nargs="+", metavar="ENV",
                        help="Run only the specified environment(s). "
                             "E.g. --env MountainCarContinuous-v0")
    args = parser.parse_args()
    run(envs=args.env)
