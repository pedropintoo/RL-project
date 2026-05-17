"""
run_efficiency_experiment.py
-----------------------------
Full PPO-RLHF pipeline runner.

Executes all five pipeline stages in sequence, writing every artifact to its
canonical location under outputs/. Re-running this script overwrites all previous
results — this is intentional, as all results must be regenerated together whenever
hyperparameters or training logic change.

Pipeline stages
---------------
1. train_reward_model.py   — Phase 1: trains one RM per (env, K, seed), records
                             per-seed wall-clock time in outputs/efficiency_results/rm_timing.json
2. train_ppo_rlhf.py       — Phase 2: fine-tunes one PPO/SAC agent per (env, K, seed)
                             with early stopping, records per-seed wall-clock time and
                             gradient steps in outputs/efficiency_results/ppo_timing.json
3. aggregate_efficiency.py — merges the two timing files into
                             outputs/efficiency_results/efficiency_results.json
                             (mean ± std over 5 seeds per (env, K) for both phases)
4. evaluate_results.py     — evaluates all trained policies over 50 deterministic
                             episodes and saves outputs/evaluation_results/results_{env}.json
5. plot_results.py         — reads the evaluation JSON and generates scaling plots
                             in outputs/plots/beta{BETA}/

Outputs (canonical paths, no isolation sub-folders)
----------------------------------------------------
    outputs/reward_models/                        ← Phase 1 model weights
    outputs/ppo_rlhf_results/beta{BETA}/          ← Phase 2 policy checkpoints
    outputs/logs/beta{BETA}/                      ← TensorBoard logs
    outputs/efficiency_results/                   ← timing JSONs + efficiency_results.json
    outputs/evaluation_results/beta{BETA}/        ← evaluation JSON
    outputs/plots/beta{BETA}/                     ← scaling plots

Usage
-----
    cd rlhf
    python run_efficiency_experiment.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "train_reward_model.py",
    "train_ppo_rlhf.py",
    "aggregate_efficiency.py",
    "evaluate_results.py",
    "plot_results.py",
]


def run():
    rlhf_dir = Path(__file__).resolve().parent

    for script in SCRIPTS:
        print(f"\n{'=' * 60}")
        print(f"  Running: {script}")
        print(f"{'=' * 60}\n")
        subprocess.run(
            [sys.executable, script],
            cwd=rlhf_dir,
            check=True,
        )

    print("\nPipeline complete.")
    print(f"  Efficiency results : {rlhf_dir / 'outputs' / 'efficiency_results' / 'efficiency_results.json'}")
    print(f"  Evaluation results : {rlhf_dir / 'outputs' / 'evaluation_results'}")
    print(f"  Scaling plots      : {rlhf_dir / 'outputs' / 'plots'}")


if __name__ == "__main__":
    run()
