"""
run_efficiency_experiment.py
-----------------------------
Orchestrator for the wall-clock timing experiment.

Runs the full PPO-RLHF pipeline (Phase 1: reward model training,
Phase 2: policy optimisation) with timing instrumentation, then aggregates
the results into outputs/efficiency_results/efficiency_results.json.

Results are saved to outputs/efficiency_results/ and do NOT overwrite:
  - Model artifacts in outputs/reward_models/ or outputs/ppo_rlhf_results/
    (re-generated with the same fixed seeds, so values are identical)
  - Evaluation JSONs in outputs/evaluation_results/
    (this script does not run evaluate_results.py)

Usage
-----
    cd rlhf
    python run_efficiency_experiment.py

The RLHF_BETA environment variable is respected (defaults to the value in
config_rlhf.py, currently 0.1).
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "train_reward_model.py",   # Phase 1 — trains RMs, saves rm_timing.json
    "train_ppo_rlhf.py",       # Phase 2 — fine-tunes policies, saves ppo_timing.json
    "aggregate_efficiency.py", # Merges both timing files → efficiency_results.json
]


def run():
    rlhf_dir = Path(__file__).resolve().parent

    # TIMING_RUN=1 tells train_reward_model.py and train_ppo_rlhf.py to redirect
    # all model artifacts to outputs/efficiency_results/ so that the canonical
    # beta0.1/ artifacts used by evaluate_results.py are never overwritten.
    env = os.environ.copy()
    env["TIMING_RUN"] = "1"

    for script in SCRIPTS:
        print(f"\n{'=' * 60}")
        print(f"  Running: {script}")
        print(f"{'=' * 60}\n")
        subprocess.run(
            [sys.executable, script],
            cwd=rlhf_dir,
            env=env,
            check=True,
        )

    print("\n✅  Efficiency experiment complete.")
    print(f"    Results: {rlhf_dir / 'outputs' / 'efficiency_results' / 'efficiency_results.json'}")


if __name__ == "__main__":
    run()
