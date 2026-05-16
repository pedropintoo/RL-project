"""
aggregate_efficiency.py
-----------------------
Reads rm_timing.json and ppo_timing.json produced by train_reward_model.py
and train_ppo_rlhf.py, then writes efficiency_results.json with per-(env, K)
mean and std wall-clock times for Phase 1, Phase 2, and the combined total.

All three files live in outputs/efficiency_results/ and are kept separate from
the evaluation JSONs in outputs/evaluation_results/ so prior experiment results
are never overwritten.
"""

import json
import numpy as np
from pathlib import Path

RLHF_DIR = Path(__file__).resolve().parent
TIMING_DIR = RLHF_DIR / "outputs" / "efficiency_results"


def aggregate():
    rm_path  = TIMING_DIR / "rm_timing.json"
    ppo_path = TIMING_DIR / "ppo_timing.json"

    if not rm_path.exists() or not ppo_path.exists():
        raise FileNotFoundError(
            "Timing files not found. Run train_reward_model.py and "
            "train_ppo_rlhf.py first (or use run_efficiency_experiment.py)."
        )

    with open(rm_path) as f:
        rm_timing = json.load(f)
    with open(ppo_path) as f:
        ppo_timing = json.load(f)

    results: dict = {}

    for env_id in rm_timing:
        results[env_id] = {}
        for k_str in rm_timing[env_id]:
            rm_seeds  = list(rm_timing[env_id][k_str].values())
            ppo_seeds = list(ppo_timing[env_id][k_str].values())

            # Pair seeds by position; both dicts must have the same seed count.
            assert len(rm_seeds) == len(ppo_seeds), (
                f"Seed count mismatch for {env_id} K={k_str}: "
                f"RM={len(rm_seeds)}, PPO={len(ppo_seeds)}"
            )
            total_seeds = [r + p for r, p in zip(rm_seeds, ppo_seeds)]

            def stats(values: list) -> dict:
                return {
                    "mean": round(float(np.mean(values)), 2),
                    "std":  round(float(np.std(values)),  2),
                    "seeds": [round(v, 3) for v in values],
                }

            results[env_id][k_str] = {
                "phase1_time_s": stats(rm_seeds),
                "phase2_time_s": stats(ppo_seeds),
                "total_time_s":  stats(total_seeds),
            }

    out_path = TIMING_DIR / "efficiency_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved efficiency results to {out_path}")

    # Human-readable summary
    print()
    print(f"{'Env':<28} {'K':>5}  {'Phase1 (s)':>12}  {'Phase2 (s)':>12}  {'Total (s)':>12}")
    print("-" * 72)
    for env_id, env_data in results.items():
        for k_str, v in sorted(env_data.items(), key=lambda x: int(x[0])):
            p1 = v["phase1_time_s"]
            p2 = v["phase2_time_s"]
            tt = v["total_time_s"]
            print(
                f"{env_id:<28} {k_str:>5}  "
                f"{p1['mean']:>7.1f}±{p1['std']:<4.1f}  "
                f"{p2['mean']:>7.1f}±{p2['std']:<4.1f}  "
                f"{tt['mean']:>7.1f}±{tt['std']:<4.1f}"
            )


if __name__ == "__main__":
    aggregate()
