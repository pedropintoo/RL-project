"""
aggregate_dpo_efficiency.py
----------------------------
Reads per-seed summary.json files produced by run_dpo_efficiency_experiment.py
and computes mean/std of wall_clock_s and grad_steps per (env, K).

Saves outputs/dpo_efficiency_results/dpo_efficiency_results.json with the
same schema used by rlhf/outputs/efficiency_results/efficiency_results.json.

Usage
-----
    cd dpo
    python aggregate_dpo_efficiency.py
"""

import json
import statistics
from pathlib import Path

DPO_DIR = Path(__file__).resolve().parent
EFF_DIR = DPO_DIR / "outputs" / "dpo_efficiency_results"

ENVS = ["CartPole-v1", "Pendulum-v1", "MountainCarContinuous-v0"]
DATASET_SIZES = [50, 200, 1000]
SEEDS = [1, 2, 3, 4, 5]


def _stats(values):
    mean = round(statistics.mean(values), 2)
    std = round(statistics.stdev(values) if len(values) > 1 else 0.0, 2)
    return {"mean": mean, "std": std, "seeds": [round(v, 3) for v in values]}


def aggregate():
    results = {}

    for env_id in ENVS:
        results[env_id] = {}
        for k in DATASET_SIZES:
            wall_times, grad_steps_list = [], []
            for seed in SEEDS:
                summary_path = EFF_DIR / env_id / f"K{k}" / f"seed{seed}" / "summary.json"
                if not summary_path.exists():
                    print(f"[Warning] Missing: {summary_path}")
                    continue
                with summary_path.open() as f:
                    data = json.load(f)
                wall_times.append(data["wall_clock_s"])
                grad_steps_list.append(data["grad_steps"])

            if not wall_times:
                print(f"[Warning] No seeds found for {env_id} K={k}, skipping.")
                continue

            results[env_id][str(k)] = {
                "wall_clock_s": _stats(wall_times),
                "grad_steps": _stats(grad_steps_list),
            }

    out_path = EFF_DIR / "dpo_efficiency_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved aggregated DPO efficiency results to {out_path}")
    _print_summary(results)


def _print_summary(results):
    print(f"\n{'Env':<30} {'K':>5}  {'T(s) mean±std':>18}  {'Grad steps mean±std':>22}")
    print("-" * 80)
    for env_id, env_data in results.items():
        for k_str, metrics in sorted(env_data.items(), key=lambda x: int(x[0])):
            wc = metrics["wall_clock_s"]
            gs = metrics["grad_steps"]
            label = f"{env_id} K={k_str}"
            print(f"{label:<30} {k_str:>5}  {wc['mean']:>8.1f} ± {wc['std']:<7.1f}  "
                  f"{gs['mean']:>10.0f} ± {gs['std']:<9.0f}")


if __name__ == "__main__":
    aggregate()
