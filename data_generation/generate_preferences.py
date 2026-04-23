"""
Build preference datasets from the trained pi_1 and pi_2 policies.

Pipeline per environment and per dataset size K:
  1. Load pi_1 (expert) and pi_2 (mid) from disk.
  2. Roll out K trajectories with each policy (stochastic actions, so that
     two rollouts of the same policy are not identical — this matters for
     DPO/PPO-RLHF to see some intra-policy variance).
  3. Form K pairs by zipping (tau_1[i], tau_2[i]).
  4. For each pair, label the preferred trajectory by sampling from the
     Bradley-Terry distribution
           p(tau_1 preferred) = exp(R(tau_1)) / (exp(R(tau_1)) + exp(R(tau_2))).
     Implemented as sigmoid(R_1 - R_2) for numerical stability.
  5. Dump everything to JSON (full data) and CSV (summary).

We generate each K independently rather than taking prefixes of the largest
K, because this makes the seed handling explicit and easy to iterate
on a small K without touching the large one. For monotonic subsets,
shuffle / truncate the K=max JSON instead.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from tqdm import tqdm

from config import (
    DATASET_SIZES,
    ENVIRONMENTS,
    POLICY_DIR,
    PREFERENCE_DIR,
    ROLLOUT_SEED,
    EnvConfig,
)
from utils import (
    rollout_trajectory,
    sample_preference,
    save_csv_summary,
    save_json,
)


ALGO_REGISTRY = {"PPO": PPO, "SAC": SAC}


def load_policies(cfg: EnvConfig):
    AlgoCls = ALGO_REGISTRY[cfg.algo]
    pi1 = AlgoCls.load(POLICY_DIR / f"{cfg.env_id}_expert")
    pi2 = AlgoCls.load(POLICY_DIR / f"{cfg.env_id}_mid")
    return pi1, pi2


DEFAULT_N_SEEDS = 5


def build_dataset(cfg: EnvConfig, K: int, seed: int, base_seed: int) -> dict:
    """Create one preference dataset of size K for env `cfg.env_id`.

    `seed` is the effective seed used to drive rollouts and labeling; it is
    typically derived from `base_seed` via a hash of (env_id, K) so that each
    (env, K, base_seed) slot is independent. `base_seed` is recorded so the
    sweep slot is identifiable from the JSON alone.
    """
    env = gym.make(cfg.env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    pi1, pi2 = load_policies(cfg)
    rng = np.random.default_rng(seed)

    pairs = []
    for i in tqdm(range(K), desc=f"{cfg.env_id} K={K} base_seed={base_seed}",
                  leave=False):
        # Use stochastic sampling so repeated rollouts differ.
        tau1 = rollout_trajectory(env, pi1, deterministic=False)
        tau2 = rollout_trajectory(env, pi2, deterministic=False)

        preferred, p_tau1 = sample_preference(tau1["return"], tau2["return"], rng)

        pairs.append(
            {
                "tau1": tau1,
                "tau2": tau2,
                "p_tau1_preferred": p_tau1,
                "preferred": preferred,  # 0 → tau1, 1 → tau2
            }
        )

    env.close()

    # Summary statistics — handy for the report.
    returns_1 = np.array([p["tau1"]["return"] for p in pairs])
    returns_2 = np.array([p["tau2"]["return"] for p in pairs])
    frac_tau1_preferred = float(np.mean([p["preferred"] == 0 for p in pairs]))

    return {
        "env_id": cfg.env_id,
        "K": K,
        "seed": seed,
        "base_seed": base_seed,
        "policies": {
            "pi1": f"{cfg.env_id}_expert",
            "pi2": f"{cfg.env_id}_mid",
        },
        "stats": {
            "mean_R_tau1": float(returns_1.mean()),
            "mean_R_tau2": float(returns_2.mean()),
            "std_R_tau1": float(returns_1.std()),
            "std_R_tau2": float(returns_2.std()),
            "fraction_tau1_preferred": frac_tau1_preferred,
        },
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="*", default=None,
                        help="Subset of env_ids (default: all).")
    parser.add_argument("--sizes", nargs="*", type=int, default=None,
                        help="Override dataset sizes (default: config.DATASET_SIZES).")
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help=f"Base seeds to sweep over. Default: {DEFAULT_N_SEEDS} consecutive "
             "seeds starting at config.ROLLOUT_SEED. One dataset per "
             "(env, K, seed) triple is written.",
    )
    args = parser.parse_args()

    configs = ENVIRONMENTS
    if args.envs:
        configs = [c for c in ENVIRONMENTS if c.env_id in set(args.envs)]
    sizes = args.sizes or DATASET_SIZES
    seeds = args.seeds if args.seeds else [ROLLOUT_SEED + i
                                           for i in range(DEFAULT_N_SEEDS)]

    for cfg in configs:
        for K in sizes:
            for base_seed in seeds:
                # Derive a unique seed per (env, K, base_seed) so the three
                # dataset sizes under one base_seed don't trivially overlap.
                derived = base_seed + hash((cfg.env_id, K)) % 10_000
                dataset = build_dataset(cfg, K, seed=derived, base_seed=base_seed)

                stem = f"{cfg.env_id}_K{K}_s{base_seed}"
                json_path = PREFERENCE_DIR / f"{stem}.json"
                csv_path = PREFERENCE_DIR / f"{stem}.csv"

                save_json(dataset, json_path)
                save_csv_summary(dataset["pairs"], csv_path)

                s = dataset["stats"]
                print(
                    f"[{cfg.env_id} | K={K} | seed={base_seed}] "
                    f"R(pi_1)={s['mean_R_tau1']:.1f}±{s['std_R_tau1']:.1f}, "
                    f"R(pi_2)={s['mean_R_tau2']:.1f}±{s['std_R_tau2']:.1f}, "
                    f"frac τ1 preferred={s['fraction_tau1_preferred']:.2f}  → "
                    f"{json_path.name}"
                )


if __name__ == "__main__":
    main()
