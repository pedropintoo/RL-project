import os
import sys
import json
import time
import gymnasium as gym
import torch
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

ALGO_REGISTRY = {"PPO": PPO, "SAC": SAC}

# Add the data_generation folder to Python's path
data_gen_path = Path(__file__).resolve().parent.parent / "data_generation"
sys.path.append(str(data_gen_path))

from config import ENVIRONMENTS, DATASET_SIZES, POLICY_DIR
from reward_model import RewardModel
from rlhf_env import RLHFEnvWrapper

# Define local outputs folders inside the rlhf directory
RLHF_DIR = Path(__file__).resolve().parent

from config_rlhf import BETA

# When TIMING_RUN=1 (set by run_efficiency_experiment.py), all artifacts are
# redirected to efficiency_results/ so the canonical beta0.1/ artifacts used by
# evaluate_results.py are never overwritten.  Phase 2 loads reward models from
# the same redirected folder so the two phases stay consistent.
_TIMING_RUN = os.environ.get("TIMING_RUN", "0") == "1"
_EFF_BASE = RLHF_DIR / "outputs" / "efficiency_results"

PPO_RLHF_DIR = (
    _EFF_BASE / "ppo_rlhf_results" / f"beta{BETA}"
    if _TIMING_RUN else
    RLHF_DIR / "outputs" / "ppo_rlhf_results" / f"beta{BETA}"
)
RM_DIR = (
    _EFF_BASE / "reward_models"
    if _TIMING_RUN else
    RLHF_DIR / "outputs" / "reward_models"
)
LOG_DIR = (
    _EFF_BASE / "logs" / f"beta{BETA}"
    if _TIMING_RUN else
    RLHF_DIR / "outputs" / "logs" / f"beta{BETA}"
)
for d in (PPO_RLHF_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Timing results are written here; kept separate from model artifacts and
# evaluation JSONs so this script can be re-run without touching prior results.
TIMING_DIR = RLHF_DIR / "outputs" / "efficiency_results"
TIMING_DIR.mkdir(parents=True, exist_ok=True)


def run_ppo_rlhf(cfg, K: int, num_seeds: int = 5):
    """Fine-tune one policy per seed against its paired reward model.

    Returns
    -------
    dict[str, float]
        Wall-clock time (seconds) of the active_model.learn() call for each
        seed, keyed by str(seed).  Setup (loading models, creating the env) is
        excluded; only the policy-optimisation phase is timed so that Phase 2
        timing is directly comparable to DPO training time.
    """
    print(f"\n=== Running PPO-RLHF for {cfg.env_id} | K={K} ===")

    seed_times: dict[str, float] = {}

    for seed in range(1, num_seeds + 1):
        print(f"\n--- Training PPO Seed {seed}/{num_seeds} ---")

        # 1. Load the SPECIFIC Reward Model for this seed!
        rm_path = RM_DIR / f"{cfg.env_id}_K{K}_seed{seed}_reward_model.pth"
        if not rm_path.exists():
            print(f"Reward model not found at {rm_path}. Skipping.")
            continue

        reward_model = RewardModel(cfg.env_id)
        reward_model.load_state_dict(torch.load(rm_path))
        reward_model.eval()

        # 2. Load the mid-performing policy to act as our Anchor
        mid_policy_path = POLICY_DIR / f"{cfg.env_id}_mid"
        AlgoCls = ALGO_REGISTRY[cfg.algo]  # Dynamically get PPO or SAC
        ref_model = AlgoCls.load(mid_policy_path, device="cpu")
        ref_policy = ref_model.policy
        ref_policy.eval()

        # 3. Create and Wrap the Environment
        raw_env = gym.make(cfg.env_id)
        raw_env = Monitor(raw_env)
        raw_env.reset(seed=seed)

        rlhf_env = RLHFEnvWrapper(raw_env, reward_model, ref_policy, beta=BETA)

        # 4. Initialize Active Model
        active_model = AlgoCls.load(
            mid_policy_path,
            env=rlhf_env,
            seed=seed,
            tensorboard_log=str(LOG_DIR),
            device="cpu"
        )
        rlhf_env.set_active_policy(active_model.policy)

        # 5. Train against this specific Reward Model — timed separately so
        #    Phase 2 wall-clock reflects only policy optimisation, not setup.
        tune_budget = int(cfg.total_timesteps * 0.5)
        run_name = f"{cfg.env_id}_K{K}_seed{seed}"

        t_start = time.perf_counter()
        active_model.learn(total_timesteps=tune_budget, tb_log_name=run_name)
        seed_times[str(seed)] = round(time.perf_counter() - t_start, 3)
        print(f"Seed {seed} Phase 2 wall-clock time: {seed_times[str(seed)]}s")

        # 6. Save the final aligned model
        save_path = PPO_RLHF_DIR / f"{cfg.env_id}_K{K}_seed{seed}"
        active_model.save(save_path)
        print(f"Saved aligned model to {save_path}.zip")

        raw_env.close()

    return seed_times


if __name__ == "__main__":
    all_timing: dict = {}

    for cfg in ENVIRONMENTS:
        all_timing[cfg.env_id] = {}
        for K in DATASET_SIZES:
            seed_times = run_ppo_rlhf(cfg, K, num_seeds=5)
            all_timing[cfg.env_id][str(K)] = seed_times

    timing_path = TIMING_DIR / "ppo_timing.json"
    with open(timing_path, "w") as f:
        json.dump(all_timing, f, indent=4)
    print(f"\nSaved Phase 2 (PPO) timing to {timing_path}")