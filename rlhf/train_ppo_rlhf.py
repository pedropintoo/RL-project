import os
import sys
import json
import time
import gymnasium as gym
import torch
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

ALGO_REGISTRY = {"PPO": PPO, "SAC": SAC}

# Add the data_generation folder to Python's path
data_gen_path = Path(__file__).resolve().parent.parent / "data_generation"
sys.path.append(str(data_gen_path))

from config import ENVIRONMENTS, DATASET_SIZES, POLICY_DIR
from reward_model import RewardModel
from rlhf_env import RLHFEnvWrapper
from rm_config import RM_TRAIN_CONFIG

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

# Early-stopping hyperparameters — mirror DPO's defaults exactly.
N_EVAL_EPISODES = 20   # episodes per evaluation checkpoint (DPO uses 20)
PLATEAU_WINDOW  = 12   # stop if no improvement in this many consecutive evals (DPO plateau_window=12)


def _grad_steps_from_timesteps(active_model, cfg, actual_ts: int) -> int:
    """Compute exact gradient steps from the actual timesteps run."""
    if cfg.algo == "PPO":
        n_steps     = active_model.n_steps     # rollout buffer size (default 2048)
        batch_size  = active_model.batch_size  # minibatch size (default 64)
        n_epochs    = active_model.n_epochs    # SGD passes per update (default 10)
        return (actual_ts // n_steps) * (n_steps // batch_size) * n_epochs
    else:  # SAC
        tf = active_model.train_freq.frequency   # env steps between updates (32)
        gs = active_model.gradient_steps         # gradient steps per update  (32)
        return (actual_ts // tf) * gs


def run_ppo_rlhf(cfg, K: int, num_seeds: int = 5):
    """Fine-tune one policy per seed against its paired reward model.

    Returns
    -------
    dict[str, dict]
        Per-seed {"time_s": float, "grad_steps": int}, keyed by str(seed).
        time_s covers the full active_model.learn() call (training + env
        evaluations for checkpoint selection), matching how DPO is timed.
        grad_steps is exact and accounts for early stopping.
    """
    print(f"\n=== Running PPO-RLHF for {cfg.env_id} | K={K} ===")

    seed_results: dict[str, dict] = {}

    for seed in range(1, num_seeds + 1):
        print(f"\n--- Training PPO Seed {seed}/{num_seeds} ---")

        # 1. Load the SPECIFIC Reward Model for this seed
        rm_path = RM_DIR / f"{cfg.env_id}_K{K}_seed{seed}_reward_model.pth"
        if not rm_path.exists():
            print(f"Reward model not found at {rm_path}. Skipping.")
            continue

        reward_model = RewardModel(cfg.env_id)
        reward_model.load_state_dict(torch.load(rm_path))
        reward_model.eval()

        # 2. Load the mid-performing policy as the reference anchor
        mid_policy_path = POLICY_DIR / f"{cfg.env_id}_mid"
        AlgoCls = ALGO_REGISTRY[cfg.algo]
        ref_model = AlgoCls.load(mid_policy_path, device="cpu")
        ref_policy = ref_model.policy
        ref_policy.eval()

        # 3. Create and wrap the training environment
        raw_env = gym.make(cfg.env_id)
        raw_env = Monitor(raw_env)
        raw_env.reset(seed=seed)

        rlhf_env = RLHFEnvWrapper(raw_env, reward_model, ref_policy, beta=BETA)

        # 4. Initialize active model from the mid policy
        active_model = AlgoCls.load(
            mid_policy_path,
            env=rlhf_env,
            seed=seed,
            tensorboard_log=str(LOG_DIR),
            device="cpu"
        )
        rlhf_env.set_active_policy(active_model.policy)

        # 5. Early-stopping callbacks — mirror DPO: evaluate in the raw env,
        #    keep the best checkpoint by mean return, stop when plateau_window
        #    consecutive evaluations show no improvement.
        eval_env = Monitor(gym.make(cfg.env_id))
        ckpt_dir = PPO_RLHF_DIR / f"{cfg.env_id}_K{K}_seed{seed}_ckpt"
        stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=PLATEAU_WINDOW,
            min_evals=PLATEAU_WINDOW,
            verbose=1,
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir),
            eval_freq=cfg.eval_freq,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            callback_after_eval=stop_cb,
            verbose=1,
        )

        # 6. Train — timed end-to-end (training + eval overhead), matching DPO timing.
        tune_budget = cfg.total_timesteps
        run_name = f"{cfg.env_id}_K{K}_seed{seed}"

        t_start = time.perf_counter()
        active_model.learn(total_timesteps=tune_budget, tb_log_name=run_name, callback=eval_cb)
        wall_clock_s = round(time.perf_counter() - t_start, 3)

        actual_ts  = active_model.num_timesteps
        grad_steps = _grad_steps_from_timesteps(active_model, cfg, actual_ts)
        print(f"Seed {seed} | wall-clock: {wall_clock_s}s | actual_ts: {actual_ts} | grad_steps: {grad_steps}")

        # 7. Save the best checkpoint (not the last weights after early stopping).
        save_path = PPO_RLHF_DIR / f"{cfg.env_id}_K{K}_seed{seed}"
        best_zip = ckpt_dir / "best_model.zip"
        if best_zip.exists():
            AlgoCls.load(str(ckpt_dir / "best_model"), device="cpu").save(str(save_path))
        else:
            active_model.save(str(save_path))
        print(f"Saved aligned model to {save_path}.zip")

        raw_env.close()
        eval_env.close()

        seed_results[str(seed)] = {"time_s": wall_clock_s, "grad_steps": grad_steps}

    return seed_results


if __name__ == "__main__":
    all_timing: dict = {}

    for cfg in ENVIRONMENTS:
        all_timing[cfg.env_id] = {}
        for K in DATASET_SIZES:
            all_timing[cfg.env_id][str(K)] = run_ppo_rlhf(cfg, K, num_seeds=5)

    if os.environ.get("ABLATION_RUN", "0") != "1":
        timing_path = TIMING_DIR / "ppo_timing.json"
        with open(timing_path, "w") as f:
            json.dump(all_timing, f, indent=4)
        print(f"\nSaved Phase 2 (PPO) timing to {timing_path}")
