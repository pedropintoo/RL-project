import sys
import gymnasium as gym
import torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Add the data_generation folder to Python's path
data_gen_path = Path(__file__).resolve().parent.parent / "data_generation"
sys.path.append(str(data_gen_path))

# Now import from the config
from config import ENVIRONMENTS, DATASET_SIZES, POLICY_DIR
from reward_model import RewardModel
from rlhf_env import RLHFEnvWrapper

# Define local outputs folders inside the rlhf directory
RLHF_DIR = Path(__file__).resolve().parent
PPO_RLHF_DIR = RLHF_DIR / "outputs" / "ppo_rlhf_results"
PPO_RLHF_DIR.mkdir(parents=True, exist_ok=True)
RM_DIR = RLHF_DIR / "outputs" / "reward_models"
LOG_DIR = RLHF_DIR / "outputs" / "logs" # <--- ADD THIS
LOG_DIR.mkdir(parents=True, exist_ok=True) # <--- ADD THIS

def run_ppo_rlhf(cfg, K: int, num_seeds: int = 1):
    print(f"\n=== Running PPO-RLHF for {cfg.env_id} | K={K} ===")
    
    # 1. Load the frozen aggregated Reward Model
    rm_path = RM_DIR / f"{cfg.env_id}_K{K}_reward_model.pth"
    if not rm_path.exists():
        print(f"Reward model not found at {rm_path}. Run train_reward_model.py first.")
        return
        
    reward_model = RewardModel(cfg.env_id)
    reward_model.load_state_dict(torch.load(rm_path))
    reward_model.eval()

    # Run across multiple seeds for robust evaluation
    for seed in range(num_seeds):
        print(f"--- Training Seed {seed+1}/{num_seeds} ---")
        
        # 2. Load the mid-performing policy to act as our Reference Model (Frozen Anchor)
        mid_policy_path = POLICY_DIR / f"{cfg.env_id}_mid"
        ref_model = PPO.load(mid_policy_path)
        ref_policy = ref_model.policy
        ref_policy.eval() # Freeze reference policy

        # 3. Create and Wrap the Environment
        raw_env = gym.make(cfg.env_id)
        raw_env = Monitor(raw_env)
        raw_env.reset(seed=seed)
        
        rlhf_env = RLHFEnvWrapper(raw_env, reward_model, ref_policy, beta=0.1)

        # 4. Initialize the Active PPO Model (starting from the mid-performing weights)
        # We load the zip file again to create a separate updating copy.
        # We also override the tensorboard_log to fix the Mac/Linux path issue!
        active_model = PPO.load(
            mid_policy_path, 
            env=rlhf_env, 
            seed=seed,
            tensorboard_log=str(LOG_DIR) # <--- ADD THIS
        )
        
        # Link the active policy to the environment wrapper so it can compute KL
        rlhf_env.set_active_policy(active_model.policy)

        # 5. Train PPO against the Reward Model
        # We use a smaller budget for fine-tuning, e.g., 50% of the original budget
        tune_budget = int(cfg.total_timesteps * 0.5)
        active_model.learn(total_timesteps=tune_budget)

        # 6. Save the final aligned model
        save_path = PPO_RLHF_DIR / f"{cfg.env_id}_K{K}_seed{seed}"
        active_model.save(save_path)
        print(f"Saved aligned model to {save_path}.zip")
        
        raw_env.close()

if __name__ == "__main__":
    for cfg in ENVIRONMENTS:
        for K in DATASET_SIZES:
            run_ppo_rlhf(cfg, K, num_seeds=3)