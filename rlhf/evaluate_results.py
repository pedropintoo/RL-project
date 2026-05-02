import sys
import json
import numpy as np
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

# Create the algorithm registry
ALGO_REGISTRY = {"PPO": PPO, "SAC": SAC}

# Add data_generation to path
data_gen_path = Path(__file__).resolve().parent.parent / "data_generation"
sys.path.append(str(data_gen_path))
from config import ENVIRONMENTS, DATASET_SIZES, POLICY_DIR


RLHF_DIR = Path(__file__).resolve().parent

from config_rlhf import BETA

RLHF_MODELS_DIR = RLHF_DIR / "outputs" / "ppo_rlhf_results" / f"beta{BETA}"

# New dynamic evaluation directory
EVAL_DIR = RLHF_DIR / "outputs" / "evaluation_results" / f"beta{BETA}"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_FILE = EVAL_DIR / f"evaluation_results_beta{BETA}.json"

def evaluate_agent(model_path: Path, env_id: str, algo: str, n_episodes: int = 50):
    """Loads a model and evaluates it on the RAW environment."""
    if not model_path.exists():
        print(f"  [Warning] Model not found: {model_path}")
        return None
    
    # Dynamically load using PPO or SAC
    AlgoCls = ALGO_REGISTRY[algo]
    model = AlgoCls.load(model_path, device="cpu")
    env = gym.make(env_id)
    
    # deterministic=True removes exploration noise for true evaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes, deterministic=True)
    env.close()
    
    return {"mean": float(mean_reward), "std": float(std_reward)}

def run_all_evaluations():
    results = {}
    
    # --- NEW: Load existing results if the file already exists ---
    if EVAL_FILE.exists():
        print(f"Loading existing data from {EVAL_FILE}...")
        with open(EVAL_FILE, "r") as f:
            results = json.load(f)
    # -------------------------------------------------------------

    for cfg in ENVIRONMENTS:
        env_id = cfg.env_id
        algo = cfg.algo # Extract the algorithm from config

        print(f"\n=== Evaluating {env_id} ===")
        results[env_id] = {"baselines": {}, "ppo_rlhf": {}}
        
        # 1. Evaluate Baselines (Expert and Mid)
        print("Evaluating Expert Policy (pi_1)...")
        results[env_id]["baselines"]["expert"] = evaluate_agent(POLICY_DIR / f"{env_id}_expert.zip", env_id, algo)
        
        print("Evaluating Mid Policy (pi_2)...")
        results[env_id]["baselines"]["mid"] = evaluate_agent(POLICY_DIR / f"{env_id}_mid.zip", env_id, algo)
        
        # 2. Evaluate PPO-RLHF Models (Averaging across seeds)
        for K in DATASET_SIZES:
            print(f"Evaluating {algo}-RLHF (K={K})...")
            k_results = []
            for seed in range(1, 6):  # Loop from 1 to 5
                model_path = RLHF_MODELS_DIR / f"{env_id}_K{K}_seed{seed}.zip"
                res = evaluate_agent(model_path, env_id, algo)
                if res:
                    k_results.append(res["mean"])
            
            if k_results:
                # Calculate the overall mean and std deviation ACROSS the 5 seeds
                results[env_id]["ppo_rlhf"][str(K)] = {
                    "mean": float(np.mean(k_results)),
                    "std": float(np.std(k_results)),
                    "raw_seeds": k_results
                }

    # Save to JSON
    with open(EVAL_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nEvaluation complete! Data saved to {EVAL_FILE}")

if __name__ == "__main__":
    run_all_evaluations()