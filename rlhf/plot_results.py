import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config_rlhf import BETA

RLHF_DIR = Path(__file__).resolve().parent

# Read from the dynamic evaluation directory
EVAL_FILE = RLHF_DIR / "outputs" / "evaluation_results" / f"beta{BETA}" / f"evaluation_results_beta{BETA}.json"

# Save to a dynamic plot directory
PLOT_DIR = RLHF_DIR / "outputs" / "plots" / f"beta{BETA}"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def plot_environment_results(env_id, data):
    plt.figure(figsize=(8, 6))
    
    # 1. Extract baseline data
    expert_mean = data["baselines"]["expert"]["mean"]
    mid_mean = data["baselines"]["mid"]["mean"]
    
    # 2. Extract PPO-RLHF data
    k_values = sorted([int(k) for k in data["ppo_rlhf"].keys()])
    ppo_means = [data["ppo_rlhf"][str(k)]["mean"] for k in k_values]
    ppo_stds = [data["ppo_rlhf"][str(k)]["std"] for k in k_values]
    
    # 3. Plot Baselines as horizontal dashed lines
    plt.axhline(y=expert_mean, color='gold', linestyle='--', linewidth=2, label=rf'Expert ($\pi_1$)')
    plt.axhline(y=mid_mean, color='gray', linestyle='--', linewidth=2, label=rf'Mid ($\pi_2$ Anchor)')
    
    # 4. Plot PPO-RLHF scaling line with error bands
    plt.plot(k_values, ppo_means, marker='o', color='royalblue', linewidth=2, markersize=8, label='PPO-RLHF (Ours)')
    plt.fill_between(k_values, 
                     np.array(ppo_means) - np.array(ppo_stds), 
                     np.array(ppo_means) + np.array(ppo_stds), 
                     color='royalblue', alpha=0.2)
    
    # 5. Formatting
    plt.title(f"RLHF Performance Scaling - {env_id}", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Preference Pairs (K)", fontsize=12)
    plt.ylabel("True Environment Return", fontsize=12)
    plt.xscale('log') # Log scale for dataset sizes (50 -> 200 -> 1000)
    plt.xticks(k_values, labels=[str(k) for k in k_values])
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right' if env_id == 'CartPole-v1' else 'upper left', fontsize=11)
    
    # 6. Save Plot
    plot_path = PLOT_DIR / f"{env_id}_scaling_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")
    plt.close()

def generate_plots():
    if not EVAL_FILE.exists():
        print(f"Evaluation file not found: {EVAL_FILE}. Run evaluate_results.py first.")
        return
        
    with open(EVAL_FILE, "r") as f:
        results = json.load(f)
        
    for env_id, data in results.items():
        plot_environment_results(env_id, data)

if __name__ == "__main__":
    generate_plots()