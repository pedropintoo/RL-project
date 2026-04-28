import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Config matching your orchestrator
BETAS = [0.01, 0.1, 0.5, 2.0] # You can adjust this list to include more or fewer beta values as needed

RLHF_DIR = Path(__file__).resolve().parent
EVAL_BASE_DIR = RLHF_DIR / "outputs" / "evaluation_results"

# Create a dedicated folder for these comparison plots
PLOT_DIR = RLHF_DIR / "outputs" / "plots" / "ablation_comparisons"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Distinct colors for the different beta lines
COLORS = ['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6']

def plot_beta_ablation_for_env(env_id):
    plt.figure(figsize=(10, 6))
    
    # 1. Extract baselines from the first beta file 
    # (Baselines are the same across all betas, so we only need to read them once)
    first_json = EVAL_BASE_DIR / f"beta{BETAS[0]}" / f"evaluation_results_beta{BETAS[0]}.json"
    if not first_json.exists():
        print(f"Skipping {env_id}, missing baseline data at {first_json}")
        return
        
    with open(first_json, "r") as f:
        base_data = json.load(f)[env_id]
        
    expert_mean = base_data["baselines"]["expert"]["mean"]
    mid_mean = base_data["baselines"]["mid"]["mean"]
    
    # 2. Plot Baselines as horizontal dashed lines
    plt.axhline(y=expert_mean, color='gold', linestyle='--', linewidth=2, label=rf'Expert ($\pi_1$)')
    plt.axhline(y=mid_mean, color='gray', linestyle='--', linewidth=2, label=rf'Mid ($\pi_2$ Anchor)')
    
    # 3. Loop through each beta and plot its scaling line
    k_values = None
    for idx, beta in enumerate(BETAS):
        json_path = EVAL_BASE_DIR / f"beta{beta}" / f"evaluation_results_beta{beta}.json"
        if not json_path.exists():
            print(f"  [Warning] Missing data for Beta {beta} in {env_id}")
            continue
            
        with open(json_path, "r") as f:
            data = json.load(f)[env_id]["ppo_rlhf"]
            
        # Extract K values and means
        k_values = sorted([int(k) for k in data.keys()])
        ppo_means = [data[str(k)]["mean"] for k in k_values]
        
        # Plot the line for this specific Beta
        plt.plot(k_values, ppo_means, marker='o', color=COLORS[idx % len(COLORS)], 
                 linewidth=2.5, markersize=8, label=rf'PPO-RLHF ($\beta={beta}$)')

    # 4. Formatting
    plt.title(f"Impact of KL Penalty ($\beta$) on RLHF Scaling - {env_id}", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Preference Pairs (K)", fontsize=12)
    plt.ylabel("True Environment Return", fontsize=12)
    plt.xscale('log') 
    
    if k_values:
        plt.xticks(k_values, labels=[str(k) for k in k_values])
        
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='lower right' if env_id == 'CartPole-v1' else 'upper left', fontsize=11)
    
    # 5. Save Plot
    plot_path = PLOT_DIR / f"{env_id}_beta_ablation_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Saved ablation plot to {plot_path}")
    plt.close()

def generate_ablation_plots():
    # Read the first json just to get the list of environments dynamically
    first_json = EVAL_BASE_DIR / f"beta{BETAS[0]}" / f"evaluation_results_beta{BETAS[0]}.json"
    if not first_json.exists():
        print(f"Cannot find {first_json}. Make sure you ran the orchestrator first.")
        return
        
    with open(first_json, "r") as f:
        envs = list(json.load(f).keys())
        
    for env_id in envs:
        plot_beta_ablation_for_env(env_id)

if __name__ == "__main__":
    generate_ablation_plots()