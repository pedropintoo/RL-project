import os
import subprocess

BETAS = [0.01, 0.1, 0.5, 2.0] # You can adjust this list to include more or fewer beta values as needed

def run_experiment():
    for beta in BETAS:
        print(f"\nSTARTING ABLATION: BETA = {beta}")
        
        # Set the environment variable for this sub-process
        env = os.environ.copy()
        env["RLHF_BETA"] = str(beta)
        
        # 1. Train PPO Agents for this beta
        print(f"--- Training PPO (Beta={beta}) ---")
        subprocess.run(["python3", "train_ppo_rlhf.py"], env=env, check=True)
        
        # 2. Evaluate results for this beta
        print(f"--- Evaluating (Beta={beta}) ---")
        subprocess.run(["python3", "evaluate_results.py"], env=env, check=True)

    print("\n✅ Ablation Study Complete. All data is in outputs/evaluation_results/")

if __name__ == "__main__":
    run_experiment()