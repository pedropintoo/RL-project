import sys
import json
import torch
import torch.optim as optim
from pathlib import Path

# Add the data_generation folder to Python's path so we can import config.py
data_gen_path = Path(__file__).resolve().parent.parent / "data_generation"
sys.path.append(str(data_gen_path))

# Now these imports will work perfectly!
from config import PREFERENCE_DIR, ENVIRONMENTS, DATASET_SIZES
from reward_model import RewardModel

# Define a local outputs folder inside the rlhf directory
RLHF_DIR = Path(__file__).resolve().parent
RM_DIR = RLHF_DIR / "outputs" / "reward_models"
RM_DIR.mkdir(parents=True, exist_ok=True)


def train_reward_model_for_k(env_id: str, K: int, num_seeds: int = 5, epochs: int = 10, lr: float = 3e-4):
    print(f"\n=== Training Reward Models for {env_id} | K={K} ===")
    
    for seed in range(num_seeds):
        print(f"\n--- Training RM Seed {seed} ---")
        
        # 1. Load ONLY the specific dataset for this seed
        file_path = PREFERENCE_DIR / f"{env_id}_K{K}_s{seed}.json"
        if not file_path.exists():
            print(f"Data not found at {file_path}. Skipping seed {seed}.")
            continue
            
        with open(file_path, "r") as f:
            data = json.load(f)
            pairs = data["pairs"]
            
        print(f"Loaded {len(pairs)} pairs from seed {seed}.")

        # 2. Setup Model and Optimizer
        model = RewardModel(env_id)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 3. Training Loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            
            for pair in pairs:
                obs1 = torch.tensor(pair["tau1"]["states"])
                act1 = torch.tensor(pair["tau1"]["actions"])
                obs2 = torch.tensor(pair["tau2"]["states"])
                act2 = torch.tensor(pair["tau2"]["actions"])
                
                preferred = pair["preferred"]

                optimizer.zero_grad()
                
                R1_total = model(obs1, act1).sum()
                R2_total = model(obs2, act2).sum()
                
                if preferred == 0:
                    diff = R1_total - R2_total
                else:
                    diff = R2_total - R1_total
                    
                loss = -torch.log(torch.sigmoid(diff))
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(pairs)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        # 4. Save this specific seed's reward model
        save_path = RM_DIR / f"{env_id}_K{K}_seed{seed}_reward_model.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved reward model to {save_path}")

if __name__ == "__main__":
    for cfg in ENVIRONMENTS:
        for K in DATASET_SIZES:
            train_reward_model_for_k(cfg.env_id, K, num_seeds=5)