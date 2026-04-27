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


def train_reward_model_for_k(env_id: str, K: int, epochs: int = 10, lr: float = 3e-4):
    print(f"\n--- Training Reward Model for {env_id} | K={K} ---")
    
    # 1. Aggregate all seeds for this K
    all_pairs = []
    files = list(PREFERENCE_DIR.glob(f"{env_id}_K{K}_s*.json"))
    if not files:
        print(f"No data found for {env_id} K={K}. Skipping.")
        return
        
    for file_path in files:
        with open(file_path, "r") as f:
            data = json.load(f)
            all_pairs.extend(data["pairs"])
            
    print(f"Aggregated {len(files)} files. Total pairs: {len(all_pairs)}")

    # 2. Setup Model and Optimizer
    model = RewardModel(env_id)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        
        for pair in all_pairs:
            # Extract tau1
            obs1 = torch.tensor(pair["tau1"]["states"])
            act1 = torch.tensor(pair["tau1"]["actions"])
            # Extract tau2
            obs2 = torch.tensor(pair["tau2"]["states"])
            act2 = torch.tensor(pair["tau2"]["actions"])
            
            preferred = pair["preferred"] # 0 for tau1, 1 for tau2

            optimizer.zero_grad()
            
            # Predict rewards for every step in the trajectory and sum them
            r1_steps = model(obs1, act1)
            R1_total = r1_steps.sum()
            
            r2_steps = model(obs2, act2)
            R2_total = r2_steps.sum()
            
            # Bradley-Terry loss: -log(sigmoid(R_winner - R_loser))
            if preferred == 0:
                diff = R1_total - R2_total
            else:
                diff = R2_total - R1_total
                
            loss = -torch.log(torch.sigmoid(diff))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(all_pairs)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # 4. Save the aggregated reward model
    save_path = RM_DIR / f"{env_id}_K{K}_reward_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved reward model to {save_path}")

if __name__ == "__main__":
    # Train aggregated reward models for all environments and K sizes
    for cfg in ENVIRONMENTS:
        for K in DATASET_SIZES:
            train_reward_model_for_k(cfg.env_id, K)