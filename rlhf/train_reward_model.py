import os
import sys
import json
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

# Add the data_generation folder to Python's path so we can import config.py
data_gen_path = Path(__file__).resolve().parent.parent / "data_generation"
sys.path.append(str(data_gen_path))

from config import PREFERENCE_DIR, ENVIRONMENTS, DATASET_SIZES
from reward_model import RewardModel
from rm_config import RM_TRAIN_CONFIG

# Force PyTorch CUDA initialisation once at import time so that the one-time
# ~4s overhead does not inflate the first seed's timing measurement.
torch.zeros(1)

# Define a local outputs folder inside the rlhf directory
RLHF_DIR = Path(__file__).resolve().parent

# When TIMING_RUN=1 (set by run_efficiency_experiment.py), model artifacts are
# redirected to efficiency_results/ so the canonical beta0.1/ artifacts used by
# evaluate_results.py are never overwritten.
_TIMING_RUN = os.environ.get("TIMING_RUN", "0") == "1"
RM_DIR = (
    RLHF_DIR / "outputs" / "efficiency_results" / "reward_models"
    if _TIMING_RUN else
    RLHF_DIR / "outputs" / "reward_models"
)
RM_DIR.mkdir(parents=True, exist_ok=True)

# Timing results are written here; kept separate from model artifacts and
# evaluation JSONs so this script can be re-run without touching prior results.
TIMING_DIR = RLHF_DIR / "outputs" / "efficiency_results"
TIMING_DIR.mkdir(parents=True, exist_ok=True)


def train_reward_model_for_k(env_id: str, K: int, num_seeds: int = 5, epochs: int = 10, lr: float = 3e-4):
    """Train one reward model per seed for the given (env_id, K) pair.

    Returns
    -------
    dict[str, float]
        Wall-clock training time (seconds) for each seed, keyed by str(seed).
        Timing covers data loading + model training + model saving — the full
        Phase 1 cost a practitioner would experience.
    """
    print(f"\n=== Training Reward Models for {env_id} | K={K} ===")

    seed_times: dict[str, float] = {}

    for seed in range(1, num_seeds + 1):
        print(f"\n--- Training RM Seed {seed} ---")

        # Start timing for this seed (includes I/O + training + save).
        t_start = time.perf_counter()

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

                # Use PyTorch's numerically stable log-sigmoid
                loss = -F.logsigmoid(diff)

                loss.backward()

                # Clip the gradients so they can never explode to infinity
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(pairs)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        # 4. Save this specific seed's reward model
        save_path = RM_DIR / f"{env_id}_K{K}_seed{seed}_reward_model.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved reward model to {save_path}")

        seed_times[str(seed)] = round(time.perf_counter() - t_start, 3)
        print(f"Seed {seed} wall-clock time: {seed_times[str(seed)]}s")

    return seed_times


if __name__ == "__main__":
    all_timing: dict = {}

    for cfg in ENVIRONMENTS:
        all_timing[cfg.env_id] = {}
        rm_cfg = RM_TRAIN_CONFIG[cfg.env_id]
        for K in DATASET_SIZES:
            seed_times = train_reward_model_for_k(
                cfg.env_id, K, num_seeds=5,
                epochs=rm_cfg["epochs"], lr=rm_cfg["lr"],
            )
            all_timing[cfg.env_id][str(K)] = seed_times

    timing_path = TIMING_DIR / "rm_timing.json"
    with open(timing_path, "w") as f:
        json.dump(all_timing, f, indent=4)
    print(f"\nSaved Phase 1 (RM) timing to {timing_path}")