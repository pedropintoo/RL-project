"""
plot_rm_convergence.py
----------------------
Trains reward models for all (env, K, seed) combinations for a larger number
of epochs than the canonical 10, records the per-epoch average loss, and plots
the loss curves.

The plots reveal whether 10 epochs (current setting) is too few (underfitting),
sufficient (loss plateau before epoch 10), or too many (loss plateau well before
epoch 10).  A vertical dashed line marks epoch 10 on every subplot.

Outputs
-------
    outputs/plots/rm_convergence/<env_id>_rm_convergence.png
    outputs/plots/rm_convergence/rm_loss_curves.json   (raw data)

Usage
-----
    cd rlhf
    python plot_rm_convergence.py
"""

import json
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

RLHF_DIR = Path(__file__).resolve().parent
data_gen_path = RLHF_DIR.parent / "data_generation"
sys.path.insert(0, str(data_gen_path))

from config import PREFERENCE_DIR, ENVIRONMENTS, DATASET_SIZES
from reward_model import RewardModel
from rm_config import RM_TRAIN_CONFIG as ENV_CONFIG

PLOT_DIR = RLHF_DIR / "outputs" / "plots" / "rm_convergence"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

N_EPOCHS  = 50   # always train longer than the canonical cutoff to see the full trajectory
NUM_SEEDS = 5


def train_and_record(env_id: str, K: int, seed: int) -> list[float]:
    """Train one reward model for N_EPOCHS and return per-epoch average loss."""
    file_path = PREFERENCE_DIR / f"{env_id}_K{K}_s{seed}.json"
    if not file_path.exists():
        print(f"  [Warning] Missing: {file_path}")
        return []

    with open(file_path) as f:
        pairs = json.load(f)["pairs"]

    lr = ENV_CONFIG[env_id]["lr"]
    model = RewardModel(env_id)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    epoch_losses = []
    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        for pair in pairs:
            obs1 = torch.tensor(pair["tau1"]["states"])
            act1 = torch.tensor(pair["tau1"]["actions"])
            obs2 = torch.tensor(pair["tau2"]["states"])
            act2 = torch.tensor(pair["tau2"]["actions"])

            optimizer.zero_grad()
            R1 = model(obs1, act1).sum()
            R2 = model(obs2, act2).sum()
            diff = R1 - R2 if pair["preferred"] == 0 else R2 - R1
            loss = -F.logsigmoid(diff)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(pairs)
        epoch_losses.append(avg)
        print(f"    Epoch {epoch+1:2d}/{N_EPOCHS} | Loss: {avg:.4f}")

    return epoch_losses


def plot_env(env_id: str, all_curves: dict):
    """One figure with three subplots (K=50, 200, 1000) for a single environment."""
    canonical = ENV_CONFIG[env_id]["epochs"]
    lr        = ENV_CONFIG[env_id]["lr"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle(f"Reward Model Loss Curves — {env_id}  (lr={lr})", fontsize=13)

    for ax, K in zip(axes, DATASET_SIZES):
        curves = all_curves.get(str(K), {})
        colors = cm.tab10(np.linspace(0, 0.5, NUM_SEEDS))

        for i, (seed_key, losses) in enumerate(sorted(curves.items())):
            epochs = list(range(1, len(losses) + 1))
            ax.plot(epochs, losses, color=colors[i], alpha=0.85,
                    label=f"seed {seed_key}")

        ax.axvline(x=canonical, color="red", linestyle="--", linewidth=1.2,
                   label=f"canonical cutoff (epoch {canonical})")
        ax.set_title(f"K = {K}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg. Bradley-Terry Loss")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    lr_str = str(lr).replace("0.", "").rstrip("0") if lr < 1 else str(lr)
    out_path = PLOT_DIR / f"{env_id}_rm_convergence_epochs{canonical}_lr{lr}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved plot: {out_path}")


def main():
    all_data = {}

    for cfg in ENVIRONMENTS:
        env_id = cfg.env_id
        all_data[env_id] = {}
        print(f"\n{'='*60}")
        print(f"  {env_id}")
        print(f"{'='*60}")

        for K in DATASET_SIZES:
            all_data[env_id][str(K)] = {}
            for seed in range(1, NUM_SEEDS + 1):
                print(f"\n  K={K} | seed={seed}")
                losses = train_and_record(env_id, K, seed)
                all_data[env_id][str(K)][str(seed)] = losses

        plot_env(env_id, all_data[env_id])

    json_path = PLOT_DIR / "rm_loss_curves.json"
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved raw loss data: {json_path}")
    print("\nDone. Check outputs/plots/rm_convergence/ for the plots.")


if __name__ == "__main__":
    main()
