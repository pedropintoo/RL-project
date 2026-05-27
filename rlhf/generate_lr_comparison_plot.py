"""
generate_lr_comparison_plot.py
-------------------------------
Produces a single figure with two subplots (one per environment) comparing
reward model convergence at two learning rates (3e-4 vs 3e-5), K=1000:

  Subplot 1 — Pendulum-v1
  Subplot 2 — MountainCarContinuous-v0

  - lr=3e-5 curves are read from the existing rm_loss_curves.json.
  - lr=3e-4 curves are trained fresh, kept in memory only, and never
    written to rm_loss_curves.json or any other pre-existing file.

Output (new file, never overwrites anything existing):
    outputs/plots/rm_convergence/lr_comparison_K1000.png

Safe to run multiple times: exits cleanly if the output already exists.
"""

import json
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RLHF_DIR = Path(__file__).resolve().parent
data_gen_path = RLHF_DIR.parent / "data_generation"
sys.path.insert(0, str(data_gen_path))

from config import PREFERENCE_DIR
from reward_model import RewardModel

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ENVS      = ["Pendulum-v1", "MountainCarContinuous-v0"]
K         = 1000
N_EPOCHS  = 30
NUM_SEEDS = 5
LR_HIGH   = 3e-4
LR_LOW    = 3e-5

EXISTING_JSON = RLHF_DIR / "outputs" / "plots" / "rm_convergence" / "rm_loss_curves.json"
PLOT_DIR      = RLHF_DIR / "outputs" / "plots" / "rm_convergence"
OUT_PATH      = PLOT_DIR / "lr_comparison_K1000.png"

if OUT_PATH.exists():
    print(f"Output already exists: {OUT_PATH}")
    print("Delete it manually if you want to regenerate.")
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# Training — structured identically to train_and_record() in
# plot_rm_convergence.py, but with an explicit lr parameter instead of
# reading from ENV_CONFIG.
# ---------------------------------------------------------------------------
def train_and_record(env_id: str, K: int, seed: int, lr: float) -> list[float]:
    file_path = PREFERENCE_DIR / f"{env_id}_K{K}_s{seed}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing: {file_path}")

    with open(file_path) as f:
        pairs = json.load(f)["pairs"]

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

            preferred = pair["preferred"]

            optimizer.zero_grad()

            R1 = model(obs1, act1).sum()
            R2 = model(obs2, act2).sum()

            if preferred == 0:
                diff = R1 - R2
            else:
                diff = R2 - R1

            loss = -F.logsigmoid(diff)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(pairs)
        epoch_losses.append(avg)
        print(f"    Epoch {epoch+1:2d}/{N_EPOCHS} | Loss: {avg:.4f}")

    return epoch_losses


def mean_std(curves: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array(curves)   # (n_seeds, n_epochs)
    return arr.mean(axis=0), arr.std(axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    with open(EXISTING_JSON) as f:
        saved = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = np.arange(1, N_EPOCHS + 1)

    for ax, env_id in zip(axes, ENVS):

        # lr=3e-5: reuse existing JSON data (no retraining needed)
        curves_low = [
            saved[env_id][str(K)][str(seed)][:N_EPOCHS]
            for seed in range(1, NUM_SEEDS + 1)
        ]

        # lr=3e-4: train fresh, result kept in memory only
        print(f"\nTraining lr={LR_HIGH:.0e} for {env_id}, K={K} …")
        curves_high = [
            train_and_record(env_id, K, seed, LR_HIGH)
            for seed in range(1, NUM_SEEDS + 1)
        ]

        mean_high, std_high = mean_std(curves_high)
        mean_low,  std_low  = mean_std(curves_low)

        ax.plot(epochs, mean_high, color="#d62728", linewidth=2,
                label=r"lr $= 3{\times}10^{-4}$ (too high)")
        ax.fill_between(epochs,
                        mean_high - std_high,
                        mean_high + std_high,
                        color="#d62728", alpha=0.18)

        ax.plot(epochs, mean_low, color="#1f77b4", linewidth=2,
                label=r"lr $= 3{\times}10^{-5}$ (chosen)")
        ax.fill_between(epochs,
                        mean_low - std_low,
                        mean_low + std_low,
                        color="#1f77b4", alpha=0.18)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Avg. Bradley–Terry Loss", fontsize=11)
        ax.set_title(env_id, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Reward Model Convergence — K={K} (mean ± 1 std, {NUM_SEEDS} seeds)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=180)
    plt.close()
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
