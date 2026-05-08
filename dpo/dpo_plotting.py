import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_environment_results(env_id: str, data: Dict, plot_dir: Path) -> Path:
    """Create RLHF-style scaling plot for DPO results."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    expert_mean = data["baselines"]["expert"]["mean"]
    mid_mean = data["baselines"]["mid"]["mean"]

    k_values = sorted([int(k) for k in data["dpo"].keys()])
    dpo_means = [data["dpo"][str(k)]["mean"] for k in k_values]
    dpo_stds = [data["dpo"][str(k)]["std"] for k in k_values]
    
    has_rlhf = "ppo_rlhf" in data
    if has_rlhf:
        rlhf_means = [data["ppo_rlhf"][str(k)]["mean"] for k in k_values]
        rlhf_stds = [data["ppo_rlhf"][str(k)]["std"] for k in k_values]

    plt.figure(figsize=(8, 6))
    plt.axhline(y=expert_mean, color="gold", linestyle="--", linewidth=2, label=r"Expert ($\pi_1$)")
    plt.axhline(y=mid_mean, color="gray", linestyle="--", linewidth=2, label=r"Mid ($\pi_2$ Anchor)")

    plt.plot(
        k_values,
        dpo_means,
        marker="o",
        color="royalblue",
        linewidth=2,
        markersize=8,
        label="DPO (Ours)",
    )
    plt.fill_between(
        k_values,
        np.array(dpo_means) - np.array(dpo_stds),
        np.array(dpo_means) + np.array(dpo_stds),
        color="royalblue",
        alpha=0.2,
    )

    if has_rlhf:
        plt.plot(
            k_values,
            rlhf_means,
            marker="s",
            color="red",
            linewidth=2,
            markersize=8,
            label="RLHF",
        )
        plt.fill_between(
            k_values,
            np.array(rlhf_means) - np.array(rlhf_stds),
            np.array(rlhf_means) + np.array(rlhf_stds),
            color="red",
            alpha=0.2,
        )

    plt.title(f"DPO {'vs RLHF' if has_rlhf else ''} Performance Scaling - {env_id}", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Preference Pairs (K)", fontsize=12)
    plt.ylabel("True Environment Return", fontsize=12)
    plt.xscale("log")
    plt.xticks(k_values, labels=[str(k) for k in k_values])
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="lower right" if env_id == "CartPole-v1" else "upper left", fontsize=11)

    plot_path = plot_dir / f"{env_id}_scaling_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()

    return plot_path


def plot_training_curves(env_id: str, data: Dict, plot_dir: Path) -> Path:
    """Plot mean return vs training epoch for each dataset size K."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    training_curves = data.get("training_curves", {})
    if not training_curves:
        print(f"No training curves available for {env_id}.")
        return None

    mid_mean = data["baselines"]["mid"]["mean"]
    expert_mean = data["baselines"]["expert"]["mean"]

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(training_curves)))
    k_values = sorted(training_curves.keys(), key=int)

    plt.figure(figsize=(8, 6))
    plt.axhline(y=expert_mean, color="gold", linestyle="--", linewidth=2, label=r"Expert ($\pi_1$)")
    plt.axhline(y=mid_mean, color="gray", linestyle="--", linewidth=2, label=r"Mid ($\pi_2$ Anchor)")

    for color, k in zip(colors, k_values):
        curve = training_curves[k]
        epochs = curve["epochs"]
        means = np.array(curve["mean_returns"])
        stds = np.array(curve["std_returns"])
        plt.plot(epochs, means, marker="o", color=color, linewidth=2, markersize=5, label=f"K={k}")
        plt.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.15)

    plt.title(f"DPO Training Curve - {env_id}", fontsize=14, fontweight="bold")
    plt.xlabel("Training Epoch", fontsize=12)
    plt.ylabel("Mean Environment Return (eval)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=11)

    plot_path = plot_dir / f"{env_id}_training_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()

    return plot_path


def plot_from_results_file(results_file: Path, plot_dir: Path) -> None:
    with results_file.open("r", encoding="utf-8") as f:
        results = json.load(f)

    for env_id, env_data in results.items():
        plot_path = plot_environment_results(env_id, env_data, plot_dir)
        print(f"Saved plot to {plot_path}")
        curve_path = plot_training_curves(env_id, env_data, plot_dir)
        if curve_path:
            print(f"Saved training curve to {curve_path}")
