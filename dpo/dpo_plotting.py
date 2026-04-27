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

    plt.title(f"DPO Performance Scaling - {env_id}", fontsize=14, fontweight="bold")
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


def plot_from_results_file(results_file: Path, plot_dir: Path) -> None:
    with results_file.open("r", encoding="utf-8") as f:
        results = json.load(f)

    for env_id, env_data in results.items():
        plot_path = plot_environment_results(env_id, env_data, plot_dir)
        print(f"Saved plot to {plot_path}")
