import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
from stable_baselines3 import PPO

from policy import SB3ContinuousPolicyAdapter, SB3DiscretePolicyAdapter
from utils import evaluate_policy_returns, load_preference_dataset, preference_pair_logps


def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute DPO loss for one preference pair."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def train_dpo(
    policy_model,
    reference_model,
    optimizer: optim.Optimizer,
    preference_data: Dict,
    n_epochs: int = 20,
    print_every: int = 1,
    beta: float = 0.1,
    early_stop: bool = True,
    plateau_window: int = 10,
    checkpoint_dir: Path = Path("./checkpoints"),
) -> List[float]:
    """Train DPO and restore the best checkpoint at the end."""
    scores: List[float] = []
    pairs = preference_data["pairs"]

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "dpo_best.pt"

    reference_model.eval()

    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, n_epochs + 1):
        epoch_losses: List[float] = []

        for pair in pairs:
            policy_chosen_logp, policy_rejected_logp = preference_pair_logps(policy_model, pair)
            with torch.no_grad():
                reference_chosen_logp, reference_rejected_logp = preference_pair_logps(reference_model, pair)

            loss, _, _ = dpo_loss(
                policy_chosen_logps=policy_chosen_logp,
                policy_rejected_logps=policy_rejected_logp,
                reference_chosen_logps=reference_chosen_logp,
                reference_rejected_logps=reference_rejected_logp,
                beta=beta,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        scores.append(mean_loss)

        epoch_ckpt_path = checkpoint_dir / f"dpo_epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": policy_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_loss": mean_loss,
            },
            epoch_ckpt_path,
        )

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": policy_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mean_loss": mean_loss,
                },
                best_ckpt_path,
            )

        if epoch % print_every == 0:
            print(f"Epoch {epoch}\tAverage Loss: {mean_loss:.4f}\tBest: {best_loss:.4f} (epoch {best_epoch})")

        if early_stop and len(scores) >= plateau_window:
            recent = scores[-plateau_window:]
            not_going_down = all(recent[i] >= recent[i - 1] for i in range(1, len(recent)))
            if not_going_down:
                print(f"Early stop at epoch {epoch}: last {plateau_window} epochs are not decreasing.")
                break

    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=policy_model.device)
        policy_model.load_state_dict(best_ckpt["model_state_dict"])
        print(
            f"Restored best checkpoint from epoch {best_ckpt['epoch']} with loss {best_ckpt['mean_loss']:.4f}."
        )

    return scores


def _build_adapter_for_env(env_id: str, sb3_policy, device):
    env = gym.make(env_id)
    is_discrete = isinstance(env.action_space, spaces.Discrete)
    env.close()

    if is_discrete:
        return SB3DiscretePolicyAdapter(device, sb3_policy).to(device)
    return SB3ContinuousPolicyAdapter(device, sb3_policy).to(device)


def _evaluate_sb3_checkpoint(model_path: Path, env_id: str, device, n_episodes: int = 50) -> Dict[str, float]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    sb3_model = PPO.load(model_path)
    adapter = _build_adapter_for_env(env_id, sb3_model.policy, device)

    _, mean_r, std_r = evaluate_policy_returns(
        adapter,
        env_name=env_id,
        n_episodes=n_episodes,
        max_t=500 if env_id == "CartPole-v1" else 200,
        deterministic=True,
    )
    return {"mean": float(mean_r), "std": float(std_r)}


def run_dpo_scaling_experiment(
    env_id: str,
    dataset_sizes: Iterable[int],
    seeds: Iterable[int],
    *,
    preference_dir: Path,
    policy_dir: Path,
    output_dir: Path,
    device,
    n_epochs: int = 80,
    lr: float = 1e-4,
    beta: float = 0.1,
    early_stop: bool = True,
    plateau_window: int = 12,
    n_eval_episodes: int = 50,
) -> Dict:
    """Run DPO for all K x seeds and return aggregated metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        env_id: {
            "baselines": {},
            "dpo": {},
        }
    }

    expert_path = policy_dir / f"{env_id}_expert.zip"
    mid_path = policy_dir / f"{env_id}_mid.zip"

    print(f"Evaluating baselines for {env_id}...")
    results[env_id]["baselines"]["expert"] = _evaluate_sb3_checkpoint(
        expert_path, env_id, device, n_episodes=n_eval_episodes
    )
    results[env_id]["baselines"]["mid"] = _evaluate_sb3_checkpoint(
        mid_path, env_id, device, n_episodes=n_eval_episodes
    )

    for k in dataset_sizes:
        seed_returns: List[float] = []
        print(f"\n=== DPO | env={env_id} | K={k} ===")

        for seed in seeds:
            pref_path = preference_dir / f"{env_id}_K{k}_s{seed}.json"
            if not pref_path.exists():
                print(f"[Warning] Missing preference file: {pref_path}")
                continue

            preference_data = load_preference_dataset(pref_path)

            sb3_model = PPO.load(mid_path)
            policy_model = _build_adapter_for_env(env_id, sb3_model.policy, device)
            reference_model = copy.deepcopy(policy_model).to(device)
            reference_model.eval()

            optimizer = optim.Adam(policy_model.parameters(), lr=lr, weight_decay=0.0)

            run_dir = output_dir / env_id / f"K{k}" / f"seed{seed}"
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            print(f"Training DPO for K={k}, seed={seed}...")
            scores = train_dpo(
                policy_model=policy_model,
                reference_model=reference_model,
                optimizer=optimizer,
                preference_data=preference_data,
                n_epochs=n_epochs,
                print_every=1,
                beta=beta,
                early_stop=early_stop,
                plateau_window=plateau_window,
                checkpoint_dir=ckpt_dir,
            )

            _, mean_r, _ = evaluate_policy_returns(
                policy_model,
                env_name=env_id,
                n_episodes=n_eval_episodes,
                max_t=500 if env_id == "CartPole-v1" else 200,
                deterministic=True,
            )
            seed_returns.append(float(mean_r))

            with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "env_id": env_id,
                        "K": k,
                        "seed": seed,
                        "epochs_ran": len(scores),
                        "final_loss": float(scores[-1]) if scores else None,
                        "mean_return": float(mean_r),
                    },
                    f,
                    indent=2,
                )

        if seed_returns:
            results[env_id]["dpo"][str(k)] = {
                "mean": float(np.mean(seed_returns)),
                "std": float(np.std(seed_returns)),
                "raw_seeds": seed_returns,
            }

    results_path = output_dir / f"dpo_scaling_{env_id}.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved experiment results to {results_path}")
    return results
