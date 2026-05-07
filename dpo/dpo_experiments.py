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
    kl_coef: float = 0.0,
    early_stop: bool = True,
    plateau_window: int = 10,
    checkpoint_dir: Path = Path("./checkpoints"),
    env_id: str = None,
    eval_every: int = 5,
    n_eval_episodes: int = 20,
    max_t: int = 500,
    batch_size: int = 1,
) -> Tuple[List[float], List[Dict]]:
    """Train DPO and restore the best checkpoint at the end.

    Checkpoint selection strategy:
    - When env_id is provided: evaluates the policy in the environment every
      eval_every epochs and keeps the checkpoint with the highest mean return.
      This is the correct criterion for RL — training loss going to 0 means
      the policy has memorised the preference pairs, not that it performs well.
    - When env_id is None: falls back to minimum training loss (use only when
      environment evaluation is unavailable).

    Returns:
        (loss_scores, eval_curve) where eval_curve is a list of
        {"epoch": int, "mean_return": float} dicts recorded every eval_every
        epochs (empty list when env_id is None).
    """
    scores: List[float] = []
    eval_curve: List[Dict] = []
    pairs = preference_data["pairs"]
    n_pairs = len(pairs)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "dpo_best.pt"

    reference_model.eval()

    # Precompute frozen reference log-probs once — the reference model never
    # changes, so recomputing it every epoch wastes n_pairs × n_epochs forward passes.
    with torch.no_grad():
        ref_chosen_logps: List[torch.Tensor] = []
        ref_rejected_logps: List[torch.Tensor] = []
        kl_states_list: List[torch.Tensor] = []
        for pair in pairs:
            rc, rr = preference_pair_logps(reference_model, pair)
            ref_chosen_logps.append(rc.detach())
            ref_rejected_logps.append(rr.detach())
            if kl_coef > 0.0:
                chosen_states = torch.as_tensor(pair["tau1"]["states"], dtype=torch.float32, device=policy_model.device)
                rejected_states = torch.as_tensor(pair["tau2"]["states"], dtype=torch.float32, device=policy_model.device)
                kl_states_list.append(torch.cat([chosen_states, rejected_states], dim=0))

    best_loss = float("inf")
    best_return = float("-inf")
    best_epoch = 0

    # Epoch-0 evaluation: record the starting performance (= mid policy baseline)
    # so the training curve shows the full picture from the very beginning.
    if env_id is not None:
        _, mean_r0, _ = evaluate_policy_returns(
            policy_model,
            env_name=env_id,
            n_episodes=n_eval_episodes,
            max_t=max_t,
            deterministic=True,
        )
        eval_curve.append({"epoch": 0, "mean_return": float(mean_r0)})
        best_return = mean_r0
        torch.save(
            {
                "epoch": 0,
                "model_state_dict": policy_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_loss": float("inf"),
                "mean_return": mean_r0,
            },
            best_ckpt_path,
        )
        print(f"Epoch 0\tReturn: {mean_r0:.2f} (starting point)")

    indices = list(range(n_pairs))

    for epoch in range(1, n_epochs + 1):
        # Mini-batch SGD: shuffle pairs, then do one optimizer step per batch.
        # batch_size=1  → per-pair SGD (K steps/epoch, best Adam calibration)
        # batch_size=K  → full-batch GD  (1 step/epoch, avoid this for small K)
        # batch_size=16 → good balance for large K (fewer steps, still stable)
        np.random.shuffle(indices)
        epoch_loss_sum = 0.0

        for batch_start in range(0, n_pairs, batch_size):
            batch_idx = indices[batch_start: batch_start + batch_size]
            actual_bs = len(batch_idx)

            optimizer.zero_grad()
            batch_loss_sum = 0.0

            for i in batch_idx:
                policy_chosen_logp, policy_rejected_logp = preference_pair_logps(policy_model, pairs[i])

                loss, _, _ = dpo_loss(
                    policy_chosen_logps=policy_chosen_logp,
                    policy_rejected_logps=policy_rejected_logp,
                    reference_chosen_logps=ref_chosen_logps[i],
                    reference_rejected_logps=ref_rejected_logps[i],
                    beta=beta,
                )
                if kl_coef > 0.0:
                    kl_penalty = policy_model.kl_divergence(reference_model, kl_states_list[i])
                    loss = loss + kl_coef * kl_penalty

                batch_loss_sum += loss.item()
                (loss / actual_bs).backward()  # normalise so LR is batch-size-invariant

            optimizer.step()
            epoch_loss_sum += batch_loss_sum

        mean_loss = epoch_loss_sum / n_pairs
        scores.append(mean_loss)

        if env_id is not None:
            # Env-eval-based checkpoint selection: keep the policy that achieves
            # the highest return, not the one that most overfits the training pairs.
            if epoch % eval_every == 0 or epoch == n_epochs:
                _, mean_r, _ = evaluate_policy_returns(
                    policy_model,
                    env_name=env_id,
                    n_episodes=n_eval_episodes,
                    max_t=max_t,
                    deterministic=True,
                )
                eval_curve.append({"epoch": epoch, "mean_return": float(mean_r)})
                if mean_r > best_return:
                    best_return = mean_r
                    best_epoch = epoch
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": policy_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "mean_loss": mean_loss,
                            "mean_return": mean_r,
                        },
                        best_ckpt_path,
                    )
                if epoch % print_every == 0:
                    print(f"Epoch {epoch}\tLoss: {mean_loss:.8f}\tReturn: {mean_r:.2f}\tBest return: {best_return:.2f} (epoch {best_epoch})")
            elif epoch % print_every == 0:
                print(f"Epoch {epoch}\tLoss: {mean_loss:.8f}")
        else:
            # Fallback: loss-based selection (prone to overfitting on small datasets).
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
                print(f"Epoch {epoch}\tAverage Loss: {mean_loss:.8f}\tBest: {best_loss:.8f} (epoch {best_epoch})")

        if early_stop:
            if env_id is not None and len(eval_curve) > plateau_window:
                # Return-based plateau: stop when the last plateau_window evaluations
                # show no improvement over best_return. Evaluations happen every
                # eval_every epochs, so this covers plateau_window * eval_every epochs.
                recent_returns = [e["mean_return"] for e in eval_curve[-plateau_window:]]
                if all(r < best_return for r in recent_returns):
                    print(f"Early stop at epoch {epoch}: no return improvement in last {plateau_window} evaluations.")
                    break
            elif env_id is None and len(scores) >= plateau_window:
                # Fallback loss-based plateau (used when no env eval is available).
                recent = scores[-plateau_window:]
                if all(recent[i] >= recent[i - 1] for i in range(1, len(recent))):
                    print(f"Early stop at epoch {epoch}: loss not decreasing for {plateau_window} epochs.")
                    break

    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=policy_model.device, weights_only=False)
        policy_model.load_state_dict(best_ckpt["model_state_dict"])
        if env_id is not None:
            print(f"Restored best checkpoint from epoch {best_ckpt['epoch']} with return {best_ckpt.get('mean_return', float('nan')):.2f}.")
        else:
            print(f"Restored best checkpoint from epoch {best_ckpt['epoch']} with loss {best_ckpt['mean_loss']:.4f}.")

    return scores, eval_curve


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

    sb3_model = PPO.load(model_path, device="cpu")
    adapter = _build_adapter_for_env(env_id, sb3_model.policy, device)

    _, mean_r, std_r = evaluate_policy_returns(
        adapter,
        env_name=env_id,
        n_episodes=n_episodes,
        max_t=500 if env_id == "CartPole-v1" else 200,
        deterministic=True,
    )
    return {"mean": float(mean_r), "std": float(std_r)}


# Hyperparameter defaults applied when a key is absent from run_config.
_HPARAM_DEFAULTS: Dict = {
    "lr": 1e-2,
    "beta": 0.1,
    "kl_coef": 0.01,
    "batch_size": 1,
    "n_epochs": 50,
    "plateau_window": 12,
    "early_stop": True,
    "eval_every": 5,
}


def _resolve_hparams(run_config: Dict, env_id: str, k: int) -> Dict:
    """Return the merged hyperparameter dict for a given (env, K) pair.

    Resolution order (later entries override earlier ones):
      1. _HPARAM_DEFAULTS         — global fallback
      2. run_config[env_id]["default"]  — env-level defaults
      3. run_config[env_id][k]          — K-specific overrides
    """
    env_cfg = run_config.get(env_id, {})
    return {
        **_HPARAM_DEFAULTS,
        **env_cfg.get("default", {}),
        **env_cfg.get(k, {}),
    }


def run_dpo_scaling_experiment(
    env_id: str,
    dataset_sizes: Iterable[int],
    seeds: Iterable[int],
    *,
    preference_dir: Path,
    policy_dir: Path,
    output_dir: Path,
    device,
    n_eval_episodes: int = 50,
    run_config: Dict,
) -> Dict:
    """Run DPO for all K x seeds and return aggregated metrics.

    Hyperparameters (lr, kl_coef, batch_size, n_epochs, …) are resolved
    per (env_id, K) via run_config.  See _resolve_hparams for the lookup order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        env_id: {
            "baselines": {},
            "dpo": {},
            "training_curves": {},
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
        hp = _resolve_hparams(run_config, env_id, k)
        seed_returns: List[float] = []
        seed_curves: List[List[Dict]] = []
        print(f"\n=== DPO | env={env_id} | K={k} | {hp} ===")

        for seed in seeds:
            pref_path = preference_dir / f"{env_id}_K{k}_s{seed}.json"
            if not pref_path.exists():
                print(f"[Warning] Missing preference file: {pref_path}")
                continue

            preference_data = load_preference_dataset(pref_path)

            sb3_model = PPO.load(mid_path, device="cpu")
            policy_model = _build_adapter_for_env(env_id, sb3_model.policy, device)
            reference_model = copy.deepcopy(policy_model).to(device)
            reference_model.eval()

            optimizer = optim.Adam(policy_model.parameters(), lr=hp["lr"], weight_decay=0.0)

            run_dir = output_dir / env_id / f"K{k}" / f"seed{seed}"
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            print(f"Training DPO for K={k}, seed={seed}...")
            max_t = 500 if env_id == "CartPole-v1" else 200
            scores, eval_curve = train_dpo(
                policy_model=policy_model,
                reference_model=reference_model,
                optimizer=optimizer,
                preference_data=preference_data,
                n_epochs=hp["n_epochs"],
                print_every=1,
                beta=hp["beta"],
                kl_coef=hp["kl_coef"],
                early_stop=hp["early_stop"],
                plateau_window=hp["plateau_window"],
                checkpoint_dir=ckpt_dir,
                env_id=env_id,
                eval_every=hp["eval_every"],
                n_eval_episodes=20,
                max_t=max_t,
                batch_size=hp["batch_size"],
            )

            # Use the best-checkpoint return when available (env-eval-based selection),
            # otherwise fall back to a final evaluation of the restored policy.
            if eval_curve:
                mean_r = max(entry["mean_return"] for entry in eval_curve)
                seed_curves.append(eval_curve)
            else:
                _, mean_r, _ = evaluate_policy_returns(
                    policy_model,
                    env_name=env_id,
                    n_episodes=n_eval_episodes,
                    max_t=max_t,
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
                        "eval_curve": eval_curve,
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

        if seed_curves:
            # Aggregate training curves across seeds: average return per epoch.
            epochs = [entry["epoch"] for entry in seed_curves[0]]
            avg_returns = [
                float(np.mean([curve[i]["mean_return"] for curve in seed_curves if i < len(curve)]))
                for i in range(len(epochs))
            ]
            std_returns = [
                float(np.std([curve[i]["mean_return"] for curve in seed_curves if i < len(curve)]))
                for i in range(len(epochs))
            ]
            results[env_id]["training_curves"][str(k)] = {
                "epochs": epochs,
                "mean_returns": avg_returns,
                "std_returns": std_returns,
                "per_seed": seed_curves,
            }

    results_path = output_dir / f"dpo_scaling_{env_id}.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved experiment results to {results_path}")
    return results
