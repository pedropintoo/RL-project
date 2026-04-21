"""
Shared helpers for rollouts, serialization, and Bradley-Terry labeling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Trajectory rollout
# ---------------------------------------------------------------------------
def rollout_trajectory(env, policy, deterministic: bool = False) -> Dict[str, Any]:
    """Run one episode and return the full (state, action, reward) sequence.

    We keep rewards (even though DPO does not need them) because:
      * the return R(tau) = sum(rewards) drives the Bradley-Terry label, and
      * Person 3's reward-model code may want per-step rewards for debugging.

    `policy` is any object exposing a Stable-Baselines3-style
    `predict(obs, deterministic=...)` returning (action, state).
    """
    obs, _ = env.reset()
    states, actions, rewards = [], [], []
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = policy.predict(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        states.append(np.asarray(obs, dtype=np.float32).tolist())
        # Discrete envs give a 0-d numpy int; continuous envs give a vector.
        actions.append(np.asarray(action).tolist())
        rewards.append(float(reward))

        obs = next_obs

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "return": float(np.sum(rewards)),
        "length": len(rewards),
    }


# ---------------------------------------------------------------------------
# Bradley-Terry preference labeling
# ---------------------------------------------------------------------------
def bradley_terry_probability(r1: float, r2: float) -> float:
    """P(tau_1 preferred over tau_2) = exp(r1) / (exp(r1) + exp(r2)).

    Implemented as a numerically stable sigmoid of the return difference —
    naive `exp(r)` overflows for large returns (e.g. CartPole at 500 gives
    exp(500) ≈ 1.4e217, which already underflows the denominator in float64).
    """
    # sigmoid(r1 - r2) is algebraically identical to the BT expression.
    diff = r1 - r2
    # Clip just to keep sigmoid numerically well-behaved in extreme cases.
    diff = float(np.clip(diff, -50.0, 50.0))
    return 1.0 / (1.0 + np.exp(-diff))


def sample_preference(r1: float, r2: float, rng: np.random.Generator) -> Tuple[int, float]:
    """Sample a 0/1 preference label under Bradley-Terry.

    Returns
    -------
    preferred : int
        0 if tau_1 is preferred, 1 if tau_2 is preferred.
    p_tau1 : float
        The probability of tau_1 being preferred (stored for debugging).
    """
    p_tau1 = bradley_terry_probability(r1, r2)
    preferred = 0 if rng.random() < p_tau1 else 1
    return preferred, p_tau1


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f)


def save_csv_summary(pairs: List[Dict[str, Any]], path: Path) -> None:
    """Write a quick-look CSV: one row per preference pair.

    The full state-action data lives in the JSON file — the CSV is just
    for eyeballing the distribution of returns and preferences.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "pair_id,R_tau1,R_tau2,len_tau1,len_tau2,p_tau1_preferred,preferred\n"
    with path.open("w") as f:
        f.write(header)
        for i, pair in enumerate(pairs):
            f.write(
                f"{i},{pair['tau1']['return']:.4f},{pair['tau2']['return']:.4f},"
                f"{pair['tau1']['length']},{pair['tau2']['length']},"
                f"{pair['p_tau1_preferred']:.6f},{pair['preferred']}\n"
            )
