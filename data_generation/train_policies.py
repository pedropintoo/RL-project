"""
Train the expert (pi_1) and mid-performing (pi_2) policies.

For each environment in `config.ENVIRONMENTS`:
  1. Train a single PPO agent end-to-end (this becomes pi_1).
  2. During training, an `EvalCallback` periodically evaluates the policy.
     A custom `MidCheckpointCallback` watches those evaluations and, the
     first time the mean return crosses the halfway mark between random
     and expert performance, dumps the current weights as pi_2.

Only one training run is needed because:
  - It guarantees pi_2 and pi_1 live on the *same learning trajectory*, so
    pi_2 is a "less trained" version of pi_1 rather than a
    different algorithm, to match the project description.
  - Halves compute.

Outputs land in `outputs/policies/<ENV_ID>_{expert,mid}.zip`.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from config import ENVIRONMENTS, LOG_DIR, POLICY_DIR, TRAIN_SEED, EnvConfig

WANDB_PROJECT = "rl-course-project"


ALGO_REGISTRY = {"PPO": PPO, "SAC": SAC}


class MidCheckpointCallback(BaseCallback):
    """Save the model the first time eval-return crosses `target_return`.

    This callback relies on `EvalCallback` having already written its latest
    evaluation into `self.parent.last_mean_reward` — that's why we attach it
    as a child callback (see `EvalCallback(..., callback_after_eval=...)`).
    """

    def __init__(self, target_return: float, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.target_return = target_return
        self.save_path = save_path
        self.saved = False

    def _on_step(self) -> bool:
        if self.saved:
            return True
        # parent is the EvalCallback; last_mean_reward is set after each eval.
        mean_r = getattr(self.parent, "last_mean_reward", -np.inf)
        if mean_r >= self.target_return:
            self.model.save(self.save_path)
            if self.verbose:
                print(
                    f"[MidCheckpoint] mean eval return {mean_r:.2f} "
                    f">= target {self.target_return:.2f} — saved pi_2 to "
                    f"{self.save_path}"
                )
            self.saved = True
        return True


def _make_env(env_id: str, seed: int) -> gym.Env:
    """Wrap the env in Monitor so episode returns show up in SB3 logs."""
    env = gym.make(env_id)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def measure_random_return(env_id: str, n_episodes: int = 20, seed: int = 0) -> float:
    """Roll a uniform-random policy to sanity-check the config's random_return."""
    env = gym.make(env_id)
    env.action_space.seed(seed)
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=seed)
        done = trunc = False
        total = 0.0
        while not (done or trunc):
            action = env.action_space.sample()
            obs, r, done, trunc, _ = env.step(action)
            total += r
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def train_one_environment(cfg: EnvConfig, seed: int) -> None:
    print(f"\n=== Training on {cfg.env_id} ({cfg.algo}) ===")

    random_ret = measure_random_return(cfg.env_id, seed=seed)
    print(f"Measured random baseline: {random_ret:.2f} "
          f"(config said {cfg.random_return:.2f})")

    # Anchor the mid-target on whichever is worse: the measured random return
    # or the config default. This prevents a lucky random-policy run from
    # biasing the threshold too high.
    random_anchor = min(random_ret, cfg.random_return)
    target_mid = random_anchor + cfg.mid_fraction * (cfg.expert_return - random_anchor)
    print(f"pi_2 target return = {target_mid:.2f}")

    train_env = _make_env(cfg.env_id, seed=seed)
    eval_env = _make_env(cfg.env_id, seed=seed + 1_000)

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{cfg.env_id}-{cfg.algo}-seed{seed}",
        config={**cfg.__dict__, "seed": seed, "target_mid": target_mid,
                "measured_random_return": random_ret},
        sync_tensorboard=True,
        reinit=True,
    )

    AlgoCls = ALGO_REGISTRY[cfg.algo]
    model = AlgoCls(
        "MlpPolicy",
        train_env,
        seed=seed,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
    )

    mid_path = str(POLICY_DIR / f"{cfg.env_id}_mid")
    expert_path = str(POLICY_DIR / f"{cfg.env_id}_expert")

    mid_cb = MidCheckpointCallback(target_return=target_mid, save_path=mid_path)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=str(LOG_DIR),
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        callback_after_eval=mid_cb,
        verbose=0,
    )
    wandb_cb = WandbCallback(verbose=0)

    model.learn(total_timesteps=cfg.total_timesteps, callback=[eval_cb, wandb_cb])
    model.save(expert_path)
    print(f"[Expert] saved pi_1 to {expert_path}")

    # Final sanity check — evaluate both checkpoints.
    for label, path in [("pi_1 (expert)", expert_path), ("pi_2 (mid)", mid_path)]:
        try:
            loaded = AlgoCls.load(path, env=eval_env)
            mean, std = evaluate_policy(loaded, eval_env, n_eval_episodes=10,
                                        deterministic=True)
            print(f"  {label}: mean return = {mean:.2f} ± {std:.2f}")
            wandb.log({f"final/{label}/mean_return": mean,
                       f"final/{label}/std_return": std})
        except FileNotFoundError:
            print(f"  {label}: NOT FOUND at {path} — "
                  "target return may have been too high; increase "
                  "total_timesteps or lower mid_fraction.")

    train_env.close()
    eval_env.close()
    run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--envs",
        nargs="*",
        default=None,
        help="Subset of env_ids to train (default: all in config).",
    )
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    args = parser.parse_args()

    configs = ENVIRONMENTS
    if args.envs:
        configs = [c for c in ENVIRONMENTS if c.env_id in set(args.envs)]
        if not configs:
            raise SystemExit(f"No matching envs for {args.envs}")

    for cfg in configs:
        train_one_environment(cfg, seed=args.seed)


if __name__ == "__main__":
    main()
