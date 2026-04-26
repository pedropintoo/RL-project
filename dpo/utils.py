import json
import gymnasium as gym
import gym as gym_old
import torch
import numpy as np
from pathlib import Path
from typing import Dict

# For visualization
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display
import glob
import base64, io

from policy import Policy

def load_preference_dataset(path: Path):
    """Load an offline preference dataset from JSON."""
    with path.open() as f:
        data = json.load(f)
    return data

def preference_pair_logps(policy: Policy, pair: Dict):
    """Return policy log-probabilities for the preferred and rejected trajectories."""
    if pair['preferred'] == 0:
        chosen = pair['tau1']
        rejected = pair['tau2']
    else:
        chosen = pair['tau2']
        rejected = pair['tau1']

    chosen_logp = trajectory_logp(policy, chosen)
    rejected_logp = trajectory_logp(policy, rejected)
    return chosen_logp, rejected_logp

def trajectory_logp(policy: Policy, trajectory: Dict) -> torch.Tensor:
    """Sum log-probabilities of the actions taken in one trajectory."""
    # Get trajectory (state, action) pairs
    states = torch.as_tensor(trajectory['states'], dtype=torch.float32, device=policy.device)
    if policy.is_discrete:
        actions = torch.as_tensor(trajectory['actions'], dtype=torch.long, device=policy.device)
    else:
        actions = torch.as_tensor(trajectory['actions'], dtype=torch.float32, device=policy.device)

    per_step_logp = policy.log_prob_actions(states, actions) # Compute log-probabilities for each step
    return per_step_logp.sum()

def evaluate_policy_returns(policy: Policy, env_name: str = 'Pendulum-v1', n_episodes: int = 50, max_t: int = 500, deterministic: bool = True):
    """Run evaluation episodes and return mean/std episode return."""
    eval_env = gym.make(env_name)
    returns = []

    for _ in range(n_episodes):
        reset_out = eval_env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        episode_return = 0.0

        for _ in range(max_t):
            action, _ = policy.act(state, deterministic=deterministic)
            step_out = eval_env.step(action)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_out

            episode_return += reward
            state = next_state
            if done:
                break

        returns.append(episode_return)

    eval_env.close()
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))

    return returns, mean_return, std_return

# Visualization utilities (gym_old is needed for video recording)
def show_video(save_path, env_name):
    """Display a video of the trained model in Colab."""
    mp4list = glob.glob(save_path+'*.mp4')
    if len(mp4list) > 0:
        mp4 = save_path+'{}.mp4'.format(env_name)
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

def show_video_of_model(save_path, policy, env_name):
    env = gym_old.make(env_name)
    vid = video_recorder.VideoRecorder(env, path=save_path+'{}.mp4'.format(env_name))
    state = env.reset()
    done = False
    for _ in range(100000):
        vid.capture_frame()
        action, _ = policy.act(state)
        next_state, _, done, _ = env.step(action)
        state = next_state
        if done:
            break
    vid.close()
    env.close()
