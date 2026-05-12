import json
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
from typing import Dict

# For visualization
from gymnasium.wrappers import RecordVideo
from IPython.display import HTML
from IPython import display
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
    # Normalize by trajectory length to avoid bias towards longer trajectories
    # (preference datasets may contain trajectories with very different lengths).
    # Return the average per-step log-probability so comparisons are length-invariant.
    length = float(per_step_logp.numel()) if hasattr(per_step_logp, 'numel') else 1.0
    return per_step_logp.sum() / max(length, 1.0)

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

def show_video(save_path, env_name):
    """Display a video of the trained model in Colab."""
    save_path_obj = Path(save_path)
    video_dir = save_path_obj.parent if save_path_obj.parent != Path("") else Path(".")
    prefix = f"{save_path_obj.name}_{env_name}"
    run_dir = video_dir / prefix
    mp4list = sorted(run_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime) if run_dir.exists() else []

    if len(mp4list) == 0:
        print("Could not find video")
        return

    mp4 = mp4list[-1]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    display.display(HTML(data='''<video alt="test" autoplay
            loop controls style="height: 400px;">
            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
         </video>'''.format(encoded.decode('ascii'))))

def show_video_of_model(save_path, policy, env_name):
    save_path_obj = Path(save_path)
    video_dir = save_path_obj.parent if save_path_obj.parent != Path("") else Path(".")
    video_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{save_path_obj.name}_{env_name}"
    run_dir = video_dir / prefix
    run_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous recordings for this specific (name, env) pair to avoid overwrite warnings.
    for stale_file in run_dir.glob("*.mp4"):
        stale_file.unlink()

    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(run_dir),
        episode_trigger=lambda episode_id: episode_id == 0,
        name_prefix=prefix,
        disable_logger=True,
    )

    reset_out = env.reset()
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    done = False
    
    for _ in range(100000):
        # Get action from policy
        action, _ = policy.act(state)

        # Convert action to the format expected by the current action space.
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if hasattr(env.action_space, "n"):
            action = int(np.asarray(action).squeeze())
        else:
            action = np.asarray(action).flatten()
        
        # Step environment
        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            next_state, _, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            next_state, _, done, _ = step_out
        state = next_state
        if done:
            break

    env.close()
