import gymnasium as gym
import torch
import numpy as np

class RLHFEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_model, ref_policy, beta: float = 0.1):
        super().__init__(env)
        self.reward_model = reward_model
        self.ref_policy = ref_policy
        self.beta = beta
        self.active_policy = None # Will be set after PPO initialization
        
        self.reward_model.eval() # Ensure it's frozen

    def set_active_policy(self, active_policy):
        """Allows us to link the updating PPO policy to the environment for KL calculation."""
        self.active_policy = active_policy

    def step(self, action):
        # 1. Take the step in the real environment
        obs_next, real_reward, terminated, truncated, info = self.env.step(action)
        
        # We need the previous observation (current state) to evaluate the reward
        # To do this cleanly in a Wrapper, we save the obs from reset() and step()
        obs_current = self.current_obs
        self.current_obs = obs_next
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs_current).float().unsqueeze(0)
            act_tensor = torch.tensor(action).unsqueeze(0)
            
            # 2. Get the reward from the frozen Reward Model
            r_phi = self.reward_model(obs_tensor, act_tensor).item()
            
            # 3. Calculate KL Penalty
            if self.active_policy is not None:
                # Get log probabilities from both policies
                _, log_prob_ref, _ = self.ref_policy.evaluate_actions(obs_tensor, act_tensor)
                _, log_prob_active, _ = self.active_policy.evaluate_actions(obs_tensor, act_tensor)
                
                # KL Divergence estimate
                kl_penalty = (log_prob_active - log_prob_ref).item()
            else:
                kl_penalty = 0.0
                
            # 4. Final RLHF Reward
            rlhf_reward = r_phi - (self.beta * kl_penalty)

        # Record original reward in info dict for tracking
        info["real_reward"] = real_reward 
        
        return obs_next, rlhf_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs
        return obs, info