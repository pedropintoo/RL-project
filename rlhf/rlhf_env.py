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
            
            # 3. Calculate KL Penalty dynamically based on the algorithm (PPO vs SAC)
            kl_penalty = 0.0
            if self.active_policy is not None:
                # --- PPO ---
                if hasattr(self.active_policy, "evaluate_actions"):
                    # Get log probabilities from both policies
                    _, log_prob_ref, _ = self.ref_policy.evaluate_actions(obs_tensor, act_tensor)
                    _, log_prob_active, _ = self.active_policy.evaluate_actions(obs_tensor, act_tensor)
                    
                    # KL Divergence estimate
                    kl_penalty = (log_prob_active - log_prob_ref).item()
                
                # --- SAC ---
                elif hasattr(self.active_policy, "actor"):
                    mean_act, log_std_act, kwargs_act = self.active_policy.actor.get_action_dist_params(obs_tensor)
                    mean_ref, log_std_ref, kwargs_ref = self.ref_policy.actor.get_action_dist_params(obs_tensor)
                    
                    if "latent_sde" in kwargs_act:
                        # gSDE case: Multiply the latent features by the log_std matrix
                        var_act = torch.mm(kwargs_act["latent_sde"]**2, torch.exp(log_std_act)**2)
                        var_ref = torch.mm(kwargs_ref["latent_sde"]**2, torch.exp(log_std_ref)**2)
                        std_act = torch.sqrt(var_act + 1e-8)
                        std_ref = torch.sqrt(var_ref + 1e-8)
                    else:
                        # Standard Gaussian case
                        std_act = torch.exp(log_std_act)
                        std_ref = torch.exp(log_std_ref)
                    
                    # Closed-form KL Divergence for diagonal Gaussians
                    kl = torch.log(std_ref / std_act) + (std_act**2 + (mean_act - mean_ref)**2) / (2.0 * std_ref**2) - 0.5
                    kl_penalty = kl.sum(dim=-1).item()
                
            # 4. Final RLHF Reward
            rlhf_reward = r_phi - (self.beta * kl_penalty)

        # Record original reward in info dict for tracking
        info["real_reward"] = real_reward 
        
        return obs_next, rlhf_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs
        return obs, info