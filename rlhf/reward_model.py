import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    def __init__(self, env_id: str, hidden_dim: int = 64):
        super().__init__()
        env = gym.make(env_id)
        self.obs_dim = env.observation_space.shape[0]
        
        # Check if action space is discrete or continuous
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if self.is_discrete:
            self.act_dim = env.action_space.n
        else:
            self.act_dim = env.action_space.shape[0]
            
        env.close()

        # The input is state + action
        input_dim = self.obs_dim + self.act_dim
        
        # Sensible default architecture: 2 hidden layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Outputs a single scalar reward
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        # Format action based on environment type
        if self.is_discrete:
            # Squeeze extra dimensions and one-hot encode
            act = act.view(-1).long()
            act_processed = F.one_hot(act, num_classes=self.act_dim).float()
        else:
            act_processed = act.view(-1, self.act_dim).float()
            
        obs_processed = obs.view(-1, self.obs_dim).float()
        
        # Concatenate state and action
        x = torch.cat([obs_processed, act_processed], dim=-1)
        return self.net(x).squeeze(-1)