import numpy as np
import gym
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal


def _reduce_action_log_prob(log_prob: torch.Tensor) -> torch.Tensor:
    """Return one log-probability per sample, regardless of action dimensionality."""
    return log_prob.sum(dim=-1) if log_prob.ndim > 1 else log_prob

# Simple Policy Network for testing both discrete and continuous action spaces.
#
# state size
#   |
#   v
# hidden size
#   |
#   v
# action size (discrete) or action size * 2 (continuous, mean and std)

# [abstract class]
class Policy(nn.Module):
    def __init__(self, device, is_discrete):
        super(Policy, self).__init__()
        self.device = device
        self.is_discrete = is_discrete
    def forward(self, state):
        raise NotImplementedError
    def act(self, state):
        raise NotImplementedError
    def log_prob_actions(self, states, actions):
        raise NotImplementedError

class DiscretePolicy(Policy):
    def __init__(self, device, state_size=4, action_size=2, hidden_size=32):
        super(DiscretePolicy, self).__init__(device, is_discrete=True)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        # we just consider 1 dimensional probability of action
        return F.softmax(x, dim=1)

    def act(self, state, deterministic=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        model = Categorical(probs)
        if deterministic:
            action = probs.argmax(dim=1) # greedy action
        else:
            action = model.sample()
        return action.item(), model.log_prob(action)
    
    def log_prob_actions(self, states, actions):
        probs = self.forward(states)
        model = Categorical(probs)
        return model.log_prob(actions)


class ContinuousPolicy(Policy):
    def __init__(self, device, state_size=4, action_size=2, hidden_size=32):
        super(ContinuousPolicy, self).__init__(device, is_discrete=False)
        self.device = device
        # fc1
        self.fc1 = nn.Linear(state_size, hidden_size)
        
        # fc2
        self.mu = nn.Linear(hidden_size, action_size) # our fc2 is now mu, the mean of the action distribution + standard deviation
        self.log_std = nn.Parameter(torch.zeros(1, action_size)) # log of standard deviation, we use a parameter so it can be learned

    def forward(self, state):
        x = F.relu(self.fc1(state))
        mu = torch.tanh(self.mu(x)) # we use tanh to bound the action between -1 and 1 (! THIS POLICY IS FOR CONTINUOUS ACTION SPACE BETWEEN -1 AND 1 !)
        std = self.log_std.exp()
        return mu, std

    def act(self, state, deterministic=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        mu, std = self.forward(state)
        model = Normal(mu, std)
        if deterministic:
            action = mu # mean deterministic action
        else:
            action = model.sample()
        step_log_prob = _reduce_action_log_prob(model.log_prob(action))
        return action.detach().cpu().numpy().flatten(), step_log_prob

    def log_prob_actions(self, states, actions):
        mu, std = self.forward(states)
        model = Normal(mu, std)
        return _reduce_action_log_prob(model.log_prob(actions))

class SB3DiscretePolicyAdapter(Policy):
    """Adapter to make SB3 ActorCriticPolicy conform to the Policy abstract class."""
    
    def __init__(self, device, sb3_policy):
        super(SB3DiscretePolicyAdapter, self).__init__(device, is_discrete=True)
        self.device = device
        self.sb3_policy = sb3_policy
    
    def forward(self, state):
        # SB3 policies expect a batch of states, so we add a batch dimension
        distribution = self.sb3_policy.get_distribution(state)
        return distribution.distribution.probs.cpu()
    
    def act(self, state, deterministic=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            # SB3 handles the distribution creation
            distribution = self.sb3_policy.get_distribution(state_tensor)
            
        if deterministic:
            action = distribution.mode() # Greedy action
        else:
            action = distribution.sample() # Sample action
            
        log_prob = distribution.log_prob(action)
        
        return action.item(), log_prob.cpu()
    
    def log_prob_actions(self, states, actions):
        # Get the SB3 distribution
        distribution = self.sb3_policy.get_distribution(states)
        
        # SB3 distributions have a built-in log_prob method that returns 
        # the log probability of the actions while preserving the gradient graph.
        return distribution.log_prob(actions)

class SB3ContinuousPolicyAdapter(Policy):
    """Adapter to make SB3 ActorCriticPolicy conform to the Policy abstract class."""
    
    def __init__(self, device, sb3_policy):
        super(SB3ContinuousPolicyAdapter, self).__init__(device, is_discrete=False)
        self.device = device
        self.sb3_policy = sb3_policy
    
    def forward(self, state):
        distribution = self.sb3_policy.get_distribution(state)
        return distribution.distribution.mean.cpu(), distribution.distribution.stddev.cpu()
    
    def act(self, state, deterministic=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            distribution = self.sb3_policy.get_distribution(state_tensor)
            
        if deterministic:
            action = distribution.mode() # Greedy action (mean for Normal distribution)
        else:
            action = distribution.sample() # Sample action
            
        log_prob = distribution.log_prob(action)

        return action.cpu().numpy().flatten(), _reduce_action_log_prob(log_prob).cpu()
    
    def log_prob_actions(self, states, actions):
        distribution = self.sb3_policy.get_distribution(states)
        return _reduce_action_log_prob(distribution.log_prob(actions))
