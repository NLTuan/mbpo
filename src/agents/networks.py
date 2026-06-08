import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym

# It's standard practice to bound the log_std to prevent numerical instability
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class QNetwork(nn.Module):
    def __init__(self, env: gym.Env, hidden_dim=32):
        super().__init__()
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # SAC requires Double Q-Learning to prevent overestimation bias, 
        # so we actually construct two separate Q-networks inside this class!
        
        # Q1 architecture
        self.net1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 architecture
        self.net2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        inp = torch.cat([state, action], dim=1)
        q1 = self.net1(inp)
        q2 = self.net2(inp)
        return q1, q2

class Actor(nn.Module):
    def __init__(self, env: gym.Env, hidden_dim=32):
        super().__init__()
        
        state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Base network
        self.base_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate heads for mean and log_std (a bit cleaner than chunking)
        self.mean_linear = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, self.action_dim)
        
        # Action rescaling
        # Environments expect actions in specific bounds (e.g. [-1, 1] or [-2, 2]).
        # register_buffer ensures these tensors are moved to the GPU if you call .to('cuda')
        self.register_buffer("action_scale", torch.tensor(
            (env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32
        ))
        self.register_buffer("action_bias", torch.tensor(
            (env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32
        ))
    
    def forward(self, state):
        x = self.base_net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        
        # Clamp log_std to prevent extreme values that cause NaNs during training
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        """
        Samples an action, returns its log probability, and the deterministic mean action.
        This is the most critical function for the Actor!
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Create a normal distribution
        normal = Normal(mean, std)
        
        # Sample using the reparameterization trick (rsample)
        # This allows gradients to flow backwards through the random sample
        x_t = normal.rsample()  
        
        # Squash the action to [-1, 1] using tanh
        y_t = torch.tanh(x_t)
        
        # Scale the action to the specific environment's bounds
        action = y_t * self.action_scale + self.action_bias
        
        # Calculate the log probability of the sampled action
        log_prob = normal.log_prob(x_t)
        
        # Enforcing Action Bound (Squashed Gaussian)
        # Because we applied tanh, we must correct the log probability using the Jacobian
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        # Also return the deterministic action (just the mean squashed)
        # This is used for evaluation (when you want the agent to stop exploring and just perform well)
        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, deterministic_action