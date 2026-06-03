import torch
from torch import nn

import gymnasium as gym

from networks import QNetwork, Actor

class SAC(nn.Module):
    def __init__(self, env: gym.Env, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(env).to(self.device)
        self.target = QNetwork(env).to(self.device)
        self.target.load_state_dict(self.critic.state_dict())

        self.actor = Actor(env).to(self.device)

        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
    def update(self):
        pass