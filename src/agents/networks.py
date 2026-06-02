import torch
from torch import nn

import gymnasium as gym

class QNetwork(nn.Module):
    def __init__(self, env, hidden_dim=32):
        self.net = nn.Sequential(
            
            
        )
    
    def forward(self, state, action):
        inp = torch.cat((state, action), dim=1)
        
        return self.net(inp)
    
class Actor(nn.Module):
    def __init__(self, env: gym.Env , hidden_dim=32):
        
        self.net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim)
            
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, state):
        return self.net(state).chunk(dim=1)
    