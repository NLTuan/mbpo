import torch
from torch import nn

import torch.nn.functional as F

import gymnasium as gym

from src.agents.networks import QNetwork, Actor

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
        
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        state_batch = torch.FloatTensor(state_batch).to(self.device)                                                                                
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)                                                                      
        action_batch = torch.FloatTensor(action_batch).to(self.device)                                                                              
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state_batch)
            q1_target, q2_target = self.target(next_state_batch, next_action)
            min_q_target = torch.min(q1_target, q2_target)
            target = reward_batch + self.gamma * (1 - done_batch) * (min_q_target - self.alpha * next_log_prob)
            
        # 1. Critic Update
        q1, q2 = self.critic(state_batch, action_batch)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()
        
        # 2. Actor Update
        action, log_probs, _ = self.actor.sample(state_batch)
        actor_loss = -(torch.min(*self.critic(state_batch, action)) - self.alpha * log_probs).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        for target_param, param in zip(self.target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)