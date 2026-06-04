import gymnasium as gym
import torch
import numpy as np
import sys
import os

# Add the project root to the Python path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.sac import SAC
from src.buffers.replay_buffer import ReplayBuffer

def train():
    env = gym.make("HalfCheetah-v4")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SAC(env)
    buffer = ReplayBuffer(state_dim, action_dim)
    
    # Hyperparameters
    max_steps = 1_000_000   
    start_steps = 10_000    
    update_after = 1_000    
    batch_size = 256
    
    # Initialization
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    
    for step in range(max_steps):
        # 1. Select Action
        if step < start_steps:
            # Completely random action for initial exploration
            action = env.action_space.sample()
        else:
            # Ask the neural network for an action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action_tensor, _, _ = agent.actor.sample(state_tensor)
            action = action_tensor.cpu().numpy()[0]
            
        # 2. Step Environment
        next_state, reward, terminated, truncated, info = env.step(action)
        # In Gymnasium, an episode ends if it naturally terminates OR hits a time limit (truncated)
        done = terminated or truncated
        
        # 3. Store in Buffer
        buffer.push(state, action, reward, next_state, done)
        
        episode_reward += reward
        episode_steps += 1
        
        # 4. Update Agent
        if step >= update_after:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = buffer.sample(batch_size)
            # This calls the method you are going to write in sac.py!
            agent.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            
        # 5. Handle Episode End
        if done:
            print(f"Step {step} | Episode Steps: {episode_steps} | Reward: {episode_reward:.2f}")
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
        else:
            state = next_state

if __name__ == "__main__":
    train()
