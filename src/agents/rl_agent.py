"""An agent that is rotating in the center of the simulation area"""
from . import BaseAgent
from src.model.net import ActorCritic

import numpy as np
from typing import *
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from collections import namedtuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RLAgent(BaseAgent):
    """RLAgent has an ability to learn
    
    will be created in main
    """
    network: nn.Module
    mode = str
    
    def __init__(self, 
        action_space: spaces.Box,
        network: nn.Module,
        mode: str = 'training',
        learning_rate: float = 1e-4, 
        gamma: float = 0.99,
        load_pretrain = ''
        ) -> None:
        super(RLAgent, self).__init__(action_space)
        
        self.mode = mode
        self.network = network
        self.network.to(device)
        self.gamma = gamma
        
        if load_pretrain:
            self.load_pretrain()
            
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
            
        self.memory_log_probs = []
        self.memory_values = []
        self.memory_rewards = []
       
    def load_pretrain(self, path):
        pass

    def act(self, obs) -> np.ndarray:
        
        means, stds, values = self.network(obs)
        distribution = Normal(means, stds)
        
        action = distribution.sample()
        
        log_prob = distribution.log_prob(action)
        self.memory_log_probs.append(log_prob)
        self.memory_values.append(values)
        
        return action
    
    def remeber_reward(self, reward):
        self.memory_rewards.append(reward)

    def compute_discounted_returns(self):
        R = 0
        size = len(self.memory_rewards)
        returns = torch.zeros(size, dtype=torch.float32).to(device)

        for i in range(size)[::-1]:
            R = self.memory_rewards[i] + self.gamma * R
            returns[i] = R

        return returns
    
    def estimate_loss(self) -> torch.Tensor:
        policy_losses = []
        value_losses  = []
        
        returns = self.compute_discounted_returns()
        
        for i, current_return in enumerate(returns):
            log_prob = self.memory_log_probs[i]
            value    = self.memory_values[i]
            
            advantage = current_return - value
            value_losses.append(F.smooth_l1_loss(value, current_return.unsqueeze(0)))
            policy_losses.append(-log_prob * advantage)
            
        policy_losses = torch.stack(policy_losses).mean()
        value_losses  = torch.stack(value_losses).mean()
        loss = policy_losses + value_losses
                
        return loss

    def update(self):
        loss = self.estimate_loss()        
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.clear_memory()
        
    def clear_memory(self):
        del self.memory_log_probs[:]
        del self.memory_values[:]
        del self.memory_rewards[:]
