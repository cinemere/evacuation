"""An agent that is rotating in the center of the simulation area"""
from . import BaseAgent

import numpy as np
from typing import *
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from collections import namedtuple

class BaseRLAgent(BaseAgent):
    """RLAgent has an ability to learn
    
    will be created in main
    and then trained in Trainer
    """
    network: nn.Module
    mode: str
    
    def __init__(self, 
        action_space: spaces.Box,
        network: nn.Module,
        mode: Literal['training', 'inference'] = 'training',
        learning_rate: float = 1e-4, 
        gamma: float = 0.99,
        load_pretrain: str = '',
        device: str = 'cpu',
        ) -> None:
        super(BaseRLAgent, self).__init__(action_space)
        
        self.mode = mode
        self.network = network
        self.network.to(device)
        self.gamma = gamma
        
        if load_pretrain:
            self.load_pretrain()
            
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
            
        self.memory = []
       
    def load_pretrain(self, path):
        raise NotImplementedError

    def act(self, obs: spaces.Space) -> np.ndarray:        
        """        
        return action in the same shape as action_space
        """
        raise NotImplementedError
    
    def receive_reward(self, reward):
        raise NotImplementedError

    def update(self):
        loss = self.estimate_loss()        
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()        