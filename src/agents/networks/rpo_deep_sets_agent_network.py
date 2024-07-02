import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

import numpy as np
from math import sqrt

from dataclasses import dataclass

from .utils import layer_init
from .rpo_linear_agent_network import RPOLinearNetwork, RPOLinearNetworkConfig

@dataclass
class RPODeepSetsEmbeddingConfig:
    
    network: RPOLinearNetworkConfig
    """params of linear network"""
    
    dim_hidden: float = 24
    """dimensionality of hidden layer in embedding"""
        
    
class DeepSets(nn.Module):
    def __init__(
            self, 
            set_elem_dim: int,
            output_dim: int, 
            hidden_dim: int,  # 16 
        ) -> None:
        super(DeepSets, self).__init__()
        self.transform_phi = nn.Sequential(
            nn.Linear(set_elem_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.transform_rho = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor):             # [batch, set_size, set_dim]
        assert x.dim() == 3, "Input shape should be: (batch_size, set_size, set_dim)"\
                             f"Instead DeepSets module received {x.shape=}"
        assert x.shape[1] > 1, "Only one element in set. No point in applying deep sets"
        
        # transform each element of set
        x = self.transform_phi(x)     # [batch, set_size, hidden_dim]
        # aggregate the set dimension
        x = x.sum(dim=-2)             # [batch, hidden_dim]
        # transform aggregated
        x = self.transform_rho(x)     # [batch, output_dim]
        return x


class RPODeepSetsEmbedding(RPOLinearNetwork):
    def __init__(self, 
                 envs, 
                 number_of_pedestrians: int,
                 config: RPODeepSetsEmbeddingConfig,
                 device: torch.DeviceObjType,
                 ):
        super().__init__(envs, config.network, device)
        
        obs_size = envs.single_observation_space.shape[-1]
        self.set_element_dim = obs_size // (number_of_pedestrians + 2)
        
        self.deep_sets = DeepSets(
            set_elem_dim=self.set_element_dim,
            hidden_dim=config.dim_hidden,
            output_dim=obs_size,
        )
        
    def get_value(self, x):
        shape = x.shape
        x = x.view(shape[0], -1, self.set_element_dim)
        x = self.deep_sets(x)
        x = x.view(shape)
        return super().get_value(x)
    
    def get_action_and_value(self, x, action=None):
        shape = x.shape
        x = x.view(shape[0], -1, self.set_element_dim)
        x = self.deep_sets(x)
        x = x.view(shape)
        return super().get_action_and_value(x, action)
