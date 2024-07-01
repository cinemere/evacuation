import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from dataclasses import dataclass

from .utils import layer_init

@dataclass
class RPOAgentNetworkConfig:

    num_hidden: int = 64
    """number of hidden layers"""

    rpo_alpha: float = 0.5
    """the alpha parameter for RPO"""


class RPOAgentNetwork(nn.Module):
    def __init__(self, envs, config: RPOAgentNetworkConfig, device: torch.DeviceObjType):
        super().__init__()
        self.rpo_alpha = config.rpo_alpha
        self.critic = nn.Sequential(
            layer_init(nn.Linear(
                np.array(envs.single_observation_space.shape).prod(), 
                config.num_hidden
            )),
            nn.Tanh(),
            layer_init(nn.Linear(config.num_hidden, config.num_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(config.num_hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(
                np.array(envs.single_observation_space.shape).prod(), 
                config.num_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(config.num_hidden, config.num_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(config.num_hidden, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.device = device

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:  # new to RPO
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(self.device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
