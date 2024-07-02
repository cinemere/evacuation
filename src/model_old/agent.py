import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from .encoding import DeepSets, TransformerBlock


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentRPO(nn.Module):
    """RPO agent. Taken from https://github.com/vwxyzjn/cleanrl"""
    def __init__(self, envs, rpo_alpha, device):
        super().__init__()
        self.device = device
        self.rpo_alpha = rpo_alpha
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

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


class DeepSetsAgent(AgentRPO):
    def __init__(self, envs, rpo_alpha):
        super().__init__(envs, rpo_alpha)
        self.deep_sets = DeepSets(
            input_dim=envs.observation_space.shape[-1],
            embedding_dim=24,
            output_dim=envs.observation_space.shape[-1]
        )
        
    def get_value(self, x):
        x = self.deep_sets(x)
        return super().get_value(x)
    
    def get_action_and_value(self, x, action=None):
        x = self.deep_sets(x)
        return super().get_action_and_value(x, action)
    

class TransformerAgent(AgentRPO):
    def __init__(self, envs, rpo_alpha):
        super().__init__(envs, rpo_alpha)
        self.transformer = nn.Sequential(
        TransformerBlock(
                d_model=envs.observation_space.shape[-1],
                num_heads=3, #2, # 3,
                d_ff=96, #34,  # 24,
            ),
        TransformerBlock(
                d_model=envs.observation_space.shape[-1],
                num_heads=3, #2, # 3,
                d_ff=96, #34,  # 24,
            ),            
        )
        
    def get_value(self, x):
        x = self.transformer(x)
        return super().get_value(x)
    
    def get_action_and_value(self, x, action=None):
        x = self.transformer(x)
        return super().get_action_and_value(x, action)
