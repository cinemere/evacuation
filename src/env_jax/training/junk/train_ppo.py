# %%
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional, Sequence, TypedDict

# import distrax
import flax
import flax.linen as nn
import gym
import imageio
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro
import wandb
from flax import struct
from flax.jax_utils import replicate, unreplicate
from flax.linen.initializers import constant, glorot_normal, orthogonal, zeros_init
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

# import envpool

# %% Network


class ActorNetwork(nn.Module):
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x):
    
        x = nn.Dense(self.num_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.num_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.action_dim)(x)        
        return x

# %%


import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.random import PRNGKey
import numpy as np

class RPOLinearNetwork(nn.Module):
    config: dict
    action_dim: int  # np.prod(self.envs.single_action_space.shape)
    rpo_alpha: float
    num_hidden: int

    def setup(self):
        self.actor_logstd = self.param('actor_logstd', nn.initializers.zeros, (1, self.action_dim))

    @nn.compact
    def __call__(self, x, action=None, rng=None):
        def critic():
            x = nn.Dense(self.num_hidden)(x)
            x = nn.tanh(x)
            x = nn.Dense(self.num_hidden)(x)
            x = nn.tanh(x)
            x = nn.Dense(1)(x)
            return x

        def actor_mean():
            x = nn.Dense(self.num_hidden)(x)
            x = nn.tanh(x)
            x = nn.Dense(self.num_hidden)(x)
            x = nn.tanh(x)
            x = nn.Dense(self.action_dim)(x)
            return x

        value = critic()
        action_mean = actor_mean()
        action_logstd = jnp.broadcast_to(self.actor_logstd, action_mean.shape)
        action_std = jnp.exp(action_logstd)

        if action is None:
            if rng is None:
                raise ValueError("RNG must be provided when action is None")
            action = action_mean + action_std * jax.random.normal(rng, action_mean.shape)
        else:
            rng = rng if rng is not None else PRNGKey(0)
            z = jax.random.uniform(rng, action_mean.shape, minval=-self.rpo_alpha, maxval=self.rpo_alpha)
            action_mean = action_mean + z
            action = action_mean + action_std * jax.random.normal(rng, action_mean.shape)

        log_prob = jnp.sum(-0.5 * ((action - action_mean) / action_std) ** 2 - action_logstd - 0.5 * jnp.log(2 * jnp.pi), axis=-1)
        entropy = jnp.sum(action_logstd + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)

        return action, log_prob, entropy, value

    def get_value(self, x):
        return self.apply(self, x, method=self.critic)

    def get_action_and_value(self, x, action=None, rng=None):
        return self.apply(self, x, action=action, rng=rng)
