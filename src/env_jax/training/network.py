# Adapted from PureJaxRL implementation and minigrid baselines, source:
# https://github.com/lupuandr/explainable-policies/blob/50acbd777dc7c6d6b8b7255cd1249e81715bcb54/purejaxrl/ppo_rnn.py#L4
# https://github.com/lcswillems/rl-starter-files/blob/master/model.py
from typing import Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                ),
                activation,
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                ),
                activation,
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(0.01),
                    bias_init=constant(0.0),
                ),
            ]
        )
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        critic = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                ),
                activation,
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                ),
                activation,
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(1.0),
                    bias_init=constant(0.0),
                ),
            ]
        )

        action_mean = actor_mean(x).astype(jnp.float32)
        action_std = jnp.exp(actor_logtstd)

        dist = distrax.MultivariateNormalDiag(action_mean, action_std)
        values = critic(x)

        return dist, jnp.squeeze(values, axis=-1)