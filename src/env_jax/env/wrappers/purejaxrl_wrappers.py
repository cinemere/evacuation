import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from functools import partial
from typing import Dict, Optional, Tuple, Union, Any

from ..env import Environment
from ..params import EnvParamsT
from ..core.constants import PEDESTRIANS_INIT_POSITIONS, Status
from ..types import State, TimeStep
from .base_wrappers import Wrapper, ObservationWrapper, ObservationWrapper2

#     TODO NormalizeVecObservation,
#     TODO NormalizeVecReward,


@struct.dataclass
class TimeStepWithLog:
    timestep: TimeStep
    episode_returns: float
    episode_lengths: int


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def reset(self, params: Any, key: jax.Array) -> TimeStepWithLog:
        timestep = self._env.reset(key, params)
        timestep_with_log = TimeStepWithLog(timestep, 0, 0)
        return timestep_with_log

    def step(
        self, params: Any, timestep_with_log: TimeStepWithLog, action: jax.Array
    ) -> TimeStepWithLog:
        new_timestep = self._env.step(params, timestep_with_log.timestep, action)

        new_episode_returns = timestep_with_log.episode_returns + new_timestep.reward
        new_episode_lengths = timestep_with_log.episode_lengths + 1

        new_timestep_with_log = TimeStepWithLog(
            timestep=new_timestep,
            episode_returns=new_episode_returns * new_timestep.discount,
            episode_lengths=new_episode_lengths * new_timestep.discount,
        )
        return new_timestep_with_log


class ClipAction(Wrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, params: Any, timestep: TimeStep, action: jax.Array) -> TimeStep:
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        clipped_action = jnp.clip(action, self.low, self.high)
        return self._env.step(params, timestep, clipped_action)


class VecEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(None, 0))
        self.step = jax.vmap(self._env.step, in_axes=(None, 0, 0))


@struct.dataclass
class RunningMeanStd:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float  # 1e-4
    timestep: TimeStep


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def update_rms_from_batch(rms: RunningMeanStd, batch: jnp.ndarray):
    batch_mean = jnp.mean(batch, axis=0)
    batch_val = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]

    new_rms = update_mean_var_count_from_moments(
        mean=rms.mean,
        var=rms.var,
        count=rms.count,
        batch_mean=batch_mean,
        batch_var=batch_val,
        batch_count=batch_count,
    )
    return new_rms


class NormalizeVecObservation(ObservationWrapper2):
    
    def observation_shape(self, params: EnvParamsT) -> int | Dict[str, Any]:
        return self._env.observation_shape(params)
    
    def observation(self, params: EnvParamsT, timestep: TimeStep) -> jax.Array:
        base_obs = timestep.observation
        
        extended_obs = base_obs
        
        return base_obs