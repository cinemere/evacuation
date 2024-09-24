from __future__ import annotations

from typing import Any

import jax

from ..env import Environment
from ..core.constants import PEDESTRIANS_INIT_POSITIONS
from ..params import EnvParamsT
from ..types import TimeStep, State, EnvCarryT
from .base_wrappers import ObservationWrapper


def define_observation_type(): ...


class RelativePosition(ObservationWrapper):
    def __init__(self, env: Environment):
        super().__init__(env)

        assert jax.numpy.sqrt(2) == jax.numpy.sqrt(
            PEDESTRIANS_INIT_POSITIONS[0] ** 2 + PEDESTRIANS_INIT_POSITIONS[1] ** 2
        )

    def observation(self, params: EnvParamsT, state: State) -> jax.Array:
        obs = self._env._get_observation(params, state)
        
        rel_pedestrians_positions = (
            obs["pedestrians_position"] - obs["agent_position"]
        ) / jax.numpy.sqrt(2)

        rel_exit_position = (
            obs["exit_position"] - obs["agent_position"]
        ) / jax.numpy.sqrt(2)

        new_obs = {
            "agent_position": obs["agent_position"],
            "pedestrians_position": rel_pedestrians_positions,
            "exit_position": rel_exit_position,
        }
        return new_obs
