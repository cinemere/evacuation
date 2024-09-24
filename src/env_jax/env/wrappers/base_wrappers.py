from typing import Any

import jax

from ..env import Environment
from ..params import EnvParamsT
from ..types import TimeStep, State


class Wrapper(Environment[EnvParamsT]):
    def __init__(self, env: Environment[EnvParamsT]):
        self._env = env

    # Question: what if wrapper adds new parameters to the dataclass?
    # Solution: do this after applying the wrapper:
    #   env_params = wrapped_env.default_params(**dataclasses.asdict(original_params))
    def default_params(self, **kwargs) -> EnvParamsT:
        return self._env.default_params(**kwargs)

    def observation_shape(self, params: EnvParamsT) -> int | dict[str, Any]:
        return self._env.observation_shape(params)

    def _generate_problem(self, params: EnvParamsT, key: jax.Array) -> State:
        return self._env._generate_problem(params, key)

    def reset(self, params: EnvParamsT, key: jax.Array) -> TimeStep:
        return self._env.reset(params, key)

    def step(
        self, params: EnvParamsT, timestep: TimeStep, action: jax.Array
    ) -> TimeStep:
        print("Type of env_params:", type(params))
        print("Type of timestep:", type(timestep))
        print("Type of action:", type(action))

        return self._env.step(params, timestep, action)

    def render(self, params: EnvParamsT, timestep: TimeStep):
        return self._env.render(params, timestep)


class ObservationWrapper(Wrapper):
    def reset(self, params: Any, key: jax.Array) -> TimeStep:
        timestep = super().reset(params, key)
        new_observation = self.observation(params, timestep.state)
        new_timestep = timestep.replace(observation=new_observation)
        return new_timestep

    def step(
        self, params: Any, timestep: TimeStep, action: jax.Array
    ) -> TimeStep:
        timestep = super().step(params, timestep, action)
        new_observation = self.observation(params, timestep.state)
        new_timestep = timestep.replace(observation=new_observation)
        return new_timestep

    def observation(self, params: EnvParamsT, state: State) -> jax.Array:
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """
        raise NotImplementedError