from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar, Tuple, Dict

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from typing_extensions import TypeAlias

from .core.constants import EXIT, Status, PEDESTRIANS_INIT_POSITIONS
from .core.rewards import estimate_agent_reward
from .core.step import agent_step, pedestrians_step
from .core.utils import update_statuses
from .render.plot import render_plot
from .params import EnvParams, EnvParamsT
from .types import (
    AgentState,
    EnvCarryT,
    PedestriansState,
    State,
    StepType,
    TimeStep,
)


class Environment(Generic[EnvParamsT]):  # (abc.ABC, Generic[EnvParamsT]):
    # @abc.abstractmethod
    def default_params(self, **kwargs: Any) -> EnvParamsT:
        params = EnvParams()
        params = params.replace(**kwargs)
        return params

    def observation_shape(self, params: EnvParamsT) -> Tuple[int] | Dict[str, Any]:
        return {
            "agent_position": (2,),
            "pedestrians_position": (params.number_of_pedestrians, 2),
            "exit_position": (2,),
        }
    
    @property
    def action_dim(self) -> int:
        return 2

    # @abc.abstractmethod
    def _generate_problem(self, params: EnvParamsT, key: jax.Array) -> State[EnvCarryT]:
        key, pedestrians_positions_key, pedestrians_direction_key = jax.random.split(
            key, num=3
        )

        # init agent
        agent = AgentState(
            position=jnp.asarray((0.0, 0.0)),
            direction=jnp.asarray((0.0, 0.0)),
            enslaving_degree=jnp.asarray(params.enslaving_degree),
        )

        # init pedestrians
        pedestrians_positions = jax.random.uniform(
            pedestrians_positions_key,
            shape=(params.number_of_pedestrians, 2),
            minval=PEDESTRIANS_INIT_POSITIONS[0],
            maxval=PEDESTRIANS_INIT_POSITIONS[1],
        )
        pedestrians_directions = jax.random.uniform(
            pedestrians_direction_key,
            shape=(params.number_of_pedestrians, 2),
            minval=PEDESTRIANS_INIT_POSITIONS[0] * params.step_size,
            maxval=PEDESTRIANS_INIT_POSITIONS[0] * params.step_size,
        )
        statuses = jnp.zeros((params.number_of_pedestrians), dtype=jnp.uint8)
        statuses = update_statuses(statuses, pedestrians_positions, agent.position)
        pedestrians = PedestriansState(
            positions=pedestrians_positions,
            directions=pedestrians_directions,
            statuses=statuses,
        )

        state = State(
            key=key,
            step_num=jnp.asarray(0),
            pedestrians=pedestrians,
            agent=agent,
            # state_encoding=...,
            # carry=EnvCarryT(),
        )
        return state

    # @abc.abstractmethod
    def observation(self, params: EnvParamsT, state: State) -> jax.Array:
        obs = {}
        obs["agent_position"] = state.agent.position
        obs["pedestrians_position"] = state.pedestrians.positions
        obs["exit_position"] = EXIT
        return obs

    def reset(self, params: EnvParamsT, key: jax.Array) -> TimeStep[EnvCarryT]:
        # create state by generating pedestrians positions
        state = self._generate_problem(params, key)
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=self.observation(params, state),
        )
        return timestep

    # Why timestep + state at once, and not like in Jumanji? # To be able to do autoresets in gym and envpools styles
    def step(
        self, params: EnvParamsT, timestep: TimeStep[EnvCarryT], action: jax.Array
    ) -> TimeStep[EnvCarryT]:

        state_key, pedestrians_step_key = jax.random.split(timestep.state.key, num=2)

        # Agent step
        new_agent, agent_step_info = agent_step(
            action=action,
            agent=timestep.state.agent,
            step_size=params.step_size,
            eps=params.eps,
            width=params.width,
            height=params.height,
        )
        terminated_agent, reward_agent = estimate_agent_reward(
            agent_step_info=agent_step_info,
            is_termination_agent_wall_collision=params.is_termination_agent_wall_collision,
        )

        # Pedestrians step
        (
            new_pedestrians,
            terminated_pedestrians,
            reward_pedestrians,
            intrinsic_reward,
        ) = pedestrians_step(
            pedestrians=timestep.state.pedestrians,
            agent=timestep.state.agent,
            now=timestep.state.step_num,
            step_size=params.step_size,
            noise_coef=params.noise_coef,
            width=params.width,
            height=params.height,
            eps=params.eps,
            num_pedestrians=params.number_of_pedestrians,
            init_reward_each_step=params.init_reward_each_step,
            is_new_exiting_reward=params.is_new_exiting_reward,
            is_new_followers_reward=params.is_new_followers_reward,
            key=pedestrians_step_key,
        )

        # Collect rewards
        reward = (
            reward_agent
            + reward_pedestrians
            + params.intrinsic_reward_coef * intrinsic_reward
        )

        new_state = timestep.state.replace(
            pedestrians=new_pedestrians,
            agent=new_agent,
            step_num=timestep.state.step_num + 1,
            key=state_key,
        )

        # Record observation
        new_observation = self.observation(params, new_state)

        # assert params.max_steps is not None
        truncated = jnp.equal(new_state.step_num, params.max_timesteps)

        print(f"{terminated_agent=} {terminated_pedestrians=}")
        terminated = terminated_agent | terminated_pedestrians

        step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_observation,
        )
        return timestep

    def render(
        self, params: EnvParamsT, timestep: TimeStep[EnvCarryT]
    ) -> np.ndarray | str:
        if params.render_mode == "plot":
            return render_plot(params, timestep)
        # elif params.render_mode == "rich_text":
        #    return text_render(timestep.state.grid, timestep.state.agent)
        # else:
        #    raise RuntimeError("Unknown render mode. Should be one of: ['rgb_array', 'rich_text']")
        ...
