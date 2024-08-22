from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from typing_extensions import TypeAlias

# import abc

from .types import EnvCarryT, AgentState, PedestriansState, State, StepType, TimeStep, IntOrArray
from .core.step import agent_step, pedestrians_step
from .core.statuses import Status, update_statuses


class EnvParams(struct.PyTreeNode):

    max_steps: Optional[None] = struct.field(pytree_node=False, default=None)
    render_mode: str = struct.field(pytree_node=False, default="rgb_array")
 
    number_of_pedestrians: int = struct.field(pytree_node=False, default=10)
    """number of pedestrians in the simulation"""

    width: float = struct.field(pytree_node=False, default=1.)
    """geometry of environment space: width"""

    height: float = struct.field(pytree_node=False, default=1.)
    """geometry of environment space: height"""

    step_size: float = struct.field(pytree_node=False, default=0.01)
    """length of pedestrian\'s and agent\'s step
    Typical expected values: 0.1, 0.05, 0.01"""

    noise_coef: float = struct.field(pytree_node=False, default=0.2)
    """noise coefficient of randomization in viscek model"""

    eps: float = struct.field(pytree_node=False, default=1e-8)
    """eps"""

    # ---- Leader params ----

    enslaving_degree: float = struct.field(pytree_node=False, default=1.)
    """enslaving degree of leader in generalized viscek model
    vary in (0; 1], where 1 is full enslaving.
    Typical expected values: 0.1, 0.5, 1."""

    # ---- Reward params ----

    is_new_exiting_reward: bool = struct.field(pytree_node=False, default=False)
    """if True, positive reward will be given for each pedestrian,
    entering the exiting zone"""

    is_new_followers_reward: bool = struct.field(pytree_node=False, default=True)
    """if True, positive reward will be given for each pedestrian,
    entering the leader\'s zone of influence"""

    intrinsic_reward_coef: float = struct.field(pytree_node=False, default=0.)
    """coefficient in front of intrinsic reward"""

    is_termination_agent_wall_collision: bool = struct.field(pytree_node=False, default=False)
    """if True, agent\'s wall collision will terminate episode"""

    init_reward_each_step: float = struct.field(pytree_node=False, default=-1)
    """constant reward given on each step of agent. 
    Typical expected values: 0, -1."""

    # # ---- Timing in the environment ----

    max_timesteps: int = struct.field(pytree_node=False, default=2_000)
    """max timesteps before truncation"""

    # n_episodes: int = 0
    # """number of episodes already done (for pretrained models)"""

    # n_timesteps: int = 0
    # """number of timesteps already done (for pretrained models)"""

    # # ---- Logging params ----

    # render_mode: str | None = None
    # """render mode (has no effect)"""

    # draw: bool = False
    # """enable saving of animation at each step"""    

    # verbose: bool = False
    # """enable debug mode of logging"""

    # giff_freq: int = 500
    # """frequency of logging the giff diagram"""

    # wandb_enabled: bool = True
    # """enable wandb logging (if True wandb.init() should be called before 
    # initializing the environment)"""

    # # ---- Logging artifacts dirs ----

    # path_giff: str = 'saved_data/giff'
    # """path to save giff animations: {path_giff}/{experiment_name}"""

    # path_png: str = 'saved_data/png'
    # """path to save png images of episode trajectories: {path_png}/{experiment_name}"""

    # path_logs: str = 'saved_data/logs'
    # """path to save logs: {path_logs}/{experiment_name}"""

EnvParamsT = TypeVar("EnvParamsT", bound="EnvParams")

# class Environment(abc.ABC, Generic[EnvParamsT]):    
class Environment(Generic[EnvParamsT]):    
    # @abc.abstractmethod
    def default_params(self, **kwargs: Any) -> EnvParamsT:
        params = EnvParams()
        params = params.replace(**kwargs)
        return params

    def observation_shape(self, params: EnvParamsT) -> int: #-> tuple[int, int, int] | dict[str, Any]:
    # TODO should be dependent on the selected observation wrapper
        return params.number_of_pedestrians * 2

    # @abc.abstractmethod
    def _generate_problem(self, params: EnvParamsT, key: jax.Array) -> State[EnvCarryT]:
        key, *keys = jax.random.split(key, num=2)
    
        # init agent    
        agent = AgentState(
            position=jnp.asarray((0, 0)),
            direction=jnp.asarray((0, 0)),
            enslaving_degree=jnp.asarray(params.enslaving_degree)
        )

        # init pedestrians
        pedestrians_positions = jax.random.uniform(keys[0], shape=(params.number_of_pedestrians, 2), minval=-1, maxval=1)
        pedestrians_directions = jax.random.uniform(keys[1], shape=(params.number_of_pedestrians, 2), minval=-1, maxval=1)
        statuses = jnp.zeros((params.number_of_pedestrians), dtype=jnp.uint8)
        statuses = update_statuses(statuses, agent.position)
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
            carry=EnvCarryT(),
        )
        return state
                        
    # @abc.abstractmethod
    def _get_observation(self, params: EnvParamsT, state: State) -> jax.Array:
        ...

    def reset(self, params: EnvParamsT, key: jax.Array) -> TimeStep[EnvCarryT]:
        # create state by generating pedestrians positions
        state = self._generate_problem(params, key)
        timestep = TimeStep(
        state=state,
        step_type=StepType.FIRST,
        reward=jnp.asarray(0.0),
        discount=jnp.asarray(1.0),
        observation=self._get_observation(params, state),
        )
        return timestep

    # Why timestep + state at once, and not like in Jumanji? # To be able to do autoresets in gym and envpools styles
    def step(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT], action: IntOrArray) -> TimeStep[EnvCarryT]:
        
        # Agent step
        new_agent, terminated_agent, reward_agent = agent_step(action, timestep.state.agent)
        
        # Pedestrians step
        new_pedestrians, terminated_pedestrians, reward_pedestrians, intrinsic_reward = \
            pedestrians_step(timestep.state.pedestrians, timestep.state.agent, timestep.state.time.now)
    
        # Collect rewards
        reward = reward_agent + reward_pedestrians + params.intrinsic_reward_coef * intrinsic_reward
        
        new_state = timestep.state.replace(
            pedestrians=new_pedestrians,
            agent=new_agent,
            step_num=timestep.state.step_num + 1,
        )
        
        # Record observation
        new_observation = self._get_observation(params, new_state)
        
        assert params.max_steps is not None
        truncated = jnp.equal(new_state.step_num, params.max_steps)
        
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

    def render(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT]) -> np.ndarray | str:
        # if params.render_mode == "rgb_array":
        #    return rgb_render(np.asarray(timestep.state.grid), timestep.state.agent, params.view_size)
        # elif params.render_mode == "rich_text":
        #    return text_render(timestep.state.grid, timestep.state.agent)
        # else:
        #    raise RuntimeError("Unknown render mode. Should be one of: ['rgb_array', 'rich_text']")
        ...
    
# %%
# import jax
# import xminigrid
# from xminigrid.wrappers import GymAutoResetWrapper
# from xminigrid.experimental.img_obs import RGBImgObservationWrapper

# key = jax.random.key(0)
# reset_key, ruleset_key = jax.random.split(key)

# # to list available benchmarks: xminigrid.registered_benchmarks()
# benchmark = xminigrid.load_benchmark(name="trivial-1m")
# # choosing ruleset, see section on rules and goals
# ruleset = benchmark.sample_ruleset(ruleset_key)

# # to list available environments: xminigrid.registered_environments()
# env, env_params = xminigrid.make("XLand-MiniGrid-R9-25x25")
# env_params = env_params.replace(ruleset=ruleset)

# # auto-reset wrapper
# env = GymAutoResetWrapper(env)

# # render obs as rgb images if needed (warn: this will affect speed greatly)
# env = RGBImgObservationWrapper(env)

# # fully jit-compatible step and reset methods
# timestep = jax.jit(env.reset)(env_params, reset_key)
# timestep = jax.jit(env.step)(env_params, timestep, action=0)

# # optionally render the state
# env.render(env_params, timestep)