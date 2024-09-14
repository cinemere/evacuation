from __future__ import annotations

from typing import TypeVar

from flax import struct


class EnvParams(struct.PyTreeNode):

    # max_steps: Optional[None] = struct.field(pytree_node=False, default=None)
    render_mode: str = struct.field(pytree_node=False, default="plot")

    number_of_pedestrians: int = struct.field(pytree_node=False, default=10)
    """number of pedestrians in the simulation"""

    width: float = struct.field(pytree_node=False, default=1.0)
    """geometry of environment space: width"""

    height: float = struct.field(pytree_node=False, default=1.0)
    """geometry of environment space: height"""

    step_size: float = struct.field(pytree_node=False, default=0.01)
    """length of pedestrian\'s and agent\'s step
    Typical expected values: 0.1, 0.05, 0.01"""

    noise_coef: float = struct.field(pytree_node=False, default=0.2)
    """noise coefficient of randomization in viscek model"""

    eps: float = struct.field(pytree_node=False, default=1e-8)
    """eps"""

    # ---- Leader params ----

    enslaving_degree: float = struct.field(pytree_node=False, default=1.0)
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

    intrinsic_reward_coef: float = struct.field(pytree_node=False, default=0.0)
    """coefficient in front of intrinsic reward"""

    is_termination_agent_wall_collision: bool = struct.field(
        pytree_node=False, default=False
    )
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
