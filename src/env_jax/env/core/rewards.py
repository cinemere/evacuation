import jax.numpy as jnp
from typing import Dict, Tuple

from .constants import Status
from .utils import mean_distance


def estimate_exit_reward(
    old_statuses: jnp.ndarray,
    new_statuses: jnp.ndarray,
    timesteps: int,
    num_pedestrians: int,
) -> float:
    """This reward is given for each new `exiting` pedestrian

    VISCEK or FOLLOWER --> EXITING  :  (15 +  5 * time_factor)
    """
    time_factor = 1 - timesteps / (200 * num_pedestrians)
    prev = jnp.logical_or(
        old_statuses == Status.VISCEK, old_statuses == Status.FOLLOWER
    )
    curr = new_statuses == Status.EXITING
    n = sum(jnp.logical_and(prev, curr))
    return (15 + 10 * time_factor) * n


def estimate_follow_reward(
    old_statuses: jnp.ndarray,
    new_statuses: jnp.ndarray,
    timesteps: int,
    num_pedestrians: int,
) -> float:
    """This reward is given for each new `following` pedestrian

    VISCEK             --> FOLLOWER :  (10 + 10 * time_factor)
    """
    time_factor = 1 - timesteps / (200 * num_pedestrians)
    prev = old_statuses == Status.VISCEK
    curr = new_statuses == Status.FOLLOWER
    n = sum(jnp.logical_and(prev, curr))
    return (10 + 5 * time_factor) * n


def estimate_status_reward(
    old_statuses: jnp.ndarray,
    new_statuses: jnp.ndarray,
    timesteps: int,
    num_pedestrians: int,
    init_reward_each_step: int,
    is_new_exiting_reward: bool,
    is_new_followers_reward: bool,
) -> float:
    """This reward is based on how pedestrians update their status

    VISCEK or FOLLOWER --> EXITING  :  (15 +  5 * time_factor)
    VISCEK             --> FOLLOWER :  (10 + 10 * time_factor)
    """
    # Each step reward
    reward = init_reward_each_step

    # Reward for new exiting
    if is_new_exiting_reward:
        reward += estimate_exit_reward(
            old_statuses=old_statuses,
            new_statuses=new_statuses,
            timesteps=timesteps,
            num_pedestrians=num_pedestrians,
        )

    # Reward for new followers
    if is_new_followers_reward:
        reward += estimate_follow_reward(
            old_statuses=old_statuses,
            new_statuses=new_statuses,
            timesteps=timesteps,
            num_pedestrians=num_pedestrians,
        )

    return reward


def estimate_intrinsic_reward(
    pedestrians_positions: jnp.ndarray,
    exit_position: jnp.ndarray,
) -> float:
    """This is intrinsic reward, which can be given to the agent at each step"""
    return 0 - mean_distance(pedestrians_positions, exit_position)

def estimate_agent_reward(
    agent_step_info: Dict[str, bool],
    is_termination_agent_wall_collision: bool, 
) -> Tuple[float, bool]:
    if agent_step_info['wall_collision']:
        return is_termination_agent_wall_collision, -5.
    else:
        return False, 0.