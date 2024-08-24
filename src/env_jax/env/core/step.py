from typing import Tuple
import jax.numpy as jnp
import jax
from functools import reduce

from ..types import AgentState, PedestriansState, IntOrArray
from .constants import Status, EXIT, SwitchDistance
from .utils import update_statuses
from .rewards import estimate_intrinsic_reward, estimate_status_reward


def _is_wall_collision(
    pt: jnp.ndarray,  # point coordinates
    width: float,  # width of simulation area
    height: float,  # height of simulation area
) -> bool:
    # Check for wall collisions by comparing the point's coordinates to the boundaries
    left = pt[0] < -width
    right = pt[0] > width
    down = pt[1] < -height
    up = pt[1] > height

    # Return True if any collision condition is met
    return jnp.any(jnp.array([left, right, down, up]))


def agent_step(
    action: IntOrArray,
    agent: AgentState,
    step_size: float,
    eps: float,
    width: float,  # width of simulation area
    height: float,  # height of simulation area
) -> Tuple[AgentState, dict]:
    """
    Perform agent step:
        1. Read & preprocess (scale or clip) action
        2. Check wall collision
        3. Return updated agent
    """
    print(f"running `agent_step` with {action=} {agent=}")

    # Scale action
    action = jnp.array(action)
    action /= jnp.linalg.norm(action) + eps

    # Estimate new position
    new_direction = step_size * action
    new_position = agent.position + new_direction

    # Check wall collision
    wall_collision = _is_wall_collision(new_position, width, height)
    new_position = agent.position if wall_collision else new_position

    # Write resulting position
    new_agent = agent.replace(
        direction=new_direction,
        position=new_position,
    )
    info = {"wall_collision": wall_collision}
    return new_agent, info


def pedestrians_step(
    pedestrians: PedestriansState,
    agent: AgentState,
    now: int,
    step_size: float,
    noise_coef: float,
    width: float,  # width of simulation area
    height: float,  # height of simulation area
    num_pedestrians: int,
    key: jax.Array,
) -> Tuple[PedestriansState, bool, float, float]:
    print(f"running `pedestrians_step` with {pedestrians=} {agent=} {now=}")

    # Check evacuated pedestrians & record new directions and positions of escaped pedestrians
    escaped = pedestrians.statuses == Status.ESCAPED
    pedestrians.directions = pedestrians.directions.at[escaped].set(0)
    pedestrians.positions = pedestrians.positions.at[escaped].set(EXIT)

    # Check exiting pedestrians & record new directions for exiting pedestrians
    exiting = pedestrians.statuses == Status.EXITING
    if jnp.any(exiting):
        vec2exit = EXIT - pedestrians.positions[exiting]
        len2exit = jnp.linalg.norm(vec2exit, axis=1)
        vec_size = jnp.minimum(len2exit, step_size)
        vec2exit = (vec2exit.T / len2exit * vec_size).T
        pedestrians.directions = pedestrians.directions.at[exiting].set(vec2exit)

    # Check following and viscek pedestrians
    following = pedestrians.statuses == Status.FOLLOWER
    viscek = pedestrians.statuses == Status.VISCEK

    # Use all moving particles (efv -- exiting, following, viscek) to estimate the movement of viscek particles
    efv = reduce(jnp.logical_or, (exiting, following, viscek))
    efv_directions = pedestrians.directions[efv]
    efv_directions = (efv_directions.T / jnp.linalg.norm(efv_directions, axis=1)).T

    # Find neighbours between following and viscek (fv) particles and all other moving particles
    fv = reduce(jnp.logical_or, (following, viscek))
    dm = jnp.linalg.norm(
        pedestrians.positions[fv][:, None] - pedestrians.positions[efv][None, :], axis=2
    )
    intersection = jnp.where(dm < SwitchDistance.to_other_pedestrian, 1, 0)
    n_intersections = jnp.maximum(1, intersection.sum(axis=1))

    def estimate_mean_direction_among_neighbours(
        intersection,  # [f+v, f+v+e]  boolean matrix
        efv_directions,  # [f+v+e, 2]    vectors of directions of pedestrians
        n_intersections,  # [f+v]         amount of neighbouring pedestrians
    ):
        """Viscek model"""
        fv_directions_x = (intersection * efv_directions[:, 0]).sum(
            axis=1
        ) / n_intersections
        fv_directions_y = (intersection * efv_directions[:, 1]).sum(
            axis=1
        ) / n_intersections
        fv_theta = jnp.arctan2(fv_directions_y, fv_directions_x)

        # Create randomization noise to obtained directions
        noise = jax.random.uniform(
            key,
            (len(n_intersections),),
            minval=-noise_coef / 2,
            maxval=noise_coef / 2,
        )

        # New direction = estimated_direction + noise
        fv_theta = fv_theta + noise

        return jnp.vstack((jnp.cos(fv_theta), jnp.sin(fv_theta)))

    fv_directions = estimate_mean_direction_among_neighbours(
        intersection, efv_directions, n_intersections
    )

    # Record new directions of following and viscek pedestrians
    pedestrians.directions = pedestrians.directions.at[fv].set(
        fv_directions.T * step_size
    )

    # Add enslaving factor of leader's direction to following particles
    f_directions = pedestrians.directions[following]
    l_directions = agent.direction
    f_directions = (
        agent.enslaving_degree * l_directions
        + (1.0 - agent.enslaving_degree) * f_directions
    )
    pedestrians.directions = pedestrians.directions.at[following].set(f_directions)

    # Record new positions of exiting, following and viscek pedestrians
    pedestrians.positions = pedestrians.positions.at[efv].add(
        pedestrians.directions[efv]
    )

    # Handling of wall collisions
    clipped = jnp.clip(pedestrians.positions, [-width, -height], [width, height])
    miss = pedestrians.positions - clipped
    pedestrians.positions -= 2 * miss
    pedestrians.directions *= jnp.where(miss != 0, -1, 1)

    # Estimate pedestrians statuses, reward & update statuses
    old_statuses = pedestrians.statuses.copy()
    new_pedestrians_statuses = update_statuses(
        statuses=pedestrians.statuses,
        pedestrian_positions=pedestrians.positions,
        agent_position=agent.position,
        exit_position=EXIT,
    )
    reward_pedestrians = estimate_status_reward(
        old_statuses=old_statuses,
        new_statuses=new_pedestrians_statuses,
        timesteps=now,
        num_pedestrians=num_pedestrians,
    )
    intrinsic_reward = estimate_intrinsic_reward(
        pedestrians_positions=pedestrians.positions, exit_position=EXIT
    )
    pedestrians.statuses = new_pedestrians_statuses

    # Termination due to all pedestrians escaped
    termination = jnp.sum(pedestrians.statuses == Status.ESCAPED) == pedestrians.num

    return pedestrians, termination, reward_pedestrians, intrinsic_reward
