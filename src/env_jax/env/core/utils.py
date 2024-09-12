import jax
import jax.numpy as jnp
from typing import Optional

from .constants import Status, SwitchDistance, EXIT


def is_distance_low(
    pedestrians_positions: jnp.ndarray,
    destination: jnp.ndarray,
    radius: float,
) -> jnp.ndarray:
    """Get boolean matrix showing pedestrians,
    which are closer to destination than raduis

    Args:
        pedestrians_positions (jnp.ndarray): coordinates of pedestrians
        (dim: [n x 2])

        destination (jnp.ndarray): coordinates of destination
        (dim: [2])

        radius (float): max distance to destination

    Returns:
        jnp.ndarray: boolean matrix
    """
    distances = jnp.linalg.norm(pedestrians_positions - destination[None, :], axis=1)
    return distances < radius


def mean_distance(
    pedestrians_positions: jnp.ndarray,
    destination: jnp.ndarray,
) -> float:
    """Mean distance between pedestrians and destination

    Args:
        pedestrians_positions (jnp.ndarray): coordinates of pedestrians
        (dim: [n x 2])

        destination (jnp.ndarray): coordinates of destination
        (dim: [2])

    Returns:
        float: mean distance
    """
    distances = jnp.linalg.norm(pedestrians_positions - destination[None, :], axis=1)
    return distances.mean()


def update_statuses(
    statuses: jnp.ndarray,
    pedestrian_positions: jnp.ndarray,
    agent_position: jnp.ndarray,
    exit_position: Optional[jnp.ndarray] = None,
):
    """Measure statuses of all pedestrians based on their position"""
    new_statuses = statuses.copy()

    if exit_position is None:
        exit_position = EXIT

    following = is_distance_low(
        pedestrian_positions, agent_position, SwitchDistance.to_leader
    )
    new_statuses = jnp.where(following, Status.FOLLOWER, new_statuses)

    exiting = is_distance_low(
        pedestrian_positions, exit_position, SwitchDistance.to_exit
    )
    new_statuses = jnp.where(exiting, Status.EXITING, new_statuses)

    escaped = is_distance_low(
        pedestrian_positions, exit_position, SwitchDistance.to_escape
    )
    new_statuses = jnp.where(escaped, Status.ESCAPED, new_statuses)

    viscek = jnp.logical_not(
        jnp.logical_or(jnp.logical_or(exiting, following), escaped)
    )
    new_statuses = jnp.where(viscek, Status.VISCEK, new_statuses)

    return new_statuses


def estimate_mean_direction_among_neighbours(
    intersection: jax.Array,  # [f+v, f+v+e]  boolean matrix
    efv_directions: jax.Array,  # [f+v+e, 2]    vectors of directions of pedestrians
    n_intersections: jax.Array,  # [f+v]         amount of neighbouring pedestrians
    noise_coef: float,
    key: jax.Array,
):
    """Viscek model"""
    fv_directions_x = (intersection * efv_directions.at[:, 0].get()).sum(
        axis=1
    ) / n_intersections
    fv_directions_y = (intersection * efv_directions.at[:, 1].get()).sum(
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
