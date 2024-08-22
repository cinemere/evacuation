from flax import struct
import jax.numpy as jnp
from typing import Optional

EXIT = jnp.asarray((0, 0), dtype=jnp.float32)

class SwitchDistance(struct.PyTreeNode):
    to_leader: float = struct.field(pytree_node=False, default=0.2)             # radius of catch by leader
    to_other_pedestrian: float = struct.field(pytree_node=False, default=0.1)   # SWITCH_DISTANCE_TO_LEADER
    to_exit: float = struct.field(pytree_node=False, default=0.4)
    to_escape: float = struct.field(pytree_node=False, default=0.01)


class Status(struct.PyTreeNode):
    VISCEK: int = struct.field(pytree_node=False, default=0)
    "Pedestrian under Viscek rules."

    FOLLOWER: int = struct.field(pytree_node=False, default=1)
    "Follower of the leader particle (agent)."

    EXITING: int = struct.field(pytree_node=False, default=2)
    "Pedestrian in exit zone."

    ESCAPED: int = struct.field(pytree_node=False, default=3)
    "Evacuated pedestrian."    


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

    following = is_distance_low(pedestrian_positions, agent_position, SwitchDistance.to_leader)
    new_statuses = jnp.where(following, Status.FOLLOWER, new_statuses)

    exiting = is_distance_low(pedestrian_positions, exit_position, SwitchDistance.to_exit)
    new_statuses = jnp.where(exiting, Status.EXITING, new_statuses)

    escaped = is_distance_low(pedestrian_positions, exit_position, SwitchDistance.to_escape)
    new_statuses = jnp.where(escaped, Status.ESCAPED, new_statuses)

    viscek = jnp.logical_not(jnp.logical_or(jnp.logical_or(exiting, following), escaped))
    new_statuses = jnp.where(viscek, Status.VISCEK, new_statuses)

    return new_statuses