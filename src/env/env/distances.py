import numpy as np
from scipy.spatial import distance_matrix

from ..constants import (
    SWITCH_DISTANCE_TO_LEADER,
    SWITCH_DISTANCE_TO_EXIT,
    SWITCH_DISTANCE_TO_ESCAPE,
    SWITCH_DISTANCE_TO_OTHER_PEDESTRIAN,
)

__all__= [
    "SwitchDistances",
    "is_distance_low",
    "sum_distance"
]

class SwitchDistances:
    to_leader: float     = SWITCH_DISTANCE_TO_LEADER
    to_exit: float       = SWITCH_DISTANCE_TO_EXIT
    to_escape: float     = SWITCH_DISTANCE_TO_ESCAPE
    to_pedestrian: float = SWITCH_DISTANCE_TO_OTHER_PEDESTRIAN


def is_distance_low(
    pedestrians_positions: np.ndarray, 
    destination: np.ndarray, 
    radius: float
    ) -> np.ndarray:
    """Get boolean matrix showing pedestrians,
    which are closer to destination than raduis 

    Args:
        pedestrians_positions (npt.NDArray): coordinates of pedestrians 
        (dim: [n x 2])
        
        destination (npt.NDArray): coordinates of destination
        (dim: [2])
        
        radius (float): max distance to destination

    Returns:
        npt.NDArray: boolean matrix
    """
    
    distances = distance_matrix(
        pedestrians_positions, np.expand_dims(destination, axis=0), 2
    )
    return np.where(distances < radius, True, False).squeeze()


def sum_distance(pedestrians_positions, destination):
    """Mean distance between pedestrians and destination"""
    distances = distance_matrix(
        pedestrians_positions, np.expand_dims(destination, axis=0), 2
    )
    return distances.sum() / pedestrians_positions.shape[0]
