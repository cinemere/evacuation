from enum import Enum, auto
import numpy as np
from functools import reduce

from .distances import SwitchDistances, is_distance_low

class UserEnum(Enum):
    @classmethod
    def all(cls):
        return list(map(lambda c: c, cls))
    
    @classmethod
    def __len__(cls):
        return len(cls.all())

class Status(UserEnum):
    VISCEK = auto()
    "Pedestrian under Viscek rules."

    FOLLOWER = auto()
    "Follower of the leader particle (agent)."

    EXITING = auto()
    "Pedestrian in exit zone."

    ESCAPED = auto()
    "Evacuated pedestrian."
    
def update_statuses(statuses, pedestrian_positions, agent_position, exit_position):
    """Measure statuses of all pedestrians based on their position"""
    new_statuses = statuses.copy()

    following = is_distance_low(
        pedestrian_positions, agent_position, SwitchDistances.to_leader)
    new_statuses[following] = Status.FOLLOWER
    
    exiting = is_distance_low(
        pedestrian_positions, exit_position, SwitchDistances.to_exit)
    new_statuses[exiting] = Status.EXITING
    
    escaped = is_distance_low(
        pedestrian_positions, exit_position, SwitchDistances.to_escape)
    new_statuses[escaped] = Status.ESCAPED
    
    viscek = np.logical_not(reduce(np.logical_or, (exiting, following, escaped)))
    new_statuses[viscek] = Status.VISCEK
    
    return new_statuses

def count_new_statuses(old_statuses, new_statuses):
    """Get number of pedestrians, who have updated their status"""
    count = {}
    for status in Status.all():
        count[status] = sum(
            np.logical_and(new_statuses == status, old_statuses != status)
        )
    return count