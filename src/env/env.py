# %%
import numpy as np
import gym
from gym import spaces
import logging; log = logging.getLogger(__name__)

from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib as mpl
from enum import Enum, auto

import constants as const

class UserEnum(Enum):
    @classmethod
    def all(cls):
        return list(map(lambda c: c, cls))

class Status(UserEnum):
    VISCEK = auto()
    "Pedestrian under Viscek rules."

    FOLLOWER = auto()
    "Follower of the leader particle (agent)."

    EXITING = auto()
    "Pedestrian in exit zone."

    ESCAPED = auto()
    "Evacuated pedestrian."

class SwitchDistances:
    to_leader = const.SWITCH_DISTANCE_TO_LEADER
    to_exit   = const.SWITCH_DISTANCE_TO_EXIT
    to_escape = const.SWITCH_DISTANCE_TO_ESCAPE


def is_distance_low(pedestrians_positions, destination, radius):
    distances = distance_matrix(pedestrians_positions, np.expand_dims(destination, axis=0), 2)
    return np.where(distances < radius, True, False).squeeze()


def update_statuses(statuses, pedestrian_positions, agent_position, exit_position):
    condition = is_distance_low(
        pedestrian_positions, agent_position, SwitchDistances.to_leader)
    statuses[condition] = Status.FOLLOWER
    
    condition = is_distance_low(
        pedestrian_positions, exit_position, SwitchDistances.to_exit)
    statuses[condition] = Status.EXITING
    
    condition = is_distance_low(
        pedestrian_positions, exit_position, SwitchDistances.to_escape)
    statuses[condition] = Status.ESCAPED
    return statuses


def count_new_statuses(old_statuses, new_statuses):
    count = {}
    for status in Status.all():
        count[status] = sum(
            np.logical_and(new_statuses == status, old_statuses != status)
        )
    return count

class Pedestrians:
    num : int # number of pedestrians
    
    positions : np.ndarray
    directions : np.ndarray
    statuses : np.ndarray
    
    def __init__(self, num : int):
        self.num = num

    def reset(self, agent_position, exit_position):
        self.positions = np.random.uniform(-1.0, 1.0, size=(self.num, 2))
        self.directions = np.zeros((self.num, 2), dtype=np.float32)

        self.statuses = np.array([Status.VISCEK for _ in range(self.num)])
        self.statuses = update_statuses(
            self.statuses,
            self.positions,
            agent_position,
            exit_position
        ) 


class Agent:
    start_position : np.ndarray
    start_direction : np.ndarray
    position : np.ndarray
    direction : np.ndarray
    
    def __init__(self):
        self.start_position = np.zeros(2, dtype=np.float32)
        self.start_direction = np.zeros(2, dtype=np.float32)
        
    def reset(self):
        self.position = self.start_position.copy()
        self.direction = self.start_position.copy()


class Exit:
    position : np.ndarray
    def __init__(self):
        self.position = np.array([0, -1], dtype=np.float32)

class Time:
    def __init__(self, max_timesteps):
        self.now = 0
        self.max_timesteps = max_timesteps

    def reset(self):
        self.now = 0

    def step(self):
        self.now += 1
        return self.now
        
    def terminated(self):
        return self.now >= self.max_timesteps 


class Area:
    def __init__(self, 
        width = const.WIDTH, 
        height = const.HEIGHT,
        step_size = const.STEP_SIZE
        ):
        self.width = width
        self.height = height
        self.step_size = step_size
        self.exit = Exit()
    
    def reset(self):
        pass

    def step(self, pedestrian_positions, pedestrian_directions):
        
        corrected_pedestrian_positions = pedestrian_positions
        corrected_pedestrian_directions = pedestrian_directions

        return (
            corrected_pedestrian_positions,
            corrected_pedestrian_directions
        )


class EvacuationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    """
    Evacuation Game Enviroment for Gym
    Continious Action and Observation Space
    """
    def __init__(self, 
        render_mode=None,
        number_of_pedestrians = const.NUM_PEDESTRIANS,
        ) -> None:
        super(EvacuationEnv, self).__init__()
            
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.area = Area()
        self.agent = Agent()
        self.pedestrians = Pedestrians(num=number_of_pedestrians)
        self.time = Time(max_timesteps=const.MAX_TIMESTEPS)

    def reset(self):
        self.time.reset()
        self.area.reset()
        self.agent.reset()
        self.pedestrians.reset(agent_position=self.agent.position,
                               exit_position=self.area.exit.position)

    def step(self):
        pass

    def render(self):
        fig, ax = plt.subplots(figsize=(5, 5))

        exit_coordinates = (self.area.exit.position[0], self.area.exit.position[1])
        agent_coordinates = (self.agent.position[0], self.agent.position[1])

        # Draw exiting zone
        exiting_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_exit, 
            0, 180, alpha = 0.2, color='green'
        )
        ax.add_patch(exiting_zone)

        # Draw escaping zone
        escaping_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_escape, 
            0, 180, color='white'
        )
        ax.add_patch(escaping_zone)
        
        # Draw exit
        ax.plot(exit_coordinates[0], exit_coordinates[1], marker='X', color='green')

        # Draw following zone
        following_zone = mpatches.Wedge(
            agent_coordinates, 
            SwitchDistances.to_leader, 
            0, 360, alpha=0.1, color='blue'
        )
        ax.add_patch(following_zone)
        
        # Draw pedestrians
        for status in Status.all():
            selected_pedestrians = self.pedestrians.statuses == status
            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(self.pedestrians.positions[selected_pedestrians, 0], 
                    self.pedestrians.positions[selected_pedestrians, 1],
                lw=0, marker='.', color=color)

        # Draw agent
        ax.plot(agent_coordinates[0], agent_coordinates[1], marker='+', color='red')

        plt.xlim([ -1.1 * self.area.width, 1.1 * self.area.width])
        plt.ylim([ -1.1 * self.area.height, 1.1 * self.area.height])

        plt.hlines([self.area.height, -self.area.height], 
            -self.area.width, self.area.width, color='grey')
        plt.vlines([self.area.width, -self.area.width], 
            -self.area.height, self.area.height, color='grey')
        # plt.axhline(-self.area.height, -self.area.width, self.area.width, color='grey')
        # plt.axvline(self.area.width, -self.area.height, self.area.height, color='grey')
        # plt.axvline(-self.area.width, -self.area.height, self.area.height, color='grey')

        plt.tight_layout()
        plt.title(f"Simulation area. Timesteps: {self.time.now}")
        plt.savefig('test.png')
        plt.show()

    def close(self):
        pass

# %%
e = EvacuationEnv(number_of_pedestrians=100)

# %%
e.reset()
# %%
e.render()
# %%
