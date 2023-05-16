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

from typing import Tuple

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
    to_pedestrian = const.SWITCH_DISTANCE_TO_OTHER_PEDESTRIAN


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
        self.positions  = np.random.uniform(-1.0, 1.0, size=(self.num, 2))
        self.directions = np.random.uniform(-1.0, 1.0, size=(self.num, 2))
        self.normirate_directions()
        self.statuses = np.array([Status.VISCEK for _ in range(self.num)])
        self.statuses = update_statuses(
            self.statuses,
            self.positions,
            agent_position,
            exit_position
        )
    
    def normirate_directions(self) -> None:
        x = self.directions
        self.directions = (x.T / np.linalg.norm(x, axis=1)).T


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

    def pedestrians_step(self, pedestrians : Pedestrians, agent : Agent) -> Tuple[Pedestrians, bool, float]:

        termination, reward = False, 0

        escaped = pedestrians.statuses == Status.ESCAPED
        pedestrians.positions[escaped] = self.exit.position

        exiting = pedestrians.statuses == Status.EXITING
        if not any(exiting):
            vec2exit = self.exit.position - pedestrians.positions[exiting]
            len2exit = np.linalg.norm(vec2exit, axis=1)
            vec_size = np.minimum(len2exit, self.step_size)
            vec2exit = (vec2exit.T / len2exit * vec_size).T # TODO do we really need this transpose?
            pedestrians.positions[exiting] += vec2exit
            # TODO check direction

        following = pedestrians.statuses == Status.FOLLOWER
        pedestrians.directions[following] = agent.direction

        viscek = pedestrians.statuses == Status.VISCEK
        fv = np.logical_or(following, viscek)
        fv_directions = pedestrians.directions[fv]

        dm = distance_matrix(pedestrians.positions[viscek], # add exitors
                             pedestrians.positions[fv], 2)
        
        intersection = np.where(dm < SwitchDistances.to_pedestrian, 1, 0)
        n_intersections = np.maximum(1, intersection.sum(axis=1))

        # pedestrians.normirate_directions()
        fv_directions = (fv_directions.T / np.linalg.norm(fv_directions, axis=1)).T 
        
        v_directions_x = (intersection * fv_directions[:, 0]).sum(axis=1) / n_intersections
        v_directions_y = (intersection * fv_directions[:, 1]).sum(axis=1) / n_intersections
        v_directions = np.concatenate((v_directions_x[np.newaxis, :], 
                                       v_directions_y[np.newaxis, :])).T
        v_directions = (v_directions.T / np.linalg.norm(v_directions, axis=1)).T
        
        eps = 1e-8 # TODO add to constants
        noise_coef = 0.2

        randomization = (np.random.rand(sum(viscek), 2) - 0.5) * 2 * self.step_size# norm distribution! TODO
        randomization = (randomization.T / (np.linalg.norm(randomization, axis=1) + eps)).T
        
        v_directions = (v_directions + noise_coef * randomization) #/ (1 + noise_coef)
        v_directions = (v_directions.T / np.linalg.norm(v_directions, axis=1)).T

        pedestrians.directions[viscek] = v_directions * self.step_size
        pedestrians.positions[fv] += pedestrians.directions[fv] 

        clipped = np.clip(pedestrians.positions, 
                    [-self.width, -self.height], [self.width, self.height])
        miss = pedestrians.positions - clipped
        pedestrians.positions -= 2 * miss
        pedestrians.directions *= np.where(miss!=0, -1, 1)

        return pedestrians

    def agent_step(self, action : list, agent : Agent) -> Tuple[Agent, bool, float]:
        """
        Perform agent step:
            1. Read & preprocess action
            2. Check wall collision
            3. Return (updated agent, termination, reward)
        """
        action = np.array(action)
        np.clip(action, -1, 1, out=action)
        agent.direction = self.step_size * action
        
        if not self._if_wall_collision(agent):
            agent.position += agent.direction
            print(agent.position, agent.direction)
            return agent, False, 0.
        else:
            return agent, True, -5.

    def _if_wall_collision(self, agent : Agent):
        pt = agent.position + agent.direction

        left  = pt[0] < -self.width
        right = pt[0] > self.width
        down  = pt[1] < -self.height  
        up    = pt[1] > self.height
        
        if left or right or down or up:
            return True
        return False


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

    def step(self, action : list):
        self.time.step()
        self.agent, termination, episode_reward_agent = self.area.agent_step(action, self.agent)
        self.pedestrians = self.area.pedestrians_step(self.pedestrians, self.agent)
        self.pedestrians.statuses = update_statuses(
            statuses=self.pedestrians.statuses,
            pedestrian_positions=self.pedestrians.positions,
            agent_position=self.agent.position,
            exit_position=self.area.exit.position
        )
        # TODO get reward

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
            # for i in range(self.pedestrians.directions.shape[0]):
            #     ax.plot(self.pedestrians.positions[i],
            #     self.pedestrians.positions[i] + self.pedestrians.directions[i])

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
e.step([1, 0])
# %%
e.render()

# %%
from time import sleep
for i in range(150):
    e.step([np.sin(i*0.1), np.cos(i*0.1)])
    e.render()
    sleep(0.1)
# %%
