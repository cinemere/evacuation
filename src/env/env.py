# ## %%
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging; log = logging.getLogger(__name__)

from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib as mpl
from functools import reduce
from enum import Enum, auto

from src.env import constants as const

from typing import Tuple, List, Dict

import wandb

def setup_logging(verbose, experiment_name):
    logs_folder = const.SAVE_PATH_LOGS
    if not os.path.exists(logs_folder): os.makedirs(logs_folder)

    logs_filename = os.path.join(logs_folder, f"logs_{experiment_name}.log")

    logging.basicConfig(
        filename=logs_filename, filemode="w",
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO
    )

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
    distances = distance_matrix(
        pedestrians_positions, np.expand_dims(destination, axis=0), 2
    )
    return np.where(distances < radius, True, False).squeeze()

def sum_distance(pedestrians_positions, destination):
    distances = distance_matrix(
        pedestrians_positions, np.expand_dims(destination, axis=0), 2
    )
    return distances.sum() / pedestrians_positions.shape[0]

def estimate_intrinsic_reward(pedestrians_positions, exit_position):
    return (1 - sum_distance(pedestrians_positions, exit_position)) / const.MAX_TIMESTEPS * 10

def estimate_reward(old_statuses, new_statuses, timesteps, num_pedestrians):
    reward = 0
    time_factor = 1 - timesteps / (200 * num_pedestrians)

    # Reward for new exiting
    prev = np.logical_or(old_statuses == Status.VISCEK, old_statuses == Status.FOLLOWER)
    curr = new_statuses == Status.EXITING
    n = sum(np.logical_and(prev, curr))
    reward += (15 + 10 * time_factor) * n

    # Reward for new followers
    prev = old_statuses == Status.VISCEK
    curr = new_statuses == Status.FOLLOWER
    n = sum(np.logical_and(prev, curr))
    reward += (10 + 5 * time_factor) * n
    return reward


def update_statuses(statuses, pedestrian_positions, agent_position, exit_position):
    # n_viscek = sum(statuses == Status.VISCEK)
    # n_follow = sum(statuses == Status.FOLLOWER)
    # n_exitin = sum(statuses == Status.EXITING)
    # n_escape = sum(statuses == Status.ESCAPED)
    # print(n_viscek, n_follow, n_exitin, n_escape)
    new_statuses = statuses.copy()

    condition = is_distance_low(
        pedestrian_positions, agent_position, SwitchDistances.to_leader)
    new_statuses[condition] = Status.FOLLOWER
    
    condition = is_distance_low(
        pedestrian_positions, exit_position, SwitchDistances.to_exit)
    new_statuses[condition] = Status.EXITING
    
    condition = is_distance_low(
        pedestrian_positions, exit_position, SwitchDistances.to_escape)
    new_statuses[condition] = Status.ESCAPED
    return new_statuses


def count_new_statuses(old_statuses, new_statuses):
    """Get number of pedestrians, who have updated their status"""
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
    memory : Dict[str, List[np.ndarray]]

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
        self.memory = {'positions' : [], 'statuses' : []}
    
    def normirate_directions(self) -> None:
        x = self.directions
        self.directions = (x.T / np.linalg.norm(x, axis=1)).T

    def save(self):
        self.memory['positions'].append(self.positions.copy())
        self.memory['statuses'].append(self.statuses.copy())


class Agent:
    start_position : np.ndarray
    start_direction : np.ndarray
    position : np.ndarray
    direction : np.ndarray
    memory : Dict[str, List[np.ndarray]]

    def __init__(self):
        self.start_position = np.zeros(2, dtype=np.float32)
        self.start_direction = np.zeros(2, dtype=np.float32)
        
    def reset(self):
        self.position = self.start_position.copy()
        self.direction = self.start_position.copy()
        self.memory = {'position' : []}

    def save(self):
        self.memory['position'].append(self.position.copy())


class Exit:
    position : np.ndarray
    def __init__(self):
        self.position = np.array([0, -1], dtype=np.float32)


class Time:
    def __init__(self, max_timesteps, n_episodes):
        self.now = 0
        self.max_timesteps = max_timesteps
        self.n_episodes = n_episodes

    def reset(self):
        self.now = 0
        self.n_episodes += 1

    def step(self):
        self.now += 1
        return self.truncated()
        
    def truncated(self):
        return self.now >= self.max_timesteps 


def grad_potential_pedestrians(agent: Agent, pedestrians: Pedestrians, alpha: float):
    R = agent.position[np.newaxis, :] - pedestrians.positions
    R = R[pedestrians.statuses == Status.VISCEK]

    if len(R) != 0:
        norm = np.linalg.norm(R, axis = 1)[:, np.newaxis] + const.EPS
        grad = - alpha / norm ** (alpha + 2) * R
        grad = grad.sum(axis = 0)
    else:
        grad = np.zeros(2)
    return grad


def grad_potential_exit(agent: Agent, pedestrians: Pedestrians, exit: Exit, alpha: float):
    R = agent.position - exit.position
    norm = np.linalg.norm(R) + const.EPS
    grad = - alpha / norm ** (alpha + 2) * R
    grad *= sum(pedestrians.statuses == Status.FOLLOWER)
    return grad


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

    def pedestrians_step(self, pedestrians : Pedestrians, agent : Agent, now : int) -> Tuple[Pedestrians, bool, float]:
        old_statuses = pedestrians.statuses.copy()

        escaped = pedestrians.statuses == Status.ESCAPED
        pedestrians.directions[escaped] = 0
        pedestrians.positions[escaped] = self.exit.position

        exiting = pedestrians.statuses == Status.EXITING
        if any(exiting):
            vec2exit = self.exit.position - pedestrians.positions[exiting]
            len2exit = np.linalg.norm(vec2exit, axis=1)
            vec_size = np.minimum(len2exit, self.step_size)
            vec2exit = (vec2exit.T / len2exit * vec_size).T
            pedestrians.directions[exiting] = vec2exit
            # pedestrians.positions[exiting] += vec2exit #uncomment

        following = pedestrians.statuses == Status.FOLLOWER
        pedestrians.directions[following] = agent.direction

        viscek = pedestrians.statuses == Status.VISCEK
        # fv = np.logical_or(following, viscek) #uncomment begin
        # fv_directions = pedestrians.directions[fv]
        # dm = distance_matrix(pedestrians.positions[viscek], # add exitors
        #                      pedestrians.positions[fv], 2)
        # intersection = np.where(dm < SwitchDistances.to_pedestrian, 1, 0)
        # n_intersections = np.maximum(1, intersection.sum(axis=1))
        # # pedestrians.normirate_directions()
        # fv_directions = (fv_directions.T / np.linalg.norm(fv_directions, axis=1)).T 
        # v_directions_x = (intersection * fv_directions[:, 0]).sum(axis=1) / n_intersections
        # v_directions_y = (intersection * fv_directions[:, 1]).sum(axis=1) / n_intersections
        # v_directions = np.concatenate((v_directions_x[np.newaxis, :], 
        #                                v_directions_y[np.newaxis, :])).T
        # v_directions = (v_directions.T / np.linalg.norm(v_directions, axis=1)).T #uncomment end

        efv = reduce(np.logical_or, (exiting, following, viscek))
        efv_directions = pedestrians.directions[efv]
        dm = distance_matrix(pedestrians.positions[viscek],
                             pedestrians.positions[efv], 2)
        
        intersection = np.where(dm < SwitchDistances.to_pedestrian, 1, 0)
        n_intersections = np.maximum(1, intersection.sum(axis=1))

        # pedestrians.normirate_directions()
        efv_directions = (efv_directions.T / np.linalg.norm(efv_directions, axis=1)).T 
        
        v_directions_x = (intersection * efv_directions[:, 0]).sum(axis=1) / n_intersections
        v_directions_y = (intersection * efv_directions[:, 1]).sum(axis=1) / n_intersections
        v_directions = np.concatenate((v_directions_x[np.newaxis, :], 
                                       v_directions_y[np.newaxis, :])).T
        v_directions = (v_directions.T / np.linalg.norm(v_directions, axis=1)).T

        # randomization = (np.random.rand(sum(viscek), 2) - 0.5) * 2 * self.step_size  # norm distribution! TODO
        randomization = np.random.normal(loc=0.0, scale=self.step_size, size=(sum(viscek), 2))
        randomization = (randomization.T / (np.linalg.norm(randomization, axis=1) + const.EPS)).T
        
        v_directions = (v_directions + const.NOISE_COEF * randomization) #/ (1 + const.NOISE_COEF)
        v_directions = (v_directions.T / np.linalg.norm(v_directions, axis=1)).T

        pedestrians.directions[viscek] = v_directions * self.step_size
        pedestrians.positions[efv] += pedestrians.directions[efv] 

        clipped = np.clip(pedestrians.positions, 
                    [-self.width, -self.height], [self.width, self.height])
        miss = pedestrians.positions - clipped
        pedestrians.positions -= 2 * miss
        pedestrians.directions *= np.where(miss!=0, -1, 1)

        # Estimate pedestrians reward and update statuses
        new_pedestrians_statuses = update_statuses(
            statuses=pedestrians.statuses,
            pedestrian_positions=pedestrians.positions,
            agent_position=agent.position,
            exit_position=self.exit.position
        )
        reward_pedestrians = estimate_reward(
            old_statuses=old_statuses,
            new_statuses=new_pedestrians_statuses,
            timesteps=now,
            num_pedestrians=pedestrians.num
        )
        intrinsic_reward = estimate_intrinsic_reward(
            pedestrians_positions=pedestrians.positions,
            exit_position=self.exit.position
        )
        pedestrians.statuses = new_pedestrians_statuses
        if sum(pedestrians.statuses == Status.ESCAPED) == pedestrians.num:
            termination = True
        else: 
            termination = False

        return pedestrians, termination, reward_pedestrians + intrinsic_reward

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
            return agent, False, 0.
        else:
            # return agent, True, -5.
            return agent, False, -5.

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
    Continious Action and Observation Space.
    """
    def __init__(self, 
        render_mode=None,
        number_of_pedestrians = const.NUM_PEDESTRIANS,
        experiment_name='test',
        verbose=False,
        draw=False,
        enable_gravity_embedding=True,
        n_episodes=0
        ) -> None:
        super(EvacuationEnv, self).__init__()
            
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.draw = draw
        self.experiment_name = experiment_name
        setup_logging(verbose, experiment_name)

        self.area = Area()
        self.agent = Agent()
        self.pedestrians = Pedestrians(num=number_of_pedestrians)
        self.time = Time(max_timesteps=const.MAX_TIMESTEPS, n_episodes=n_episodes)
        log.info('Env is initialized.')        
        
        self.enabled_gravity_embedding = enable_gravity_embedding
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = self._get_observation_space()
        
        self.save_next_episode_anim = False

    def _get_observation_space(self):        
        observation_space = {
            'agent_position' : spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
        }
        
        if self.enabled_gravity_embedding:
            observation_space['grad_potential_pedestrians'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
            observation_space['grad_potential_exit'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
    
        else:
            observation_space['pedestrians_positions'] = \
                spaces.Box(low=-1, high=1, shape=(self.pedestrians.num, 2), dtype=np.float32)
            
        return spaces.Dict(observation_space)

    def _get_observation(self):
        observation = {}
        observation['agent_position'] = self.agent.position
        
        if self.enabled_gravity_embedding:
            observation['grad_potential_pedestrians'] = grad_potential_pedestrians(
                agent=self.agent, 
                pedestrians=self.pedestrians, 
                alpha=const.ALPHA
            )
            observation['grad_potential_exit'] = grad_potential_exit(
                agent=self.agent,
                pedestrians=self.pedestrians,
                exit=self.area.exit,
                alpha=const.ALPHA
            )
        else:
            observation['pedestrians_positions'] = self.pedestrians.positions
            
        return observation

    def reset(self, seed=None):
        if self.save_next_episode_anim:
            self.draw = True

        if self.time.n_episodes > 0:
            wandb.log({
                "escaped_pedestrians" : sum(self.pedestrians.statuses == Status.ESCAPED),
                "exiting_pedestrians" : sum(self.pedestrians.statuses == Status.EXITING),
                "following_pedestrians" : sum(self.pedestrians.statuses == Status.FOLLOWER),
                "viscek_pedestrians" : sum(self.pedestrians.statuses == Status.VISCEK)
            })
        
        self.time.reset()
        self.area.reset()
        self.agent.reset()
        self.pedestrians.reset(agent_position=self.agent.position,
                               exit_position=self.area.exit.position)
        self.pedestrians.save()
        log.info('Env is reseted.')
        return self._get_observation(), {}

    def step(self, action: list):
        # Incrimenate time
        truncated = self.time.step()

        # Agent step
        self.agent, terminated_agent, reward_agent = self.area.agent_step(action, self.agent)
        
        # Pedestrians step
        self.pedestrians, terminated_pedestrians, reward_pedestrians = \
            self.area.pedestrians_step(self.pedestrians, self.agent, self.time.now)
        
        # Save positions for rendering an animation
        if self.draw:
            self.pedestrians.save()
            self.agent.save()

        # Collect rewards
        reward = reward_agent + reward_pedestrians
        
        # Record observation
        observation = self._get_observation()
        
        if (terminated_agent or terminated_pedestrians or truncated) and self.draw:
            self.save_animation()

        return observation, reward, terminated_agent or terminated_pedestrians, truncated, {}

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
        plt.xticks([]); plt.yticks([])
        plt.hlines([self.area.height, -self.area.height], 
            -self.area.width, self.area.width, linestyle='--', color='grey')
        plt.vlines([self.area.width, -self.area.width], 
            -self.area.height, self.area.height, linestyle='--', color='grey')

        plt.title(f"{self.experiment_name}. Timesteps: {self.time.now}")

        plt.tight_layout()
        if not os.path.exists(const.SAVE_PATH_PNG): os.makedirs(const.SAVE_PATH_PNG)
        filename = os.path.join(const.SAVE_PATH_PNG, f'{self.experiment_name}.png')
        plt.savefig(filename)
        plt.show()
        log.info(f"Env is rendered and pnd image is saved to {filename}")

    def save_animation(self):
        
        fig, ax = plt.subplots(figsize=(5, 5))

        plt.title(f"{self.experiment_name}\nn_episodes = {self.time.n_episodes}")
        plt.hlines([self.area.height, -self.area.height], 
            -self.area.width, self.area.width, linestyle='--', color='grey')
        plt.vlines([self.area.width, -self.area.width], 
            -self.area.height, self.area.height, linestyle='--',  color='grey')
        plt.xlim([ -1.1 * self.area.width, 1.1 * self.area.width])
        plt.ylim([ -1.1 * self.area.height, 1.1 * self.area.height])
        plt.xticks([]); plt.yticks([])

        exit_coordinates = (self.area.exit.position[0], self.area.exit.position[1])
        agent_coordinates = (self.agent.memory['position'][0][0], self.agent.memory['position'][0][1])

        # Draw exiting zone
        exiting_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_exit, 
            0, 180, alpha = 0.2, color='green'
        )
        ax.add_patch(exiting_zone)

        # Draw following zone
        following_zone = mpatches.Wedge(
            agent_coordinates, 
            SwitchDistances.to_leader, 
            0, 360, alpha=0.1, color='blue'
        )
        following_zone_plots = ax.add_patch(following_zone)

        # Draw escaping zone
        escaping_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_escape, 
            0, 180, color='white'
        )
        ax.add_patch(escaping_zone)
        
        # Draw exit
        ax.plot(exit_coordinates[0], exit_coordinates[1], marker='X', color='green')
        
        # Draw pedestrians
        pedestrian_position_plots = {}
        for status in Status.all():
            selected_pedestrians = self.pedestrians.memory['statuses'][0] == status
            color = next(ax._get_lines.prop_cycler)['color']
            pedestrian_position_plots[status] = \
                ax.plot(self.pedestrians.memory['positions'][0][selected_pedestrians, 0], 
                self.pedestrians.memory['positions'][0][selected_pedestrians, 1],
                lw=0, marker='.', color=color)[0]

        # Draw agent
        agent_position_plot = ax.plot(agent_coordinates[0], agent_coordinates[1], marker='+', color='red')[0]

        def update(i):

            agent_coordinates = (self.agent.memory['position'][i][0], self.agent.memory['position'][i][1])
            following_zone_plots.set_center(agent_coordinates)

            for status in Status.all():
                selected_pedestrians = self.pedestrians.memory['statuses'][i] == status
                pedestrian_position_plots[status].set_xdata(self.pedestrians.memory['positions'][i][selected_pedestrians, 0])
                pedestrian_position_plots[status].set_ydata(self.pedestrians.memory['positions'][i][selected_pedestrians, 1])
                 
            # agent_position_plot.set_xdata(agent_coordinates[0])
            # agent_position_plot.set_ydata(agent_coordinates[1])
            agent_position_plot.set_data(agent_coordinates)

        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.time.now, interval=20)
        
        if not os.path.exists(const.SAVE_PATH_GIFF): os.makedirs(const.SAVE_PATH_GIFF)
        filename = os.path.join(const.SAVE_PATH_GIFF, f'{self.experiment_name}_ep-{self.time.n_episodes}.gif')
        ani.save(filename=filename, writer='pillow')
        log.info(f"Env is rendered and gif animation is saved to {filename}")

        if self.save_next_episode_anim:
            self.save_next_episode_anim = False
            self.draw = False

    def close(self):
        pass
# # %%
# e = EvacuationEnv(number_of_pedestrians=100)

# e.reset()
# e.step([1, 0])

# for i in range(300):
#     e.step([np.sin(i*0.1), np.cos(i*0.1)])
# e.save_animation()
# e.render()
# # %%
