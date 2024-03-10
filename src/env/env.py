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

from src.env import constants
from src.params import WALK_DIAGRAM_LOGGING_FREQUENCY

from typing import Tuple, List, Dict
import numpy.typing as npt

import wandb


def setup_logging(verbose: bool, experiment_name: str) -> None:
    logs_folder = constants.SAVE_PATH_LOGS
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
    to_leader: float     = constants.SWITCH_DISTANCE_TO_LEADER
    to_exit: float       = constants.SWITCH_DISTANCE_TO_EXIT
    to_escape: float     = constants.SWITCH_DISTANCE_TO_ESCAPE
    to_pedestrian: float = constants.SWITCH_DISTANCE_TO_OTHER_PEDESTRIAN


def is_distance_low(
    pedestrians_positions: npt.NDArray, 
    destination: npt.NDArray, 
    radius: float
    ) -> npt.NDArray:
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

class Reward:
    def __init__(self, 
        is_new_exiting_reward: bool,
        is_new_followers_reward: bool,
        is_termination_agent_wall_collision: bool,
        init_reward_each_step: float
        ) -> None:
        
        self.init_reward_each_step = init_reward_each_step
        self.is_new_exiting_reward = is_new_exiting_reward
        self.is_new_followers_reward = is_new_followers_reward
        self.is_termination_agent_wall_collision = is_termination_agent_wall_collision

    def estimate_intrinsic_reward(self, pedestrians_positions, exit_position):
        """This is intrinsic reward, which is given to the agent at each step"""
        return (0 - sum_distance(pedestrians_positions, exit_position))

    def estimate_status_reward(self, old_statuses, new_statuses, timesteps, num_pedestrians):
        """This reward is based on how pedestrians update their status
        
        VISCEK or FOLLOWER --> EXITING  :  (15 +  5 * time_factor)
        VISCEK             --> FOLLOWER :  (10 + 10 * time_factor)
        """
        reward = self.init_reward_each_step
        time_factor = 1 - timesteps / (200 * num_pedestrians)

        # Reward for new exiting
        if self.is_new_exiting_reward:
            prev = np.logical_or(old_statuses == Status.VISCEK, 
                                 old_statuses == Status.FOLLOWER)
            curr = new_statuses == Status.EXITING
            n = sum(np.logical_and(prev, curr))
            reward += (15 + 10 * time_factor) * n

        # Reward for new followers
        if self.is_new_followers_reward:
            prev = old_statuses == Status.VISCEK
            curr = new_statuses == Status.FOLLOWER
            n = sum(np.logical_and(prev, curr))
            reward += (10 + 5 * time_factor) * n

        return reward


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


class Pedestrians:
    num : int                                   # number of pedestrians
    
    positions : np.ndarray                      # [num x 2], r_x and r_y
    directions : np.ndarray                     # [num x 2], v_x and v_y
    statuses : np.ndarray                       # [num], status : Status 
    memory : Dict[str, List[np.ndarray]]        # needed for animation drawing (positions and statuses)

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

    @property
    def status_stats(self):
        return {
            'escaped'  : sum(self.statuses == Status.ESCAPED),
            'exiting'  : sum(self.statuses == Status.EXITING),
            'following': sum(self.statuses == Status.FOLLOWER),
            'viscek'   : sum(self.statuses == Status.VISCEK),
        }

class Agent:
    start_position : np.ndarray                 # [2], r0_x and r0_y (same each reset)
    start_direction : np.ndarray                # [2], v0_x and v0_x (same each reset)
    position : np.ndarray                       # [2], r_x and r_y
    direction : np.ndarray                      # [2], v_x and v_y
    memory : Dict[str, List[np.ndarray]]        # needed for animation drawing (position)
    enslaving_degree: float                     # 0 < enslaving_degree <= 1

    def __init__(self, enslaving_degree):
        self.start_position = np.zeros(2, dtype=np.float32)
        self.start_direction = np.zeros(2, dtype=np.float32)
        self.enslaving_degree = enslaving_degree
        
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
    def __init__(self, 
        max_timesteps : int = constants.MAX_TIMESTEPS,
        n_episodes : int = constants.N_EPISODES,
        n_timesteps : int = constants.N_TIMESTEPS
        ) -> None:
        self.now = 0
        self.max_timesteps = max_timesteps
        self.n_episodes = n_episodes
        self.overall_timesteps = n_timesteps

    def reset(self):
        self.now = 0
        self.n_episodes += 1

    def step(self):
        self.now += 1
        self.overall_timesteps += 1
        return self.truncated()
        
    def truncated(self):
        return self.now >= self.max_timesteps 


def grad_potential_pedestrians(
        agent: Agent, pedestrians: Pedestrians, alpha: float = constants.ALPHA
    ) -> np.ndarray:
    R = agent.position[np.newaxis, :] - pedestrians.positions
    R = R[pedestrians.statuses == Status.VISCEK]

    if len(R) != 0:
        norm = np.linalg.norm(R, axis = 1)[:, np.newaxis] + constants.EPS
        grad = - alpha / norm ** (alpha + 2) * R
        grad = grad.sum(axis = 0)
    else:
        grad = np.zeros(2)
    return grad


def grad_potential_exit(
        agent: Agent, pedestrians: Pedestrians, exit: Exit, alpha: float = constants.ALPHA
    ) -> np.ndarray:
    R = agent.position - exit.position
    norm = np.linalg.norm(R) + constants.EPS
    grad = - alpha / norm ** (alpha + 2) * R
    grad *= sum(pedestrians.statuses == Status.FOLLOWER)
    return grad


class Area:
    def __init__(self, 
        reward: Reward,
        width = constants.WIDTH, 
        height = constants.HEIGHT,
        step_size = constants.STEP_SIZE,
        noise_coef = constants.NOISE_COEF,
        ):
        self.reward = reward
        self.width = width
        self.height = height
        self.step_size = step_size
        self.noise_coef = noise_coef
        self.exit = Exit()
    
    def reset(self):
        pass

    def pedestrians_step(self, pedestrians : Pedestrians, agent : Agent, now : int) -> Tuple[Pedestrians, bool, float, float]:

        # Check evacuated pedestrians & record new directions and positions of escaped pedestrians
        escaped = pedestrians.statuses == Status.ESCAPED
        pedestrians.directions[escaped] = 0
        pedestrians.positions[escaped] = self.exit.position
        
        # Check exiting pedestrians & record new directions for exiting pedestrians
        exiting = pedestrians.statuses == Status.EXITING
        if any(exiting):
            vec2exit = self.exit.position - pedestrians.positions[exiting]
            len2exit = np.linalg.norm(vec2exit, axis=1)
            vec_size = np.minimum(len2exit, self.step_size)
            vec2exit = (vec2exit.T / len2exit * vec_size).T
            pedestrians.directions[exiting] = vec2exit

        # Check following pedestrians & record new directions for following pedestrians
        following = pedestrians.statuses == Status.FOLLOWER

        # Check viscek pedestrians
        viscek = pedestrians.statuses == Status.VISCEK
        
        # Use all moving particles (efv -- exiting, following, viscek) to estimate the movement of viscek particles
        efv = reduce(np.logical_or, (exiting, following, viscek))
        efv_directions = pedestrians.directions[efv]
        efv_directions = (efv_directions.T / np.linalg.norm(efv_directions, axis=1)).T 
        
        # Find neighbours between following and viscek (fv) particles and all other moving particles
        fv = reduce(np.logical_or, (following, viscek))
        dm = distance_matrix(pedestrians.positions[fv],
                             pedestrians.positions[efv], 2)
        intersection = np.where(dm < SwitchDistances.to_pedestrian, 1, 0) 
        n_intersections = np.maximum(1, intersection.sum(axis=1))

        def estimate_mean_direction_among_neighbours(
                intersection,           # [f+v, f+v+e]  boolean matrix
                efv_directions,         # [f+v+e, 2]    vectors of directions of pedestrians
                n_intersections         # [f+v]         amount of neighbouring pedestrians
            ):
            """Viscek model"""
        
            # Estimate the contibution if each neighbouring particle 
            fv_directions_x = (intersection * efv_directions[:, 0]).sum(axis=1) / n_intersections
            fv_directions_y = (intersection * efv_directions[:, 1]).sum(axis=1) / n_intersections
            fv_theta = np.arctan2(fv_directions_y, fv_directions_x)
                                    
            # Create randomization noise to obtained directions
            # noise = np.random.normal(loc=0., scale=constants.NOISE_COEF, size=len(n_intersections))
            noise = np.random.uniform(low=-self.noise_coef/2, high=self.noise_coef/2, size=len(n_intersections))
            
            # New direction = estimated_direction + noise
            fv_theta = fv_theta + noise
            
            return np.vstack((np.cos(fv_theta), np.sin(fv_theta)))

        fv_directions = estimate_mean_direction_among_neighbours(
            intersection, efv_directions, n_intersections
        )            

        # Record new directions of following and viscek pedestrians
        pedestrians.directions[fv] = fv_directions.T * self.step_size
        
        # Add enslaving factor of leader's direction to following particles
        f_directions = pedestrians.directions[following]
        l_directions = agent.direction
        f_directions = agent.enslaving_degree * l_directions + (1. - agent.enslaving_degree) * f_directions
        pedestrians.directions[following] = f_directions
        
        # Record new positions of exiting, following and viscek pedestrians
        pedestrians.positions[efv] += pedestrians.directions[efv] 

        # Handling of wall collisions
        clipped = np.clip(pedestrians.positions, 
                    [-self.width, -self.height], [self.width, self.height])
        miss = pedestrians.positions - clipped
        pedestrians.positions -= 2 * miss
        pedestrians.directions *= np.where(miss!=0, -1, 1)

        # Estimate pedestrians statues, reward & update statuses
        old_statuses = pedestrians.statuses.copy()
        new_pedestrians_statuses = update_statuses(
            statuses=pedestrians.statuses,
            pedestrian_positions=pedestrians.positions,
            agent_position=agent.position,
            exit_position=self.exit.position
        )
        reward_pedestrians = self.reward.estimate_status_reward(
            old_statuses=old_statuses,
            new_statuses=new_pedestrians_statuses,
            timesteps=now,
            num_pedestrians=pedestrians.num
        )
        intrinsic_reward = self.reward.estimate_intrinsic_reward(
            pedestrians_positions=pedestrians.positions,
            exit_position=self.exit.position
        )
        pedestrians.statuses = new_pedestrians_statuses
        
        # Termination due to all pedestrians escaped
        if sum(pedestrians.statuses == Status.ESCAPED) == pedestrians.num:
            termination = True
        else: 
            termination = False

        return pedestrians, termination, reward_pedestrians, intrinsic_reward

    def agent_step(self, action : list, agent : Agent) -> Tuple[Agent, bool, float]:
        """
        Perform agent step:
            1. Read & preprocess action
            2. Check wall collision
            3. Return (updated agent, termination, reward)
        """
        action = np.array(action)
        action /= np.linalg.norm(action) + constants.EPS # np.clip(action, -1, 1, out=action)

        agent.direction = self.step_size * action
        
        if not self._if_wall_collision(agent):
            agent.position += agent.direction
            return agent, False, 0.
        else:
            return agent, self.reward.is_termination_agent_wall_collision, -5.

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
        experiment_name='test',
        number_of_pedestrians=constants.NUM_PEDESTRIANS,
        
        # leader params
        enslaving_degree=constants.ENSLAVING_DEGREE,
        
        # area params
        width=constants.WIDTH,
        height=constants.HEIGHT,
        step_size=constants.STEP_SIZE,
        noise_coef=constants.NOISE_COEF,
        
        # reward params
        is_termination_agent_wall_collision=constants.TERMINATION_AGENT_WALL_COLLISION,
        is_new_exiting_reward=constants.IS_NEW_EXITING_REWARD,
        is_new_followers_reward=constants.IS_NEW_FOLLOWERS_REWARD,
        intrinsic_reward_coef=constants.INTRINSIC_REWARD_COEF,
        init_reward_each_step=constants.INIT_REWARD_EACH_STEP, 
        
        # time params
        max_timesteps=constants.MAX_TIMESTEPS,
        n_episodes=constants.N_EPISODES,
        n_timesteps=constants.N_TIMESTEPS,
        
        # gravity embedding params
        enabled_gravity_embedding=constants.ENABLED_GRAVITY_EMBEDDING,
        alpha=constants.ALPHA,
        
        # logging params
        verbose=False,
        render_mode=None,
        draw=False
        
        ) -> None:
        super(EvacuationEnv, self).__init__()
        
        # setup env
        self.pedestrians = Pedestrians(num=number_of_pedestrians)
        
        reward = Reward(
            is_new_exiting_reward=is_new_exiting_reward,
            is_new_followers_reward=is_new_followers_reward,
            is_termination_agent_wall_collision=is_termination_agent_wall_collision,
            init_reward_each_step=init_reward_each_step)        
        
        self.area = Area(
            reward=reward, 
            width=width, height=height, 
            step_size=step_size, noise_coef=noise_coef)
        
        self.time = Time(
            max_timesteps=max_timesteps, 
            n_episodes=n_episodes, n_timesteps=n_timesteps)
        
        # setup agent
        self.agent = Agent(enslaving_degree=enslaving_degree)
        
        self.intrinsic_reward_coef = intrinsic_reward_coef
        self.episode_reward = 0
        self.episode_intrinsic_reward = 0
        self.episode_status_reward = 0        
        self.enabled_gravity_embedding = enabled_gravity_embedding
        self.alpha = alpha

        self.action_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.observation_space = self._get_observation_space()
        
        # logging
        self.render_mode = render_mode
        self.experiment_name = experiment_name
        setup_logging(verbose, experiment_name)

        # drawing
        self.draw = draw
        self.save_next_episode_anim = False
        log.info(f'Env {self.experiment_name} is initialized.')        
        
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
            observation_space['exit_position'] = \
                spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)

        return spaces.Dict(observation_space)

    def _get_observation(self):
        observation = {}
        observation['agent_position'] = self.agent.position
        
        if self.enabled_gravity_embedding:
            observation['grad_potential_pedestrians'] = grad_potential_pedestrians(
                agent=self.agent, 
                pedestrians=self.pedestrians, 
                alpha=self.alpha
            )
            observation['grad_potential_exit'] = grad_potential_exit(
                agent=self.agent,
                pedestrians=self.pedestrians,
                exit=self.area.exit,
                alpha=self.alpha
            )
        else:
            observation['pedestrians_positions'] = self.pedestrians.positions
            observation['exit_position'] = self.area.exit.position

        return observation

    def reset(self, seed=None, options=None):
        if self.save_next_episode_anim or (self.time.n_episodes + 1) % WALK_DIAGRAM_LOGGING_FREQUENCY == 0:
            self.draw = True
            self.save_next_episode_anim = True

        if self.time.n_episodes > 0:
            logging_dict = {
                "episode_intrinsic_reward" : self.episode_intrinsic_reward,
                "episode_status_reward" : self.episode_status_reward,
                "episode_reward" : self.episode_reward,
                "episode_length" : self.time.now,
                "escaped_pedestrians" : sum(self.pedestrians.statuses == Status.ESCAPED),
                "exiting_pedestrians" : sum(self.pedestrians.statuses == Status.EXITING),
                "following_pedestrians" : sum(self.pedestrians.statuses == Status.FOLLOWER),
                "viscek_pedestrians" : sum(self.pedestrians.statuses == Status.VISCEK),
                "overall_timesteps" : self.time.overall_timesteps
            }
            log.info('\t'.join([f'{key}={value}' for key, value in logging_dict.items()]))
            wandb.log(logging_dict)
        
        self.episode_reward = 0
        self.episode_intrinsic_reward = 0
        self.episode_status_reward = 0 
        self.time.reset()
        self.area.reset()
        self.agent.reset()
        self.pedestrians.reset(agent_position=self.agent.position,
                               exit_position=self.area.exit.position)
        self.pedestrians.save()
        log.info('Env is reseted.')
        return self._get_observation(), {}

    def step(self, action: list):
        # Increment time
        truncated = self.time.step()

        # Agent step
        self.agent, terminated_agent, reward_agent = self.area.agent_step(action, self.agent)
        
        # Pedestrians step
        self.pedestrians, terminated_pedestrians, reward_pedestrians, intrinsic_reward = \
            self.area.pedestrians_step(self.pedestrians, self.agent, self.time.now)
        
        # Save positions for rendering an animation
        if self.draw:
            self.pedestrians.save()
            self.agent.save()

        # Collect rewards
        reward = reward_agent + reward_pedestrians + self.intrinsic_reward_coef * intrinsic_reward
        
        # Record observation
        observation = self._get_observation()
        
        # Draw animation
        if (terminated_agent or terminated_pedestrians or truncated) and self.draw:
            self.save_animation()

        # Log reward
        self.episode_reward += reward
        self.episode_intrinsic_reward += intrinsic_reward
        self.episode_status_reward += reward_agent + reward_pedestrians
        return observation, reward, terminated_agent or terminated_pedestrians, truncated, {}

    def render(self):
        fig, ax = plt.subplots(figsize=(5, 5))

        exit_coordinates = (self.area.exit.position[0], self.area.exit.position[1])
        agent_coordinates = (self.agent.position[0], self.agent.position[1])

        # Draw exiting zone
        exiting_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_exit, 
            0, 180, alpha=0.2, color='green'
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

        from itertools import cycle
        colors = cycle([item['color'] for item in ax._get_lines._cycler_items])
        
        # Draw pedestrians
        for status in Status.all():
            selected_pedestrians = self.pedestrians.statuses == status
            color = next(colors)
            # color = next(ax._get_lines.prop_cycler)['color']
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
        if not os.path.exists(constants.SAVE_PATH_PNG): os.makedirs(constants.SAVE_PATH_PNG)
        filename = os.path.join(constants.SAVE_PATH_PNG, f'{self.experiment_name}_{self.time.now}.png')
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
            0, 180, alpha=0.2, color='green'
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
        
        from itertools import cycle
        colors = cycle([item['color'] for item in ax._get_lines._cycler_items])
        
        # Draw pedestrians
        pedestrian_position_plots = {}
        for status in Status.all():
            selected_pedestrians = self.pedestrians.memory['statuses'][0] == status
            color = next(colors)
            # color = next(ax._get_lines.prop_cycler)['color']
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
        
        if not os.path.exists(constants.SAVE_PATH_GIFF): os.makedirs(constants.SAVE_PATH_GIFF)
        filename = os.path.join(constants.SAVE_PATH_GIFF, f'{self.experiment_name}_ep-{self.time.n_episodes}.gif')
        ani.save(filename=filename, writer='pillow')
        log.info(f"Env is rendered and gif animation is saved to {filename}")

        if self.save_next_episode_anim:
            self.save_next_episode_anim = False
            self.draw = False

    def close(self):
        pass
    
    def seed(self, seed=None):
        from gym.utils.seeding import np_random
        return np_random(seed)
# # %%
# e = EvacuationEnv(number_of_pedestrians=100)

# e.reset()
# e.step([1, 0])

# for i in range(300):
#     e.step([np.sin(i*0.1), np.cos(i*0.1)])
# e.save_animation()
# e.render()
# # %%
