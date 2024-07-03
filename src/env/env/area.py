import numpy as np
from typing import Tuple, List, Dict
from scipy.spatial import distance_matrix
from functools import reduce

from .pedestrians import Pedestrians
from .reward import Reward
from .statuses import Status, update_statuses
from .distances import SwitchDistances


class Agent:
    """Dummy class to initialize leader and save its trajectory in memory"""
    start_position : np.ndarray                 # [2], r0_x and r0_y (same each reset)
    start_direction : np.ndarray                # [2], v0_x and v0_x (same each reset)
    position : np.ndarray                       # [2], r_x and r_y
    direction : np.ndarray                      # [2], v_x and v_y
    memory : Dict[str, List[np.ndarray]]        # needed for animation drawing (position)
    enslaving_degree: float                     # 0 < enslaving_degree <= 1

    def __init__(self, enslaving_degree):
        self.start_position = np.zeros(2, dtype=np.float32)
        # self.start_position = np.random.rand(2).astype(np.float32)
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
    def __init__(self, max_timesteps: int, n_episodes: int, n_timesteps: int) -> None:
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
    

class Area:
    def __init__(self, reward: Reward, width: float,  height: float,
                 step_size: float, noise_coef: float, eps: float):
        self.reward = reward
        self.width = width
        self.height = height
        self.step_size = step_size
        self.noise_coef = noise_coef
        self.eps = eps
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
        action /= np.linalg.norm(action) + self.eps # np.clip(action, -1, 1, out=action)

        agent.direction = self.step_size * action
        
        if not self._if_wall_collision(agent):
            agent.position += agent.direction
            return agent, False, 0.
        else:
            return agent, self.reward.is_termination_agent_wall_collision, -5.

    def _if_wall_collision(self, agent: Agent):
        pt = agent.position + agent.direction

        left  = pt[0] < -self.width
        right = pt[0] > self.width
        down  = pt[1] < -self.height  
        up    = pt[1] > self.height
        
        if left or right or down or up:
            return True
        return False