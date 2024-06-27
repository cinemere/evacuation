import numpy as np
from gymnasium import Env, ObservationWrapper
from gymnasium.spaces import Box, Dict

from . import EvacuationEnv, constants
from .env import Status

def grad_potential_pedestrians(
        agent_position: np.ndarray, 
        pedestrians_positions: np.ndarray,
        status_viscek: np.ndarray,
        alpha: float = constants.ALPHA, 
        eps: float = constants.EPS
    ) -> np.ndarray:
    R = agent_position[np.newaxis, :] - pedestrians_positions[status_viscek, :]
    # R = agent_position[np.newaxis, :] - pedestrians_positions
    # R = R[status_viscek]

    if len(R) != 0:
        norm = np.linalg.norm(R, axis = 1)[:, np.newaxis] + eps
        grad = - alpha / norm ** (alpha + 2) * R
        grad = grad.sum(axis = 0)
    else:
        grad = np.zeros(2)
    return grad


def grad_potential_exit(
        agent_position: np.ndarray, 
        num_followers: int, 
        exit_position: np.ndarray, 
        alpha: float = constants.ALPHA, 
        eps: float = constants.EPS
    ) -> np.ndarray:
    R = agent_position - exit_position
    norm = np.linalg.norm(R) + eps
    grad = - alpha / norm ** (alpha + 2) * R
    return grad * num_followers


class GravityEncoding(ObservationWrapper):
    def __init__(self, 
                 env: EvacuationEnv, 
                 alpha: float = constants.ALPHA, 
                 eps: float = None
        ) -> None:
        super().__init__(env)
        
        self.alpha = alpha
        self.eps = self.env.unwrapped.area.eps if isinstance(eps, type(None)) else eps
        
        self.observation_space = Dict({
            'agent_position' : self.observation_space['agent_position'],
            'grad_potential_pedestrians': Box(low=-1, high=1, shape=(2, ), dtype=np.float32),
            'grad_potential_exit': Box(low=-1, high=1, shape=(2, ), dtype=np.float32),
        })
        self.observation_space = Dict(self.observation_space)
        
    def observation(self, obs: Dict):
        pedestrians_positions = obs.pop("pedestrians_positions")
        exit_position = obs.pop("exit_position")
        agent_position = obs['agent_position']
        
        num_followers = sum(self.env.unwrapped.pedestrians.statuses == Status.FOLLOWER)
        status_viscek = self.env.unwrapped.pedestrians.statuses == Status.VISCEK
        
        obs['grad_potential_pedestrians'] = grad_potential_pedestrians(
            agent_position=agent_position,
            pedestrians_positions=pedestrians_positions,
            status_viscek=status_viscek,
            alpha=self.alpha,
            eps=self.eps,
        )
        obs['grad_potential_exit'] = grad_potential_exit(
            agent_position=agent_position,
            exit_position=exit_position,
            num_followers=num_followers,
            alpha=self.alpha,
            eps=self.eps,
        )
        return obs

class RelativePosition(ObservationWrapper):
    def __init__(self, env: EvacuationEnv):
        super().__init__(env)
        
        self._pedestrains_hippotenuse = np.sqrt(
            self.observation_space['pedestrians_positions'].low**2 + \
            self.observation_space['pedestrians_positions'].high**2)
        
        self._exit_hippotenuse = np.sqrt(
            self.observation_space['exit_position'].low**2 + \
            self.observation_space['exit_position'].high**2)

    def observation(self, obs: Dict) -> Dict:
        obs['pedestrians_positions'] = (
            obs['pedestrians_positions'] - obs['agent_position']
            ) / self._pedestrains_hippotenuse
        obs['exit_position'] = (
            obs['exit_position'] - obs['agent_position']
            ) / self._exit_hippotenuse 
        return obs

class PedestriansStatuses(ObservationWrapper):
    def __init__(self, env: EvacuationEnv, type: str = 'ohe'):
        super().__init__(env)
        self.type = type        
        if self.type == 'ohe':
            self.observation_space['pedestrians_statuses'] = \
                Box(low=0, high=1, shape=(env.unwrapped.pedestrians.num, len(Status)), 
                    dtype=np.float32)
        elif self.type == 'cat':
            self.observation_space['pedestrians_statuses'] = \
                Box(low=0, high=1, shape=(env.unwrapped.pedestrians.num, ), 
                    dtype=np.float32)
        else:
            raise ValueError(f"Invalid value of `type`='{self.type}'. Must be 'ohe' or 'cat'.")
    
    def observation(self, obs: Dict) -> Dict:
        statuses = self.env.unwrapped.pedestrians.statuses
        statuses = list(map(lambda x: x.value, statuses))
        statuses = len(Status) - np.array(statuses)
        if self.type == 'ohe':
            obs['pedestrians_statuses'] = np.zeros((self.env.unwrapped.pedestrians.num, len(Status)))
            for index, _status in enumerate(statuses):
                obs['pedestrians_statuses'][index][_status] = 1
        elif self.type == 'cat':
            obs['pedestrians_statuses'] = statuses / len(Status)
        else:
            raise ValueError(f"Invalid value of `type`='{self.type}'. Must be 'ohe' or 'cat'.")
        return obs
    
class MatrixObs(PedestriansStatuses):
    def __init__(self, env: EvacuationEnv, type: str = 'no'):
        super().__init__(env, type=type)
        if self.type == 'ohe':
            self.observation_space = \
                Box(low=-1, high=1, shape=(env.unwrapped.pedestrians.num + 2, 2 + len(Status)), 
                    dtype=np.float32)
        elif self.type == 'cat':
            self.observation_space = \
                Box(low=-1, high=1, shape=(env.unwrapped.pedestrians.num + 2, 3), 
                    dtype=np.float32)
        elif self.type == 'no':
            self.observation_space = \
                Box(low=-1, high=1, shape=(env.unwrapped.pedestrians.num + 2, 2), 
                    dtype=np.float32)
        else:
            raise ValueError(f"Invalid value of `type`='{self.type}'. Must be 'no', 'ohe' or 'cat'.")
    
    def observation(self, obs: Dict) -> Box:
        obs = super().observation(obs)
        pos = np.vstack((obs['agent_position'], 
                         obs['exit_position'], 
                         obs['pedestrians_positions']))
        if self.type == 'ohe':
            stat_agent = np.array([0, 0, 0, 0], dtype=np.float32)
            stat_exit = np.array([1, 0, 0, 0], dtype=np.float32)
            stat = np.vstack((stat_agent, 
                            stat_exit, 
                            obs['pedestrians_statuses']))
            vec = np.hstack((pos, stat)).astype(np.float32)
        elif self.type == 'cat':        
            stat = np.hstack(([0, 1], obs['pedestrians_statuses']))
            vec = np.hstack((pos, stat[:,np.newaxis])).astype(np.float32)
        elif self.type == 'no':
            vec = pos
        else:
            raise ValueError(f"Invalid value of `type`='{self.type}'. Must be 'no', 'ohe' or 'cat'.")
        return vec
    