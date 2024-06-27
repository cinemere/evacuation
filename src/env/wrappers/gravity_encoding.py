import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict

from .. import EvacuationEnv, constants
from ..env.env import Status

def grad_potential_pedestrians(
        agent_position: np.ndarray, 
        pedestrians_positions: np.ndarray,
        status_viscek: np.ndarray,
        alpha: float, 
        eps: float
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
        alpha: float, 
        eps: float
    ) -> np.ndarray:
    R = agent_position - exit_position
    norm = np.linalg.norm(R) + eps
    grad = - alpha / norm ** (alpha + 2) * R
    return grad * num_followers


class GravityEncoding(ObservationWrapper):
    def __init__(self, 
                 env: EvacuationEnv, 
                 alpha: float, 
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