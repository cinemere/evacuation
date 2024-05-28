import numpy as np
from gymnasium import Env, ObservationWrapper
from gymnasium.spaces import Box

from . import EvacuationEnv
from .env import Status

class RelativePosition(ObservationWrapper):
    def __init__(self, env: EvacuationEnv):
        super().__init__(env)
        
        self._pedestrains_hippotenuse = np.sqrt(
            self.observation_space['pedestrians_positions'].low**2 + \
            self.observation_space['pedestrians_positions'].high**2)
        
        self._exit_hippotenuse = np.sqrt(
            self.observation_space['exit_position'].low**2 + \
            self.observation_space['exit_position'].high**2)

    def observation(self, obs):
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
                Box(low=0, high=1, shape=(env.pedestrians.num, len(Status)), 
                    dtype=np.float32)
        else:
            self.observation_space['pedestrians_statuses'] = \
                Box(low=0, high=1, shape=(env.pedestrians.num, ), 
                    dtype=np.float32)
    
    def observation(self, obs):
        statuses = self.env.pedestrians.statuses
        statuses = list(map(lambda x: x.value, statuses))
        statuses = len(Status) - np.array(statuses)
        if self.type == 'ohe':
            obs['pedestrians_statuses'] = np.zeros((self.env.pedestrians.num, len(Status)))
            for index, _status in enumerate(statuses):
                obs['pedestrians_statuses'][index][_status] = 1
        else:
            obs['pedestrians_statuses'] = statuses / len(Status)
        return obs
    
class MatrixObs(PedestriansStatuses):
    def __init__(self, env: Env, type: str = 'ohe'):
        super().__init__(env, type=type)
        if self.type == 'ohe':
            self.observation_space = \
                Box(low=-1, high=1, shape=(env.pedestrians.num + 2, 2 + len(Status)), 
                    dtype=np.float32)
        else:
            self.observation_space = \
                Box(low=-1, high=1, shape=(env.pedestrians.num + 2, 3), 
                    dtype=np.float32)
    
    def observation(self, obs):
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
        else:        
            stat = np.hstack(([0, 1], obs['pedestrians_statuses']))
            vec = np.hstack((pos, stat[:,np.newaxis])).astype(np.float32)
        return vec
    