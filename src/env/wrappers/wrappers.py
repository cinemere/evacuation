import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict

from .. import EvacuationEnv
from ..env.env import Status

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
    