import numpy as np
from gymnasium import ObservationWrapper

class RelativePosition(ObservationWrapper):
    def __init__(self, env):
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
