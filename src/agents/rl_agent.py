"""An agent that is rotating in the center of the simulation area"""
from . import BaseAgent
import numpy as np

class RLAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""
    def __init__(self, action_space, parameter=0.05, mode='training'):
        super(RLAgent, self).__init__(action_space)
        self.i = 0
        self.mode = mode

    def act(self, obs):
        self.i += 1
        step = self.i*self.parameter
        action = [np.sin(step), np.cos(step)]
        return action

    def update(self):
        pass