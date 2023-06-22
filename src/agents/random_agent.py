"""An agent that preforms a random action each step"""
from . import BaseAgent


class RandomAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs):
        return self.action_space.sample()