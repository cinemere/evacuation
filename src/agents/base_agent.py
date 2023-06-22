"""This is the base abstraction for agents.
All agents should inherent from this class"""


class BaseAgent:
    """Parent abstract Agent."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        raise NotImplementedError()