import numpy as np

from src.agents import BaseAgent
from src.env.constants import SWITCH_DISTANCE_TO_LEADER
from src.env import EvacuationEnv

class WacuumCleaner(BaseAgent):
    def __init__(self, env : EvacuationEnv):
        self.up = np.array([0., 1.], dtype=np.float32)
        self.right = np.array([1., 0.], dtype=np.float32) 
        self.left = np.array([-1., 0.], dtype=np.float32) 
        self.down = np.array([0., -1.], dtype=np.float32)

        self.exit_position = env.area.exit.position
        self.step_size = env.area.step_size
        self.current_strategy = 0
        self.strategy_condition = {
            'go_up' :
            lambda pos : pos[1] < env.area.height - SWITCH_DISTANCE_TO_LEADER/2 + env.area.step_size,
            
            'go_right' :
            lambda pos : pos[0] < env.area.width - SWITCH_DISTANCE_TO_LEADER/2 + env.area.step_size,

            'go_left' :
            lambda pos : pos[0] > -env.area.width + SWITCH_DISTANCE_TO_LEADER/2 - env.area.step_size,

            'go_down' :
            lambda pos : pos[1] > -env.area.height + SWITCH_DISTANCE_TO_LEADER/2 - env.area.step_size
        }
        self.task_done = [False, False, False]
        self.task_1_direction = self.right
        self.task_1_condition = 'go_right'
        self.task_1_time_to_go_down = 0

    def task_0(self, pos):
        # reach up
        if self.task_done[0]:
            return self.task_1(pos)

        if self.strategy_condition['go_up'](pos):
            return self.up
        else:
            self.task_done[0] = True
            return self.right

    def task_1(self, pos):
        # go right and left
        if self.task_done[1]:
            return self.task_2(pos)
        
        if self.task_1_time_to_go_down > 0:
            self.task_1_time_to_go_down -= 1

            if self.strategy_condition['go_down'](pos):
                return self.down
            else:
                self.task_done[1] = True
                return self.task_2(pos)

        if self.strategy_condition[self.task_1_condition](pos):            
            return self.task_1_direction
        else:
            self.switch_task_1_direction()
            self.task_1_time_to_go_down = 25
            return self.down
    
    def switch_task_1_direction(self):
        if (self.task_1_direction == self.right).all():
            self.task_1_direction = self.left
            self.task_1_condition = 'go_left'
        else:
            self.task_1_direction = self.right
            self.task_1_condition = 'go_right'

    def task_2(self, pos):
        return self.exit_position - pos

    def act(self, obs):
        pos = obs['agent_position']
        action = self.task_0(pos)
        return action