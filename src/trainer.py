from src.agents import RLAgent
from src.env import EvacuationEnv
from src.params import *

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import os

def setup_tensorboard(experiment_name):
    if not os.path.exists(SAVE_PATH_TBLOGS): os.makedirs(SAVE_PATH_TBLOGS)
    writer = SummaryWriter(log_dir=os.path.join(SAVE_PATH_TBLOGS, experiment_name))
    return writer

class Trainer:
    def __init__(self, env: EvacuationEnv, agent: RLAgent, 
        experiment_name: str, 
        verbose: int = VERBOSE
        ) -> None:
    
        self.experiment_name = experiment_name    

        self.env = env
        self.agent = agent
        
        self.verbose = verbose
        self.writer = setup_tensorboard(experiment_name)
        
    def logger(self, episode_number: int) -> None:
        # logger in model_contin
        # log reward and save .pkl 
        # tensorboard
        pass
        
    def learn(self, number_of_episodes: int) -> None:
        for episode_number in range(number_of_episodes):

            obs, _ = self.env.reset()

            self.one_episode_loop(obs, episode_number)

            # tensorbord
            if self.verbose > 0: self.logger(episode_number)

            # optimization
            self.agent.update()
            
    def obs2tensor(self, obs):
        
        tensor = [value for value in obs.values()]
        return torch.from_numpy(np.concatenate(tensor)).float() #.to(torch.device(DEVICE))
    
    def tensor2action(self, tensor):
        return tensor #.detach()
            
    def one_episode_loop(self, obs, episode_number):
        for i in range(50_000):

            tensor_obs = self.obs2tensor(obs)
            tensor_action = self.agent.act(tensor_obs)
            action = self.tensor2action(tensor_action)
            obs, reward, terminated, truncated, _ = self.env.step(action)

            self.agent.remeber_reward(reward)

            if terminated or truncated:
                print('Episode: {}, Score: {}'.format(episode_number, i))
                break

        if (self.verbose == 1 and episode_number % WALK_DIAGRAM_LOGGING_FREQUENCY == 0) or \
            self.verbose == 2:
            self.env.save_next_episode_anim = True