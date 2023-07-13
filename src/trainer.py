from src.agents import RLAgent
from src.env import EvacuationEnv
from src.params import *

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import os
from collections import deque

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

class Trainer:
    def __init__(self, env: EvacuationEnv, agent: RLAgent, 
        experiment_name: str, 
        verbose: int = VERBOSE,
        buffer_maxlen: int = LOGGER_BUFFER_MAXLEN
        ) -> None:
    
        self.experiment_name = experiment_name    

        self.env = env
        self.agent = agent
        
        self.verbose = verbose
        self.writer = None        
        self.buffer = None
        self.max_reward = -np.inf
        
        if verbose > 0: self.setup_logger(maxlen=buffer_maxlen)
        
    def setup_logger(self, maxlen):
        self.setup_tensorboard()
        self.setup_buffer(maxlen)

    def setup_tensorboard(self):
        if not os.path.exists(SAVE_PATH_TBLOGS): 
            os.makedirs(SAVE_PATH_TBLOGS)
        
        self.writer = SummaryWriter(
            log_dir=os.path.join(SAVE_PATH_TBLOGS, self.experiment_name)
        )
        
    def setup_buffer(self, maxlen) -> None:
        self.buffer = {
            'reward/last_reward' : deque(maxlen=maxlen),
            'reward/mean_reward' : deque(maxlen=maxlen),
            'episode_length/steps' : deque(maxlen=maxlen)
        }
        
    def update_buffer(self) -> float:
        
        last_reward = self.agent.memory_rewards[-1]
        mean_reward = np.mean(self.agent.memory_rewards)
        episode_length = len(self.agent.memory_rewards)
  
        self.buffer['reward/last_reward'].append(last_reward)
        self.buffer['reward/mean_reward'].append(mean_reward)
        self.buffer['episode_length/steps'].append(episode_length)
        return mean_reward
    
    def save_network(self) -> None:
        if not os.path.exists(SAVE_PATH_MODELS):
            os.makedirs(SAVE_PATH_MODELS)

        path = os.path.join(SAVE_PATH_MODELS, 
                            f'bestmodel_{self.experiment_name}.pkl')
        torch.save(self.agent.network, path)
        
    def logger(self, episode_number: int) -> None:
        mean_reward = self.update_buffer()

        for key, buffer in self.buffer.items(): 
            self.writer.add_scalar(
                key, np.mean(buffer), episode_number
            )
        
        if episode_number > 100 and mean_reward > self.max_reward:
            self.max_reward = mean_reward
            self.save_network()        
        
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
        return torch.from_numpy(np.concatenate(tensor)).float().to(device)
    
    def tensor2action(self, tensor):
        return tensor.cpu()
            
    def one_episode_loop(self, obs, episode_number):
        for nstep in range(50_000):

            tensor_obs = self.obs2tensor(obs)
            tensor_action = self.agent.act(tensor_obs)
            action = self.tensor2action(tensor_action)
            obs, reward, terminated, truncated, _ = self.env.step(action)

            self.agent.remeber_reward(reward)

            if terminated or truncated:
                print(f'Episode: {episode_number}, Reward: {reward}, Length: {nstep}')
                break

        if (self.verbose == 1 and episode_number % WALK_DIAGRAM_LOGGING_FREQUENCY == 0) or \
            self.verbose == 2:
            self.env.save_next_episode_anim = True