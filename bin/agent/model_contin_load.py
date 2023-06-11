import matplotlib.pyplot as plt
import torch, torch.nn as nn
from time import time
# from IPython import display
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal
import os
import shutil

from collections import namedtuple

SavedAV = namedtuple('SavedAV', ['log_prob', 'value'])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Trainer:
    def __init__(self, env, network, experiment_name, writer=None,
                 learning_rate=1e-4, gamma=0.99, 
                 model_name=None, model_path=None, verbose=True):

        self.env = env
        self.network = network

        self.network = torch.load(model_path + model_name)
        self.network.eval()

        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.gamma = gamma
        self.verbose = verbose
        self.writer = writer

        self.eps = np.finfo(np.float32).eps.item()
        self.best_reward = 0
        self.experiment_name = experiment_name

        torch.manual_seed(73)

    def compute_returns(self):
        R = 0  # self.network.saved_av[-1].value
        returns = []
        for r in self.network.saved_r[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + self.eps)  # нормализовать advantade
        return returns

    def select_action(self, state):
        state = state.float()
        means, stds, value = self.network(state)

        distribution = Normal(means, stds)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        self.network.saved_av.append(SavedAV(log_prob, value))

        # print(action)

        return action #.item()

    def losses_and_backprop(self):

        policy_losses = []
        value_losses = []

        returns = self.compute_returns()

        #normalized advantage -- BEGIN
        # advantage = []
        # for (log_prob, value), current_return in zip(self.network.saved_av, returns):
        #     advantage.append(current_return - value.item())
        #     value_losses.append(F.smooth_l1_loss(value, torch.tensor([current_return])))  # may be mse here

        # advantage = torch.tensor(advantage)
        # advantage = (advantage - advantage.mean()) / (advantage.std() + self.eps)

        # for (log_prob, value), adv in zip(self.network.saved_av, advantage):
        #     policy_losses.append(-log_prob * adv)
        #normalized advantage -- END

        #non-normalized advantage -- BEGIN
        for (log_prob, value), current_return in zip(self.network.saved_av, returns):
            advantage = current_return - value.item()
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([current_return])))  # may be mse here
            policy_losses.append(-log_prob * advantage)
        #non-normalized advantage -- END

        policy_losses = torch.stack(policy_losses).mean()
        value_losses = torch.stack(value_losses).mean()
        loss = policy_losses + value_losses

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del self.network.saved_av[:]
        del self.network.saved_r[:]

    def one_episode_loop(self, state, episode_number):

        for i in range(50_000):

            state = torch.FloatTensor(state).to(device)

            action = self.select_action(state)

            # update state by taking an action
            # state, reward, done, _ = env.step(action.cpu().numpy())
            state, reward, done, _ = self.env.step(action)

            self.network.saved_r.append(reward)

            if episode_number % 50 == 0:
                path = '{}/plots/'.format(self.experiment_name)
                # env.render()
                # self.env.create_walk_diagram(mode='animation', path=path, n_episode=episode_number)
                self.env.create_walk_diagram(mode='one_image', path=path, n_episode=episode_number)

            if done:
                print('Episode: {}, Score: {}'.format(episode_number, i))

                if episode_number % 50 ==0:
                    path = '{}/plots/'.format(self.experiment_name)

                    if not os.path.exists(path):
                        os.makedirs(path)

                    self.env.record(mode='one_image', path=path, n_episode=episode_number)
                    # self.env.record(mode='animation', path=path, n_episode=episode_number)
                    # self.env.record(mode='animation', path=path, n_episode=episode_number)
                break

    def logger(self, reward_for_log, episode_number):

        if episode_number < 100:
            reward_for_log = 0.5 * self.network.saved_r[-1] + 0.5 * reward_for_log

        else:
            reward_for_log = 0.05 * self.network.saved_r[-1] + 0.95 * reward_for_log

            if self.best_reward < reward_for_log:
                path = '{}/model/'.format(self.experiment_name)

                if not os.path.exists(path):
                    os.makedirs(path)

                torch.save(self.network, '{}best_{}.pkl'.format(path, self.experiment_name))

        phase = 'mean'
        self.writer.add_scalar("reward/{}".format(phase),
                          reward_for_log, episode_number)
        phase = 'n_steps'
        self.writer.add_scalar("length/{}".format(phase),
                          len(self.network.saved_r), episode_number)

        return reward_for_log

    def learn(self, number_of_episodes):

        reward_for_log = 0

        for episode_number in range(number_of_episodes):

            state = self.env.reset()

            self.one_episode_loop(state, episode_number)

            # tensorbord
            if self.writer is not None:
                reward_for_log = self.logger(reward_for_log, episode_number)

                # optimization
            self.losses_and_backprop()