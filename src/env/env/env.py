# ## %%
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging; log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib as mpl

from .. import constants
from .config import EnvConfig
from .statuses import Status
from .distances import SwitchDistances
from .pedestrians import Pedestrians
from .area import Area, Agent, Time
from .reward import Reward

import wandb    


def setup_logging(verbose: bool, experiment_name: str, logs_folder: str) -> None:
    if not os.path.exists(logs_folder): os.makedirs(logs_folder)
    logs_filename = os.path.join(logs_folder, f"logs_{experiment_name}.log")

    logging.basicConfig(
        filename=logs_filename, filemode="w",
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO
    )


class EvacuationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    """
    Evacuation Game Enviroment for Gym
    Continious Action and Observation Space.
    """
    def __init__(self, cfg: EnvConfig) -> None:
        super(EvacuationEnv, self).__init__()
        
        # setup env
        self.pedestrians = Pedestrians(num=cfg.number_of_pedestrians)
        
        reward = Reward(
            is_new_exiting_reward=cfg.is_new_exiting_reward,
            is_new_followers_reward=cfg.is_new_followers_reward,
            is_termination_agent_wall_collision=cfg.is_termination_agent_wall_collision,
            init_reward_each_step=cfg.init_reward_each_step)        
        
        self.area = Area(
            reward=reward, width=cfg.width, height=cfg.height, 
            step_size=cfg.step_size, noise_coef=cfg.noise_coef, eps=cfg.eps)
        
        self.time = Time(
            max_timesteps=cfg.max_timesteps, 
            n_episodes=cfg.n_episodes, n_timesteps=cfg.n_timesteps)
        
        # setup agent
        self.agent = Agent(enslaving_degree=cfg.enslaving_degree)
        
        self.intrinsic_reward_coef = cfg.intrinsic_reward_coef
        self.episode_reward = 0
        self.episode_intrinsic_reward = 0
        self.episode_status_reward = 0

        self.action_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.observation_space = self._get_observation_space()
        
        # logging
        self.render_mode = cfg.render_mode
        self.experiment_name = cfg.experiment_name
        setup_logging(cfg.verbose, self.experiment_name, cfg.path_logs)
        self.wandb_enabled = cfg.wandb_enabled

        # drawing
        self.path_giff = cfg.path_giff
        self.path_png = cfg.path_png
        self.draw = cfg.draw
        self.giff_freq = cfg.giff_freq
        self.save_next_episode_anim = False
        log.info(f'Env {self.experiment_name} is initialized.')
        
    def _get_observation_space(self):        
        observation_space = {
            'agent_position' : spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
        }
        
        observation_space['pedestrians_positions'] = \
            spaces.Box(low=-1, high=1, shape=(self.pedestrians.num, 2), dtype=np.float32)
        observation_space['exit_position'] = \
            spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)

        return spaces.Dict(observation_space)

    def _get_observation(self):
        observation = {}
        observation['agent_position'] = self.agent.position        
        observation['pedestrians_positions'] = self.pedestrians.positions
        observation['exit_position'] = self.area.exit.position

        return observation

    def reset(self, seed=None, options=None):
        # to seed self.np_random
        super().reset(seed=seed)
        
        if self.save_next_episode_anim or (self.time.n_episodes + 1) % self.giff_freq == 0:
            self.draw = True
            self.save_next_episode_anim = True

        if self.time.n_episodes > 0:
            logging_dict = {
                "episode_intrinsic_reward" : self.episode_intrinsic_reward,
                "episode_status_reward" : self.episode_status_reward,
                "episode_reward" : self.episode_reward,
                "episode_length" : self.time.now,
                "escaped_pedestrians" : sum(self.pedestrians.statuses == Status.ESCAPED),
                "exiting_pedestrians" : sum(self.pedestrians.statuses == Status.EXITING),
                "following_pedestrians" : sum(self.pedestrians.statuses == Status.FOLLOWER),
                "viscek_pedestrians" : sum(self.pedestrians.statuses == Status.VISCEK),
                "overall_timesteps" : self.time.overall_timesteps
            }
            log.info('\t'.join([f'{key}={value}' for key, value in logging_dict.items()]))
            if self.wandb_enabled: wandb.log(logging_dict)
        
        self.episode_reward = 0
        self.episode_intrinsic_reward = 0
        self.episode_status_reward = 0 
        self.time.reset()
        self.area.reset()
        self.agent.reset()
        self.pedestrians.reset(agent_position=self.agent.position,
                               exit_position=self.area.exit.position)
        self.pedestrians.save()
        log.info('Env is reseted.')
        return self._get_observation(), {}

    def step(self, action: list):
        # Increment time
        truncated = self.time.step()

        # Agent step
        self.agent, terminated_agent, reward_agent = self.area.agent_step(action, self.agent)
        
        # Pedestrians step
        self.pedestrians, terminated_pedestrians, reward_pedestrians, intrinsic_reward = \
            self.area.pedestrians_step(self.pedestrians, self.agent, self.time.now)
        
        # Save positions for rendering an animation
        if self.draw:
            self.pedestrians.save()
            self.agent.save()

        # Collect rewards
        reward = reward_agent + reward_pedestrians + self.intrinsic_reward_coef * intrinsic_reward
        
        # Record observation
        observation = self._get_observation()
        
        # Draw animation
        if (terminated_agent or terminated_pedestrians or truncated) and self.draw:
            self.save_animation()

        # Log reward
        self.episode_reward += reward
        self.episode_intrinsic_reward += intrinsic_reward
        self.episode_status_reward += reward_agent + reward_pedestrians
        return observation, reward, terminated_agent or terminated_pedestrians, truncated, {}

    def render(self):
        fig, ax = plt.subplots(figsize=(5, 5))

        exit_coordinates = (self.area.exit.position[0], self.area.exit.position[1])
        agent_coordinates = (self.agent.position[0], self.agent.position[1])

        # Draw exiting zone
        exiting_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_exit, 
            0, 180, alpha=0.2, color='green'
        )
        ax.add_patch(exiting_zone)

        # Draw escaping zone
        escaping_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_escape, 
            0, 180, color='white'
        )
        ax.add_patch(escaping_zone)
        
        # Draw exit
        ax.plot(exit_coordinates[0], exit_coordinates[1], marker='X', color='green')

        # Draw following zone
        following_zone = mpatches.Wedge(
            agent_coordinates, 
            SwitchDistances.to_leader, 
            0, 360, alpha=0.1, color='blue'
        )
        ax.add_patch(following_zone)

        from itertools import cycle
        colors = cycle([item['color'] for item in ax._get_lines._cycler_items])
        
        # Draw pedestrians
        for status in Status.all():
            selected_pedestrians = self.pedestrians.statuses == status
            color = next(colors)
            # color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(self.pedestrians.positions[selected_pedestrians, 0], 
                    self.pedestrians.positions[selected_pedestrians, 1],
                lw=0, marker='.', color=color)
            # for i in range(self.pedestrians.directions.shape[0]):
            #     ax.plot(self.pedestrians.positions[i],
            #     self.pedestrians.positions[i] + self.pedestrians.directions[i])

        # Draw agent
        ax.plot(agent_coordinates[0], agent_coordinates[1], marker='+', color='red')

        plt.xlim([ -1.1 * self.area.width, 1.1 * self.area.width])
        plt.ylim([ -1.1 * self.area.height, 1.1 * self.area.height])
        plt.xticks([]); plt.yticks([])
        plt.hlines([self.area.height, -self.area.height], 
            -self.area.width, self.area.width, linestyle='--', color='grey')
        plt.vlines([self.area.width, -self.area.width], 
            -self.area.height, self.area.height, linestyle='--', color='grey')

        plt.title(f"{self.experiment_name}. Timesteps: {self.time.now}")

        plt.tight_layout()
        if not os.path.exists(constants.SAVE_PATH_PNG): os.makedirs(constants.SAVE_PATH_PNG)
        filename = os.path.join(constants.SAVE_PATH_PNG, f'{self.experiment_name}_{self.time.now}.png')
        plt.savefig(filename)
        plt.show()
        log.info(f"Env is rendered and pnd image is saved to {filename}")

    def save_animation(self):
        
        fig, ax = plt.subplots(figsize=(5, 5))

        plt.title(f"{self.experiment_name}\nn_episodes = {self.time.n_episodes}")
        plt.hlines([self.area.height, -self.area.height], 
            -self.area.width, self.area.width, linestyle='--', color='grey')
        plt.vlines([self.area.width, -self.area.width], 
            -self.area.height, self.area.height, linestyle='--',  color='grey')
        plt.xlim([ -1.1 * self.area.width, 1.1 * self.area.width])
        plt.ylim([ -1.1 * self.area.height, 1.1 * self.area.height])
        plt.xticks([]); plt.yticks([])

        exit_coordinates = (self.area.exit.position[0], self.area.exit.position[1])
        agent_coordinates = (self.agent.memory['position'][0][0], self.agent.memory['position'][0][1])

        # Draw exiting zone
        exiting_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_exit, 
            0, 180, alpha=0.2, color='green'
        )
        ax.add_patch(exiting_zone)

        # Draw following zone
        following_zone = mpatches.Wedge(
            agent_coordinates, 
            SwitchDistances.to_leader, 
            0, 360, alpha=0.1, color='blue'
        )
        following_zone_plots = ax.add_patch(following_zone)

        # Draw escaping zone
        escaping_zone = mpatches.Wedge(
            exit_coordinates, 
            SwitchDistances.to_escape, 
            0, 180, color='white'
        )
        ax.add_patch(escaping_zone)
        
        # Draw exit
        ax.plot(exit_coordinates[0], exit_coordinates[1], marker='X', color='green')
        
        from itertools import cycle
        colors = cycle([item['color'] for item in ax._get_lines._cycler_items])
        
        # Draw pedestrians
        pedestrian_position_plots = {}
        for status in Status.all():
            selected_pedestrians = self.pedestrians.memory['statuses'][0] == status
            color = next(colors)
            # color = next(ax._get_lines.prop_cycler)['color']
            pedestrian_position_plots[status] = \
                ax.plot(self.pedestrians.memory['positions'][0][selected_pedestrians, 0], 
                self.pedestrians.memory['positions'][0][selected_pedestrians, 1],
                lw=0, marker='.', color=color)[0]

        # Draw agent
        agent_position_plot = ax.plot(agent_coordinates[0], agent_coordinates[1], marker='+', color='red')[0]

        def update(i):

            agent_coordinates = (self.agent.memory['position'][i][0], self.agent.memory['position'][i][1])
            following_zone_plots.set_center(agent_coordinates)

            for status in Status.all():
                selected_pedestrians = self.pedestrians.memory['statuses'][i] == status
                pedestrian_position_plots[status].set_xdata(self.pedestrians.memory['positions'][i][selected_pedestrians, 0])
                pedestrian_position_plots[status].set_ydata(self.pedestrians.memory['positions'][i][selected_pedestrians, 1])
                 
            # agent_position_plot.set_xdata(agent_coordinates[0])
            # agent_position_plot.set_ydata(agent_coordinates[1])
            agent_position_plot.set_data(agent_coordinates)

        ani = animation.FuncAnimation(fig=fig, func=update, frames=self.time.now, interval=20)
        
        if not os.path.exists(constants.SAVE_PATH_GIFF): os.makedirs(constants.SAVE_PATH_GIFF)
        filename = os.path.join(constants.SAVE_PATH_GIFF, f'{self.experiment_name}_ep-{self.time.n_episodes}.gif')
        ani.save(filename=filename, writer='pillow')
        log.info(f"Env is rendered and gif animation is saved to {filename}")

        if self.save_next_episode_anim:
            self.save_next_episode_anim = False
            self.draw = False

    def close(self):
        pass
    
    def seed(self, seed=None):
        from gymnasium.utils.seeding import np_random
        return np_random(seed)

# # %%
# e = EvacuationEnv(number_of_pedestrians=100)

# e.reset()
# e.step([1, 0])

# for i in range(300):
#     e.step([np.sin(i*0.1), np.cos(i*0.1)])
# e.save_animation()
# e.render()
# # %%
