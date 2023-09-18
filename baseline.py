import os
import numpy as np
from stable_baselines3 import PPO
import wandb

from src.env import EvacuationEnv
from src.env import constants
from src import params
from src.utils import get_experiment_name, parse_args


def setup_env(args, experiment_name):
    env = EvacuationEnv(
        experiment_name = experiment_name,
        number_of_pedestrians = args.number_of_pedestrians,
        width=args.width,
        height=args.height,
        step_size=args.step_size,
        noise_coef=args.noise_coef,
        intrinsic_reward_coef=args.intrinsic_reward_coef,
        max_timesteps=args.max_timesteps,
        n_episodes=args.n_episodes,
        n_timesteps=args.n_timesteps,
        enabled_gravity_embedding=args.enabled_gravity_embedding,
        alpha=args.alpha,
        verbose=args.verbose,
        render_mode=None,
        draw=args.draw
    )
    return env

def setup_model(args, env):
    model = PPO(
        "MultiInputPolicy", 
        env, verbose=1, 
        tensorboard_log=params.SAVE_PATH_TBLOGS,
        device=args.device,
        learning_rate=args.learning_rate,
        gamma=args.gamma
    )
    return model

# def setup_wandb(args, experiment_name):
#     config_args = vars(args)
#     config_env = {key : value for key, value in constants.__dict__.items() if key[0] != '_'}
#     config_model = {key : value for key, value in params.__dict__.items() if key[0] != '_'}

#     wandb.init(
#         project="evacuation",
#         name=args.exp_name,
#         notes=experiment_name,
#         config=dict(config_args, **config_env, **config_model)
#     )

from src.agents import BaseAgent
from src.env.constants import SWITCH_DISTANCE_TO_LEADER

class BaselineAgent(BaseAgent):
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
            print('task_0 is done!')
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
        print(self.exit_position - pos)
        return self.exit_position - pos

    def act(self, obs):
        pos = obs['agent_position']
        print(pos, self.task_done)
        action = self.task_0(pos)
        return action
    
if __name__ == "__main__":
    args = parse_args()
    experiment_name = get_experiment_name(args)
    # setup_wandb(args, experiment_name)
    args.draw = True
    args.number_of_pedestrians = 60
    env = setup_env(args, experiment_name)

    # model = setup_model(args, env)
    # model.learn(args.learn_timesteps, tb_log_name=experiment_name)
    # model.save(os.path.join(params.SAVE_PATH_MODELS, f"{experiment_name}.zip"))

    from src.agents import RandomAgent

    # agent = BaselineAgent(env)
    agent = RandomAgent(env.action_space)

    obs, _ = env.reset()
    terminated, truncated = False, False

    # pos_x, pos_y = [], []
    # i = 0
    while not (terminated or truncated):
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # pos_x.append(obs['agent_position'][0])
        # pos_y.append(obs['agent_position'][1])
        # i+=1
        # if i > 2000:
        #     break
        # env.render()
        # collect_stats.append(env.pedestrians.status_stats)

    # import matplotlib.pyplot as plt
    # plt.plot(pos_x, pos_y)
    # plt.savefig('tmp.png')