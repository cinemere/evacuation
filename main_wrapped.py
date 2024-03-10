# %%
import os
import numpy as np
from stable_baselines3 import PPO, SAC
import wandb

from src.env import EvacuationEnv
from src.env import constants
from src import params
from src.utils import get_experiment_name, parse_args

def setup_env(args, experiment_name):
    env = EvacuationEnv(
        experiment_name=experiment_name,
        number_of_pedestrians=args.number_of_pedestrians,
        enslaving_degree=args.enslaving_degree, 
        width=args.width,
        height=args.height,
        step_size=args.step_size,
        noise_coef=args.noise_coef,
        is_termination_agent_wall_collision=args.is_termination_agent_wall_collision,
        is_new_exiting_reward=args.is_new_exiting_reward,
        is_new_followers_reward=args.is_new_followers_reward,
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
    
    if args.origin == 'ppo':
        model = PPO(
            "MlpPolicy", 
            env, verbose=1, 
            tensorboard_log=params.SAVE_PATH_TBLOGS,
            device=args.device,
            learning_rate=args.learning_rate,
            gamma=args.gamma
        )
    elif args.origin == 'sac':
        model = SAC(
            "MlpPolicy",
            env, verbose=1,
            tensorboard_log=params.SAVE_PATH_TBLOGS,
            device=args.device,
            learning_rate=args.learning_rate,
            gamma=args.gamma
        )
    else:
        raise NotImplementedError
    return model

def setup_wandb(args, experiment_name):
    config_args = vars(args)
    # config_env = {key : value for key, value in constants.__dict__.items() if key[0] != '_'}
    # config_model = {key : value for key, value in params.__dict__.items() if key[0] != '_'}
    # save_config = dict(config_args, **config_env, **config_model)
    from src.env.env import SwitchDistances as sd
    config_switch_distances = {k : vars(sd)[k] for k in sd.__annotations__.keys()}
    save_config = dict(config_args, **config_switch_distances)

    wandb.init(
        project="evacuation",
        name=args.exp_name,
        notes=experiment_name,
        config=save_config
    )
# %%
# if __name__ == "__main__":
args = parse_args(True, [
    "--exp-name", "test-distance-wrapper",
    # "-e", "true",
    "-e", "false",
    "--intrinsic-reward-coef", "0",
    ])
# %%
vars(args)
# %%
experiment_name = get_experiment_name(args)
experiment_name
# %%

setup_wandb(args, experiment_name)
# %%
env = setup_env(args, experiment_name)
env, env.observation_space
# %%
# %%
from src.env import RelativePosition
# %%
env = RelativePosition(env)
# %%
env.reset()
# %%
for i in range(10):
    print(env.step(np.array([1., 1.]))[0])

# %%
A = 1 if False else 2 if False else 3
A
# %%
