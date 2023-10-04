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
        experiment_name=experiment_name,
        number_of_pedestrians=args.number_of_pedestrians,
        enslaving_degree=args.enslaving_degree, 
        width=args.width,
        height=args.height,
        step_size=args.step_size,
        noise_coef=args.noise_coef,
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
            "MultiInputPolicy", 
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
    config_env = {key : value for key, value in constants.__dict__.items() if key[0] != '_'}
    config_model = {key : value for key, value in params.__dict__.items() if key[0] != '_'}

    wandb.init(
        project="evacuation",
        name=args.exp_name,
        notes=experiment_name,
        config=dict(config_args, **config_env, **config_model)
    )

if __name__ == "__main__":
    args = parse_args()

    experiment_name = get_experiment_name(args)

    setup_wandb(args, experiment_name)
    env = setup_env(args, experiment_name)

    model = setup_model(args, env)
    model.learn(args.learn_timesteps, tb_log_name=experiment_name)
    model.save(os.path.join(params.SAVE_PATH_MODELS, f"{experiment_name}.zip"))
