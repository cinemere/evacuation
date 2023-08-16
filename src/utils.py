from gymnasium import spaces
import numpy as np
import argparse

from src.env import constants
from src.params import *


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    model = parser.add_argument_group('model')
    model.add_argument('--origin', type=str, default='ppo',
        choices=['ppo', 'a2c'], help="which model to use")
    model.add_argument('--learn-timesteps', type=int, default=5_000_000,
        help='number of timesteps to learn the model')
    model.add_argument('--learning-rate', type=float, default=0.0003,
        help='learning rate for stable baselines ppo model')
    model.add_argument('--gamma', type=float, default=0.99,
        help='gammma for stable baselines ppo model')
    model.add_argument('--device', type=str, default=DEVICE,
        choices=['cpu', 'cuda'], help='device for the model')

    experiment = parser.add_argument_group('experiment')
    experiment.add_argument('--exp-name', type=str, default='test',
        help='prefix of the experiment name for logging results')
    experiment.add_argument('-v', '--verbose', action='store_true',
        help='debug mode of logging')
    experiment.add_argument('--draw', action='store_true',
        help='save animation at each step')

    env = parser.add_argument_group('env')
    env.add_argument('-n', '--number-of-pedestrians', type=int, default=constants.NUM_PEDESTRIANS,
        help='number of pedestrians in the simulation')
    env.add_argument('--width', type=float, default=constants.WIDTH,
        help='geometry of environment space: width')
    env.add_argument('--height', type=float, default=constants.HEIGHT,
        help='geometry of environment space: height')
    env.add_argument('--step-size', type=float, default=constants.STEP_SIZE,
        help='length of pedestrian\'s and agent\'s step')
    env.add_argument('--noise-coef', type=float, default=constants.NOISE_COEF,
        help='noise coefficient of randomization in viscek model')
    env.add_argument('--intrinsic-reward-coef', type=float, default=constants.INTRINSIC_REWARD_COEF,
        help='coefficient in front of intrinsic reward')
    
    time = parser.add_argument_group('time')
    time.add_argument('--max-timesteps', type=int, default=constants.MAX_TIMESTEPS,
        help = 'max timesteps before truncation')
    time.add_argument('--n-episodes', type=int, default=constants.N_EPISODES,
        help = 'number of episodes already done (for pretrained models)')
    time.add_argument('--n-timesteps', type=int, default=constants.N_TIMESTEPS,
        help = 'number of timesteps already done (for pretrained models)')
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    gravity_embedding_params = parser.add_argument_group('gravity embedding params')    
    gravity_embedding_params.add_argument('-e', '--enabled-gravity-embedding', type=str2bool,
        default=constants.ENABLED_GRAVITY_EMBEDDING,
        help='if True use gravity embedding')
    gravity_embedding_params.add_argument('--alpha', type=float, default=constants.ALPHA,
        help='alpha parameter of gravity gradient embedding')
    
    args = parser.parse_args()
    return args

def get_experiment_name(args):
    prefix = args.exp_name
    params = [
        f"n-{args.number_of_pedestrians}",
        f"lr-{args.learning_rate}",
        f"gamma-{args.gamma}",
        f"s-{f'gra_a-{args.alpha}' if args.enabled_gravity_embedding else 'ped'}",
        f"ss-{args.step_size}",
        f"vr-{constants.SWITCH_DISTANCE_TO_OTHER_PEDESTRIAN}"
    ]
    return f"{prefix}_{'_'.join(params)}_{NOW}"

def get_act_size(act_space: spaces.Box):
    return act_space.shape[0]

def get_obs_size(obs_space: spaces.Dict):
    return np.sum([v.shape[0] for v in obs_space.values()])    
