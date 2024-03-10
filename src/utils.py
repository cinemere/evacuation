from gymnasium import spaces
import numpy as np
import argparse

from src.env import constants
from src.params import *

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(inline_mode=False, request=""):

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    model_params = parser.add_argument_group('model')
    model_params.add_argument('--origin', type=str, default='ppo',
        choices=['ppo', 'a2c', 'sac'], help="which model to use")
    model_params.add_argument('--learn-timesteps', type=int, default=5_000_000,
        help='number of timesteps to learn the model')
    model_params.add_argument('--learning-rate', type=float, default=0.0003,
        help='learning rate for stable baselines ppo model')
    model_params.add_argument('--gamma', type=float, default=0.99,
        help='gammma for stable baselines ppo model')
    model_params.add_argument('--device', type=str, default=DEVICE,
        choices=['cpu', 'cuda'], help='device for the model')

    experiment_params = parser.add_argument_group('experiment')
    experiment_params.add_argument('--exp-name', type=str, default='test',
        help='prefix of the experiment name for logging results')
    experiment_params.add_argument('-v', '--verbose', action='store_true',
        help='debug mode of logging')
    experiment_params.add_argument('--draw', action='store_true',
        help='save animation at each step')

    env_params = parser.add_argument_group('env')
    env_params.add_argument('-n', '--number-of-pedestrians', type=int, default=constants.NUM_PEDESTRIANS,
        help='number of pedestrians in the simulation')
    env_params.add_argument('--width', type=float, default=constants.WIDTH,
        help='geometry of environment space: width')
    env_params.add_argument('--height', type=float, default=constants.HEIGHT,
        help='geometry of environment space: height')
    env_params.add_argument('--step-size', type=float, default=constants.STEP_SIZE,
        help='length of pedestrian\'s and agent\'s step')
    env_params.add_argument('--noise-coef', type=float, default=constants.NOISE_COEF,
        help='noise coefficient of randomization in viscek model')
    env_params.add_argument('--num-obs-stacks', type=int, default=constants.NUM_OBS_STACKS,
        help="number of times to stack observation")
    env_params.add_argument('-rel', '--use-relative-positions', type=str2bool, default=constants.USE_RELATIVE_POSITIONS,
        help="add relative positions wrapper (can be use only WITHOUT gravity embedding)")
    
    leader_params = parser.add_argument_group('leader params')
    leader_params.add_argument('--enslaving-degree', type=float, default=constants.ENSLAVING_DEGREE,
        help='enslaving degree of leader in generalized viscek model')
    
    reward_params = parser.add_argument_group('reward params')
    reward_params.add_argument('--is-new-exiting-reward', type=str2bool, default=constants.IS_NEW_EXITING_REWARD,
        help="if True, positive reward will be given for each pedestrian, "\
             "entering the exiting zone")
    reward_params.add_argument('--is-new-followers-reward', type=str2bool, default=constants.IS_NEW_FOLLOWERS_REWARD,
        help="if True, positive reward will be given for each pedestrian, "\
             "entering the leader\'s zone of influence")
    reward_params.add_argument('--intrinsic-reward-coef', type=float, default=constants.INTRINSIC_REWARD_COEF,
        help='coefficient in front of intrinsic reward')
    reward_params.add_argument('--is-termination-agent-wall-collision', type=str2bool, default=constants.TERMINATION_AGENT_WALL_COLLISION,
        help='if True, agent\'s wall collision will terminate episode')
    reward_params.add_argument('--init-reward-each-step', type=float, default=constants.INIT_REWARD_EACH_STEP,
        help='constant reward given on each step of agent')
    
    time_params = parser.add_argument_group('time')
    time_params.add_argument('--max-timesteps', type=int, default=constants.MAX_TIMESTEPS,
        help = 'max timesteps before truncation')
    time_params.add_argument('--n-episodes', type=int, default=constants.N_EPISODES,
        help = 'number of episodes already done (for pretrained models)')
    time_params.add_argument('--n-timesteps', type=int, default=constants.N_TIMESTEPS,
        help = 'number of timesteps already done (for pretrained models)')

    gravity_embedding_params = parser.add_argument_group('gravity embedding params')    
    gravity_embedding_params.add_argument('-e', '--enabled-gravity-embedding', type=str2bool,
        default=constants.ENABLED_GRAVITY_EMBEDDING,
        help='if True use gravity embedding')
    gravity_embedding_params.add_argument('--alpha', type=float, default=constants.ALPHA,
        help='alpha parameter of gravity gradient embedding')
    
    if inline_mode:
        args = parser.parse_args(request)
    else:
        args = parser.parse_args()
        
    if args.enabled_gravity_embedding:
        assert not args.use_relative_positions, \
            "Relative positions wrapper can NOT be used while enabled gravity embedding"
    
    return args

def get_experiment_name(args):
    prefix = args.exp_name
    params = [
        f"n-{args.number_of_pedestrians}",
        f"lr-{args.learning_rate}",
        f"gamma-{args.gamma}",
        f"s-{f'gra_a-{args.alpha}' if args.enabled_gravity_embedding else 'rel' if args.use_relative_positions else 'ped'}",
        f"ss-{args.step_size}",
        f"vr-{constants.SWITCH_DISTANCE_TO_OTHER_PEDESTRIAN}"
    ]
    return f"{prefix}_{'_'.join(params)}_{NOW}"

def get_act_size(act_space: spaces.Box):
    return act_space.shape[0]

def get_obs_size(obs_space: spaces.Dict):
    return np.sum([v.shape[0] for v in obs_space.values()])    
