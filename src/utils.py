from src.params import *
from gymnasium import spaces
import numpy as np

def get_experiment_name(prefix):
    # # parameters to vary:
    # learning_rate = 1e-4
    # gamma = 0.99
    # number_of_pedestrians = 60  # 2, 3, 5, 10, 50, 100
    # case = 'continious'  # 'continious', 'grid'
    # obs_type = 's_vigrad'  # 'vigrad', 'coord', 'vigrad'
    # norm = 'no'  # 'no', 'adv', 'ret'/home/skoguest/aklepach/pr_evac/load_continious_viscekobs_nonorm_N60_lr1e-06_gamma0.99_vr0.05_a4--18-05-2022-19-26-38
    # vision_radius = 0.1  # 0.2
    # alpha = 2  # 2
    # step_size = 0.01  # 0.01
    
    params = [
        f"lr-{LEARNING_RATE}",
        f"n-{NUMBER_OF_PEDESTRIANS}",
        f"gamma-{GAMMA}",
        f"s-{'gra' if ENABLE_GRAVITY_EMBEDDING else 'ped'}"
    ]
    return f"{prefix}_{NOW}_{'_'.join(params)}"

def get_act_size(act_space: spaces.Box):
    return act_space.shape[0]

def get_obs_size(obs_space: spaces.Dict):
    return np.sum([v.shape[0] for v in obs_space.values()])    
