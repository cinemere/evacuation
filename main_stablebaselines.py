from stable_baselines3 import PPO

from src.env import EvacuationEnv
import numpy as np

from src.utils import *
from src.params import *

experiment_prefix = 'stablebaselines_nowallcollisions_withintrinsicreward'
experiment_name = get_experiment_name(experiment_prefix)

env = EvacuationEnv(
    number_of_pedestrians = NUMBER_OF_PEDESTRIANS,
    experiment_name=experiment_name,
    verbose=VERBOSE_GYM,
    draw=DRAW,
    enable_gravity_embedding=ENABLE_GRAVITY_EMBEDDING
    )

from stable_baselines3 import PPO

model = PPO(
    "MultiInputPolicy", 
    env, verbose=1, 
    tensorboard_log="/home/cinemere/work/evacuation/saved_data/tb-logs/",
    device='cpu'
    )

import wandb
from src.env import constants
config = {key : value for key, value in constants.__dict__.items() if key[0] != '_'}

wandb.init(
    project="evacuation",
    notes=experiment_name,
    config=config
)

model.learn(5_000_000)