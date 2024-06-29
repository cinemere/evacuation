# Imports
import os
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.env import EvacuationEnv
from src.env import constants
from src import params
from main import setup_env
from src.utils import parse_args
from src.plotting.paths2models import *

import wandb
wandb.init(mode="disabled")

# Define params
path_save = 'saved_data/plot_efficiency'
os.makedirs(path_save, exist_ok=True)

DEVICE = 'cpu'
N_REPEATS = 5_000

MODE = "FULL_ENSLAVING"

if MODE == "FULL_ENSLAVING":
    experiment_name = f'plot-dec-PROB-60ped-{N_REPEATS}reps-fullens'

    n_ped = 60
    noisecoef = 0.5
    ensldegree = 1.0
    max_timesteps = int(1e4)

    exitrew_state = True
    follrew_state = True
    intrrew = 0
    initrew = -1

    emb_state = True
    alpha = 2
    learn_timesteps = 2_000_000

# Prepare request 
env_setup = \
    f"-n {n_ped} "\
    f"--noise-coef {noisecoef} "\
    f"--enslaving-degree {ensldegree}"\
    f"--max-timesteps {max_timesteps}"

reward_setup = \
    f"--is-new-exiting-reward {exitrew_state} "\
    f"--is-new-followers-reward {follrew_state} "\
    f"--intrinsic-reward-coef {intrrew} "\
    f"--init-reward-each-step={initrew}"

learn_setup = \
    f"--learn-timesteps {learn_timesteps} "\
    f"-e ${emb_state} "\
    f"--alpha ${alpha}"

request = f"--exp-name ${experiment_name} ${reward_setup} ${env_setup} ${learn_setup}"

args = parse_args(inline_mode=True, request=request)

# Function for preprocessing of states
def preprocess_stats(stats):
    maxlen = max(len(episode) for episode in stats['escaped'])
    for i in range(len(stats['escaped'])):
        for key in stats.keys():
            begin = len(stats[key][i])
            for j in range(begin, maxlen):
                stats[key][i].append(stats[key][i][-1])
    mean_stats = {key : np.array(val).mean(axis=0) for key, val in stats.items()}
    std_stats = {key : np.array(val).std(axis=0) for key, val in stats.items()}
    return mean_stats, std_stats

# Prepare Evacuation env
env = setup_env(args, experiment_name)

# Prepare PPO model
model = PPO.load(FULLENSL_MODEL_1, env, device=DEVICE)

# Collect statistics
ppo_stats = {'escaped' : [], 'exiting' : [], 'following' : [], 'viscek' : []}
for _ in tqdm(range(2)): #N_REPEATS)):
    collect_stats = []
    obs, _ = env.reset()
    terminated, truncated = False, False

    while not (terminated or truncated):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        collect_stats.append(env.pedestrians.status_stats)

    if len(collect_stats) > 0:
        collect_stats = {k: [d[k] for d in collect_stats] for k in collect_stats[0]}
        for key in ppo_stats.keys():
            ppo_stats[key].append(collect_stats[key])

