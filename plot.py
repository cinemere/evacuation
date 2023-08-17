# %% Imports
import os
import numpy as np
from stable_baselines3 import PPO
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.env import EvacuationEnv
from src.env import constants
from src import params

path_save = 'saved_data/plot_efficiency'
if not os.path.exists(path_save): os.makedirs(path_save)

DEVICE='cpu'
N_REPEATS = 5_000
experiment_name=f'plot-60ped-{N_REPEATS}reps'

# %% Function / utils
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
    
    gravity_embedding_params = parser.add_argument_group('gravity embedding params')    
    gravity_embedding_params.add_argument('-e', '--enabled-gravity-embedding', type=bool, 
        default=constants.ENABLED_GRAVITY_EMBEDDING,
        help='if True use gravity embedding')
    gravity_embedding_params.add_argument('--alpha', type=float, default=constants.ALPHA,
        help='alpha parameter of gravity gradient embedding')
    
    args = parser.parse_args("")
    return args

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

# %% Setup environment

args = parse_args()
args.number_of_pedestrians = 60
env = setup_env(args, experiment_name)

# %% Prepare PPO stats

model_path = "/home/cinemere/work/evacuation/saved_data/models/60pedestrians_n-60_lr-0.0001_gamma-0.99_s-gra_a-3_ss-0.01_vr-0.1_17-Aug-00-54-45.zip"
model = PPO.load(model_path, env, device=DEVICE)

ppo_stats = {'escaped' : [], 'exiting' : [], 'following' : [], 'viscek' : []}
for _ in tqdm(range(N_REPEATS)):
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

ppo_mean_stats, ppo_std_stats = preprocess_stats(ppo_stats)
np.save(os.path.join(path_save, f'{experiment_name}_ppo_mean.np'), ppo_mean_stats['escaped'])
np.save(os.path.join(path_save, f'{experiment_name}_ppo_std.np'), ppo_std_stats['escaped'])
# %% Prepare random stats

from src.agents import RandomAgent

random_agent = RandomAgent(env.action_space)
rand_stats = {'escaped' : [], 'exiting' : [], 'following' : [], 'viscek' : []}
for _ in tqdm(range(N_REPEATS)):
    collect_stats = []
    obs, _ = env.reset()
    terminated, truncated = False, False

    while not (terminated or truncated):
        action, _ = random_agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        collect_stats.append(env.pedestrians.status_stats)

    if len(collect_stats) > 0:
        collect_stats = {k: [d[k] for d in collect_stats] for k in collect_stats[0]}
        for key in rand_stats.keys():
            rand_stats[key].append(collect_stats[key])

rand_mean_stats, rand_std_stats = preprocess_stats(rand_stats)
np.save(os.path.join(path_save, f'{experiment_name}_rand_mean.np'), rand_mean_stats['escaped'])
np.save(os.path.join(path_save, f'{experiment_name}_rand_std.np'), rand_std_stats['escaped'])

# %% Prepare plot

plt.title("Quantification of evacuation efficiency")
plt.xlabel('Time (timesteps)'); plt.ylabel("% NOT evacuated")
plt.errorbar(
    y = 1 - ppo_mean_stats['escaped'] / args.number_of_pedestrians,
    x = np.arange(len(ppo_mean_stats['escaped'])),
    yerr = 1 - ppo_std_stats['escaped'] / args.number_of_pedestrians,
    label = 'RL-trained leader', alpha=0.2, errorevery=100)
plt.errorbar(
    y = 1 - rand_mean_stats['escaped'] / args.number_of_pedestrians,
    x = np.arange(len(rand_mean_stats['escaped'])),
    yerr = 1 - rand_std_stats['escaped'] / args.number_of_pedestrians,
    label = 'Random leader', alpha=0.2, errorevery=100)
plt.legend()
plt.savefig(os.path.join(path_save, f"plot_{experiment_name}.png"))
plt.savefig(os.path.join(path_save, f"plot_{experiment_name}.pdf"))

# %%

plt.title("Quantification of evacuation efficiency")
plt.xlabel('Time (timesteps)'); plt.ylabel("% NOT evacuated")
plt.plot(
    np.arange(len(rand_mean_stats['escaped'])),
    1 - rand_mean_stats['escaped'] / args.number_of_pedestrians,
    # yerr = 1 - rand_std_stats['escaped'] / args.number_of_pedestrians,
    label = 'Random leader', alpha=0.5)
plt.plot(
    np.arange(len(ppo_mean_stats['escaped'])),
    1 - ppo_mean_stats['escaped'] / args.number_of_pedestrians,
    # yerr = 1 - ppo_std_stats['escaped'] / args.number_of_pedestrians,
    label = 'RL-trained leader', alpha=0.5)
plt.legend()
plt.savefig(os.path.join(path_save, f"plot_{experiment_name}.png"))
plt.savefig(os.path.join(path_save, f"plot_{experiment_name}.pdf"))

# %%
