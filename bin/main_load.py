from datetime import datetime
from operator import mod
import sys
from torch.utils.tensorboard import SummaryWriter
import os

from old_files.env import GameEnv
# from env_vicsek import GameEnv as GameEnv_vicsek
from env_vicsek_obstacle import GameEnv as GameEnv_vicsek
from net import ActorCritic_contin
from embedding import NNSetEmbedding
# from model import Trainer 
from model_contin_load import Trainer as Trainer_contin


def setup_experiment(title, logdir="logs"):
    experiment_name = "{}--{}".format(title, datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    writer = SummaryWriter(log_dir=os.path.join(logdir, experiment_name))
    return writer, experiment_name

# parameters to vary:
learning_rate = 1e-4
gamma = 0.99
number_of_pedestrians = 60  # 2, 3, 5, 10, 50, 100
case = 'continious'  # 'continious'
state_type = 'viscek'  # 's_vigrad'
norm = 'no'  # 'no', 'adv', 'ret'
vision_radius = 0.1  # 0.2
alpha = 2  # 2
step_size = 0.1

model_mode = 'existing' # 'new', 'existing'
# model_name = 'model/best_continious_s_vigradstate_nonorm_N60_lr0.0001_gamma0.99_vr0.1_a2_ss0.05--31-05-2022-14-50-21.pkl' # or None 
# model_path = 'continious_s_vigradstate_nonorm_N60_lr0.0001_gamma0.99_vr0.1_a2_ss0.05--31-05-2022-14-50-21/'
# model_path = 'continious_s_vigradstate_nonorm_N60_lr0.0001_gamma0.99_vr0.1_a2_ss0.05--31-05-2022-14-50-21/model/'
# model_name = 'best_continious_s_vigradstate_nonorm_N60_lr0.0001_gamma0.99_vr0.1_a2_ss0.05--31-05-2022-14-50-21.pkl'
model_path = 'continious_s_vigradstate_nonorm_N60_lr0.001_gamma0.99_vr0.1_a2--30-05-2022-08-23-49/model/'
model_name = 'best_continious_s_vigradstate_nonorm_N60_lr0.001_gamma0.99_vr0.1_a2--30-05-2022-08-23-49.pkl'

model_path = '54load_continious_viscekstate_nonorm_N60_lr0.0005_gamma0.99_vr0.1_a2_ss0.01--01-06-2022-11-39-05/model/'
model_name = 'best_54load_continious_viscekstate_nonorm_N60_lr0.0005_gamma0.99_vr0.1_a2_ss0.01--01-06-2022-11-39-05.pkl' #small obstacle_pretrained 

model_path = 'continious_s_vigradstate_nonorm_N60_lr0.001_gamma0.99_vr0.1_a2--30-05-2022-08-23-49/model/'
model_name = 'best_continious_s_vigradstate_nonorm_N60_lr0.001_gamma0.99_vr0.1_a2--30-05-2022-08-23-49.pkl' #small obstacle_pretrained 

#model_path = '/home/skoguest/aklepach/pr_evac/load_continious_viscekstate_nonorm_N60_lr1e-05_gamma0.99_vr0.1_a2_ss0.1--31-05-2022-16-54-43/model/' #!
#model_name = 'best_load_continious_viscekstate_nonorm_N60_lr1e-05_gamma0.99_vr0.1_a2_ss0.1--31-05-2022-16-54-43.pkl' #obstacle_pretrained

writer, experiment_name = setup_experiment(f'small_obstacle_2train_{case}_{state_type}state_{norm}norm_N{number_of_pedestrians}_lr{learning_rate}_gamma{gamma}_vr{vision_radius}_a{alpha}_ss{step_size}', logdir="./logs")
print("Experiment name:", experiment_name)

if case == 'continious':
    if state_type == 'viscek':
        env = GameEnv_vicsek(NUM_PEDESTRIANS=number_of_pedestrians, VISION_RADIUS=vision_radius, ALPHA=alpha, STEP_SIZE=step_size)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        # EMBEDDING IS TURNED OFF!!!
        emb = NNSetEmbedding(state_size)
        net = ActorCritic_contin(state_size, action_size, embedding=emb, state_type=state_type)
        experiment = Trainer_contin(env, net, experiment_name, writer=writer, learning_rate=learning_rate, gamma=gamma, 
        model_name=model_name, model_path=model_path)

experiment.learn(500_000)