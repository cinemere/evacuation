from datetime import datetime
import sys
from torch.utils.tensorboard import SummaryWriter
import os

from old_files.env import GameEnv
# from env_contin import GameEnv as GameEnv_contin
# from env_contin_grad_state import GameEnv as GameEnv_contin_grad_state
# from env_contin_grad_vicsek import GameEnv as GameEnv_contin_grad_vicsek
# from env_vicsek import GameEnv as GameEnv_vicsek
from env_vicsek_obstacle import GameEnv as GameEnv_vicsek
from embedding import NNSetEmbedding
from net import ActorCritic, ActorCritic_contin
# from model import Trainer 
from model_contin import Trainer as Trainer_contin


def setup_experiment(title, logdir="logs"):
    experiment_name = "{}--{}".format(title, datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    writer = SummaryWriter(log_dir=os.path.join(logdir, experiment_name))
    return writer, experiment_name

# parameters to vary:
learning_rate = 1e-4
gamma = 0.99
number_of_pedestrians = 60  # 2, 3, 5, 10, 50, 100
case = 'continious'  # 'continious', 'grid'
state_type = 's_vigrad'  # 'vigrad', 'coord', 'vigrad'
norm = 'no'  # 'no', 'adv', 'ret'/home/skoguest/aklepach/pr_evac/load_continious_viscekstate_nonorm_N60_lr1e-06_gamma0.99_vr0.05_a4--18-05-2022-19-26-38
vision_radius = 0.1  # 0.2
alpha = 2  # 2
step_size = 0.01  # 0.01

writer, experiment_name = setup_experiment(f'small_obstacle_1train_{case}_{state_type}state_{norm}norm_N{number_of_pedestrians}_lr{learning_rate}_gamma{gamma}_vr{vision_radius}_a{alpha}_ss{step_size}', logdir="./logs")
print("Experiment name:", experiment_name)

if case == 'continious':
    if state_type == 'vigrad' or state_type == 's_vigrad':
        env = GameEnv_vicsek(NUM_PEDESTRIANS=number_of_pedestrians, VISION_RADIUS=vision_radius, ALPHA=alpha, STEP_SIZE=step_size)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        # EMBEDDING IS TURNED OFF!!!
        emb = NNSetEmbedding(state_size)
        net = ActorCritic_contin(state_size, action_size, embedding=emb, state_type=state_type)
        experiment = Trainer_contin(env, net, experiment_name, writer=writer, learning_rate=learning_rate, gamma=gamma)

    # elif state_type == 'grad':
    #     env = GameEnv_contin_grad_state(NUM_PEDESTRIANS=number_of_pedestrians, VISION_RADIUS=vision_radius, ALPHA=alpha)
    #     state_size = env.observation_space.shape[0]
    #     action_size = env.action_space.shape[0]
    #     # EMBEDDING IS TURNED OFF!!!
    #     emb = NNSetEmbedding(state_size)
    #     net = ActorCritic_contin(state_size, action_size, embedding=emb, state_type=state_type)
    #     experiment = Trainer_contin(env, net, experiment_name, writer=writer, learning_rate=learning_rate, gamma=gamma)

    # elif state_type == 'coord':
    #     env = GameEnv_contin(NUM_PEDESTRIANS=number_of_pedestrians, VISION_RADIUS=vision_radius)
    #     state_size = env.observation_space.shape[0]
    #     action_size = env.action_space.shape[0]
    #     emb = NNSetEmbedding(state_size)
    #     net = ActorCritic_contin(state_size, action_size, embedding=emb, state_type=state_type)
    #     experiment = Trainer_contin(env, net, experiment_name, writer=writer, learning_rate=learning_rate, gamma=gamma)


# elif case == 'grid':
#     env = GameEnv(NUM_PEDESTRIANS=number_of_pedestrians)
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
#     emb = NNSetEmbedding(state_size)
#     net = ActorCritic(state_size, action_size, embedding=emb)
#     experiment = Trainer(env, net, experiment_name, writer=writer, learning_rate=learning_rate, gamma=gamma)

experiment.learn(500_000)