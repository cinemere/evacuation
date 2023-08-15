# # # %%
# # from env import EvacuationEnv
# # from agents import RandomAgent, RotatingAgent
# # import numpy as np

# # print('starting the experiment')

# # env = EvacuationEnv(number_of_pedestrians=100, experiment_name='experiment_test', draw=True)
# # # agent = RandomAgent(env.action_space)
# # agent = RotatingAgent(env.action_space, 0.05)

# # obs, _ = env.reset()
# # for i in range(3):
# #     action = agent.act(obs)
# #     obs, reward, terminated, truncated, _ = env.step(action)
# #     if reward != 0:
# #         print('reward = ', reward)

# # # env.save_animation()
# # # env.render()

# # print('code completed succesfully')
# # %%

# from src.env import EvacuationEnv
# from src.agents import RLAgent
# import numpy as np

# from src.utils import *
# from src.params import *

# experiment_prefix = 'test'

# experiment_name = get_experiment_name(experiment_prefix)

# env = EvacuationEnv(
#     number_of_pedestrians = NUMBER_OF_PEDESTRIANS,
#     experiment_name=experiment_name,
#     verbose=VERBOSE_GYM,
#     draw=DRAW,
#     enable_gravity_embedding=ENABLE_GRAVITY_EMBEDDING
#     )

# # %%
# from src.model.net import ActorCritic

# action_space = env.action_space
# obs_space = env.observation_space

# act_size = get_act_size(action_space)
# obs_size = get_obs_size(obs_space)

# network = ActorCritic(
#     obs_size=obs_size,
#     act_size=act_size,
#     hidden_size=HIDDEN_SIZE,
#     n_layers=N_LAYERS,
#     embedding=None
# )
# # %%
# from src.agents import RLAgent

# agent = RLAgent(
#     action_space=action_space,
#     network=network,
#     mode=MODE,
#     learning_rate=LEARNING_RATE,
#     gamma=GAMMA,
#     load_pretrain=False
# )
# # %%
# from src.trainer import Trainer

# trainer = Trainer(
#     env=env,
#     agent=agent,
#     experiment_name=experiment_name,
#     verbose=VERBOSE
#     )
# # %%

# import torch
# # setting device on GPU if available, else CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# print()

# #Additional Info when using cuda
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# # %%
# import wandb
# from src.env import constants
# from src import params
# config_env = {key : value for key, value in constants.__dict__.items() if key[0] != '_'}
# config_model = {key : value for key, value in params.__dict__.items() if key[0] != '_'}

# wandb.init(
#     project="evacuation",
#     name=experiment_name
#     notes='ss0.01_nowallcollisions_withintrinsicreward',
#     config=dict(config_env, **config_model)
# )

# # %%
# trainer.learn(number_of_episodes=100_000)
