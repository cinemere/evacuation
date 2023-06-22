from env import EvacuationEnv
from agents import RandomAgent, RotatingAgent
import numpy as np

print('starting the experiment')

env = EvacuationEnv(number_of_pedestrians=100, experiment_name='experiment_test', draw=True)
# agent = RandomAgent(env.action_space)
agent = RotatingAgent(env.action_space, 0.05)

obs, _ = env.reset()
for i in range(300):
    action = agent.act(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if reward != 0:
        print('reward = ', reward)

env.save_animation()
env.render()

print('code completed succesfully')
