import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# env = GameEnv()

# state_size = env.observation_space.shape[0]
# action_size = env.action_space.n
# lr = 0.0001


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, embedding):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.embedding = embedding

        self.body1 = nn.Linear(self.state_size, 64)
        self.body2 = nn.Linear(64, 64)

        # self.actor1 = nn.Linear(64, 64)
        # self.actor2 = nn.Linear(64, self.action_size)
        self.actor = nn.Linear(64, self.action_size)

        # self.critic1 = nn.Linear(64, 64)
        # self.critic2 = nn.Linear(64, 1)
        self.critic = nn.Linear(64, 1)

        # action (log_prob + value) buffer
        self.saved_av = []
        # reward buffer
        self.saved_r = []

    def forward(self, state):
        state = self.embedding(state)

        output_body = torch.tanh(self.body1(state))  # leaky_rely --> tanh
        output_body = torch.tanh(self.body2(output_body))

        # action_head = F.tanh(self.actor1(output_body)) # delete
        # action_head = self.actor2(action_head)  # do we need Relu here???
        # action_prob = F.softmax(action_head, dim=-1)
        action_head = self.actor(output_body)
        action_prob = F.softmax(action_head, dim=-1)

        # critic_head = F.tanh(self.critic1(output_body))  # delete
        # state_values = self.critic2(critic_head)
        state_values = self.critic(output_body)

        return action_prob, state_values


class ActorCritic_contin(nn.Module):
    def __init__(self, state_size, action_size, embedding, state_type='grad'):
        super(ActorCritic_contin, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = 64
        self.embedding = embedding
        self.state_type = state_type

        self.body = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        )

        self.actor = nn.Linear(self.hidden_size, self.action_size * 2)

        self.critic = nn.Linear(self.hidden_size, 1)

        # self.logstds_param = nn.Parameter(torch.full((action_size,), 0.25))
        # self.register_parameter("logstds", self.logstds_param)

        # action (log_prob + value) buffer
        self.saved_av = []
        # reward buffer
        self.saved_r = []

    def forward(self, state):

        if self.state_type == 'viscek' or (self.state_type == 'vigrad' or self.state_type == 's_vigrad'):
            output_body = self.body(state)
            
        elif self.state_type == 'coord':
            embedded = self.embedding(state)
            output_body = self.body(embedded)
        
        # means = self.actor(output_body)  # means
        # stds = torch.clamp(self.logstds.exp(), 1e-3, 50)
        means, stds = torch.split(self.actor(output_body), self.action_size)
        stds = torch.clamp(stds.exp(), 1e-4, 50)

        state_values = self.critic(output_body)

        return means, stds, state_values
