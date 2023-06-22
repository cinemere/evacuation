import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic_contin(nn.Module):
    def __init__(self, 
        embedding: nn.Module,
        state_size: int, 
        action_size: int, 
        state_type: str = 'grad', 
        hidden_size: int = 64
        ) -> None:
        """Actor-Critic network

        Args:
            embedding (nn.Module): Embedding network
            state_size (int): input size
            action_size (int): _description_
            state_type (str, optional): _description_. Defaults to 'grad'.
            hidden_size (int, optional): _description_. Defaults to 64.
        """
        super(ActorCritic_contin, self).__init__()
        
        # Parameters
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.state_type = state_type

        # Networks
        self.embedding = embedding
        self.body = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        )
        self.actor = nn.Linear(self.hidden_size, self.action_size * 2)
        self.critic = nn.Linear(self.hidden_size, 1)

        self.saved_av = []
        self.saved_r = []

    def forward(self, state):

        if self.state_type == 'viscek' or (self.state_type == 'vigrad' or self.state_type == 's_vigrad'):
            output_body = self.body(state)
            
        elif self.state_type == 'coord':
            embedded = self.embedding(state)
            output_body = self.body(embedded)
        
        means, stds = torch.split(self.actor(output_body), self.action_size)
        stds = torch.clamp(stds.exp(), 1e-4, 50)

        state_values = self.critic(output_body)

        return means, stds, state_values
