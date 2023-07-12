import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    """Actor-Critic network"""
    
    def __init__(self, 
        obs_size: int, 
        act_size: int, 
        hidden_size: int = 64,
        n_layers: int = 2,
        embedding: nn.Module = None
        ) -> None:
        super(ActorCritic, self).__init__()

        self.act_size = act_size
        
        self.embedding = embedding

        self.body = nn.Sequential()
        for i_layer in range(n_layers):
            if i_layer == 0:
                self.body.add_module(
                    f"linear_1",
                    nn.Linear(obs_size, hidden_size)        
                )
                self.body.add_module(
                    f"tanh_1",
                    nn.Tanh()            
                )
            else:
                self.body.add_module(
                    f"linear_{i_layer+1}",
                    nn.Linear(hidden_size, hidden_size)        
                )
                self.body.add_module(
                    f"tanh_{i_layer+1}",
                    nn.Tanh()            
                )
                        
        self.actor  = nn.Linear(hidden_size, act_size * 2)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, input):

        # TODO add embedding for coordinate state
        
        output = self.body(input)
                        
        means, stds = torch.split(self.actor(output), self.act_size)
        stds = torch.clamp(stds.exp(), 1e-4, 50)

        state_values = self.critic(output)

        return means, stds, state_values
