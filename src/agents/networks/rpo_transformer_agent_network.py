import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

import numpy as np
from math import sqrt

from dataclasses import dataclass

from .utils import layer_init
from .rpo_linear_agent_network import RPOLinearNetwork, RPOLinearNetworkConfig

@dataclass
class RPOTransformerEmbeddingConfig:
    
    network: RPOLinearNetworkConfig
    """params of linear network"""
    
    num_blocks: int = 2

    num_heads: int = 3
    """number of hidden layers"""

    dim_feedforward: float = 96
    """the alpha parameter for RPO"""
    
    dropout: float = 0.1
    """dropout value"""
    
    use_resid: bool = False
    """to use or not residual connections in transformer blocks"""
    

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 d_model: int = 3,  # # dimention of arguments for 1 pedestrian
                 num_heads: int = 2
                 ) -> None:  
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.scale = sqrt(d_model)
        
        self.Wq = nn.Linear(d_model, d_model * num_heads)
        self.Wk = nn.Linear(d_model, d_model * num_heads)
        self.Wv = nn.Linear(d_model, d_model * num_heads)
        self.dense = nn.Linear(d_model * num_heads, d_model)
        
    def split_heads(self, x, batch_size):                           # [batch, seq_len, d_model * num_heads]
        x = x.view(batch_size, -1, self.num_heads, self.d_model)    # [batch, seq_len, num_heads, d_model]
        return x.permute(0, 3, 1, 2)                                # [batch, d_model, seq_len, num_heads]
    
    def forward(self, x, mask=None):
        assert x.dim() == 3, f"Input shape should be: (batch_size, seq_len, d_model). "\
                             f"Instead MultiHeadAttention module received {x.shape=}"
        batch_size = x.size(0)                              # [batch, seq_len, d_model]
        
        q = self.split_heads(self.Wq(x), batch_size)
        k = self.split_heads(self.Wk(x), batch_size)
        v = self.split_heads(self.Wv(x), batch_size)        # [batch, d_model, seq_len, num_heads]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale # [batch, d_model, seq_len, seq_len]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)       # [batch, d_model, seq_len, seq_len]
        output = torch.matmul(attention_weights, v)         # [batch, d_model, seq_len, num_heads]
        
        output = output.transpose(1, 2).contiguous()        # [batch, seq_len, d_model, num_heads]
        output = output.flatten(-2, -1)                     # [batch, seq_len, d_model * num_heads]
        output = self.dense(output)                         # [batch, seq_len, d_model]
        return output


class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: int = 6, # dimention of arguments for 1 pedestrian
                 num_heads: int = 3, 
                 d_ff: int = 64, 
                 dropout: float = 0.1,
                 use_resid: bool = False,
        ) -> None:
        super(TransformerBlock, self).__init__()
        
        # Constants
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_resid = use_resid
        
        # Attention layer
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward layer
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
            
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):                # [batch, ...]
        shape = x.shape

        # If inputs are flattened, reshape them
        if x.dim() == 2:                            # [batch, seq_len * d_model]
            x = x.view(shape[0], -1, self.d_model)  # [batch, seq_len, d_model]

        # Attention
        x_attn = self.attention(x, mask)
        # NOTE this was in previous version of model
        # attn_output = attn_output.view(batch_size, 1, -1)
        x_attn = self.dropout(x_attn)
        x = x + x_attn if self.use_resid else x_attn
        x = self.norm1(x) 

        # Feed-forward
        x_ff = self.ff(x)
        x_ff = self.dropout(x_ff)
        x = x + x_ff if self.use_resid else x_ff
        x = self.norm2(x)

        # Reshape to return same shape as input
        x = x.view(shape)
        return x

class RPOTransformerEmbedding(RPOLinearNetwork):
    def __init__(self, 
                 envs,
                 number_of_pedestrians: int,
                 config: RPOTransformerEmbeddingConfig,
                 device: torch.DeviceObjType):
        super().__init__(envs, config.network, device)
        
        embedding = []
        for emb_layer in range(config.num_blocks):
            embedding.append(
                TransformerBlock(
                    d_model=envs.single_observation_space.shape[-1] // (number_of_pedestrians + 2),
                    # d_model=envs.observation_space.shape[-1],
                    num_heads=config.num_heads,  # 3, #2, # 3,
                    d_ff=config.dim_feedforward,  # 96, #34,  # 24,
                    dropout=config.dropout,
                    use_resid=config.use_resid,
                )
            )
        self.embedding = nn.Sequential(*embedding)
        
    def get_value(self, x):
        x = self.embedding(x)
        return super().get_value(x)
    
    def get_action_and_value(self, x, action=None):
        # print(x)
        # print(x.shape)
        x = self.embedding(x)
        return super().get_action_and_value(x, action)