import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from dataclasses import dataclass

from .utils import layer_init

@dataclass
class RPOTransformerNetworkConfig:

    num_hidden: int = 64
    """number of hidden layers"""

    rpo_alpha: float = 0.5
    """the alpha parameter for RPO"""
