from dataclasses import dataclass

from .env import EvacuationEnv
from agents import BaseRLAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class TrainerRPOConfig:
    exp_name: str = "rpo-tr3-initagent0"##os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    
    total_timesteps: int = 80000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 3
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    
    # agent params
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    rpo_alpha: float = 0.5
    """the alpha parameter for RPO"""

class TrainerRPO:
    def __init__(
            self, 
            env: EvacuationEnv, 
            agent: ,
            config: TrainerRPOConfig,
        ):
        # init arguments
        # setup env
        # wrap env with cleanrl wrappers
        
        ...
    
    def learn(
            self,
            
        ):
        ...
        
