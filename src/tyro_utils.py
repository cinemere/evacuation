import os
from dataclasses import dataclass, asdict
from typing import Literal
from params import *
import tyro
import yaml
from datetime import datetime

from env import EnvConfig, EnvWrappersConfig

@dataclass
class StableBaselinesModelConfig:    
    # origin
    algorithm: Literal['ppo', 'sac', 'a2c'] = 'ppo'
    """which model ti use"""
    
    learn_timesteps: int = 5_000_000
    """number of timesteps to learn the model"""
        
    learning_rate: float = 0.0003
    """learning rate for stable baselines ppo model"""
    
    gamma: float = 0.99
    """gammma for stable baselines ppo model"""
    
    device: Literal['cpu', 'cuda'] = DEVICE
    """device for the model"""

@dataclass
class CleanRLModelConfig:
    is_embedding: bool = False

@dataclass
class Args:
    model: StableBaselinesModelConfig | CleanRLModelConfig# = StableBaselinesModelConfig
    """select the config of model"""

    env: EnvConfig
    """env params"""
    
    wrap: EnvWrappersConfig
    """env wrappers params"""

    exp_name: str = 'test'
    """prefix of the experiment name for logging results"""
    
    def __post_init__(self):
        self.check()
        self.setup_exp_name()
    
    def setup_exp_name(self):
        self.now = datetime.now().strftime(f"%d-%b-%H-%M-%S")
        self.exp_name = f"{self.exp_name}_{self.now}"
    
    def check(self):
        # Check compatibility of input arguments
        self.wrap.check()
        print("Input arguments passed compatibility checks!")
        
    def print_args(self):
        print("The following arguments will be used in the experiment:")
        print(yaml.dump(asdict(self)))
    
    def save_args(self):
        save_config_dir = 'saved_data/configs/'
        os.makedirs(save_config_dir, exist_ok=True)
        with open(os.path.join(save_config_dir, f"config_{self.exp_name}.yaml"), "w") as f:
            f.write(yaml.dump(asdict(self)))
               
def setup_env(exp_name: str, env_config: EnvConfig):
    
    from env import EvacuationEnv
    env = EvacuationEnv(exp_name, **vars(env_config))
    return env        
    
if __name__ == "__main__":
    config = tyro.cli(Args)
    config.check()
    config.print_args()
    config.save_args()
    env = setup_env(exp_name=config.exp_name, 
                    env_config=config.env)
    env.reset()
    print(f"{env.pedestrians.num=}")