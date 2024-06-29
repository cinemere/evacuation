import os
from dataclasses import dataclass, asdict
from typing import Literal
from src.params_old import *
import tyro
import yaml
import wandb
from datetime import datetime

from env import EvacuationEnv, EnvConfig, EnvWrappersConfig
    
@dataclass
class SBConfig:
    """Stable Baselines Model Config"""
        
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
class CleanRLConfig:
    """Stable Baselines Model Config"""
    
    is_embedding: bool = False
    """is embedding"""
    
@dataclass
class Args:

    env: EnvConfig
    """env params"""
    
    wrap: EnvWrappersConfig
    """env wrappers params"""
    
    # model: SBConfig | CleanRLConfig = SBConfig
    # """select the config of model"""
    
    def __post_init__(self):
        self.setup_exp_name()
        self.setup_wandb_logging()
    
    def setup_exp_name(self):
        self.now = datetime.now().strftime(f"%d-%b-%H-%M-%S")
        self.env.experiment_name = f"{self.env.experiment_name}_{self.now}"
            
    def setup_wandb_logging(self):
        if self.env.wandb_enabled:
            wandb.init(
                project="evacuation",
                name=self.env.experiment_name,
                notes=self.env.experiment_name,
                config=self,
                dir='saved_data/wandb'
        )            

    def print_args(self):
        print("The following arguments will be used in the experiment:")
        print(yaml.dump(asdict(self)))
    
    def save_args(self):
        save_config_dir = 'saved_data/configs/'
        os.makedirs(save_config_dir, exist_ok=True)
        with open(os.path.join(save_config_dir, f"config_{self.env.experiment_name}.yaml"), "w") as f:
            f.write(yaml.dump(asdict(self)))
            
    @classmethod
    def create_from_yaml(cls, path: str):
        import warnings
        warnings.warn('All arguments will be loaded from yaml file (from $CONFIG).')
        
        with open(CONFIG, 'r') as cfg:
            content = yaml.load(cfg, Loader=yaml.Loader)
        
        config = cls(
            EnvConfig(**content['env']), 
            EnvWrappersConfig(**content['wrap']))
        return config
    
               
def setup_env(env_config: EnvConfig, wrap_config: EnvWrappersConfig):
    env = EvacuationEnv(env_config)
    env = wrap_config.wrap_env(env)
    return env        

WANDB_DIR = os.getenv("WANDB_DIR", "saved_data/wandb")
CONFIG = os.getenv("CONFIG", "")
        
if __name__ == "__main__":
    
    help_text="""
    To use yaml config set the env variable `CONFIG`:
    
    `CONFIG=<path-to-yaml-config> python main.py`
    
    """
    
    config = Args.create_from_yaml(CONFIG) if CONFIG else tyro.cli(Args, description=help_text)
    
    config.print_args()
    config.save_args()
    env = setup_env(env_config=config.env,
                    wrap_config=config.wrap)
    env.reset()
    print(env)
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())
    env.reset()
    print(f"{env.unwrapped.pedestrians.num=}")
