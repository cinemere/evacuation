from dataclasses import dataclass
from typing import Literal
from src.env import constants
from src.params import *
import tyro

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
# class EnvConfig:
class EnvGeometryConfig:
    number_of_pedestrians: int = constants.NUM_PEDESTRIANS
    """number of pedestrians in the simulation"""

    width:float = constants.WIDTH
    """geometry of environment space: width"""
    
    height: float = constants.HEIGHT
    """geometry of environment space: height"""
    
    step_size: float = constants.STEP_SIZE
    """length of pedestrian\'s and agent\'s step"""
    
    noise_coef: float = constants.NOISE_COEF
    """noise coefficient of randomization in viscek model"""
    
    num_obs_stacks: int = constants.NUM_OBS_STACKS
    """number of times to stack observation"""

    use_relative_positions: bool = constants.USE_RELATIVE_POSITIONS
    """add relative positions wrapper (can be use only WITHOUT gravity embedding"""

@dataclass    
class EnvLeaderConfig:
    enslaving_degree: float = constants.ENSLAVING_DEGREE
    """enslaving degree of leader in generalized viscek model"""

@dataclass
class EnvRewardConfig:
    is_new_exiting_reward: bool = constants.IS_NEW_EXITING_REWARD
    """if True, positive reward will be given for each pedestrian,
    entering the exiting zone"""
    
    is_new_followers_reward: bool = constants.IS_NEW_FOLLOWERS_REWARD
    """if True, positive reward will be given for each pedestrian,
    entering the leader\'s zone of influence"""
    
    intrinsic_reward_coef: float = constants.INTRINSIC_REWARD_COEF
    """coefficient in front of intrinsic reward')
    is-termination-agent-wall-collision', type=str2bool, default=constants.TERMINATION_AGENT_WALL_COLLISION,
    if True, agent\'s wall collision will terminate episode')
    """
    
    init_reward_each_step: float = constants.INIT_REWARD_EACH_STEP
    """constant reward given on each step of agent"""

@dataclass
class EnvTimeConfig:
    max_timesteps: int = constants.MAX_TIMESTEPS
    """max timesteps before truncation"""
    
    n_episodes: int = constants.N_EPISODES
    """number of episodes already done (for pretrained models)"""
    
    n_timesteps: int = constants.N_TIMESTEPS
    """number of timesteps already done (for pretrained models)"""

@dataclass
class EnvGravityEmbeddingParams:
    enabled_gravity_embedding: bool = constants.ENABLED_GRAVITY_EMBEDDING
    """if True use gravity embedding"""

    alpha: float = constants.ALPHA
    """alpha parameter of gravity gradient embedding"""

# dataclass(frozen=True)
# class EnvConfig:


@dataclass(frozen=True)
class Args:
    # env: EnvConfig
    # """evacuation env params"""
    geometry: EnvGeometryConfig
    """geometry of environment"""
    
    leader: EnvLeaderConfig
    """leader params"""
    
    reward: EnvRewardConfig
    """reward params"""
    
    time: EnvTimeConfig
    """time params"""
    
    gravity: EnvGravityEmbeddingParams
    """gravity embedding params"""

    model: StableBaselinesModelConfig #| CleanRLModelConfig = StableBaselinesModelConfig
    """select the config of model"""

    exp_name: str = 'test'
    """prefix of the experiment name for logging results"""
    
    verbose: bool = False
    """enable debug mode of logging"""

    draw: bool = False
    """enable saving of animation at each step"""    

    
if __name__ == "__main__":
    config = tyro.cli(Args)  
    print(config)
