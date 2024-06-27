from dataclasses import dataclass
from typing import Literal

from . import constants
from .wrappers import *

@dataclass    
class EnvConfig:
    
    experiment_name: str = 'test'
    """prefix of the experiment name for logging results"""

    # ---- Geometry of the environment ----
    
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
    
    eps: float = constants.EPS
    """eps"""

    # ---- Leader params ----
    
    enslaving_degree: float = constants.ENSLAVING_DEGREE
    """enslaving degree of leader in generalized viscek model"""

    # ---- Reward params ----
    
    is_new_exiting_reward: bool = constants.IS_NEW_EXITING_REWARD
    """if True, positive reward will be given for each pedestrian,
    entering the exiting zone"""
    
    is_new_followers_reward: bool = constants.IS_NEW_FOLLOWERS_REWARD
    """if True, positive reward will be given for each pedestrian,
    entering the leader\'s zone of influence"""
    
    intrinsic_reward_coef: float = constants.INTRINSIC_REWARD_COEF
    """coefficient in front of intrinsic reward"""
    
    is_termination_agent_wall_collision: bool = constants.TERMINATION_AGENT_WALL_COLLISION
    """if True, agent\'s wall collision will terminate episode"""
    
    init_reward_each_step: float = constants.INIT_REWARD_EACH_STEP
    """constant reward given on each step of agent"""

    # ---- Timing in the environment ----
    
    max_timesteps: int = constants.MAX_TIMESTEPS
    """max timesteps before truncation"""
    
    n_episodes: int = constants.N_EPISODES
    """number of episodes already done (for pretrained models)"""
    
    n_timesteps: int = constants.N_TIMESTEPS
    """number of timesteps already done (for pretrained models)"""

    # ---- Logging params ----

    render_mode: str | None = None
    """render mode (has no effect)"""
    
    draw: bool = False
    """enable saving of animation at each step"""    

    verbose: bool = False
    """enable debug mode of logging"""

    giff_freq: int = 500
    """frequency of logging the giff diagram"""

    wandb_enabled: bool = True
    """enable wandb logging (if True wandb.init() should be called before 
    initializing the environment)"""
    

@dataclass
class EnvWrappersConfig:
    """Observation wrappers params"""
    
    # TODO, add to wrap_env
    num_obs_stacks: int = constants.NUM_OBS_STACKS
    """number of times to stack observation"""

    positions: Literal['abs', 'rel', 'grav'] = 'abs'
    """positions: 
        - 'abs': absolute coordinates
        - 'rel': relative coordinates
        - 'grav': gradient gravity potential encoding (GravityEncoding)
    """
    
    statuses: Literal['no', 'ohe', 'cat'] = 'no'
    """add pedestrians statuses to obeservation as one-hot-encoded columns.
    NOTE: this value has no effect when `positions`='grad' is selected.
    """
    
    type: Literal['Dict', 'Box'] = 'Dict'
    """concatenate Dict-type observation to a Box-type observation
    (with added statuses to the observation)"""

    # ---- GravityEncoding params ----
    
    alpha: float = constants.ALPHA
    """alpha parameter of GravityEncoding"""

    # ---- post-init setup ----
    
    def __post_init__(self):
        ...
        
    def wrap_env(self, env):
        """
        Possible options and how to get them:
        | pos:   | sta:  | type:  | how
        | - abs  | - no  | - dict | (which wrappers)
        | - rel  | - ohe | - box  | 
        | - grav | - cat |        | 
        |--------|-------|--------|-----
        |  abs   |  no   |  dict  | -
        |  abs   |  ohe  |  dict  | PedestriansStatuses(type='ohe')
        |  abs   |  cat  |  dict  | PedestriansStatuses(type='cat')
        |  abs   |  no   |  box   | MatrixObs(type='no')
        |  abs   |  ohe  |  box   | MatrixObs(type='ohe')
        |  abs   |  cat  |  box   | MatrixObs(type='cat')
        |  rel   |  no   |  dict  | RelativePosition()
        |  rel   |  ohe  |  dict  | RelativePosition() + PedestriansStatuses(type='ohe')
        |  rel   |  cat  |  dict  | RelativePosition() + PedestriansStatuses(type='cat')
        |  rel   |  no   |  box   | RelativePosition() + MatrixObs(type='no')
        |  rel   |  ohe  |  box   | RelativePosition() + MatrixObs(type='ohe')
        |  rel   |  cat  |  box   | RelativePosition() + MatrixObs(type='cat')
        |  grav  |  -    |  dict  | GravityEmbedding(alpha)
        |  grav  |  -    |  box   | TODO
        
        NOTE #1: `grav` position option utilizes information about state but 
        in its own way, so you don't need to add PedestriansStatuses() wrapper. 
        
        NOTE #2: to use Box version of `grav` position it is recommended to 
        just use `Flatten` observation wrapper from gymnasium.
        """
        if self.positions == 'grav':
            if self.type == 'Dict':
                return GravityEncoding(env, alpha=self.alpha)
            elif self.type == 'Box':
                raise NotImplementedError
            else:
                raise ValueError
        
        if self.positions == 'rel':
            env = RelativePosition(env)
        
        if self.type == 'Box':
            return MatrixObs(env, type=self.statuses)
    
        if self.statuses != 'no':
            env = PedestriansStatuses(env, type=self.statuses)
        
        return env