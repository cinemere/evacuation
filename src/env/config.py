from dataclasses import dataclass

from . import constants


@dataclass    
class EnvConfig:

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

    render_mode: str = None
    """render mode (has no effect)"""
    
    draw: bool = False
    """enable saving of animation at each step"""    

    verbose: bool = False
    """enable debug mode of logging"""

    giff_freq: int = 500
    """frequency of logging the giff diagram"""


@dataclass
class EnvWrappersConfig:
    
    # ---- Observation wrappers params ---
    
    num_obs_stacks: int = constants.NUM_OBS_STACKS
    """number of times to stack observation"""

    relative_positions: bool = constants.USE_RELATIVE_POSITIONS
    """add relative positions wrapper (can be use only WITHOUT gravity embedding)"""
    
    add_statuses_ohe: bool = False
    """add pedestrians statuses to obeservation as one-hot-encoded columns"""
    
    add_statuses_cat: bool = False
    """add pedestrians statuses to obeservation as categorical float column"""

    matrix_observation: bool = False
    """concatenate Dict-type observation to a Box-type observation
    (with added statuses to the observation)"""

    # ---- GravityEmbedding params ----
    
    gravity_embedding: bool = constants.ENABLED_GRAVITY_EMBEDDING
    """if True use gravity embedding"""

    alpha: float = constants.ALPHA
    """alpha parameter of GravityEncoding"""

    # ---- post-init setup ----
    
    def __post_init__(self):
        self.check()

    def check(self):
        if self.gravity_embedding:
            assert self.relative_positions == False, \
                "Relative positions wrapper can NOT be used while enabled gravity embedding"
                
@dataclass
class ObsTypeWrappers:
                    
    # 0. pos-grav  (gravity_embedding)

    # 1. pos-abs_dict (-)
    # 2. pos-rel_dict (relative_positions)

    # 3. pos-rel_stat-ohe_dict (relative_positions, add_statuses_ohe)
    # 4. pos-rel_stat-cat_dict (relative_positions, add_statuses_cat)
    
    # 5. pos-abs_stat-ohe_dict (add_statuses_ohe)
    # 6. pos-abs_stat-cat_dict (add_statuses_cat)

    # 7. pos-abs_stat-ohe_box -> (ohe stat) -> matrix_obs (matrix_observation) 
    # 8. pos-abs_stat-cat_box -> (cat stat) -> matrix_obs (matrix_observation) 

    # 7. pos-rel_stat-ohe_box -> (ohe stat) -> matrix_obs (relative_positions, matrix_observation)     
    # 7. pos-rel_stat-cat_box -> (cat stat) -> matrix_obs (relative_positions, matrix_observation)    
    """
    Possible options and how to get them:
    | pos:   | stat  | type   | how
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
    
    NOTE #1: `grav` position option utilizes information about state but 
    in its own way, so you don't need to add PedestriansStatuses() wrapper. 
    
    NOTE #2: to use Box version of `grav` position it is recommended to 
    just use `Flatten` observation wrapper from gymnasium.
    """
    


