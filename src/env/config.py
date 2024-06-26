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
    """coefficient in front of intrinsic reward')
    is-termination-agent-wall-collision', type=str2bool, default=constants.TERMINATION_AGENT_WALL_COLLISION,
    if True, agent\'s wall collision will terminate episode')
    """
    
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

    use_relative_positions: bool = constants.USE_RELATIVE_POSITIONS
    """add relative positions wrapper (can be use only WITHOUT gravity embedding"""

    # ---- GravityEmbedding params ----
    
    enabled_gravity_embedding: bool = constants.ENABLED_GRAVITY_EMBEDDING
    """if True use gravity embedding"""

    alpha: float = constants.ALPHA
    """alpha parameter of gravity gradient embedding"""

    def __post_init__(self):
        self.check()

    def check(self):
        if self.enabled_gravity_embedding:
            assert self.use_relative_positions == False, \
                "Relative positions wrapper can NOT be used while enabled gravity embedding"