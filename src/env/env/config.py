from dataclasses import dataclass

@dataclass    
class EnvConfig:
    
    experiment_name: str = 'test'
    """prefix of the experiment name for logging results"""

    # ---- Geometry of the environment ----
    
    number_of_pedestrians: int = 10
    """number of pedestrians in the simulation"""

    width: float = 1.
    """geometry of environment space: width"""
    
    height: float = 1.
    """geometry of environment space: height"""
    
    step_size: float = 0.01
    """length of pedestrian\'s and agent\'s step
    Typical expected values: 0.1, 0.05, 0.01"""
    
    noise_coef: float = 0.2
    """noise coefficient of randomization in viscek model"""
    
    eps: float = 1e-8
    """eps"""

    # ---- Leader params ----
    
    enslaving_degree: float = 1.
    """enslaving degree of leader in generalized viscek model
    vary in (0; 1], where 1 is full enslaving.
    Typical expected values: 0.1, 0.5, 1."""

    # ---- Reward params ----
    
    is_new_exiting_reward: bool = False
    """if True, positive reward will be given for each pedestrian,
    entering the exiting zone"""
    
    is_new_followers_reward: bool = True
    """if True, positive reward will be given for each pedestrian,
    entering the leader\'s zone of influence"""
    
    intrinsic_reward_coef: float = 0.
    """coefficient in front of intrinsic reward"""
    
    is_termination_agent_wall_collision: bool = False
    """if True, agent\'s wall collision will terminate episode"""
    
    init_reward_each_step: float = -1.
    """constant reward given on each step of agent. 
    Typical expected values: 0, -1."""

    # ---- Timing in the environment ----
    
    max_timesteps: int = 2_000
    """max timesteps before truncation"""
    
    n_episodes: int = 0
    """number of episodes already done (for pretrained models)"""
    
    n_timesteps: int = 0
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

    # ---- Logging artifacts dirs ----

    path_giff: str = 'saved_data/giff'
    """path to save giff animations: {path_giff}/{experiment_name}"""
    
    path_png: str = 'saved_data/png'
    """path to save png images of episode trajectories: {path_png}/{experiment_name}"""
    
    path_logs: str = 'saved_data/logs'
    """path to save logs: {path_logs}/{experiment_name}"""
    
    def __post_init__(self):
        # TODO add loading of pretrained model
        assert self.n_episodes == 0, NotImplementedError
        assert self.n_timesteps == 0, NotImplementedError