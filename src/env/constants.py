# List of params of the environment
# Note: params with (!) in comment are only default params, which
# can be changed with argparse module in main.py

NUM_PEDESTRIANS = 10                        # ! number of pedestrians
TERMINATION_AGENT_WALL_COLLISION = False    # is or no termination for agent's wall collision
EPS = 1e-8 

# Area params
WIDTH = 1.0                                 # ! geometry of environment space: width
HEIGHT = 1.0                                # ! geometry of environment space: height
STEP_SIZE = 0.01                            # ! 0.1, 0.05, 0.01
NOISE_COEF = 0.2                            # ! randomization in viscek model
INTRINSIC_REWARD_COEF = 1.                  # ! coef of intrinsic reward

# Time params
MAX_TIMESTEPS = int(2e3)                    # ! max timesteps before truncation
N_EPISODES = 0                              # ! number of episodes already done (for pretrained models)
N_TIMESTEPS = 0                             # ! number of timesteps already done (for pretrained models)

# Gravity embedding params
ENABLED_GRAVITY_EMBEDDING = True            # ! if True use gravity embedding
ALPHA = 3                                   # ! parameter of gradient state

SWITCH_DISTANCE_TO_LEADER = 0.2             # radius of catch by leader
SWITCH_DISTANCE_TO_OTHER_PEDESTRIAN = 0.1   # SWITCH_DISTANCE_TO_LEADER
SWITCH_DISTANCE_TO_EXIT   = 0.4
SWITCH_DISTANCE_TO_ESCAPE = 0.01

SAVE_PATH_GIFF = 'saved_data/giff'
SAVE_PATH_PNG  = 'saved_data/png'
SAVE_PATH_LOGS = 'saved_data/logs'