# List of params of the environment
# Note: params with (!) in comment are only default params, which
# can be changed with argparse module in main.py

NUM_PEDESTRIANS = 10                        # ! number of pedestrians
EPS = 1e-8 
ENSLAVING_DEGREE = 1.                       # ! leader's enslaving degree

# Area params
WIDTH = 1.                                  # ! geometry of environment space: width
HEIGHT = 1.                                 # ! geometry of environment space: height
STEP_SIZE = 0.01                            # ! 0.1, 0.05, 0.01
NOISE_COEF = 0.2                            # ! randomization in viscek model

# Observation params
NUM_OBS_STACKS = 1                          # ! number of times to stack observation (via FrameStack wrapper)
USE_RELATIVE_POSITIONS = False              # ! add relative positions wrapper (can be use only WITHOUT gravity embedding)

# Reward params
TERMINATION_AGENT_WALL_COLLISION = False    # ! is or no termination for agent's wall collision
INTRINSIC_REWARD_COEF = 1.                  # ! coef of intrinsic reward
IS_NEW_EXITING_REWARD = True                # ! if enable reward for new exiting
IS_NEW_FOLLOWERS_REWARD = True              # ! if enable reward for new followers
INIT_REWARD_EACH_STEP = 0.                  # ! each step give this reward

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