NUM_PEDESTRIANS = 10                # number of pedestrians
WIDTH = 1.0                         # geometry of environment space: width
HEIGHT = 1.0                        # geometry of environment space: height
ALPHA = 3                           # parameter of gradient state
STEP_SIZE = 0.01

SWITCH_DISTANCE_TO_LEADER = 0.2     # radius of catch by leader
SWITCH_DISTANCE_TO_OTHER_PEDESTRIAN = 0.1 # SWITCH_DISTANCE_TO_LEADER
SWITCH_DISTANCE_TO_EXIT   = 0.4
SWITCH_DISTANCE_TO_ESCAPE = 0.01

MAX_TIMESTEPS = int(2*10e3)

EPS = 1e-8 
NOISE_COEF = 0.2                    # randomization in viscek model

SAVE_PATH_GIFF = 'saved_data/giff'
SAVE_PATH_PNG  = 'saved_data/png'
SAVE_PATH_LOGS = 'saved_data/logs'