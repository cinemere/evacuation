# Note. params are commented to use stable-baselines implementation

# Env params
# NUMBER_OF_PEDESTRIANS = 11
# ENABLE_GRAVITY_EMBEDDING = True
# VERBOSE_GYM = 1                         # Parameter of logging
# DRAW=False                          # If True, we save data for animation plotting

# Trainer params
VERBOSE = 1
LOGGER_BUFFER_MAXLEN = 100

# RLAgent params
# LEARNING_RATE = 1e-4
# GAMMA = 0.99
# MODE = 'training'

# Network Params
# HIDDEN_SIZE = 64
# N_LAYERS = 2

from datetime import datetime
NOW = datetime.now().strftime(f"%d-%b-%H-%M-%S")

SAVE_PATH_TBLOGS = 'saved_data/tb-logs'
SAVE_PATH_MODELS = 'saved_data/models'

DEVICE = "cpu"

WALK_DIAGRAM_LOGGING_FREQUENCY = 500   # frequency in episodes  (overall_timesteps: 2_000_000)