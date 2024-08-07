
# from . import constants
from .env.env import EvacuationEnv, Status
from .env.config import EnvConfig

from .wrappers.config import EnvWrappersConfig
from .wrappers.gravity_encoding import GravityEncoding
from .wrappers.wrappers import PedestriansStatuses, RelativePosition, MatrixObs

# from gymnasium.envs.registration import register

# register(
#     id='evacuation/Evacuation-v0',
#     entry_point='env.env:EvacuationEnv',
#     # max_episode_steps=300,
# )

def setup_env(env_config: EnvConfig, wrap_config: EnvWrappersConfig):
    env = EvacuationEnv(env_config)
    env = wrap_config.wrap_env(env)
    return env        