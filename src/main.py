import tyro
from agents import RPOAgent
from utils import Args, CONFIG

if __name__ == "__main__":

    config = Args.create_from_yaml(CONFIG) if CONFIG \
        else tyro.cli(Args, description=Args.help())
    
    config.print_args()
    config.save_args()
    
    # env = setup_env(env_config=config.env,
    #                 wrap_config=config.wrap)
    # env.reset()
    # print(env)
    # env.step(env.action_space.sample())
    # env.step(env.action_space.sample())
    # env.step(env.action_space.sample())
    # env.step(env.action_space.sample())
    # env.reset()
    # print(f"{env.unwrapped.pedestrians.num=}")

    training = RPOAgent(
        env_config=config.env,
        env_wrappers_config=config.wrap,
        training_config=config.model.agent,
        network_config=config.model.network
    )
    training.learn()
    