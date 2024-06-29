# %% load config
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import main_config, create_config
from ding.config import compile_config

cfg = compile_config(main_config, create_cfg=create_config, auto=True)
# %%
cfg
# %% create env (train + eval)
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from src.env import EvacuationEnv
import gymnasium as gym

collector_env = BaseEnvManagerV2(
    env_fn=[lambda: DingEnvWrapper(
        # gym.make("CartPole-v0")
        EvacuationEnv(
            "test-ding", 10, 1,
            intrinsic_reward_coef=0,
            init_reward_each_step=-1
        )
        ) for _ in range(cfg.env.collector_env_num)],
    cfg=cfg.env.manager
)

evaluator_env = BaseEnvManagerV2(
    env_fn=[lambda: DingEnvWrapper(
        # gym.make("CartPole-v0")
        EvacuationEnv(
            "test-ding", 10, 1,
            intrinsic_reward_coef=0,
            init_reward_each_step=-1
        )

        ) for _ in range(cfg.env.evaluator_env_num)],
    cfg=cfg.env.manager
)
# %%
vars(collector_env)
# %% 
f"{cfg.env.collector_env_num=}, {cfg.env.evaluator_env_num=}"
# %% DEFINE MODEL AND POLICY
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.data import DequeBuffer

model = DQN(**cfg.policy.model)
buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
policy = DQNPolicy(cfg.policy, model=model)
# %% 
model
# %% bulid pipeline

from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, \
    interaction_evaluator, data_pusher, eps_greedy_handler, CkptSaver

with task.start(async_mode=False, ctx=OnlineRLContext()):
    
    # Evaluating, we place it on the first place to get 
    # the score of the random model as a benchmark value
    task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
    
    # Decay probability of explore-exploit
    task.use(eps_greedy_handler(cfg))
    
    # Collect environmental data
    task.use(StepCollector(cfg, policy.collect_mode, collector_env))
    
    # Push data to buffer
    task.use(data_pusher(cfg, buffer_))
    
    # Train the model
    task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
    
    # Save the model
    task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))

    # In the evaluation process, if the model is found to have 
    # exceeded the convergence value, it will end early here
    task.run()

# %%
collector_env.seed(42)
# %%
test_env = EvacuationEnv(
    "test-ding", 10, 1,
    intrinsic_reward_coef=0,
    init_reward_each_step=-1
)

# %%
test_env
# %%
