
import wandb
import jax
import numpy as np
import yaml
from dataclasses import dataclass
import pyrallis
import sys

@dataclass
class TrainConfig:
    project: str = "evacuation"
    group: str = "default"
    name: str = "ppo-classic"
    env_id: str = "evacuation-vanila"
    # agent (probably we'd better use separate encoder for different instances of obseravation)
    hidden_dim: int = 256
    # training
    num_envs: int = 2048
    num_steps: int = 10  # 16  # ???
    update_epochs: int = 4  # ???
    num_minibatches: int = 32
    total_timesteps: int = 50_000_000
    lr: float = 0.0003
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01 # 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_episodes: int = 80
    activation: str = "tanh"
    anneal_lr: bool = False
    normalize_env: bool = False
    seed: int = 42

    def __post_init__(self):
        num_devices = jax.local_device_count()
        # splitting computation across all available devices
        self.num_envs_per_device = self.num_envs // num_devices
        self.total_timesteps_per_device = self.total_timesteps // num_devices
        self.eval_episodes_per_device = self.eval_episodes // num_devices
        assert self.num_envs % num_devices == 0
        self.num_updates = (
            self.total_timesteps_per_device
            // self.num_steps
            // self.num_envs_per_device
        )
        print(f"Num devices: {num_devices}, Num updates: {self.num_updates}")

config_path = "sweep_grid_search_config.yaml"
wandb.login()

@pyrallis.wrap()
def train(config: TrainConfig):
    print(config)
    seed = np.random.randint(0, 10000)
    key = jax.random.key(seed)
    data = jax.random.normal(key, shape=(1000, 1000))
    score = (data ** 3).mean()
    return score


def config_to_args(config):
    """Converts a configuration object to a list of command-line arguments."""

    exclude_fields = [
        'num_envs_per_device',
        'total_timesteps_per_device',
        'eval_episodes_per_device',
        'num_updates'
    ]

    args = []
    for key, value in config.__dict__.items():
        if key not in exclude_fields:
            args.append(f"--{key}")
            args.append(str(value))
    return args


def main():
    wandb.init(project="my-first-sweep")
    config = TrainConfig(**wandb.config)
    config_args = config_to_args(config)
    print(config_args)
    # unmodified_argv = sys.argv.copy()
    sys.argv = sys.argv + config_args
    score = train()
    # sys.argv = unmodified_argv
    wandb.log({"reward": score})


with open(config_path, 'r') as file:
    sweep_configuration = yaml.safe_load(file)

sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
print(f"Run the sweep in parallel on all devices using 'wandb agent {sweep_id}'")
# wandb.agent(sweep_id, function=main, count=10)