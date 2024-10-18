# %%
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct

import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax

# %%
# Adapted from PureJaxRL implementation and minigrid baselines, source:
# https://github.com/lupuandr/explainable-policies/blob/50acbd777dc7c6d6b8b7255cd1249e81715bcb54/purejaxrl/ppo_rnn.py#L4
# https://github.com/lcswillems/rl-starter-files/blob/master/model.py
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import pyrallis
import wandb
from flax.jax_utils import replicate, unreplicate
from flax.training.train_state import TrainState

# this will be default in new jax versions anyway
jax.config.update("jax_threefry_partitionable", True)


# %%
import sys

sys.path.append("/Users/Klepach/work/repo/tmp/evacuation/src")
from env_jax.env.wrappers.purejaxrl_wrappers import (
    LogWrapper,
    VecEnv,
    ClipAction,
)
from env_jax.env.wrappers.gym_wrappers import GymAutoResetWrapper

#     NormalizeVecObservation,
#     NormalizeVecReward,

from dataclasses import asdict

from env_jax.env.env import Environment, EnvParams

from env_jax.env.wrappers import (
    RelativePosition,
    PedestriansStatusesCat,
    PedestriansStatusesOhe,
    MatrixObsOheStates,
    GravityEncoding,
    FlattenObservation,
)


# %%
class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                ),
                activation,
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                ),
                activation,
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(0.01),
                    bias_init=constant(0.0),
                ),
            ]
        )
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        critic = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                ),
                activation,
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                ),
                activation,
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(1.0),
                    bias_init=constant(0.0),
                ),
            ]
        )

        action_mean = actor_mean(x).astype(jnp.float32)
        action_std = jnp.exp(actor_logtstd)

        dist = distrax.MultivariateNormalDiag(action_mean, action_std)
        values = critic(x)

        return dist, jnp.squeeze(values, axis=-1)


# %%
# class Transition(NamedTuple):
#     done: jnp.ndarray
#     action: jnp.ndarray
#     value: jnp.ndarray
#     reward: jnp.ndarray
#     log_prob: jnp.ndarray
#     obs: jnp.ndarray
#     info: jnp.ndarray


class Transition(struct.PyTreeNode):
    done: jnp.Array
    action: jnp.Array
    value: jnp.Array
    reward: jnp.Array
    log_prob: jnp.Array
    # for obs
    obs: jnp.Array
    dir: jax.Array
    # info: jnp.ndarray


def calculate_gae(
    transitions: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    # single iteration for the loop
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = (
            transition.reward
            + gamma * next_value * (1 - transition.done)
            - transition.value
        )
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value


# %%
def ppo_update_networks(
    train_state: TrainState,
    transitions: Transition,
    init_hstate: jax.Array,
    advantages: jax.Array,
    targets: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    # NORMALIZE ADVANTAGES
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _loss_fn(params):
        # RERUN NETWORK
        dist, value, _ = train_state.apply_fn(
            params,
            {
                # [batch_size, seq_len, ...]
                "obs_img": transitions.obs,
                "obs_dir": transitions.dir,
                "prev_action": transitions.prev_action,
                "prev_reward": transitions.prev_reward,
            },
            init_hstate,
        )
        log_prob = dist.log_prob(transitions.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = transitions.value + (value - transitions.value).clip(
            -clip_eps, clip_eps
        )
        value_loss = jnp.square(value - targets)
        value_loss_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()

        # TODO: ablate this!
        # value_loss = jnp.square(value - targets).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
        entropy = dist.entropy().mean()

        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (value_loss, actor_loss, entropy)

    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
        train_state.params
    )
    (loss, vloss, aloss, entropy, grads) = jax.lax.pmean(
        (loss, vloss, aloss, entropy, grads), axis_name="devices"
    )
    train_state = train_state.apply_gradients(grads=grads)
    update_info = {
        "total_loss": loss,
        "value_loss": vloss,
        "actor_loss": aloss,
        "entropy": entropy,
    }
    return train_state, update_info


# %%
# for evaluation (evaluate for N consecutive episodes, sum rewards)
# N=1 single task, N>1 for meta-RL
class RolloutStats(struct.PyTreeNode):
    reward: jax.Array = jnp.asarray(0.0)
    length: jax.Array = jnp.asarray(0)
    episodes: jax.Array = jnp.asarray(0)


def rollout(
    rng: jax.Array,
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: jax.Array,
    num_consecutive_episodes: int = 1,
) -> RolloutStats:
    def _cond_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry
        return jnp.less(stats.episodes, num_consecutive_episodes)

    def _body_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry

        rng, _rng = jax.random.split(rng)
        dist, _, hstate = train_state.apply_fn(
            train_state.params,
            {
                "obs_img": timestep.observation["img"][None, None, ...],
                "obs_dir": timestep.observation["direction"][None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
            },
            hstate,
        )
        action = dist.sample(seed=_rng).squeeze()
        timestep = env.step(env_params, timestep, action)

        stats = stats.replace(
            reward=stats.reward + timestep.reward,
            length=stats.length + 1,
            episodes=stats.episodes + timestep.last(),
        )
        carry = (rng, stats, timestep, action, timestep.reward, hstate)
        return carry

    timestep = env.reset(env_params, rng)
    prev_action = jnp.asarray(0)
    prev_reward = jnp.asarray(0)
    init_carry = (rng, RolloutStats(), timestep, prev_action, prev_reward, init_hstate)

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    return final_carry[1]


# %%


def init_env(env_name: str):
    if env_name == "evacuation-classic":
        env = Environment()
        env_params = EnvParams()
        env_params = env.default_params(**asdict(env_params))

    env = GymAutoResetWrapper(env)

    env = ClipAction(env)
    env = LogWrapper(env)
    env = FlattenObservation(env)
    # env = VecEnv(env)  TODO keep an eye on here

    return env, env_params


# %%
def make_states(config: TrainConfig):
    # for learning rate scheduling
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.num_minibatches * config.update_epochs))
            / config.num_updates
        )
        return config.lr * frac

    # setup environment
    env, env_params = init_env(config.env_id)

    # setup training state
    rng = jax.random.key(config.seed)
    rng, _rng = jax.random.split(rng)

    network = ActorCritic(
        action_dim=env.action_dim,
        activation="tanh",
        hidden_dim=256,
    )

    # [batch_size, seq_len, ...]
    # shapes = env.observation_shape(env_params)

    init_obs = jnp.zeros(
        (config.num_env_per_device, 1, *env.observation_shape(env_params))
    )
    # init_obs = {key : jnp.zeros((config.num_env_per_dievice, 1, *shapes[key])) for key, value in shapes.items()}
    # init_obs["prev_action"] = jnp.zeros((config.num_envs_per_device, env.action_dim))
    # init_obs["prev_reward"] = jnp.zeros((config.num_envs_per_device, env.action_dim))
    # init_obs = {
    #     "obs_img": jnp.zeros((config.num_envs_per_device, 1, *shapes["img"])),
    #     "obs_dir": jnp.zeros((config.num_envs_per_device, 1, shapes["direction"])),
    #     "prev_action": jnp.zeros((config.num_envs_per_device, 1), dtype=jnp.int32),
    #     "prev_reward": jnp.zeros((config.num_envs_per_device, 1)),
    # }
    # init_hstate = network.initialize_carry(batch_size=config.num_envs_per_device)

    network_params = network.init(_rng, init_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(
            learning_rate=linear_schedule, eps=1e-5
        ), # eps=1e-8
    )
    train_state = TrainState.create(
        apply_fn=network.apply, params=network_params, tx=tx
    )

    return rng, env, env_params, train_state


# %%
def make_train(config):

    # -------
    # env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    # env = LogWrapper(env)
    # env = ClipAction(env)
    # env = VecEnv(env)
    # if config["NORMALIZE_ENV"]:
    #     # env = NormalizeVecObservation(env)
    #     # env = NormalizeVecReward(env, config["GAMMA"])
    #     pass
    # -------
    env, env_params = init_env(config)

    @partial(jax.pmap, axis_name="devices")
    def train(rng: jax.Array, train_state: TrainState, init_hstate: jax.Array):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs_per_device)

        timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_rng)
        prev_action = jnp.zeros(config.num_envs_per_device, env.action_dim)
        prev_reward = jnp.zeros(config.num_envs_per_device)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++YOU-STOPPED-HERE+++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


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
    ent_coef: float = 0.01
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


@pyrallis.wrap()
def train(config: TrainConfig):
    # logging to wandb
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )

    rng, env, env_params, init_hstate, train_state = make_states(config)
    # replicating args across devices
    rng = jax.random.split(rng, num=jax.local_device_count())
    train_state = replicate(train_state, jax.local_devices())
    init_hstate = replicate(init_hstate, jax.local_devices())

    print("Compiling...")
    t = time.time()
    train_fn = make_train(env, env_params, config)
    train_fn = train_fn.lower(rng, train_state, init_hstate).compile()
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    print("Training...")
    t = time.time()
    train_info = jax.block_until_ready(train_fn(rng, train_state, init_hstate))
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s")

    print("Logging...")
    loss_info = unreplicate(train_info["loss_info"])

    total_transitions = 0
    for i in range(config.num_updates):
        # summing total transitions per update from all devices
        total_transitions += (
            config.num_steps * config.num_envs_per_device * jax.local_device_count()
        )
        info = jtu.tree_map(lambda x: x[i].item(), loss_info)
        info["transitions"] = total_transitions
        wandb.log(info)

    run.summary["training_time"] = elapsed_time
    run.summary["steps_per_second"] = (
        config.total_timesteps_per_device * jax.local_device_count()
    ) / elapsed_time

    print("Final return: ", float(loss_info["eval/returns"][-1]))
    run.finish()


if __name__ == "__main__":
    train()

    # rng = jax.random.PRNGKey(30)
    # train_jit = jax.jit(make_train(config))
    # out = train_jit(rng)

# config = {
#     # "LR": 3e-4,

#     # "NUM_ENVS": 2048,
#     "NUM_STEPS": 10,
#     "TOTAL_TIMESTEPS": 5e7,
#     "UPDATE_EPOCHS": 4,
#     "NUM_MINIBATCHES": 32,

#     "GAMMA": 0.99,
#     "GAE_LAMBDA": 0.95,
#     "CLIP_EPS": 0.2,
#     "ENT_COEF": 0.0,
#     "VF_COEF": 0.5,
#     "MAX_GRAD_NORM": 0.5,
#     "ACTIVATION": "tanh",
#     "ENV_NAME": "classic",
#     "ANNEAL_LR": False,
#     "NORMALIZE_ENV": True,
#     "DEBUG": True,
# }
# num_devices = jax.local_device_count()
# # splitting computation across all available devices
# config["NUM_ENVS_PER_DEVICE"] = config["NUM_ENVS"] // num_devices
# config["TOTAL_TIMESTEPS_PER_DEVICE"] = config["TOTAL_TIMESTEPS"] // num_devices
# config["EVAL_EPISODES_PER_DEVICE"] = config["EVAL_EPISODES"] // num_devices
# assert config["NUM_ENVS"] % num_devices == 0
# config["NUM_UPDATES"] = (
#     config["TOTAL_TIMESTEPS_PER_DEVICE"]
#     // config["NUM_STEPS"]
#     // config["NUM_ENVS_PER_DEVICE"]
# )
# print(f"Num devices: {num_devices}, Num updates: {config["NUM_UPDATES"]}")
