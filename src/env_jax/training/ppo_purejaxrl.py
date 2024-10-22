# %%
from __future__ import annotations
import os 
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
import sys

# this will be default in new jax versions anyway
jax.config.update("jax_threefry_partitionable", True)



sys.path.append("/Users/Klepach/work/repo/tmp/evacuation/src")
from env_jax.env.wrappers.purejaxrl_wrappers import (
    LogWrapper,
    VecEnv,
    ClipAction,
)
from env_jax.env.wrappers.gym_wrappers import GymAutoResetWrapper

#     NormalizeVecObservation,
#     NormalizeVecReward,


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
# config = {
#     "VF_COEF": 0.5,
#     "MAX_GRAD_NORM": 0.5,
#     "ACTIVATION": "tanh",
#     "ENV_NAME": "classic",
#     "ANNEAL_LR": False,
#     "NORMALIZE_ENV": True,
#     "DEBUG": True,
# }

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
    # dir: jax.Array
    # info: jnp.ndarray
    # prev_action: jax.Array
    # prev_reward: jax.Array


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
        # unroll=16  TODO why we comment this?
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value


# %%
def ppo_update_networks(
    train_state: TrainState,
    transitions: Transition,
    # init_hstate: jax.Array,
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
        dist, value, _ = train_state.apply_fn(params, transitions.obs)
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
    num_consecutive_episodes: int = 1,
) -> RolloutStats:
    """Collect eval statistics."""
    def _cond_fn(carry):
        rng, stats, timestep = carry
        return jnp.less(stats.episodes, num_consecutive_episodes)

    def _body_fn(carry):
        rng, stats, timestep = carry

        rng, _rng = jax.random.split(rng)
        dist, _ = train_state.apply_fn(
            train_state.params, timestep.observation[None, None, ...]
        )
        action = dist.sample(seed=_rng).squeeze()
        timestep = env.step(env_params, timestep, action)

        stats = stats.replace(
            reward=stats.reward + timestep.reward,
            length=stats.length + 1,
            episodes=stats.episodes + timestep.last(),
        )
        carry = (rng, stats, timestep, action, timestep.reward)
        return carry

    timestep = env.reset(env_params, rng)
    init_carry = (rng, RolloutStats(), timestep)
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
        ),  # eps=1e-8
    )
    train_state = TrainState.create(
        apply_fn=network.apply, params=network_params, tx=tx
    )

    return rng, env, env_params, train_state


# %%
def make_train(config: TrainConfig):

    env, env_params = init_env(config)

    @partial(jax.pmap, axis_name="devices")
    def train(rng: jax.Array, train_state: TrainState):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs_per_device)

        timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_rng)
        # prev_action = jnp.zeros(config.num_envs_per_device, env.action_dim)
        # prev_reward = jnp.zeros(config.num_envs_per_device)

        # TRAIN LOOP
        def _update_step(runner_state, _):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                # rng, train_state, prev_timestep, prev_action, prev_reward, prev_hstate = runner_state
                rng, train_state, prev_timestep = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                prev_obs = prev_timestep.observation[:, None]  # TODO why add dim here??
                dist, value = train_state.apply_fn(train_state.params, prev_obs)
                action, log_prob = dist.sample_and_log_prob(seed=_rng)
                # squeeze seq_len where possible  # why??
                action, value, log_prob = (
                    action.squeeze(1),
                    value.squeeze(1),
                    log_prob.squeeze(1),
                )

                # STEP ENV
                timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(
                    env_params, prev_timestep, action
                )
                transition = Transition(
                    done=timestep.last(),
                    action=action,
                    value=value,
                    reward=timestep.reward,
                    log_prob=log_prob,
                    obs=prev_timestep.observation,
                )
                runner_state = (
                    rng,
                    train_state,
                    timestep
                )
                return runner_state, transition

            # transitions: [seq_len, batch_size, ...]
            runner_state, transitions = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            rng, train_state, timestep = runner_state
            # calculate value of the last step for bootstrapping
            _, last_val, _ = train_state.apply_fn(
                train_state.params,
                timestep.observation[:, None],
            )
            advantages, targets = calculate_gae(
                transitions, last_val.squeeze(1), config.gamma, config.gae_lambda
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch_info):
                    transitions, advantages, targets = batch_info
                    new_train_state, update_info = ppo_update_networks(
                        train_state=train_state,
                        transitions=transitions,
                        # init_hstate=init_hstate.squeeze(1),
                        advantages=advantages,
                        targets=targets,
                        clip_eps=config.clip_eps,
                        vf_coef=config.vf_coef,
                        ent_coef=config.ent_coef,
                    )
                    return new_train_state, update_info

                rng, train_state, transitions, advantages, targets = (
                    update_state
                )

                # MINIBATCHES PREPARATION
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config.num_envs_per_device)

                # [seq_len, batch_size, ...]
                batch = (init_hstate, transitions, advantages, targets)

                # [batch_size, seq_len, ...], as our model assumes
                batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                shuffled_batch = jtu.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # [num_minibatches, minibatch_size, ...]
                minibatches = jtu.tree_map(
                    lambda x: jnp.reshape(
                        x, (config.num_minibatches, -1) + x.shape[1:]
                    ),
                    shuffled_batch,
                )
                train_state, update_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                update_state = (
                    rng,
                    train_state,
                    init_hstate,
                    transitions,
                    advantages,
                    targets,
                )
                return update_state, update_info

            update_state = (
                rng,
                train_state,
                transitions,
                advantages,
                targets,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )

            # averaging over minibatches then over epochs
            loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

            rng, train_state = update_state[:2]
            # EVALUATE AGENT
            rng, _rng = jax.random.split(rng)
            eval_rng = jax.random.split(_rng, num=config.eval_episodes_per_device)

            # vmap only on rngs
            eval_stats = jax.vmap(rollout, in_axes=(0, None, None, None, None))(
                eval_rng,
                env,
                env_params,
                train_state,
                1,
            )
            eval_stats = jax.lax.pmean(eval_stats, axis_name="devices")
            loss_info.update(
                {
                    "eval/returns": eval_stats.reward.mean(0),
                    "eval/lengths": eval_stats.length.mean(0),
                    "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
                }
            )
            runner_state = (rng, train_state, timestep)
            return runner_state, loss_info

        runner_state = (rng, train_state, timestep)
        runner_state, loss_info = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )
        return {"runner_state": runner_state, "loss_info": loss_info}

    return train


@pyrallis.wrap()
def train(config: TrainConfig):
    os.environ["WANDB_DISABLED"] = "true"
    # logging to wandb
    run = wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )

    rng, env, env_params,train_state = make_states(config)
    
    # replicating args across devices
    rng = jax.random.split(rng, num=jax.local_device_count())
    train_state = replicate(train_state, jax.local_devices())

    print("Compiling...")
    t = time.time()
    train_fn = make_train(env, env_params, config)
    train_fn = train_fn.lower(rng, train_state).compile()
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    print("Training...")
    t = time.time()
    train_info = jax.block_until_ready(train_fn(rng, train_state))
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


