from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np


from ..env import Environment
from ..params import EnvParamsT
from ..core.constants import PEDESTRIANS_INIT_POSITIONS, Status
from ..types import State, TimeStep
from .base_wrappers import ObservationWrapper


def define_observation_type(): ...


class RelativePosition(ObservationWrapper):
    def __init__(self, env: Environment):
        super().__init__(env)

        assert jax.numpy.sqrt(2) == jax.numpy.sqrt(
            PEDESTRIANS_INIT_POSITIONS[0] ** 2 + PEDESTRIANS_INIT_POSITIONS[1] ** 2
        )

    def observation_shape(self, params: EnvParamsT) -> Tuple[int] | Dict[str, Any]:
        return {
            "agent_position": (2,),
            "pedestrians_position": (params.number_of_pedestrians, 2),
            "exit_position": (2,),
        }

    def transform_pos_abs2rel(self, pos_1: jax.Array, pos_2: jax.Array):
        return (pos_1 - pos_2) / jax.numpy.sqrt(2)

    def observation(self, params: EnvParamsT, timestep: TimeStep) -> jax.Array:
        print("RelativePosition observation")
        base_obs = timestep.observation

        rel_pedestrians_positions = self.transform_pos_abs2rel(
            base_obs["pedestrians_position"], base_obs["agent_position"]
        )
        rel_exit_position = self.transform_pos_abs2rel(
            base_obs["exit_position"], base_obs["agent_position"]
        )

        new_obs = {
            **base_obs,
            **{
                "pedestrians_position": rel_pedestrians_positions,
                "exit_position": rel_exit_position,
            },
        }
        return new_obs


class PedestriansStatusesCat(ObservationWrapper):
    def observation_shape(self, params: EnvParamsT) -> Tuple[int] | Dict[str, Any]:
        base_shape = self._env.observation_shape(params)
        assert type(base_shape) is dict
        extended_shape = {
            **base_shape,
            **{"pedestrians_statuses": (params.number_of_pedestrians,)},
        }
        return extended_shape

    def observation(self, params: EnvParamsT, timestep: TimeStep) -> jax.Array:
        print("PedestriansStatuseCat observation")
        statuses = timestep.state.pedestrians.statuses.astype(float) / 4
        base_obs = timestep.observation
        extended_obs = {
            **base_obs,
            **{"pedestrians_statuses": statuses},
        }
        return extended_obs


class PedestriansStatusesOhe(ObservationWrapper):
    def observation_shape(self, params: EnvParamsT) -> Tuple[int] | Dict[str, Any]:
        base_shape = self._env.observation_shape(params)
        assert type(base_shape) is dict
        extended_shape = {
            **base_shape,
            **{
                "pedestrians_statuses": (params.number_of_pedestrians, 4)
            },  # 4: viscek, follower, exiting, escaped
        }
        return extended_shape

    def observation(self, params: EnvParamsT, timestep: TimeStep) -> jax.Array:
        print("PedestriansStatusesOhe observation")
        statuses = jax.nn.one_hot(timestep.state.pedestrians.statuses, 4)
        base_obs = timestep.observation
        extended_obs = {
            **base_obs,
            **{"pedestrians_statuses": statuses},
        }
        return extended_obs


class MatrixObs(ObservationWrapper):
    def observation_shape(self, params: EnvParamsT) -> Tuple[int] | Dict[str, Any]:
        return (params.number_of_pedestrians + 2, 2)

    def observation(self, params: EnvParamsT, timestep: TimeStep) -> jax.Array:
        base_obs = timestep.observation
        pos = jnp.vstack(
            (
                base_obs["agent_position"],
                base_obs["exit_position"],
                base_obs["pedestrians_position"],
            )
        )
        return pos


class MatrixObsOheStates(PedestriansStatusesOhe):
    def observation_shape(self, params: EnvParamsT) -> Tuple[int] | Dict[str, Any]:
        return (params.number_of_pedestrians + 2, 2 + 4)  # 4 --> v,f,e,e

    def observation(self, params: EnvParamsT, timestep: TimeStep) -> jax.Array:
        base_obs = super().observation(params, timestep)
        pos = jnp.vstack(
            (
                base_obs["agent_position"],
                base_obs["exit_position"],
                base_obs["pedestrians_position"],
            )
        )
        stat_agent = jnp.asarray([1, 1, 0, 0], dtype=jnp.float32)
        stat_exit = jnp.asarray([0, 0, 1, 1], dtype=jnp.float32)
        stat = jnp.vstack((stat_agent, stat_exit, base_obs["pedestrians_statuses"]))
        vec = jnp.hstack((pos, stat)).astype(jnp.float32)
        return vec


class MatrixObsCatStates(PedestriansStatusesCat):
    def observation_shape(self, params: EnvParamsT) -> Tuple[int] | Dict[str, Any]:
        return (params.number_of_pedestrians + 2, 3)

    def observation(self, params: EnvParamsT, timestep: TimeStep) -> jax.Array:
        base_obs = super().observation(params, timestep)
        pos = jnp.vstack(
            (
                base_obs["agent_position"],
                base_obs["exit_position"],
                base_obs["pedestrians_position"],
            )
        )
        stat = jnp.hstack((jnp.asarray([0, 1], base_obs["pedestrians_statuses"])))
        vec = jnp.hstack((pos, jnp.expand_dims(stat, 1))).astype(jnp.float32)
        return vec


def grad_potential_pedestrians(
    agent_position: jnp.ndarray,
    pedestrians_positions: jnp.ndarray,
    status_viscek: jnp.ndarray,
    alpha: float,
    eps: float,
) -> jnp.ndarray:
    selected_positions = jnp.where(
        jnp.expand_dims(status_viscek, 1), pedestrians_positions, 0
    )
    R = jnp.expand_dims(agent_position, 0) - selected_positions

    def fn(R):
        norm = jnp.expand_dims(jnp.linalg.norm(R, axis=1), 1)
        grad = -alpha / (norm ** (alpha + 2) + eps) * R
        grad = grad.sum(axis=0)
        return grad

    grad = lax.cond(len(R) != 0, lambda: fn(R), lambda: jnp.zeros(2))
    return grad


def grad_potential_exit(
    agent_position: jnp.ndarray,
    num_followers: int,
    exit_position: jnp.ndarray,
    alpha: float,
    eps: float,
) -> jnp.ndarray:
    R = agent_position - exit_position
    norm = jnp.linalg.norm(R)
    grad = -alpha / (norm ** (alpha + 2) + eps) * R
    return grad * num_followers


class GravityEncoding(ObservationWrapper):
    def observation_shape(self, params: EnvParamsT) -> Tuple[int] | Dict[str, Any]:
        return {
            "agent_position": (2,),
            "grad_potential_pedestrians": (2,),
            "grad_potential_exit": (2,),
        }

    def observation(self, params: EnvParamsT, timestep: TimeStep) -> jax.Array:

        base_obs = timestep.observation

        pedestrians_positions = base_obs.pop("pedestrians_position")
        exit_position = base_obs.pop("exit_position")
        agent_position = base_obs["agent_position"]

        statuses = timestep.state.pedestrians.statuses
        num_followers = sum(statuses == Status.FOLLOWER).astype(jnp.float32)
        status_viscek = statuses == Status.VISCEK

        extended_obs = {
            **base_obs,
            **{
                "grad_potential_pedestrians": grad_potential_pedestrians(
                    agent_position=agent_position,
                    pedestrians_positions=pedestrians_positions,
                    status_viscek=status_viscek,
                    alpha=params.alpha,
                    eps=params.eps,
                ),
                "grad_potential_exit": grad_potential_exit(
                    agent_position=agent_position,
                    exit_position=exit_position,
                    num_followers=num_followers,
                    alpha=params.alpha,
                    eps=params.eps,
                ),
            },
        }
        return extended_obs


class FlattenObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(
            self._env.observation_shape, dict
        ), "FlattenObservation wrapper works only with `dict` observation "\
            f"space, however {type(self._env.observation_shape(None))=}."

    def observation_shape(self, params=None):
        shapes = self._env.observation_shape()
        return sum(np.prod(val) for val in shapes.values())

    def observation(self, params, timestep):
        base_obs = timestep.observation
        return jnp.concatenate([jnp.ravel(obs) for obs in base_obs.values()])
