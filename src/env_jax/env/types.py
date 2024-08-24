from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar, Union

import jax
import jax.numpy as jnp
from flax import struct
from typing_extensions import TypeAlias


class AgentState(struct.PyTreeNode):    
    # how to change enslaving degree ???
    # enslaving_degree: jax.Array = jnp.asarray(0, dtype=jnp.uint8)
    enslaving_degree: jax.Array
    position: jax.Array = struct.field(default_factory=lambda: jnp.asarray((0, 0)))
    direction: jax.Array = struct.field(default_factory=lambda: jnp.asarray((0, 0)))

class PedestriansState(struct.PyTreeNode):
    # how to read number of pedestrians from params???
    # num: jax.Array = jnp.asarray(10, dtype=jnp.uint8)
    positions: jax.Array
    directions: jax.Array
    statuses: jax.Array

# class AreaState(struct.PyTreeNode):
# 	...

# class EnvCarry(struct.PyTreeNode):
# 	# what is it I don't understand :(
# 	...

IntOrArray: TypeAlias = Union[int, jax.Array]
EnvCarryT = TypeVar("EnvCarryT")

class State(struct.PyTreeNode, Generic[EnvCarryT]):
    key: jax.Array
    step_num: jax.Array
    pedestrians: PedestriansState
    agent: AgentState
    # state_encoding: jax.Array
    # carry: EnvCarryT

class StepType(jnp.uint8):
    FIRST: jax.Array = jnp.asarray(0, dtype=jnp.uint8)
    MID: jax.Array = jnp.asarray(1, dtype=jnp.uint8)
    LAST: jax.Array = jnp.asarray(2, dtype=jnp.uint8)

class TimeStep(struct.PyTreeNode, Generic[EnvCarryT]):
    state: State[EnvCarryT]

    step_type: StepType
    reward: jax.Array
    discount: jax.Array
    observation: jax.Array | dict[str, jax.Array]

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST