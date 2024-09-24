import jax.numpy as jnp
from flax import struct

EXIT = jnp.asarray((0.0, -1.0), dtype=jnp.float32)
"Exit position."

PEDESTRIANS_INIT_POSITIONS = jnp.asarray((-1.0, 1.0), dtype=jnp.float32)
"X and Y low and high for uniform distribution."

class SwitchDistance(struct.PyTreeNode):
    to_leader: float = struct.field(pytree_node=False, default=0.2)
    "Radius of catch by leader."

    to_other_pedestrian: float = struct.field(pytree_node=False, default=0.1)
    "Neighbouring particles in viscek model."

    to_exit: float = struct.field(pytree_node=False, default=0.4)
    "Radius of exit zone."

    to_escape: float = struct.field(pytree_node=False, default=0.02)
    "Radius of escape zone."


class Status(struct.PyTreeNode):
    VISCEK: int = struct.field(pytree_node=False, default=0)
    "Pedestrian under Viscek rules."

    FOLLOWER: int = struct.field(pytree_node=False, default=1)
    "Follower of the leader particle (agent)."

    EXITING: int = struct.field(pytree_node=False, default=2)
    "Pedestrian in exit zone."

    ESCAPED: int = struct.field(pytree_node=False, default=3)
    "Evacuated pedestrian."
