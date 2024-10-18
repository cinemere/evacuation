# %%
# import sys; sys.path.append("/home/cinemere/work/repo/evacuation/src")
import sys; sys.path.append("/Users/Klepach/work/repo/tmp/evacuation/src")

import jax
import jax.numpy as jnp
import numpy as np
import dataclasses

from env_jax.env.env import Environment, EnvParams

from env_jax.env.wrappers import (
    RelativePosition,
    PedestriansStatusesCat,
    PedestriansStatusesOhe,
    MatrixObsOheStates,
    GravityEncoding,
)

# %%

key = jax.random.key(0)
reset_key, step_key = jax.random.split(key)

env = Environment()
env_params = EnvParams()
env_params = env.default_params(**dataclasses.asdict(env_params))

# %%
rng, _rng = jax.random.split(key)
reset_rng = jax.random.split(_rng, 10)

timestep = jax.vmap(env.reset, in_axes=(None, 0))(env_params, reset_rng)


# %% WRAP

env = GravityEncoding(env)
# env = RelativePosition(env=env)
# # env = PedestriansStatusesOhe(env=env)
# env = MatrixObsOheStates(env=env)
# %% RESET

# timestep = jax.jit(env.reset)(env_params, reset_key)
timestep = env.reset(env_params, reset_key)

# %%
# action = jnp.array(np.array([0.0, 1.0]))
action = jnp.array(np.array([[0.0, 1.0]] * 10))
action
# %%
# timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(env_params, prev_timestep, action)

timestep = jax.jit(jax.vmap(env.step, in_axes=(None, 0, 0)))(env_params, timestep, action)

# %%
# Debugging prints

# Ensure action is a valid JAX type
if isinstance(action, str):
    raise ValueError("Action should not be a string")

# %%
timestep = jax.jit(env.step)(env_params, timestep, action=action)
# timestep = env.step(env_params, timestep, action=action)
# %%
timestep.observation
# %%
timestep

# # %%
# from env_jax.env.render.anim import Animator

# animator = Animator()
# animator.update_memory(timestep.state)
# # %%

# for i in range(50):
#     action = jnp.array(np.array([0.0, 1.0]))
#     timestep = jax.jit(env.step)(env_params, timestep, action=action)
#     # timestep = env.step(env_params, timestep, action=action)
#     # timestep = env.step(env_params, timestep, action=action, key=step_key)
#     animator.update_memory(timestep.state)

# # %% PLOTTING THE CURRENT STATE

# env.render(env_params, timestep)
# # %%
# animator.anim(env_params)
# # # %%
# # selected_pedestrians = animator.memory.pedestrians_statuses == 1
# # animator.memory.pedestrians_positions[0][selected_pedestrians, 0]
# # animator.memory.pedestrians_positions[0][selected_pedestrians, 1]
# # # %%
# # animator.memory.pedestrians_statuses
# # # %%
# # for i in range(500):
# #     action = jnp.array(np.array([0.0, 1.0]))
# #     timestep = jax.jit(env.step)(env_params, timestep, action=action)

# # # %%
# # animator.memory.pedestrians_positions
# # # %%
# # import matplotlib.pyplot as plt
# # for i in range(10):
# #     plt.plot(np.vstack(animator.memory.pedestrians_statuses)[:, i])
# # # %%
# # np.vstack(animator.memory.pedestrians_statuses)[289:293, 0]

# # # %%
# # np.stack(animator.memory.pedestrians_positions)[289:293, 0, :]

# # # %%
# # np.stack(animator.memory.pedestrians_positions).shape
# # %%
# import matplotlib.pyplot as plt

# for PEDESTRIAN in range(10):
#     pos = np.array([x[PEDESTRIAN] for x in animator.memory.pedestrians_positions])
#     stats = np.array([str(x[PEDESTRIAN]) for x in animator.memory.pedestrians_statuses])
#     plt.plot(pos.T[0], pos.T[1], label=PEDESTRIAN, marker=".")
#     for i in range(50):
#         plt.text(pos.T[0, i], pos.T[1, i], stats[i])
# plt.legend()
# plt.show()
# # %%
# pos.T
# # %%
# PEDESTRIAN = 6
# pos = np.array([x[PEDESTRIAN] for x in animator.memory.pedestrians_positions])
# stats = np.array([x[PEDESTRIAN] for x in animator.memory.pedestrians_statuses])
# pos, stats
# # %%
# # %%
# from env_jax.env.core.constants import EXIT


# def get_normed_direction(positions):
#     vec2exit = EXIT - positions
#     len2exit = jnp.linalg.norm(vec2exit, axis=1)
#     vec_size = jnp.minimum(len2exit, 0.01) + 0.000001  # TODO add eps variable
#     normed_vec2exit = (vec2exit.T / len2exit * vec_size).T
#     return normed_vec2exit


# positions = jnp.array([[-0.11385489, -0.91152096], [0.1, -0.8], [-0.2, -0.2]])
# exiting = jnp.array([[True, True], [True, True], [False, False]])

# # %%
# vec2exit = EXIT - positions
# vec2exit
# # %%
# len2exit = jnp.linalg.norm(vec2exit, axis=1)  + 0.000001
# len2exit
# # %%
# vec_size = jnp.minimum(len2exit, 0.1)  # TODO add eps variable
# vec_size
# # %%
# # normed_vec2exit = (vec2exit.T / len2exit * vec_size).T
# directions = (vec2exit.T / len2exit * vec_size).T
# old_directions = jnp.array([[0, 0.01], [0, 0.01], [0, 0.01]])
# new_direction = jnp.where(exiting, directions, old_directions)
# # %%
# plt.plot(positions[:, 0], positions[:, 1], lw=0, marker=".")
# plt.plot(
#     positions[:, 0] + new_direction[:, 0],
#     positions[:, 1] + new_direction[:, 1],
#     lw=0,
#     marker=".",
# )
# plt.plot(EXIT[0], EXIT[1], lw=0, marker=".", color="green")
# # %%
# positions + new_direction
# # %%
# from env_jax.env.core.utils import update_statuses

# update_statuses(
#     statuses=jnp.array([0, 2, 2]),
#     agent_position=jnp.array([1.0, 1.0]),
#     pedestrian_positions=positions + new_direction,
# )

# # %% ----------------------------------------------------------------

# class A:
#     def _internal(self):
#         return "internal from A"

#     def func(self):
#         return self._internal()

#     def one_more_function(self):
#         return 'hello'

# class Wrapper(A):
#     def __init__(self, cl) -> None:
#         self._cl = cl

#     def change(self, output):
#         return NotImplementedError

#     def func(self):
#         output = self._cl.func()
#         return self.change(output)

# class B(Wrapper):
#     def __init__(self, cl) -> None:
#         super().__init__(cl)

#     def change(self, output):
#         return f"changed `{output}` (_internal) in B"

# a = A()
# a = B(a)
# a.func()
# a.one_more_function()
# # %%
# a._cl
# # %%

# %%
