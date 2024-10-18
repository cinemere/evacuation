import jax

from .base_wrappers import Wrapper


# gym and gymnasium style reset (on the same step with termination)
class GymAutoResetWrapper(Wrapper):
    def __auto_reset(self, params, timestep):
        key, _ = jax.random.split(timestep.state.key)
        reset_timestep = self._env.reset(params, key)

        timestep = timestep.replace(
            state=reset_timestep.state,
            observation=reset_timestep.observation,
        )
        return timestep

    # TODO: add last_obs somewhere in the timestep? add extras like in Jumanji?
    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = jax.lax.cond(
            timestep.last(),
            lambda: self.__auto_reset(params, timestep),
            lambda: timestep,
        )
        return timestep
