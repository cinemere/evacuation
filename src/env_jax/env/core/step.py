from typing import Tuple
import jax.numpy as jnp
import jax
from functools import reduce

from ..types import AgentState, PedestriansState, IntOrArray
from .constants import Status, EXIT, SwitchDistance
from .utils import update_statuses, estimate_mean_direction_among_neighbours
from .rewards import estimate_intrinsic_reward, estimate_status_reward


def _is_wall_collision(
    pt: jnp.ndarray,  # point coordinates
    width: float,  # width of simulation area
    height: float,  # height of simulation area
) -> bool:
    # Check for wall collisions by comparing the point's coordinates to the boundaries
    left = pt[0] < -width
    right = pt[0] > width
    down = pt[1] < -height
    up = pt[1] > height

    # Return True if any collision condition is met
    return jnp.any(jnp.array([left, right, down, up]))


def agent_step(
    action: jax.Array,
    agent: AgentState,
    step_size: float,
    eps: float,
    width: float,  # width of simulation area
    height: float,  # height of simulation area
) -> Tuple[AgentState, dict]:
    """
    Perform agent step:
        1. Read & preprocess (scale or clip) action
        2. Check wall collision
        3. Return updated agent
    """
    print(f"running `agent_step` with {action=} {agent=}")

    # Scale action
    # action = jnp.array(action)
    action /= jnp.linalg.norm(action) + eps
    # jax.debug.print("{action}", action=action)

    # Estimate new position
    new_direction = step_size * action
    new_position = agent.position + new_direction

    # Check wall collision
    wall_collision = _is_wall_collision(new_position, width, height)
    agent_position = jax.lax.cond(
        wall_collision, lambda: agent.position, lambda: new_position
    )
    # new_position = agent.position if wall_collision else new_position

    # Write resulting position
    new_agent = agent.replace(
        direction=new_direction,
        # position=new_position,
        position=agent_position,
    )
    info = {"wall_collision": wall_collision}
    return new_agent, info


def get_new_exiting_directions(
    exiting: jax.Array,
    pedestrians: PedestriansState,
    step_size: float,
) -> PedestriansState:
    # Estimate direction to exit for exiting pedestrians
    # vec2exit = EXIT - pedestrians.positions[exiting]

    def get_normed_direction(positions):
        vec2exit = EXIT - positions
        len2exit = jnp.linalg.norm(vec2exit, axis=1)
        vec_size = jnp.minimum(len2exit, step_size) + 0.000001  # TODO add eps variable
        normed_vec2exit = (vec2exit.T / len2exit * vec_size).T
        return normed_vec2exit

    # vec2exit = jnp.where(exiting, EXIT-pedestrians.positions, 0)

    # # Normirate the direction
    # len2exit = jnp.linalg.norm(vec2exit, axis=1)
    # vec_size = jnp.minimum(len2exit, step_size)
    # normed_vec2exit = jnp.where(vec2exit.T / len2exit * vec_size).T

    new_directions = jnp.where(
        exiting, get_normed_direction(pedestrians.positions), pedestrians.directions
    )

    # Return the updated directons
    new_pedestrians = pedestrians.replace(
        # directions=pedestrians.directions.at[exiting].set(normed_vec2exit)
        directions=new_directions
    )
    return new_pedestrians


def get_new_direction_based_on_viscek_model(
    moving_paricles_cond: jax.Array,  # (efv) the particles which are influencing the directions
    updated_particles_cond: jax.Array,  # (fv) the particles to be updated
    pedestrians: PedestriansState,
    noise_coef: float,
    step_size: float,
    key: jax.Array,
) -> PedestriansState:
    # Get the directions of all moving particles
    # # efv_directions = pedestrians.directions[moving_paricles_cond]
    # # efv_directions = pedestrians.directions[moving_paricles_cond]
    # efv_directions = pedestrians.directions.at[moving_paricles_cond].get()
    # normed_efv_directions = (
    #     efv_directions.T / jnp.linalg.norm(efv_directions, axis=1)
    # ).T

    # # Estimate distances between all moving and all updated particles
    # dm = jnp.linalg.norm(
    #     # pedestrians.positions[updated_particles_cond][:, None]
    #     # - pedestrians.positions[moving_paricles_cond][None, :],
    #     pedestrians.positions.at[updated_particles_cond].get()[:, None]
    #     - pedestrians.positions.at[moving_paricles_cond].get()[None, :],
    #     axis=2,
    # )
    # intersection = jnp.where(dm < SwitchDistance.to_other_pedestrian, 1, 0)
    # n_intersections = jnp.maximum(1, intersection.sum(axis=1))

    # # Estimate the updated directions
    # fv_directions = estimate_mean_direction_among_neighbours(
    #     intersection, normed_efv_directions, n_intersections, noise_coef, key
    # )

    # # Record new directions of following and viscek pedestrians (based on Viscek model)
    # new_directions = pedestrians.directions.at[updated_particles_cond].set(
    #     fv_directions.T * step_size  # FIXME why step size is on directions????
    # )

    def get_new_directions():
        efv_directions = pedestrians.directions
        normed_efv_directions = (
            efv_directions.T / jnp.linalg.norm(efv_directions, axis=1)
        ).T  # TODO can we remove norm here?

        dm = jnp.linalg.norm(
            # pedestrians.positions[updated_particles_cond][:, None]
            # - pedestrians.positions[moving_paricles_cond][None, :],
            pedestrians.positions[:, None] - pedestrians.positions[None, :],
            axis=2,
        )
        # mask_dm = jnp.vecdot(
        #     updated_particles_cond[:, None, None],
        #     moving_paricles_cond[None, :, None],
        #     axis=2,
        # )
        mask_dm = jnp.vecdot(
            updated_particles_cond[:, None], moving_paricles_cond[None, :], axis=2
        )

        intersection = jnp.where(
            jnp.logical_and(dm < SwitchDistance.to_other_pedestrian, mask_dm), 1, 0
        )
        n_intersections = jnp.maximum(1, intersection.sum(axis=1))

        fv_directions = estimate_mean_direction_among_neighbours(
            intersection, normed_efv_directions, n_intersections, noise_coef, key
        )

        return fv_directions.T

    # Record new directions of following and viscek pedestrians (based on Viscek model)
    new_directions = jnp.where(
        updated_particles_cond, get_new_directions() * step_size, pedestrians.positions
    )

    # Update the pedestrians variable
    new_pedestrians = pedestrians.replace(directions=new_directions)
    return new_pedestrians


def get_new_direction_based_on_leaders_enslaving(
    pedestrians: PedestriansState, agent: AgentState, following_cond: jax.Array
) -> PedestriansState:
    # f_directions = pedestrians.directions[following_cond]
    # l_directions = agent.direction
    # new_f_directions = (
    #     agent.enslaving_degree * l_directions
    #     + (1.0 - agent.enslaving_degree) * f_directions
    # )
    # new_directions = pedestrians.directions.at[following_cond].set(new_f_directions)
    
    def get_new_directions(directions, leaders_direction, enslaving_degree):
        return (
        enslaving_degree * leaders_direction
        + (1.0 - enslaving_degree) * directions
    )
    new_directions = jnp.where(
        following_cond,
        get_new_directions(pedestrians.directions, agent.direction, agent.enslaving_degree),
        pedestrians.directions,
    )
    
    # Update the pedestrians variable
    new_pedestrians = pedestrians.replace(directions=new_directions)
    return new_pedestrians


# Handle wall collisions
def handle_pedestrians_wall_collisions(
    pedestrians: PedestriansState, width: float, heigth: float
) -> PedestriansState:
    print(f"{pedestrians.positions.dtype=} {type(pedestrians.positions)=}")
    print(f"{pedestrians.positions=} {width=} {heigth=}")
    x_clipped = jnp.clip(pedestrians.positions[:, 0], -width, width)
    y_clipped = jnp.clip(pedestrians.positions[:, 1], -heigth, heigth)
    clipped = jnp.stack((x_clipped, y_clipped), axis=1)
    # clipped = jnp.clip(pedestrians.positions, [-width, -heigth], [width, heigth])
    miss = pedestrians.positions - clipped

    new_positions = pedestrians.positions - 2 * miss
    new_directions = pedestrians.directions * jnp.where(miss != 0, -1, 1)

    # Update the pedestrians variable
    new_pedestrians = pedestrians.replace(
        positions=new_positions,
        directions=new_directions,
    )
    return new_pedestrians


def pedestrians_step(
    pedestrians: PedestriansState,
    agent: AgentState,
    now: int,
    step_size: float,
    noise_coef: float,
    width: float,  # width of simulation area
    height: float,  # height of simulation area
    num_pedestrians: int,
    init_reward_each_step: float,
    is_new_exiting_reward: bool,
    is_new_followers_reward: bool,
    key: jax.Array,
) -> Tuple[PedestriansState, bool, float, float]:
    print(f"running `pedestrians_step` with {pedestrians=} {agent=} {now=}")

    # Estimate statuses for each pedestrian
    escaped_1D = pedestrians.statuses == Status.ESCAPED
    exiting_1D = pedestrians.statuses == Status.EXITING
    following_1D = pedestrians.statuses == Status.FOLLOWER
    viscek_1D = pedestrians.statuses == Status.VISCEK

    def extend_1D_to_2D(array: jax.Array):
        return jnp.repeat(array, 2).reshape(-1, 2)

    escaped = extend_1D_to_2D(escaped_1D)
    exiting = extend_1D_to_2D(exiting_1D)
    following = extend_1D_to_2D(following_1D)
    viscek = extend_1D_to_2D(viscek_1D)

    efv = reduce(
        jnp.logical_or, (exiting, following, viscek)
    )  # all moving particles (efv -- exiting, following, viscek)
    fv = reduce(
        jnp.logical_or, (following, viscek)
    )  # following and viscek (fv) particles

    # Evacuated (escaped) pedestrians: record new directions and positions
    pedestrians_00 = pedestrians.replace(
        directions=jnp.where(escaped, 0, pedestrians.directions),
        positions=jnp.where(escaped, EXIT, pedestrians.positions),  # checked
        # directions=pedestrians.directions.at[escaped].set(0),
        # positions=pedestrians.positions.at[escaped].set(EXIT),
    )

    # Exiting pedestrians: record new directions (directing to exit)
    pedestrians_01 = jax.lax.cond(
        jnp.any(exiting),
        lambda: get_new_exiting_directions(exiting, pedestrians_00, step_size),
        lambda: pedestrians_00,
    )

    # Following and viscek pedestrians (based on viscek model):
    pedestrians_02 = get_new_direction_based_on_viscek_model(
        efv, fv, pedestrians_01, noise_coef, step_size, key
    )

    # Following pedestrians: record new directions (add leader's enslaving)
    pedestrians_03 = get_new_direction_based_on_leaders_enslaving(
        pedestrians_02, agent, following
    )

    # Record new positions of exiting, following and viscek pedestrians
    pedestrians_04 = pedestrians_03.replace(
        positions=jnp.where(efv, pedestrians.positions + pedestrians.directions, pedestrians.positions)
    )

    pedestrians_05 = handle_pedestrians_wall_collisions(pedestrians_04, width, height)

    ### YOU STOPPED HERE ###
    # think about updating only directions to not to create new pedestrians so often

    # Estimate pedestrians statuses, reward & update statuses
    old_statuses = pedestrians_05.statuses.copy()
    new_pedestrians_statuses = update_statuses(
        statuses=pedestrians_05.statuses,
        pedestrian_positions=pedestrians_05.positions,
        agent_position=agent.position,
        exit_position=EXIT,
    )
    reward_pedestrians = estimate_status_reward(
        old_statuses=old_statuses,
        new_statuses=new_pedestrians_statuses,
        timesteps=now,
        num_pedestrians=num_pedestrians,
        init_reward_each_step=init_reward_each_step,
        is_new_exiting_reward=is_new_exiting_reward,
        is_new_followers_reward=is_new_followers_reward,
    )
    intrinsic_reward = estimate_intrinsic_reward(
        pedestrians_positions=pedestrians_05.positions, exit_position=EXIT
    )

    # Update statuses
    pedestrians_06 = pedestrians_05.replace(
        statuses=new_pedestrians_statuses,
    )

    # Termination due to all pedestrians escaped
    termination = jnp.sum(new_pedestrians_statuses == Status.ESCAPED) == num_pedestrians

    return pedestrians_06, termination, reward_pedestrians, intrinsic_reward
