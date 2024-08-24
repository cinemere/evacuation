# env-jax

## installation notes

select the proper version of drivers here: https://anaconda.org/nvidia/cuda-nvcc

## env

```text
ENV
    PEDESTRIANS
        - number_of_pedestrians
        + positions
        + directions
        + statuses
    AGENT
        - enslaving degree
        - random_start: bool
        + position
        + direction
    AREA    
        REWARD
            - is_new_exiting_reward
            - is_new_followers_reward
            - is_termination_agent_wall_collision
            - init_reward_each_step
            * intrinsic_reward(ped_pos, exit_pos) -> float
            * stetus_reward(ped_pos, exit_pos) -> float
        - width
        - height
        - step_size
        - noise_coef
        - eps
        * pedestrians_step(pedestrians, agent) -> Pedestrians, bool, float, float
        * agent_step(pedestrians, agent) -> Agent, bool, float
        * _if_wall_collision(agent) -> bool
    TIME
        - max_timesteps CONST
        - n_episodes CHANGING
        - n_timesteps CHANGING
```