import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os

from ..core.constants import EXIT, Status, SwitchDistance
from ..types import EnvCarryT, TimeStep, State
from ..env import EnvParamsT


class AnimatorData:
    def __init__(self):
        self.pedestrians_positions = []
        self.pedestrians_statuses = []
        self.agent_position = []

    def update(self, state: State):
        self.pedestrians_positions.append(state.pedestrians.positions.__array__())
        self.pedestrians_statuses.append(state.pedestrians.statuses.__array__())
        self.agent_position.append(state.agent.position.__array__())


class Animator:
    def __init__(
        self,
        path_giff: str = "saved_data/giff",
        experiment_name: str = "tmp",
        number_of_episoded: int = 0,
    ):
        self._init_memory()
        self.path_giff = path_giff
        self.experiment_name = experiment_name
        self.number_of_episoded = number_of_episoded

    def _init_memory(self):
        self.memory = AnimatorData()

    def update_memory(self, state: State):
        self.memory.update(state)

    def anim(self, params: EnvParamsT):

        fig, ax = plt.subplots(figsize=(5, 5))

        # plt.title(f"{self.experiment_name}\nn_episodes = {self.time.n_episodes}")
        plt.hlines(
            [params.height, -params.height],
            -params.width,
            params.width,
            linestyle="--",
            color="grey",
        )
        plt.vlines(
            [params.width, -params.width],
            -params.height,
            params.height,
            linestyle="--",
            color="grey",
        )
        plt.xlim([-1.1 * params.width, 1.1 * params.width])
        plt.ylim([-1.1 * params.height, 1.1 * params.height])
        plt.xticks([])
        plt.yticks([])

        exit_coordinates = EXIT
        agent_coordinates = (
            self.memory.agent_position[0][0],
            self.memory.agent_position[0][1],
        )

        # Draw exiting zone
        exiting_zone = mpatches.Wedge(
            exit_coordinates, SwitchDistance.to_exit, 0, 180, alpha=0.2, color="green"
        )
        ax.add_patch(exiting_zone)

        # Draw following zone
        following_zone = mpatches.Wedge(
            agent_coordinates, SwitchDistance.to_leader, 0, 360, alpha=0.1, color="blue"
        )
        following_zone_plots = ax.add_patch(following_zone)

        # Draw escaping zone
        escaping_zone = mpatches.Wedge(
            exit_coordinates, SwitchDistance.to_escape, 0, 180, color="white"
        )
        ax.add_patch(escaping_zone)

        # Draw exit
        ax.plot(exit_coordinates[0], exit_coordinates[1], marker="X", color="green")

        from itertools import cycle

        colors = cycle(mpl.colors.BASE_COLORS)

        # Draw pedestrians
        pedestrian_position_plots = {}
        for status in [Status.VISCEK, Status.FOLLOWER, Status.EXITING, Status.ESCAPED]:
            selected_pedestrians = (
                self.memory.pedestrians_statuses[0] == status
            )
            color = next(colors)
            pedestrian_position_plots[status] = ax.plot(
                self.memory.pedestrians_positions[0][selected_pedestrians, 0],
                self.memory.pedestrians_positions[0][selected_pedestrians, 1],
                lw=0,
                marker=".",
                color=color,
            )[0]

        # Draw agent
        agent_position_plot = ax.plot(
            agent_coordinates[0], agent_coordinates[1], marker="+", color="red"
        )[0]

        def update(i):

            agent_coordinates = (
                self.memory.agent_position[i][0],
                self.memory.agent_position[i][1],
            )
            following_zone_plots.set_center(agent_coordinates)

            for status in [
                Status.VISCEK,
                Status.FOLLOWER,
                Status.EXITING,
                Status.ESCAPED,
            ]:
                selected_pedestrians = self.memory.pedestrians_statuses[i] == status
                pedestrian_position_plots[status].set_xdata(
                    self.memory.pedestrians_positions[i][selected_pedestrians, 0]
                )
                pedestrian_position_plots[status].set_ydata(
                    self.memory.pedestrians_positions[i][selected_pedestrians, 1]
                )

            # agent_position_plot.set_xdata(agent_coordinates[0])
            # agent_position_plot.set_ydata(agent_coordinates[1])
            agent_position_plot.set_data(agent_coordinates)

        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=len(self.memory.agent_position), interval=20
        )

        if not os.path.exists(self.path_giff):
            os.makedirs(self.path_giff)
        filename = os.path.join(
            self.path_giff, f"{self.experiment_name}_ep-{self.number_of_episoded}.gif"
        )
        ani.save(filename=filename, writer="pillow")
        # log.info(f"Env is rendered and gif animation is saved to {filename}")

        # if self.save_next_episode_anim:
        #     self.save_next_episode_anim = False
        #     self.draw = False

        # from IPython.display import HTML
        # HTML(ani.to_jshtml())