import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..core.constants import EXIT, Status, SwitchDistance
from ..types import EnvCarryT, TimeStep
from ..params import EnvParamsT

def render_plot(params: EnvParamsT, timestep: TimeStep) -> np.ndarray | str:

    fig, ax = plt.subplots(figsize=(5, 5))

    exit_coordinates = EXIT
    agent_coordinates = timestep.state.agent.position

    # Draw exiting zone
    exiting_zone = mpatches.Wedge(
        exit_coordinates, SwitchDistance.to_exit, 0, 180, alpha=0.2, color="green"
    )
    ax.add_patch(exiting_zone)

    # Draw escaping zone
    escaping_zone = mpatches.Wedge(
        exit_coordinates, SwitchDistance.to_escape, 0, 180, color="white"
    )
    ax.add_patch(escaping_zone)

    # Draw exit
    ax.plot(exit_coordinates[0], exit_coordinates[1], marker="X", color="green")

    # Draw following zone
    following_zone = mpatches.Wedge(
        agent_coordinates, SwitchDistance.to_leader, 0, 360, alpha=0.1, color="blue"
    )
    ax.add_patch(following_zone)

    from itertools import cycle

    colors = cycle(mpl.colors.BASE_COLORS)

    # Draw pedestrians
    positions = timestep.state.pedestrians.positions.__array__()
    directions = timestep.state.pedestrians.directions.__array__()
    for status in [Status.VISCEK, Status.FOLLOWER, Status.EXITING, Status.ESCAPED]:
        selected_pedestrians = timestep.state.pedestrians.statuses == status
        color = next(colors)
        ax.plot(
            positions[selected_pedestrians, 0],
            positions[selected_pedestrians, 1],
            lw=0,
            marker=".",
            color=color,
        )
    # pedestrians directions
    xy0 = positions
    xy1 = xy0 + directions
    ax.plot([xy0[:, 0], xy1[:, 0]], [xy0[:, 1], xy1[:, 1]], color="black")

    # # Draw agent
    ax.plot(agent_coordinates[0], agent_coordinates[1], marker="+", color="red")

    plt.xlim([-1.1 * params.width, 1.1 * params.width])
    plt.ylim([-1.1 * params.height, 1.1 * params.height])
    plt.xticks([])
    plt.yticks([])
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

    # plt.title(f"{self.experiment_name}. Timesteps: {self.time.now}")

    plt.tight_layout()
    # if not os.path.exists(self.path_png): os.makedirs(self.path_png)
    # filename = os.path.join(self.path_png, f'{self.experiment_name}_{self.time.now}.png')
    # plt.savefig(filename)
    plt.show()
    # log.info(f"Env is rendered and pnd image is saved to {filename}")
