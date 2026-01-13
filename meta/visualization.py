from __future__ import annotations

from typing import Optional, Iterable
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .instance import Instance
from .solution import Solution


def plot_solution(
    inst: Instance,
    sol: Solution,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
    draw_coverage: bool = False,
    max_circles: int = 300,
    draw_comm_edges: bool = False,
    max_edges: int = 2000,
):
    """
    Visualize an instance + a solution.
    - Targets: small dots
    - Sensors: highlighted dots
    - Sink: star

    Options:
    - draw_coverage: draws circles of radius Rcapt around sensors (can be slow)
    - draw_comm_edges: draws some communication edges between selected sensors (can be slow)
    """
    xs = [p[0] for p in inst.targets]
    ys = [p[1] for p in inst.targets]

    fig, ax = plt.subplots()
    ax.scatter(xs, ys, s=8)  # targets

    if sol.sensors:
        sx = [inst.targets[i][0] for i in sol.sensors]
        sy = [inst.targets[i][1] for i in sol.sensors]
        ax.scatter(sx, sy, s=35)  # sensors

    # sink
    ax.scatter([inst.sink[0]], [inst.sink[1]], s=140, marker="*", edgecolors="black")

    if draw_coverage and sol.sensors:
        # limit circles for speed/readability
        sensors_list = list(sol.sensors)
        if len(sensors_list) > max_circles:
            sensors_list = sensors_list[:max_circles]
        for i in sensors_list:
            c = Circle(inst.targets[i], inst.rcapt, fill=False, linewidth=0.6)
            ax.add_patch(c)

    if draw_comm_edges and sol.sensors:
        # draw edges only among selected sensors (subset)
        edges_drawn = 0
        selected = set(sol.sensors)
        for u in sol.sensors:
            for v in inst.comm[u]:
                if v in selected and u < v:
                    ax.plot(
                        [inst.targets[u][0], inst.targets[v][0]],
                        [inst.targets[u][1], inst.targets[v][1]],
                        linewidth=0.3,
                    )
                    edges_drawn += 1
                    if edges_drawn >= max_edges:
                        break
            if edges_drawn >= max_edges:
                break

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
