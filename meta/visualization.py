from __future__ import annotations

from typing import Optional, List, Tuple, Dict
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .instance import Instance, Idx
from .solution import Solution, connected_to_sink


def _build_connectivity_tree(inst: Instance, sol: Solution) -> Tuple[List[Tuple[Idx, Idx]], List[Idx]]:
    """
    Builds a BFS tree on the induced subgraph of selected sensors,
    rooted at sensors directly connected to the sink.

    Returns:
      - edges (u, v) of the BFS tree among sensors
      - roots (selected sensors that are within Rcom of the sink)
    """
    selected = set(sol.sensors)
    if not selected:
        return [], []

    roots = [u for u in inst.sink_comm if u in selected]
    if not roots:
        return [], []

    parent: Dict[Idx, Optional[Idx]] = {r: None for r in roots}
    q = deque(roots)
    edges: List[Tuple[Idx, Idx]] = []

    while q:
        u = q.popleft()
        for v in inst.comm[u]:
            if v in selected and v not in parent:
                parent[v] = u
                edges.append((u, v))
                q.append(v)

    return edges, roots


def plot_solution_auto(
    inst: Instance,
    sol: Solution,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    One-command plot:
      - targets
      - selected sensors
      - sink
      - coverage circles (radius Rcapt)
      - connectivity (BFS tree + sink->roots)

    Design choices for readability:
      - single color for connectivity edges
      - auto-limit the number of coverage circles shown
    """
    fig, ax = plt.subplots()

    # Targets
    xs = [p[0] for p in inst.targets]
    ys = [p[1] for p in inst.targets]
    ax.scatter(xs, ys, s=10, label="Targets")

    # Sensors
    if sol.sensors:
        sx = [inst.targets[i][0] for i in sol.sensors]
        sy = [inst.targets[i][1] for i in sol.sensors]
        ax.scatter(sx, sy, s=45, label=f"Sensors (|S|={sol.size()})")

    # Sink
    ax.scatter([inst.sink[0]], [inst.sink[1]], s=200, marker="*", edgecolors="black", label="Sink")

    # Coverage circles (auto-limit for readability)
    EDGE_COLOR = "black"
    EDGE_LW = 1.0
    EDGE_ALPHA = 0.9
    sensors_sorted = sorted(sol.sensors)
    max_circles = 120
    shown = sensors_sorted[:max_circles]
    for i in shown:
        c = Circle(inst.targets[i], inst.rcapt, fill=False, linewidth=0.8, alpha=0.6)
        ax.add_patch(c)

    if sol.size() > max_circles:
        ax.text(
            0.01, 0.01,
            f"Coverage circles shown: {max_circles}/{sol.size()} (auto-limited)",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom"
        )

    # Connectivity: BFS tree edges among sensors + sink->roots edges
    edges, roots = _build_connectivity_tree(inst, sol)

    # Use a single style (no rainbow)
    edge_lw = 1.0
    for (u, v) in edges:
            ax.plot(
            [inst.targets[u][0], inst.targets[v][0]],
            [inst.targets[u][1], inst.targets[v][1]],
            color=EDGE_COLOR,
            linewidth=EDGE_LW,
            alpha=EDGE_ALPHA,
            zorder=2,
    )

    # Draw sink -> roots to make connectivity to sink visible
    for r in roots:
        ax.plot(
            [inst.sink[0], inst.targets[r][0]],
            [inst.sink[1], inst.targets[r][1]],
            color=EDGE_COLOR,
            linewidth=EDGE_LW,
            alpha=EDGE_ALPHA,
            zorder=2,
    )

    # Compute disconnected count (for title)
    conn = connected_to_sink(inst, sol)
    disconnected = sol.size() - len(conn)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)

    if not title:
        title = f"R=({inst.rcapt},{inst.rcom}) | sensors={sol.size()} | disc={disconnected}"
    ax.set_title(title)

    ax.legend(loc="upper right", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
