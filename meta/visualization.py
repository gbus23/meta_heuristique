from __future__ import annotations

from typing import Optional, List, Tuple, Dict
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .instance import Instance, Idx
from .solution import Solution, connected_to_sink


def _build_connectivity_tree(inst: Instance, sol: Solution) -> Tuple[List[Tuple[Idx, Idx]], List[Idx]]:
    """Construit un arbre BFS sur les capteurs sélectionnés, enraciné aux capteurs connectés au sink."""
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
    """Affiche la solution : cibles, capteurs, sink, cercles de couverture, connectivité."""
    fig, ax = plt.subplots()

    xs = [p[0] for p in inst.targets]
    ys = [p[1] for p in inst.targets]
    ax.scatter(xs, ys, s=10, label="Targets")

    if sol.sensors:
        sx = [inst.targets[i][0] for i in sol.sensors]
        sy = [inst.targets[i][1] for i in sol.sensors]
        ax.scatter(sx, sy, s=45, label=f"Sensors (|S|={sol.size()})")

    ax.scatter([inst.sink[0]], [inst.sink[1]], s=200, marker="*", edgecolors="black", label="Sink")

    # Afficher tous les cercles de captation et de communication
    for i in sorted(sol.sensors):
        # Rayon de captation (cercle plein)
        c_capt = Circle(inst.targets[i], inst.rcapt, fill=False, linewidth=0.8, alpha=0.6, color='blue')
        ax.add_patch(c_capt)
        # Rayon de communication (pointillés noirs)
        c_com = Circle(inst.targets[i], inst.rcom, fill=False, linewidth=0.5, 
                      linestyle='--', alpha=0.6, color='black')
        ax.add_patch(c_com)

    edges, roots = _build_connectivity_tree(inst, sol)

    for (u, v) in edges:
        ax.plot([inst.targets[u][0], inst.targets[v][0]],
                [inst.targets[u][1], inst.targets[v][1]],
                color="black", linewidth=1.0, alpha=0.9, zorder=2)

    for r in roots:
        ax.plot([inst.sink[0], inst.targets[r][0]],
                [inst.sink[1], inst.targets[r][1]],
                color="black", linewidth=1.0, alpha=0.9, zorder=2)

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
