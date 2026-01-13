
from __future__ import annotations

from dataclasses import dataclass
from typing import Set, Dict, Optional, List
from collections import deque

from .instance import Instance, Idx


@dataclass
class Solution:
    sensors: Set[Idx]

    def copy(self) -> "Solution":
        return Solution(set(self.sensors))

    def size(self) -> int:
        return len(self.sensors)


def covered_targets(inst: Instance, sol: Solution) -> Set[Idx]:
    cov: Set[Idx] = set()
    for s in sol.sensors:
        cov |= inst.cover[s]
    return cov


def connected_to_sink(inst: Instance, sol: Solution) -> Set[Idx]:
    if not sol.sensors:
        return set()

    start = [u for u in inst.sink_comm if u in sol.sensors]
    if not start:
        return set()

    visited: Set[Idx] = set(start)
    q = deque(start)
    while q:
        u = q.popleft()
        for v in inst.comm[u]:
            if v in sol.sensors and v not in visited:
                visited.add(v)
                q.append(v)
    return visited


def is_feasible(inst: Instance, sol: Solution) -> bool:
    if len(covered_targets(inst, sol)) != inst.n:
        return False
    if len(connected_to_sink(inst, sol)) != sol.size():
        return False
    return True


def shortest_path_in_comm_graph(inst: Instance, src: Idx, goals: Set[Idx]) -> Optional[List[Idx]]:
    """
    Shortest path in the FULL communication graph (nodes = candidates),
    independent of current selection.
    Returns [src, ..., goal] or None.
    """
    if src in goals:
        return [src]

    parent: Dict[Idx, Optional[Idx]] = {src: None}
    q = deque([src])

    while q:
        u = q.popleft()
        for v in inst.comm[u]:
            if v not in parent:
                parent[v] = u
                if v in goals:
                    # Reconstruct path
                    path = [v]
                    cur = u
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path
                q.append(v)

    return None


def repair_connectivity(inst: Instance, sol: Solution) -> Solution:
    """
    Repair heuristic: add relay sensors so that all selected sensors connect to the sink component.
    Assumes coverage is handled elsewhere (constructive / move filters).
    """
    sol = sol.copy()
    if not sol.sensors:
        return sol

    conn = connected_to_sink(inst, sol)
    if not conn:
        # Seed with a sink neighbor if possible
        if inst.sink_comm:
            sol.sensors.add(inst.sink_comm[0])
        else:
            return sol  # no possible sink adjacency

    while True:
        conn = connected_to_sink(inst, sol)
        if len(conn) == sol.size():
            return sol

        anchors = set(conn) if conn else set(inst.sink_comm)
        disconnected = [s for s in sol.sensors if s not in conn]
        if not disconnected or not anchors:
            return sol

        s = disconnected[0]
        path = shortest_path_in_comm_graph(inst, s, anchors)
        if path is None:
            return sol
        sol.sensors.update(path)
