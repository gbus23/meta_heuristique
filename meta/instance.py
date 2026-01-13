
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Set

Point = Tuple[float, float]
Idx = int


def dist2(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


@dataclass(frozen=True)
class Instance:
    """
    Candidate sensor locations are exactly the targets.
    """
    targets: List[Point]
    sink: Point
    rcapt: float
    rcom: float

    # Precomputed neighborhoods
    cover: List[Set[Idx]]     # cover[i] = set of targets covered if sensor placed at i
    comm: List[List[Idx]]     # comm[i]  = list of candidates within Rcom of i
    sink_comm: List[Idx]      # candidates within Rcom of sink

    @property
    def n(self) -> int:
        return len(self.targets)

    @staticmethod
    def build(targets: List[Point], sink: Point, rcapt: float, rcom: float) -> "Instance":
        n = len(targets)
        rc2 = rcapt * rcapt
        rcom2 = rcom * rcom

        cover: List[Set[Idx]] = [set() for _ in range(n)]
        comm: List[List[Idx]] = [[] for _ in range(n)]
        sink_comm: List[Idx] = []

        # Coverage
        for i in range(n):
            pi = targets[i]
            s = set()
            for j in range(n):
                if dist2(pi, targets[j]) <= rc2 + 1e-12:
                    s.add(j)
            cover[i] = s

        # Communication graph
        for i in range(n):
            pi = targets[i]
            neigh: List[Idx] = []
            for j in range(n):
                if i == j:
                    continue
                if dist2(pi, targets[j]) <= rcom2 + 1e-12:
                    neigh.append(j)
            comm[i] = neigh

        # Sink adjacency
        for i in range(n):
            if dist2(sink, targets[i]) <= rcom2 + 1e-12:
                sink_comm.append(i)

        return Instance(
            targets=targets,
            sink=sink,
            rcapt=rcapt,
            rcom=rcom,
            cover=cover,
            comm=comm,
            sink_comm=sink_comm,
        )
