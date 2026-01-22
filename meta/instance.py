from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Set

Point = Tuple[float, float]
Idx = int


def dist2(a: Point, b: Point) -> float:
    """Distance au carré entre deux points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


@dataclass(frozen=True)
class Instance:
    """Instance du problème : positions des cibles, rayons, voisinages précalculés."""
    targets: List[Point]
    sink: Point
    rcapt: float
    rcom: float

    cover: List[Set[Idx]]     # cover[i] = cibles couvertes si capteur en i
    comm: List[List[Idx]]      # comm[i] = candidats à portée Rcom de i
    sink_comm: List[Idx]      # candidats à portée Rcom du sink

    @property
    def n(self) -> int:
        return len(self.targets)

    @staticmethod
    def build(targets: List[Point], sink: Point, rcapt: float, rcom: float) -> "Instance":
        """Construit une instance avec voisinages précalculés."""
        n = len(targets)
        rc2 = rcapt * rcapt
        rcom2 = rcom * rcom

        cover: List[Set[Idx]] = [set() for _ in range(n)]
        comm: List[List[Idx]] = [[] for _ in range(n)]
        sink_comm: List[Idx] = []

        for i in range(n):
            pi = targets[i]
            s = set()
            for j in range(n):
                if dist2(pi, targets[j]) <= rc2 + 1e-12:
                    s.add(j)
            cover[i] = s

        for i in range(n):
            pi = targets[i]
            neigh: List[Idx] = []
            for j in range(n):
                if i == j:
                    continue
                if dist2(pi, targets[j]) <= rcom2 + 1e-12:
                    neigh.append(j)
            comm[i] = neigh

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
