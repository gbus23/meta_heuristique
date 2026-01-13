
from __future__ import annotations

from typing import Set
import random

from .instance import Instance, Idx
from .solution import Solution, repair_connectivity


def greedy_construct(inst: Instance) -> Solution:
    """
    Deterministic greedy set-cover-like construction (coverage only),
    then repair connectivity.
    """
    uncovered = set(range(inst.n))
    sensors: Set[Idx] = set()

    while uncovered:
        best_i = None
        best_gain = -1
        for i in range(inst.n):
            g = len(inst.cover[i] & uncovered)
            if g > best_gain:
                best_gain = g
                best_i = i
        if best_i is None or best_gain <= 0:
            break
        sensors.add(best_i)
        uncovered -= inst.cover[best_i]

    sol = Solution(sensors)
    return repair_connectivity(inst, sol)


def randomized_greedy_construct(inst: Instance, rng: random.Random, alpha: float = 0.3) -> Solution:
    """
    GRASP randomized greedy construction using an RCL:
      threshold = g_max - alpha*(g_max - g_min)
      RCL = {i | gain(i) >= threshold}
      pick random from RCL
    """
    uncovered = set(range(inst.n))
    sensors: Set[Idx] = set()

    while uncovered:
        gains = []
        gmax = 0
        gmin = None
        for i in range(inst.n):
            g = len(inst.cover[i] & uncovered)
            if g > 0:
                gains.append((i, g))
                gmax = max(gmax, g)
                gmin = g if gmin is None else min(gmin, g)

        if not gains:
            break

        assert gmin is not None
        threshold = gmax - alpha * (gmax - gmin)
        rcl = [i for (i, g) in gains if g >= threshold]

        chosen = rng.choice(rcl) if rcl else max(gains, key=lambda t: t[1])[0]
        sensors.add(chosen)
        uncovered -= inst.cover[chosen]

    sol = Solution(sensors)
    return repair_connectivity(inst, sol)
