
from __future__ import annotations

import time
import random

from .instance import Instance
from .constructive import greedy_construct, randomized_greedy_construct
from .local_search import local_search_vnd
from .solution import covered_targets, is_feasible, repair_connectivity, Solution


def shake(inst: Instance, sol: Solution, rng: random.Random, k: int) -> Solution:
    """
    Vk: apply k random moves among remove/add/swap.
    Then repair connectivity.
    """
    s = sol.copy()
    for _ in range(k):
        r = rng.random()
        if r < 0.34:
            if s.sensors:
                s.sensors.remove(rng.choice(tuple(s.sensors)))
        elif r < 0.67:
            s.sensors.add(rng.randrange(inst.n))
        else:
            if s.sensors:
                s.sensors.remove(rng.choice(tuple(s.sensors)))
            s.sensors.add(rng.randrange(inst.n))

    return repair_connectivity(inst, s)


def vns(inst: Instance, time_limit_s: float = 2.0, kmax: int = 4, seed: int = 0) -> Solution:
    """
    VNS:
      - start from feasible solution
      - repeat until time limit:
          k = 1..kmax:
            x'  = shake(best, k)
            x'' = local_search(x')
            if improved: best = x''; k = 1 else k++
    """
    rng = random.Random(seed)
    t0 = time.time()

    best = greedy_construct(inst)
    best = local_search_vnd(inst, best, rng=rng)

    while time.time() - t0 < time_limit_s:
        k = 1
        while k <= kmax and time.time() - t0 < time_limit_s:
            x1 = shake(inst, best, rng=rng, k=k)

            # Strictly keep feasibility: if shake broke coverage, rebuild using randomized greedy
            if len(covered_targets(inst, x1)) != inst.n:
                x1 = randomized_greedy_construct(inst, rng=rng, alpha=0.3)

            x2 = local_search_vnd(inst, x1, rng=rng)

            if is_feasible(inst, x2) and x2.size() < best.size():
                best = x2
                k = 1
            else:
                k += 1

    return best
