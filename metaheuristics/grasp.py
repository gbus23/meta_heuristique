
from __future__ import annotations

import time
import random

from .instance import Instance
from .constructive import greedy_construct, randomized_greedy_construct
from .local_search import local_search_vnd
from .solution import is_feasible, Solution


def grasp(inst: Instance, time_limit_s: float = 2.0, alpha: float = 0.3, seed: int = 0) -> Solution:
    """
    GRASP:
      repeat until time limit:
        - randomized greedy construction (RCL)
        - local search (descent / VND)
      keep best solution.
    """
    rng = random.Random(seed)
    t0 = time.time()

    best = greedy_construct(inst)
    best = local_search_vnd(inst, best, rng=rng)

    while time.time() - t0 < time_limit_s:
        s = randomized_greedy_construct(inst, rng=rng, alpha=alpha)
        s = local_search_vnd(inst, s, rng=rng)
        if is_feasible(inst, s) and s.size() < best.size():
            best = s

    return best
