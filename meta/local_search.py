
from __future__ import annotations

import random
from typing import List

from .instance import Instance
from .solution import Solution, covered_targets, is_feasible, repair_connectivity


def prune_descent(inst: Instance, sol: Solution) -> Solution:
    """
    Descent neighborhood: remove one sensor if feasibility stays true.
    First improvement strategy.
    """
    sol = repair_connectivity(inst, sol)
    improved = True

    while improved:
        improved = False
        for s in list(sol.sensors):
            trial = sol.copy()
            trial.sensors.remove(s)

            # quick coverage filter
            if len(covered_targets(inst, trial)) != inst.n:
                continue

            trial = repair_connectivity(inst, trial)
            if is_feasible(inst, trial) and trial.size() < sol.size():
                sol = trial
                improved = True
                break

    return sol


def swap_descent(inst: Instance, sol: Solution, rng: random.Random, max_trials: int = 500) -> Solution:
    """
    Descent neighborhood: 1-1 swaps (remove one, add one).
    Random sampling for speed; first improvement.
    """
    sol = repair_connectivity(inst, sol)
    if not is_feasible(inst, sol):
        return sol

    sensors_list = list(sol.sensors)
    if not sensors_list:
        return sol

    trials = 0
    while trials < max_trials:
        trials += 1
        out = rng.choice(sensors_list)
        inn = rng.randrange(inst.n)
        if inn in sol.sensors:
            continue

        trial = sol.copy()
        trial.sensors.remove(out)
        trial.sensors.add(inn)

        if len(covered_targets(inst, trial)) != inst.n:
            continue

        trial = repair_connectivity(inst, trial)
        if is_feasible(inst, trial) and trial.size() < sol.size():
            sol = trial
            sensors_list = list(sol.sensors)
            trials = 0  # restart after improvement

    return sol




def local_search_vnd(inst: Instance, sol: Solution, rng: random.Random) -> Solution:
    """
    Simple VND-like local search:
      1) prune descent
      2) swap descent
      3) prune descent again
    """
    sol = prune_descent(inst, sol)
    sol = swap_descent(inst, sol, rng=rng)
    sol = prune_descent(inst, sol)
    return sol
