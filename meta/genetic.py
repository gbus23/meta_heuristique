from __future__ import annotations

import random
import time

from .instance import Instance
from .solution import Solution, is_feasible, repair_connectivity
from .constructive import randomized_greedy_construct
from .local_search import local_search_vnd


def crossover_1point(p1: Solution, p2: Solution, rng: random.Random, n_targets: int) -> Solution:
    """Croisement 1-point : combine partie gauche de p1 et droite de p2."""
    cut = rng.randint(0, n_targets - 1)
    new_sensors = {s for s in p1.sensors if s < cut} | {s for s in p2.sensors if s >= cut}
    return Solution(new_sensors)


def mutate(sol: Solution, n_targets: int, rng: random.Random):
    """Mutation : ajoute ou supprime un capteur aléatoirement."""
    if rng.random() < 0.5:
        sol.sensors.add(rng.randint(0, n_targets - 1))
    else:
        if sol.sensors:
            sol.sensors.remove(rng.choice(list(sol.sensors)))


def genetic_algorithm(inst: Instance, time_limit_s: float, pop_size: int = 20, seed: int = 0) -> Solution:
    """Algorithme génétique mémétique."""
    rng = random.Random(seed)
    start_time = time.time()
    
    population = []
    for _ in range(pop_size):
        sol = randomized_greedy_construct(inst, rng, alpha=0.3)
        population.append(sol)
    
    best_sol = min(population, key=lambda s: s.size())
    
    while time.time() - start_time < time_limit_s:
        new_pop = []
        
        while len(new_pop) < pop_size:
            parent1 = min(rng.sample(population, k=3), key=lambda s: s.size())
            parent2 = min(rng.sample(population, k=3), key=lambda s: s.size())
            
            child = crossover_1point(parent1, parent2, rng, inst.n)
            mutate(child, inst.n, rng)
            child = repair_connectivity(inst, child)
            child = local_search_vnd(inst, child, rng)
            
            if is_feasible(inst, child):
                new_pop.append(child)
                if child.size() < best_sol.size():
                    best_sol = child
        
        population = new_pop

    return best_sol
