from __future__ import annotations

import random

from .instance import Instance
from .solution import Solution, covered_targets, is_feasible, repair_connectivity, connected_to_sink


def prune_descent(inst: Instance, sol: Solution) -> Solution:
    """Retire un capteur si la solution reste faisable. Première amélioration."""
    sol = repair_connectivity(inst, sol)
    improved = True

    while improved:
        improved = False
        
        # Calculer l'utilité de chaque capteur (couverture unique)
        sensor_utility = {}
        for s in sol.sensors:
            unique_coverage = set(inst.cover[s])
            for other in sol.sensors:
                if other != s:
                    unique_coverage -= inst.cover[other]
            sensor_utility[s] = len(unique_coverage)
        
        # Trier par utilité croissante (tester d'abord les moins utiles)
        sensors_sorted = sorted(sol.sensors, key=lambda s: sensor_utility.get(s, 0))
        
        for s in sensors_sorted:
            trial = sol.copy()
            trial.sensors.remove(s)

            # Vérifier couverture
            if len(covered_targets(inst, trial)) != inst.n:
                continue

            # Vérifier connectivité sans réparation d'abord
            if len(connected_to_sink(inst, trial)) == trial.size():
                # Connectivité OK sans réparation, on peut retirer
                if is_feasible(inst, trial):
                    sol = trial
                    improved = True
                    break
            else:
                # Connectivité cassée, réparer et vérifier si on gagne quand même
                trial_repaired = repair_connectivity(inst, trial)
                if is_feasible(inst, trial_repaired) and trial_repaired.size() < sol.size():
                    sol = trial_repaired
                    improved = True
                    break

    return sol


def swap_descent(inst: Instance, sol: Solution, rng: random.Random, max_trials: int = 1500) -> Solution:
    """Échange 1-1 : retire un capteur, en ajoute un autre. Échantillonnage aléatoire."""
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
            trials = 0

    return sol


def swap_2_1_descent(inst: Instance, sol: Solution, rng: random.Random, max_trials: int = 600) -> Solution:
    """Échange 2-1 : retire 2 capteurs, en ajoute 1."""
    sol = repair_connectivity(inst, sol)
    if not is_feasible(inst, sol) or sol.size() < 3:
        return sol

    sensors_list = list(sol.sensors)
    improved = True
    iterations = 0
    
    while improved and iterations < 20:
        improved = False
        iterations += 1
        trials = 0
        
        while trials < max_trials:
            trials += 1
            if len(sensors_list) < 2:
                break
            
            out1, out2 = rng.sample(sensors_list, 2)
            
            candidates_tried = set()
            for _ in range(min(100, inst.n * 2)):
                inn = rng.randrange(inst.n)
                if inn in sol.sensors or inn == out1 or inn == out2 or inn in candidates_tried:
                    continue
                candidates_tried.add(inn)
                
                trial = sol.copy()
                trial.sensors.remove(out1)
                trial.sensors.remove(out2)
                trial.sensors.add(inn)
                
                if len(covered_targets(inst, trial)) != inst.n:
                    continue
                
                trial = repair_connectivity(inst, trial)
                if is_feasible(inst, trial) and trial.size() < sol.size():
                    sol = trial
                    sensors_list = list(sol.sensors)
                    improved = True
                    break
            
            if improved:
                break
    
    return sol


def local_search_vnd(inst: Instance, sol: Solution, rng: random.Random) -> Solution:
    """VND : séquence prune -> swap -> prune -> swap_2_1 -> prune."""
    sol = prune_descent(inst, sol)
    sol = swap_descent(inst, sol, rng=rng)
    sol = prune_descent(inst, sol)
    sol = swap_2_1_descent(inst, sol, rng=rng)
    sol = prune_descent(inst, sol)
    return sol
