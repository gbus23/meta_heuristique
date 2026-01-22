from __future__ import annotations

import time
import random
import math
from typing import Dict

from .instance import Instance
from .constructive import greedy_construct
from .local_search import local_search_vnd, swap_2_1_descent
from .solution import covered_targets, is_feasible, repair_connectivity, connected_to_sink, Solution


def compute_sensor_utility(inst: Instance, sol: Solution) -> Dict[int, float]:
    """Calcule l'utilité de chaque capteur (couverture unique + rôle connectivité)."""
    utility: Dict[int, float] = {}
    if not sol.sensors:
        return utility
    
    for s in sol.sensors:
        unique = set(inst.cover[s])
        for other in sol.sensors:
            if other != s:
                unique -= inst.cover[other]
        utility[s] = float(len(unique))
    
    connected = connected_to_sink(inst, sol)
    for s in sol.sensors:
        if s not in connected:
            continue
        test = sol.copy()
        test.sensors.remove(s)
        new_conn = connected_to_sink(inst, test)
        loss = len(connected) - len(new_conn)
        if loss > 1:
            utility[s] += 0.5 * loss
    
    return utility


def intensify_search(inst: Instance, sol: Solution, rng: random.Random, max_iter: int = 5) -> Solution:
    """Recherche locale intensive autour d'une bonne solution."""
    best = sol.copy()
    for _ in range(max_iter):
        candidate = local_search_vnd(inst, best, rng=rng)
        candidate = swap_2_1_descent(inst, candidate, rng=rng, max_trials=800)
        candidate = local_search_vnd(inst, candidate, rng=rng)
        if is_feasible(inst, candidate) and candidate.size() < best.size():
            best = candidate
        else:
            break
    return best


def _select_low_utility(sensors_list: list, utility: Dict[int, float], count: int, rng: random.Random) -> list:
    """Sélectionne les capteurs de faible utilité."""
    if not utility:
        return rng.sample(sensors_list, min(count, len(sensors_list)))
    sorted_sensors = sorted(sensors_list, key=lambda s: utility.get(s, 0.0))
    n = max(1, len(sorted_sensors) // (len(sensors_list) // count + 1))
    candidates = sorted_sensors[:n]
    if len(candidates) >= count:
        return rng.sample(candidates, count)
    return rng.sample(sensors_list, count)


def generate_neighbor(inst: Instance, sol: Solution, rng: random.Random, 
                     temp_norm: float = 1.0, aggressive: bool = False, stuck: bool = False) -> Solution:
    """Génère un voisin avec voisinages multi-niveaux guidés par l'utilité."""
    neighbor = sol.copy()
    utility = compute_sensor_utility(inst, sol)
    sensors_list = list(neighbor.sensors)
    
    # Niveau 4: reconstruction partielle si bloqué
    if stuck and neighbor.size() >= 6 and rng.random() < 0.3:
        n_remove = min(neighbor.size() // 2, 5)
        to_remove = _select_low_utility(sensors_list, utility, n_remove, rng)
        for s in to_remove:
            neighbor.sensors.remove(s)
        
        uncovered = set(range(inst.n)) - covered_targets(inst, neighbor)
        for _ in range(min(15, len(uncovered) * 2)):
            if not uncovered:
                break
            candidates = [(i, len(inst.cover[i] & uncovered)) 
                         for i in range(inst.n) 
                         if i not in neighbor.sensors and len(inst.cover[i] & uncovered) > 0]
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                chosen = rng.choice(candidates[:min(5, len(candidates))])[0]
                neighbor.sensors.add(chosen)
                uncovered -= inst.cover[chosen]
        return repair_connectivity(inst, neighbor)
    
    # Niveau 3: swaps agressifs (3-1, 2-1, 3-2, 4-2)
    if aggressive or (temp_norm > 0.6 and rng.random() < 0.4):
        move = rng.random()
        if move < 0.5 and neighbor.size() >= 4:
            to_remove = _select_low_utility(sensors_list, utility, 3, rng)
            for s in to_remove:
                neighbor.sensors.remove(s)
            neighbor.sensors.add(rng.randrange(inst.n))
        elif move < 0.7 and neighbor.size() >= 3:
            to_remove = _select_low_utility(sensors_list, utility, 2, rng)
            for s in to_remove:
                neighbor.sensors.remove(s)
            neighbor.sensors.add(rng.randrange(inst.n))
        elif move < 0.85 and neighbor.size() >= 5:
            to_remove = _select_low_utility(sensors_list, utility, 3, rng)
            for s in to_remove:
                neighbor.sensors.remove(s)
            neighbor.sensors.add(rng.randrange(inst.n))
            neighbor.sensors.add(rng.randrange(inst.n))
        elif move < 0.95 and neighbor.size() >= 6:
            to_remove = _select_low_utility(sensors_list, utility, 4, rng)
            for s in to_remove:
                neighbor.sensors.remove(s)
            neighbor.sensors.add(rng.randrange(inst.n))
            neighbor.sensors.add(rng.randrange(inst.n))
    else:
        # Niveaux 1-2: mouvements basiques
        if temp_norm > 0.4 and rng.random() < 0.3 and neighbor.size() >= 3:
            to_remove = _select_low_utility(sensors_list, utility, 2, rng)
            for s in to_remove:
                neighbor.sensors.remove(s)
            neighbor.sensors.add(rng.randrange(inst.n))
            return repair_connectivity(inst, neighbor)
        
        n_moves = rng.randint(1, 3) if temp_norm > 0.6 else (rng.randint(1, 2) if temp_norm > 0.3 else 1)
        for _ in range(n_moves):
            r = rng.random()
            if r < 0.4 and neighbor.sensors:
                if utility:
                    sorted_s = sorted(list(neighbor.sensors), key=lambda s: utility.get(s, 0.0))
                    to_remove = rng.choice(sorted_s[:max(1, int(len(sorted_s) * 0.4))])
                else:
                    to_remove = rng.choice(tuple(neighbor.sensors))
                neighbor.sensors.remove(to_remove)
            elif r < 0.7:
                neighbor.sensors.add(rng.randrange(inst.n))
            else:
                if neighbor.sensors:
                    if utility:
                        sorted_s = sorted(list(neighbor.sensors), key=lambda s: utility.get(s, 0.0))
                        to_remove = rng.choice(sorted_s[:max(1, int(len(sorted_s) * 0.4))])
                    else:
                        to_remove = rng.choice(tuple(neighbor.sensors))
                    neighbor.sensors.remove(to_remove)
                neighbor.sensors.add(rng.randrange(inst.n))
    
    return repair_connectivity(inst, neighbor)


def simulated_annealing(
    inst: Instance,
    time_limit_s: float = 2.0,
    initial_temp: float = None,
    cooling_rate: float = 0.985,  # Optimisé via grid search (était 0.99)
    min_temp: float = 0.001,       # Optimisé via grid search (était 0.01)
    seed: int = 0,
    cooling_factors: Dict[str, float] = None,
    acceptance_thresholds: Dict[str, float] = None
) -> Solution:
    """Recuit simulé optimisé pour le placement de capteurs."""
    rng = random.Random(seed)
    t0 = time.time()
    
    # Paramètres de refroidissement par défaut (optimisés via grid search)
    if cooling_factors is None:
        cooling_factors = {
            'high_temp_slow': 0.99,  # Optimisé: 0.99 (était 0.995)
            'high_temp_fast': 1.03,  # Optimisé: 1.03 (était 1.05)
            'mid_temp_fast': 1.08,   # Optimisé: 1.08 (était 1.1)
            'low_temp_fast': 1.15    # Optimisé: 1.15 (était 1.2)
        }
    if acceptance_thresholds is None:
        acceptance_thresholds = {
            'high_accept': 0.3,
            'mid_accept': 0.15,
            'low_accept': 0.1
        }
    
    current = greedy_construct(inst)
    current = local_search_vnd(inst, current, rng=rng)
    best = current.copy()
    
    # Estimation température initiale
    if initial_temp is None:
        n_samples = min(max(20, inst.n // 10), 30)
        deltas = []
        for _ in range(n_samples):
            n = generate_neighbor(inst, current, rng=rng)
            n = repair_connectivity(inst, n)
            if is_feasible(inst, n):
                d = abs(n.size() - current.size())
                if d > 0:
                    deltas.append(d)
        
        if deltas:
            avg = sum(deltas) / len(deltas)
            var = sum((d - avg) ** 2 for d in deltas) / len(deltas) if len(deltas) > 1 else 0
            std = math.sqrt(var) if var > 0 else avg
            base = -avg / math.log(0.7) if avg > 0 else 30.0
            initial_temp = base * (1.0 + (std / max(avg, 1.0)) * 0.3) * (1.0 + (inst.n / 100.0) * 0.2)
        else:
            initial_temp = 30.0 + (inst.n / 10.0) * 5.0
    
    temp = initial_temp
    iterations = 0
    accepted_count = 0
    last_improvement = 0
    iter_per_temp = max(20, min(inst.n // 8, 60))
    
    while time.time() - t0 < time_limit_s:
        iterations += 1
        temp_iter = 0
        
        while temp_iter < iter_per_temp and time.time() - t0 < time_limit_s:
            temp_iter += 1
            
            temp_norm = temp / initial_temp if initial_temp > 0 else 0.5
            aggressive = (iterations - last_improvement > 50) or (temp_norm > 0.5 and rng.random() < 0.3)
            stuck = (iterations - last_improvement > 100)
            neighbor = generate_neighbor(inst, current, rng=rng, temp_norm=temp_norm, aggressive=aggressive, stuck=stuck)
            
            # Réparation couverture si cassée
            if len(covered_targets(inst, neighbor)) != inst.n:
                uncovered = set(range(inst.n)) - covered_targets(inst, neighbor)
                for target in list(uncovered)[:5]:
                    best_s, best_g = None, 0
                    for i in range(inst.n):
                        if target in inst.cover[i]:
                            g = len(inst.cover[i] & uncovered)
                            if g > best_g:
                                best_g, best_s = g, i
                    if best_s is not None:
                        neighbor.sensors.add(best_s)
                        uncovered -= inst.cover[best_s]
                neighbor = repair_connectivity(inst, neighbor)
            
            # Recherche locale selon température
            if temp < initial_temp * 0.3:
                neighbor = local_search_vnd(inst, neighbor, rng=rng)
                for _ in range(2):
                    neighbor = swap_2_1_descent(inst, neighbor, rng=rng, max_trials=800)
                    neighbor = local_search_vnd(inst, neighbor, rng=rng)
            else:
                neighbor = local_search_vnd(inst, neighbor, rng=rng)
            
            if not is_feasible(inst, neighbor):
                continue
            
            delta = neighbor.size() - current.size()
            
            # Acceptation
            if delta < 0:
                current = neighbor
                accepted_count += 1
                if current.size() < best.size():
                    best = current.copy()
                    intensified = intensify_search(inst, best, rng=rng, max_iter=3)
                    if is_feasible(inst, intensified) and intensified.size() < best.size():
                        best = intensified
                        current = best.copy()
                    last_improvement = iterations
            elif delta == 0:
                current = neighbor
                accepted_count += 1
            else:
                stuck_f = 1.5 if iterations - last_improvement > 100 else 1.0
                prob = math.exp(-delta / (temp * stuck_f)) if temp > 0 else 0.0
                if rng.random() < prob:
                    current = neighbor
                    accepted_count += 1
        
        # Refroidissement adaptatif
        accept_rate = accepted_count / temp_iter if temp_iter > 0 else 0
        temp_norm = temp / initial_temp if initial_temp > 0 else 0.5
        time_stuck = iterations - last_improvement
        
        if temp_norm > 0.6:
            if accept_rate > acceptance_thresholds['high_accept']:
                temp *= cooling_factors['high_temp_slow']
            elif accept_rate > acceptance_thresholds['mid_accept']:
                temp *= cooling_rate
            else:
                temp *= cooling_rate ** cooling_factors['high_temp_fast']
        elif temp_norm > 0.3:
            if accept_rate > acceptance_thresholds['low_accept']:
                temp *= cooling_rate
            else:
                temp *= cooling_rate ** cooling_factors['mid_temp_fast']
        else:
            if time_stuck <= 50:
                temp *= cooling_rate
            else:
                temp *= cooling_rate ** cooling_factors['low_temp_fast']
        
        if temp < min_temp:
            temp = min_temp
        
        # Rechauffement si bloqué
        if iterations - last_improvement > 100:
            temp = max(temp, initial_temp * 0.7)
            last_improvement = iterations
            
            if best.size() >= 5:
                current = best.copy()
                sensors_list = list(current.sensors)
                n_remove = min(5, max(3, best.size() // 3))
                utility = compute_sensor_utility(inst, current)
                to_remove = _select_low_utility(sensors_list, utility, n_remove, rng)
                for s in to_remove:
                    current.sensors.remove(s)
                
                uncovered = set(range(inst.n)) - covered_targets(inst, current)
                for _ in range(min(10, len(uncovered))):
                    if not uncovered:
                        break
                    best_s, best_g = None, 0
                    for i in range(inst.n):
                        if i not in current.sensors:
                            g = len(inst.cover[i] & uncovered)
                            if g > best_g:
                                best_g, best_s = g, i
                    if best_s is not None:
                        current.sensors.add(best_s)
                        uncovered -= inst.cover[best_s]
                
                current = repair_connectivity(inst, current)
                if is_feasible(inst, current):
                    current = local_search_vnd(inst, current, rng=rng)
            else:
                current = best.copy()
            
            accepted_count = temp_iter
        
        accepted_count = 0
    
    return best
