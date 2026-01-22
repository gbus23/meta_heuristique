from __future__ import annotations

from typing import Set
import random

from .instance import Instance, Idx
from .solution import Solution, repair_connectivity


def greedy_construct(inst: Instance) -> Solution:
    """Construction gloutonne améliorée : couverture + connectivité + difficulté cibles + pénalité bordure."""
    uncovered = set(range(inst.n))
    sensors: Set[Idx] = set()
    
    if inst.targets:
        x_coords = [p[0] for p in inst.targets]
        y_coords = [p[1] for p in inst.targets]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
    else:
        x_min = x_max = y_min = y_max = 0.0
    
    target_difficulty = [0] * inst.n
    for i in range(inst.n):
        for j in range(inst.n):
            if i in inst.cover[j]:
                target_difficulty[i] += 1
    
    if inst.sink_comm:
        best_sink_sensor = None
        best_sink_score = -1.0
        for i in inst.sink_comm:
            gain = len(inst.cover[i] & uncovered)
            if gain <= 0:
                continue
            difficulty_bonus = sum(1.0 / max(target_difficulty[t], 1) for t in inst.cover[i] & uncovered)
            score = gain + 0.3 * difficulty_bonus
            
            if inst.targets:
                pos = inst.targets[i]
                dist_to_left = pos[0] - x_min
                dist_to_right = x_max - pos[0]
                dist_to_bottom = pos[1] - y_min
                dist_to_top = y_max - pos[1]
                min_dist_to_border = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
                if min_dist_to_border < inst.rcapt:
                    wasted_ratio = max(0.0, (inst.rcapt - min_dist_to_border) / inst.rcapt)
                    score -= 0.4 * wasted_ratio * min(gain, 4)
            
            if score > best_sink_score:
                best_sink_score = score
                best_sink_sensor = i
        
        if best_sink_sensor is not None and best_sink_score > 0:
            sensors.add(best_sink_sensor)
            uncovered -= inst.cover[best_sink_sensor]

    connected_component = set(sensors)
    if sensors:
        for s in sensors:
            connected_component.update(inst.comm[s])

    while uncovered:
        best_i = None
        best_score = -1.0
        
        hard_targets = {t for t in uncovered if target_difficulty[t] <= 3}
        
        for i in range(inst.n):
            coverage_gain = len(inst.cover[i] & uncovered)
            if coverage_gain <= 0:
                continue
            
            score = float(coverage_gain)
            
            connectivity_bonus = 0.0
            if i in inst.sink_comm:
                connectivity_bonus = 0.4
            elif sensors and i in connected_component:
                connectivity_bonus = 0.25
            
            difficulty_bonus = 0.0
            hard_covered = len(inst.cover[i] & uncovered & hard_targets)
            if hard_covered > 0:
                difficulty_bonus = 0.3 * hard_covered
            
            redundancy_penalty = 0.0
            if coverage_gain > 1:
                alternative_count = 0
                for t in inst.cover[i] & uncovered:
                    alternative_count += target_difficulty[t] - 1
                avg_alternatives = alternative_count / coverage_gain if coverage_gain > 0 else 0
                if avg_alternatives > 2:
                    redundancy_penalty = 0.1 * min(avg_alternatives / 5.0, 1.0)
            
            border_penalty = 0.0
            if inst.targets:
                pos = inst.targets[i]
                dist_to_left = pos[0] - x_min
                dist_to_right = x_max - pos[0]
                dist_to_bottom = pos[1] - y_min
                dist_to_top = y_max - pos[1]
                min_dist_to_border = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
                
                if min_dist_to_border < inst.rcapt:
                    wasted_ratio = max(0.0, (inst.rcapt - min_dist_to_border) / inst.rcapt)
                    border_penalty = 0.5 * wasted_ratio * min(coverage_gain, 5)
            
            score = score + connectivity_bonus * min(coverage_gain, 4) + difficulty_bonus - redundancy_penalty - border_penalty
            
            if score > best_score:
                best_score = score
                best_i = i
        
        if best_i is None:
            break
        
        sensors.add(best_i)
        uncovered -= inst.cover[best_i]
        connected_component.update(inst.comm[best_i])

    sol = Solution(sensors)
    return repair_connectivity(inst, sol)


def randomized_greedy_construct(inst: Instance, rng: random.Random, alpha: float = 0.3) -> Solution:
    """Construction gloutonne randomisée avec RCL (GRASP)."""
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
