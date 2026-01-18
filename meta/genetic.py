import random
import time
from typing import List
from meta.instance import Instance
from meta.solution import Solution, is_feasible, repair_connectivity
from meta.constructive import randomized_greedy_construct
from meta.local_search import local_search_vnd

def crossover_1point(p1: Solution, p2: Solution, rng: random.Random, n_targets: int) -> Solution:
    """
    Opérateur de croisement 1-point adapté aux sets d'indices.
    Combine la partie 'gauche' de p1 avec la partie 'droite' de p2.
    """
    # Choix d'un point de coupure basé sur l'ID des cibles
    cut = rng.randint(0, n_targets - 1)
    
    # Construction de l'enfant par union des sous-ensembles
    new_sensors = {s for s in p1.sensors if s < cut} | {s for s in p2.sensors if s >= cut}
    return Solution(new_sensors)

def mutate(sol: Solution, n_targets: int, rng: random.Random):
    """
    Opérateur de mutation simple (Bit-flip).
    Ajoute ou supprime un capteur aléatoirement pour maintenir la diversité.
    """
    if rng.random() < 0.5:
        # Ajout d'un capteur aléatoire
        sol.sensors.add(rng.randint(0, n_targets - 1))
    else:
        # Suppression d'un capteur aléatoire (si non vide)
        if sol.sensors:
            sol.sensors.remove(rng.choice(list(sol.sensors)))

def genetic_algorithm(inst: Instance, time_limit_s: float, pop_size: int = 20, seed: int = 0) -> Solution:
    """
    Algorithme Génétique (Mémétique) pour la couverture connexe.
    Combine une population initiale gloutonne, croisement, mutation,
    réparation et recherche locale (VND).
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    # 1. Initialisation de la population avec l'heuristique constructive randomisée
    # Cela garantit des solutions de départ de bonne qualité.
    population = []
    for _ in range(pop_size):
        sol = randomized_greedy_construct(inst, rng, alpha=0.3)
        population.append(sol)
    
    # Meilleure solution trouvée globalement
    best_sol = min(population, key=lambda s: s.size())
    
    # Boucle principale (temps imparti)
    while time.time() - start_time < time_limit_s:
        new_pop = []
        
        # Génération de la nouvelle population
        while len(new_pop) < pop_size:
            # 2. Sélection par Tournoi (k=2)
            # On tire 2 individus au hasard et on garde le meilleur pour être parent
            parent1 = min(rng.sample(population, k=3), key=lambda s: s.size())
            parent2 = min(rng.sample(population, k=3), key=lambda s: s.size())
            
            # 3. Croisement
            child = crossover_1point(parent1, parent2, rng, inst.n)
            
            # 4. Mutation
            mutate(child, inst.n, rng)
            
            # 5. Réparation
            # Assure la connectivité et la couverture après les perturbations génétiques
            child = repair_connectivity(inst, child)
            
            # 6. Amélioration Locale (Hybridation / Algorithme Mémétique)
            # Optimise immédiatement l'enfant via VND
            child = local_search_vnd(inst, child, rng)
            
            # Ajout à la nouvelle population si la solution est valide
            if is_feasible(inst, child):
                new_pop.append(child)
                # Mise à jour de la meilleure solution globale
                if child.size() < best_sol.size():
                    best_sol = child
        
        # Remplacement générationnel complet
        population = new_pop

    return best_sol