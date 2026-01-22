# Projet Métaheuristiques - Couverture Connexe Minimum dans les Réseaux de Capteurs

## Table des matières

1. [Introduction](#introduction)
2. [Description du problème](#description-du-problème)
3. [Heuristiques constructives](#heuristiques-constructives)
4. [Structures de voisinage et amélioration locale](#structures-de-voisinage-et-amélioration-locale)
5. [Métaheuristiques](#métaheuristiques)
6. [Architecture du code](#architecture-du-code)
7. [Installation et utilisation](#installation-et-utilisation)
8. [Résultats et comparaisons](#résultats-et-comparaisons)

---

## Introduction

Ce projet traite le problème de **couverture connexe minimum dans les réseaux de capteurs**. L'objectif est de placer un nombre minimal de capteurs sur un terrain discrétisé de manière à :

1. **Couverture** : Chaque cible du terrain doit être dans le rayon de captation `Rcapt` d'au moins un capteur
2. **Connectivité** : Tous les capteurs doivent être connectés au puits (sink) via un chemin de capteurs où deux capteurs adjacents sont à une distance inférieure ou égale à `Rcom`

Le projet implémente plusieurs heuristiques et métaheuristiques pour résoudre ce problème d'optimisation combinatoire NP-difficile.

---

## Description du problème

### Modélisation

- **Terrain** : Plan discrétisé en un ensemble de cibles à capter
- **Capteurs** : Ne peuvent être placés que sur les positions des cibles
- **Rayon de captation** `Rcapt` : Un capteur placé en `(i,j)` peut capter toutes les cibles `(k,l)` telles que `√[(i-k)² + (j-l)²] ≤ Rcapt`
- **Rayon de communication** `Rcom` : Un capteur placé en `(i,j)` peut communiquer avec les capteurs en `(k,l)` si `√[(i-k)² + (j-l)²] ≤ Rcom` (avec `Rcom ≥ Rcapt`)
- **Puits (sink)** : Point de collecte situé en `(0,0)` qui ne nécessite pas d'être capté

### Contraintes

1. **Couverture complète** : `|covered_targets(sol)| = n` (toutes les cibles sont couvertes)
2. **Connectivité** : `|connected_to_sink(sol)| = |sol.sensors|` (tous les capteurs sont connectés au puits)

### Objectif

Minimiser le nombre de capteurs placés : `min |sol.sensors|`

---

## Heuristiques constructives

### 1. Heuristique gloutonne déterministe (`greedy_construct`)

#### Principe

Cette heuristique suit une approche gloutonne classique de type "set cover" : à chaque itération, elle sélectionne le capteur qui couvre le plus de cibles non encore couvertes.

#### Algorithme détaillé

```
ENTRÉE: Instance inst
SORTIE: Solution réalisable sol

1. Initialiser:
   - uncovered = {0, 1, ..., n-1}  (toutes les cibles non couvertes)
   - sensors = ∅  (ensemble de capteurs sélectionnés)

2. TANT QUE uncovered ≠ ∅:
   a. Pour chaque position candidate i ∈ [0, n-1]:
      - Calculer gain(i) = |inst.cover[i] ∩ uncovered|
   b. Sélectionner best_i tel que gain(best_i) = max(gain(i))
   c. Si gain(best_i) ≤ 0:
      - ARRÊTER (pas d'amélioration possible)
   d. Ajouter best_i à sensors
   e. Mettre à jour: uncovered = uncovered \ inst.cover[best_i]

3. Construire la solution: sol = Solution(sensors)

4. Réparer la connectivité: sol = repair_connectivity(inst, sol)

5. RETOURNER sol
```

#### Complexité

- **Temps** : O(n² × m) où n est le nombre de cibles et m le nombre moyen de cibles couvertes par capteur
- **Espace** : O(n)

#### Réparation de connectivité

Après la construction gloutonne, la solution peut ne pas être connexe. La fonction `repair_connectivity` :

1. Identifie les capteurs connectés au puits via un BFS
2. Pour chaque capteur déconnecté, trouve le plus court chemin vers un capteur connecté
3. Ajoute les capteurs-relais nécessaires sur ce chemin

Cette réparation garantit la faisabilité mais peut augmenter le nombre de capteurs.

---

### 2. Heuristique gloutonne randomisée (`randomized_greedy_construct`)

#### Principe GRASP avec RCL

Cette heuristique utilise le principe **GRASP (Greedy Randomized Adaptive Search Procedure)** avec une **RCL (Restricted Candidate List)** pour introduire de la diversification dans la construction.

#### Algorithme détaillé

```
ENTRÉE: Instance inst, générateur aléatoire rng, paramètre alpha ∈ [0,1]
SORTIE: Solution réalisable sol

1. Initialiser:
   - uncovered = {0, 1, ..., n-1}
   - sensors = ∅

2. TANT QUE uncovered ≠ ∅:
   a. Pour chaque position candidate i ∈ [0, n-1]:
      - Calculer gain(i) = |inst.cover[i] ∩ uncovered|
      - Si gain(i) > 0, ajouter (i, gain(i)) à la liste gains
   
   b. Calculer:
      - g_max = max(gain(i) pour tous les gains > 0)
      - g_min = min(gain(i) pour tous les gains > 0)
      - threshold = g_max - alpha × (g_max - g_min)
   
   c. Construire la RCL:
      - RCL = {i | gain(i) ≥ threshold}
   
   d. Si RCL ≠ ∅:
      - Choisir aléatoirement chosen ∈ RCL
   Sinon:
      - Choisir l'élément avec le gain maximal
   
   e. Ajouter chosen à sensors
   f. Mettre à jour: uncovered = uncovered \ inst.cover[chosen]

3. Construire la solution: sol = Solution(sensors)

4. Réparer la connectivité: sol = repair_connectivity(inst, sol)

5. RETOURNER sol
```

#### Paramètre alpha

- **alpha = 0** : Comportement déterministe (sélection du meilleur uniquement)
- **alpha = 1** : Sélection purement aléatoire parmi tous les candidats valides
- **alpha ∈ ]0,1[** : Équilibre entre qualité et diversification

**Valeur par défaut** : `alpha = 0.3` (recommandé dans la littérature GRASP)

#### Avantages de la randomisation

1. **Diversification** : Génère différentes solutions à chaque exécution
2. **Évite les optima locaux précoces** : Permet d'explorer différentes régions de l'espace de recherche
3. **Complémentarité avec amélioration locale** : Les solutions construites servent de points de départ variés pour la recherche locale

---

## Structures de voisinage et amélioration locale

### 1. Voisinage de suppression (`prune_descent`)

#### Transformation

Le voisinage de suppression consiste à **retirer un capteur** de la solution courante.

#### Définition formelle

Pour une solution `sol` avec `sensors = {s₁, s₂, ..., sₖ}`, le voisinage est :
```
V_prune(sol) = {sol' | sol'.sensors = sol.sensors \ {s} pour s ∈ sol.sensors}
```

#### Stratégie de recherche

- **Première amélioration** : Dès qu'une amélioration est trouvée, on l'accepte et on recommence
- **Ordre d'exploration** : Parcours séquentiel de tous les capteurs
- **Critère d'arrêt** : Aucune amélioration possible (optimum local atteint)

#### Algorithme

```
ENTRÉE: Instance inst, Solution sol
SORTIE: Solution améliorée sol

1. Réparer la connectivité: sol = repair_connectivity(inst, sol)

2. TANT QUE amélioration trouvée:
   a. improved = False
   b. Pour chaque capteur s ∈ sol.sensors:
      - Créer trial = sol.copy()
      - Retirer s de trial.sensors
      - Vérifier couverture: Si |covered_targets(inst, trial)| ≠ n:
           CONTINUER (solution non réalisable)
      - Réparer connectivité: trial = repair_connectivity(inst, trial)
      - Si trial est réalisable ET |trial.sensors| < |sol.sensors|:
           sol = trial
           improved = True
           BREAK (première amélioration)

3. RETOURNER sol
```

#### Complexité

- **Temps** : O(k × n × m) où k est le nombre de capteurs, n le nombre de cibles, m la complexité de la réparation
- **Espace** : O(n)

---

### 2. Voisinage d'échange (`swap_descent`)

#### Transformation

Le voisinage d'échange consiste à **remplacer un capteur** par un autre : retirer un capteur `s_out` et ajouter un capteur `s_in`.

#### Définition formelle

```
V_swap(sol) = {sol' | sol'.sensors = (sol.sensors \ {s_out}) ∪ {s_in}
                pour s_out ∈ sol.sensors, s_in ∈ [0, n-1] \ sol.sensors}
```

#### Stratégie de recherche

- **Échantillonnage aléatoire** : Pour des raisons de performance, on teste un nombre limité de swaps aléatoires
- **Première amélioration** : Acceptation immédiate d'une amélioration
- **Limite de tentatives** : `max_trials = 250` par itération

#### Algorithme

```
ENTRÉE: Instance inst, Solution sol, générateur aléatoire rng, max_trials = 250
SORTIE: Solution améliorée sol

1. Réparer la connectivité: sol = repair_connectivity(inst, sol)
2. Si sol non réalisable: RETOURNER sol

3. trials = 0
4. TANT QUE trials < max_trials:
   a. Choisir aléatoirement:
      - s_out ∈ sol.sensors
      - s_in ∈ [0, n-1] (aléatoire)
   b. Si s_in == s_out: CONTINUER
   
   c. Créer trial = sol.copy()
      - Retirer s_out de trial.sensors
      - Ajouter s_in à trial.sensors
   
   d. Vérifier couverture: Si |covered_targets(inst, trial)| ≠ n:
        CONTINUER
   
   e. Réparer connectivité: trial = repair_connectivity(inst, trial)
   
   f. Si trial réalisable ET |trial.sensors| < |sol.sensors|:
        sol = trial
        trials = 0  (redémarrer après amélioration)
   Sinon:
        trials += 1

5. RETOURNER sol
```

#### Complexité

- **Temps** : O(max_trials × n × m)
- **Espace** : O(n)

---

### 3. Variable Neighborhood Descent (VND) (`local_search_vnd`)

#### Principe

Le **VND (Variable Neighborhood Descent)** utilise une séquence de voisinages de complexité croissante pour éviter de rester bloqué dans un optimum local d'un seul voisinage.

#### Séquence de voisinages

1. **Voisinage de suppression** (`prune_descent`) : Le plus simple, permet d'éliminer les capteurs redondants
2. **Voisinage d'échange** (`swap_descent`) : Plus complexe, permet de remplacer des capteurs mal positionnés
3. **Voisinage de suppression** (à nouveau) : Pour nettoyer après les échanges

#### Algorithme

```
ENTRÉE: Instance inst, Solution sol, générateur aléatoire rng
SORTIE: Solution améliorée sol

1. sol = prune_descent(inst, sol)      // Nettoyage initial
2. sol = swap_descent(inst, sol, rng) // Échanges
3. sol = prune_descent(inst, sol)      // Nettoyage final

RETOURNER sol
```

#### Avantages

- **Diversification** : Chaque voisinage explore différentes transformations
- **Efficacité** : Les voisinages simples sont rapides, les complexes sont utilisés seulement si nécessaire
- **Robustesse** : Moins sensible aux optima locaux d'un seul voisinage

---

## Métaheuristiques

### 1. GRASP (`grasp`)

#### Principe général

**GRASP (Greedy Randomized Adaptive Search Procedure)** est une métaheuristique itérative qui combine :

1. **Construction randomisée** : Utilise une RCL pour générer des solutions variées
2. **Amélioration locale** : Applique une recherche locale (VND) sur chaque solution construite
3. **Mémorisation** : Conserve la meilleure solution rencontrée

#### Algorithme détaillé

```
ENTRÉE: Instance inst, time_limit_s, alpha, seed
SORTIE: Meilleure solution best

1. Initialiser générateur aléatoire: rng = Random(seed)
2. t0 = time()

3. Solution initiale:
   - best = greedy_construct(inst)           // Construction déterministe
   - best = local_search_vnd(inst, best, rng) // Amélioration locale

4. TANT QUE time() - t0 < time_limit_s:
   a. Construction randomisée:
      - s = randomized_greedy_construct(inst, rng, alpha)
   
   b. Amélioration locale:
      - s = local_search_vnd(inst, s, rng)
   
   c. Mise à jour:
      - Si s réalisable ET |s.sensors| < |best.sensors|:
           best = s

5. RETOURNER best
```

#### Paramètres

- **alpha** : Paramètre de la RCL (défaut: 0.3)
  - Contrôle le compromis qualité/diversification
- **time_limit_s** : Limite de temps en secondes (défaut: 2.0)
  - Détermine le nombre d'itérations
- **seed** : Graine pour la reproductibilité (défaut: 0)

#### Complexité

- **Temps** : O(time_limit_s × (construction + amélioration))
- **Espace** : O(n)

---

### 2. VNS - Variable Neighborhood Search (`vns`)

#### Principe

**VNS (Variable Neighborhood Search)** est une métaheuristique qui utilise plusieurs voisinages de "shake" (perturbation) pour échapper aux optima locaux.

#### Composants principaux

1. **Shake** : Perturbation de la solution avec k mouvements aléatoires
2. **Local Search** : Amélioration locale (VND) sur la solution perturbée
3. **Acceptation** : Acceptation stricte (seulement les améliorations)

#### Fonction Shake (`shake`)

La fonction shake applique k mouvements aléatoires parmi :
- **Remove** (probabilité 34%) : Retirer un capteur
- **Add** (probabilité 33%) : Ajouter un capteur
- **Swap** (probabilité 33%) : Retirer un capteur et en ajouter un autre

```
ENTRÉE: Instance inst, Solution sol, rng, k (intensité)
SORTIE: Solution perturbée s

1. s = sol.copy()
2. POUR i = 1 À k:
   a. r = random() ∈ [0,1]
   b. Si r < 0.34 ET s.sensors ≠ ∅:
        Retirer un capteur aléatoire de s.sensors
   c. Sinon si r < 0.67:
        Ajouter un capteur aléatoire à s.sensors
   d. Sinon ET s.sensors ≠ ∅:
        Retirer un capteur aléatoire
        Ajouter un capteur aléatoire

3. Réparer connectivité: s = repair_connectivity(inst, s)
4. RETOURNER s
```

#### Algorithme VNS

```
ENTRÉE: Instance inst, time_limit_s, kmax, seed
SORTIE: Meilleure solution best

1. Initialiser générateur aléatoire: rng = Random(seed)
2. t0 = time()

3. Solution initiale:
   - best = greedy_construct(inst)
   - best = local_search_vnd(inst, best, rng)

4. TANT QUE time() - t0 < time_limit_s:
   a. k = 1
   b. TANT QUE k ≤ kmax ET time() - t0 < time_limit_s:
      
      i. Shake:
         - x1 = shake(inst, best, rng, k)
         - Si couverture cassée:
              x1 = randomized_greedy_construct(inst, rng, alpha=0.3)
      
      ii. Local Search:
          - x2 = local_search_vnd(inst, x1, rng)
      
      iii. Acceptation:
          - Si x2 réalisable ET |x2.sensors| < |best.sensors|:
               best = x2
               k = 1  (redémarrer avec faible perturbation)
          Sinon:
               k = k + 1  (augmenter l'intensité de perturbation)

5. RETOURNER best
```

#### Paramètres

- **kmax** : Intensité maximale de shake (défaut: 4)
  - Contrôle la diversification : k=1 (faible) à kmax (forte)
- **time_limit_s** : Limite de temps en secondes (défaut: 2.0)
- **seed** : Graine pour la reproductibilité (défaut: 0)

#### Stratégie d'acceptation

- **Acceptation stricte** : Seules les améliorations sont acceptées
- **Gestion de k** : 
  - Si amélioration → k = 1 (recherche locale autour de la nouvelle solution)
  - Sinon → k++ (augmentation de la perturbation pour explorer plus loin)

#### Complexité

- **Temps** : O(time_limit_s × kmax × (shake + amélioration))
- **Espace** : O(n)

---

### 3. Recuit Simulé (`simulated_annealing`)

#### Principe

Le **Recuit Simulé (Simulated Annealing)** est une métaheuristique inspirée du processus de recuit métallurgique. Il accepte des solutions moins bonnes avec une probabilité décroissante au fil du temps, permettant d'échapper aux optima locaux.

#### Composants principaux

1. **Solution initiale** : Construction gloutonne + amélioration locale (VND)
2. **Génération de voisins** : Multi-niveaux (remove, add, swap, 2-1, 3-2, 3-1, 4-2, reconstruction partielle)
3. **Acceptation probabiliste** : `P(accepter) = exp(-Δ/T)` où Δ est la différence de qualité et T la température
4. **Refroidissement adaptatif** : Schéma multi-phase avec refroidissement logarithmique puis géométrique
5. **Rechauffement périodique** : Diversification agressive quand bloqué

#### Génération de voisins

Le recuit utilise plusieurs niveaux de voisinages :

- **Niveau 1** (probabilité 40%) : Remove, Add, Swap classiques
- **Niveau 2** (probabilité 30%) : Échanges 2-1, 3-2 (retirer plusieurs, ajouter moins)
- **Niveau 3** (probabilité 20%) : Échanges 3-1, 4-2 (plus agressifs)
- **Niveau 4** (probabilité 10%) : Reconstruction partielle (retirer plusieurs capteurs, reconstruire avec glouton)

La sélection utilise l'utilité des capteurs : les capteurs peu utiles sont préférés pour la suppression.

#### Température initiale adaptative

La température initiale est estimée en échantillonnant des voisins et en calculant :
```
T₀ = moyenne(|Δ|) / ln(acceptance_rate_desirée)
```

Cela garantit un taux d'acceptation initial d'environ 50-60%.

#### Schéma de refroidissement multi-phase

1. **Phase haute température** (T > 0.6×T₀) : Refroidissement logarithmique lent
2. **Phase moyenne** (0.3×T₀ < T ≤ 0.6×T₀) : Refroidissement géométrique standard
3. **Phase basse température** (T ≤ 0.3×T₀) : Refroidissement géométrique agressif

Le refroidissement s'adapte au taux d'acceptation : plus rapide si peu d'acceptations, plus lent sinon.

#### Intensification locale

À basse température, une recherche locale intensive est appliquée systématiquement :
- VND standard
- Plusieurs passes de `swap_2_1_descent`
- Application répétée de VND

#### Rechauffement et diversification

Quand aucune amélioration n'est trouvée depuis 100 itérations :
1. **Rechauffement** : Température remise à 70% de T₀
2. **Diversification** : Retrait de 3-5 capteurs peu utiles
3. **Reconstruction** : Reconstruction partielle avec glouton
4. **Amélioration** : Application de VND

#### Algorithme détaillé

```
ENTRÉE: Instance inst, time_limit_s, seed
SORTIE: Meilleure solution best

1. Initialiser générateur aléatoire: rng = Random(seed)
2. t0 = time()

3. Solution initiale:
   - current = greedy_construct(inst)
   - current = local_search_vnd(inst, current, rng)
   - best = current.copy()

4. Estimation température initiale T₀:
   - Échantillonner des voisins
   - T₀ = moyenne(|Δ|) / ln(0.5)

5. temp = T₀
6. iterations = 0
7. last_improvement_iter = 0

8. TANT QUE time() - t0 < time_limit_s:
   a. Pour iterations_per_temp itérations:
      i. Générer voisin:
         - Sélectionner niveau de voisinage selon température et état
         - neighbor = generate_neighbor(inst, current, rng, temp, aggressive, stuck)
      
      ii. Réparation:
          - Si couverture cassée: réparer
          - neighbor = repair_connectivity(inst, neighbor)
      
      iii. Recherche locale (selon température):
          - Si temp < 0.3×T₀: recherche locale intensive
          - Sinon si temp < 0.6×T₀: VND standard
          - Sinon: VND léger
      
      iv. Acceptation:
          - Δ = neighbor.size() - current.size()
          - Si Δ < 0: accepter (amélioration)
          - Sinon si Δ == 0: accepter (diversification)
          - Sinon: accepter avec probabilité exp(-Δ/(temp×stuck_factor))
      
      v. Si accepté:
         - current = neighbor
         - Si current.size() < best.size():
              best = current.copy()
              Intensifier autour de best
              last_improvement_iter = iterations
      
      vi. iterations += 1
   
   b. Refroidissement adaptatif:
      - Calculer taux d'acceptation
      - Ajuster temp selon phase et taux d'acceptation
      - Si temp < min_temp: temp = min_temp
   
   c. Rechauffement si bloqué:
      - Si iterations - last_improvement_iter > 100:
           temp = max(temp, 0.7×T₀)
           Diversifier current (retirer capteurs, reconstruire)
           last_improvement_iter = iterations

9. RETOURNER best
```

#### Paramètres

- **time_limit_s** : Limite de temps en secondes (défaut: 2.0)
- **seed** : Graine pour la reproductibilité (défaut: 0)
- **cooling_rate** : Taux de refroidissement géométrique (défaut: 0.985, optimisé via grid search)
- **min_temp** : Température minimale (défaut: 0.001, optimisé via grid search)
- **cooling_factors** : Dictionnaire des facteurs de refroidissement adaptatif (défaut: optimisés)
  - `high_temp_slow`: 0.99 (refroidissement lent haute température)
  - `high_temp_fast`: 1.03 (refroidissement rapide haute température)
  - `mid_temp_fast`: 1.08 (refroidissement rapide température moyenne)
  - `low_temp_fast`: 1.15 (refroidissement rapide basse température)
- **acceptance_thresholds** : Dictionnaire des seuils d'acceptation (défaut: optimisés)
  - `high_accept`: 0.3 (seuil haute température)
  - `mid_accept`: 0.15 (seuil température moyenne)
  - `low_accept`: 0.1 (seuil basse température)

#### Stratégie d'acceptation

- **Amélioration** (Δ < 0) : Toujours acceptée
- **Même qualité** (Δ = 0) : Acceptée pour diversification
- **Dégradation** (Δ > 0) : Acceptée avec probabilité `exp(-Δ/(T×stuck_factor))`
  - `stuck_factor = 1.5` si bloqué depuis > 100 itérations (plus permissif)

#### Complexité

- **Temps** : O(time_limit_s × iterations_per_temp × (génération + amélioration))
- **Espace** : O(n)

#### Avantages

- **Échappement des optima locaux** : Acceptation probabiliste permet d'explorer plus largement
- **Adaptatif** : S'adapte automatiquement à la structure du problème
- **Multi-niveaux** : Utilise plusieurs voisinages pour une exploration efficace
- **Intensification** : Recherche locale intensive à basse température

#### Optimisation des paramètres via Grid Search

Les paramètres de refroidissement du recuit simulé ont été optimisés via une recherche systématique (grid search) à l'aide du script `benchmark_sa_params.py`.

**Utilisation du script de benchmark :**

```bash
python benchmark_sa_params.py <instance_path> <rcapt> <rcom> [time_limit] [restarts]
```

Exemple :
```bash
python benchmark_sa_params.py instances/instances_grid/grille1010_1.dat 2 3 2.0 3
```

Le script exécute trois phases :
1. **Phase 1** : Test de `cooling_rate` et `min_temp`
2. **Phase 2** : Test des facteurs de refroidissement adaptatif
3. **Phase 3** : Test des seuils d'acceptation

Les résultats sont sauvegardés dans un fichier CSV avec timestamp et un résumé des meilleures configurations est affiché.

**Résultats de l'optimisation :**

Sur l'instance `grille1010_1.dat` avec R=(2,3), les paramètres optimaux identifiés sont :
- `cooling_rate`: 0.985 (au lieu de 0.99)
- `min_temp`: 0.001 (au lieu de 0.01)
- Facteurs de refroidissement légèrement plus agressifs pour une meilleure exploration

Ces valeurs sont maintenant utilisées par défaut dans `simulated_annealing()`.

---

## Architecture du code

### Structure des modules

```
meta/
├── __init__.py           # Package Python
├── instance.py          # Classe Instance (cibles, rayons, précalculs)
├── solution.py          # Classe Solution et fonctions de faisabilité
├── io_instances.py       # Chargement des instances (.dat)
├── constructive.py      # Heuristiques constructives (greedy, randomized)
├── local_search.py      # Recherche locale (prune, swap, VND)
├── vns.py               # Métaheuristique VNS
├── genetic.py           # Algorithme génétique
├── simulated_annealing.py # Recuit simulé
├── visualization.py     # Visualisation des solutions
└── comparison.py        # Comparaison des résultats (tableaux, graphiques)

main.py                  # Point d'entrée principal (CLI, batch processing)
benchmark_sa_params.py   # Script de grid search pour optimiser les paramètres SA
```

### Flux de données

```
Instance (.dat)
    ↓
io_instances.load_targets()
    ↓
Instance.build()  [précalculs: cover, comm, sink_comm]
    ↓
┌──────────────┬──────────────┬──────────────┬──────────────────┐
│     VNS      │  Génétique   │ Recuit Sim.  │                  │
│              │              │              │                  │
│ shake()      │ crossover()  │ generate_    │                  │
│              │ mutate()     │ neighbor()   │                  │
│      ↓       │      ↓       │      ↓       │                  │
│ local_search │ local_search │ local_search │                  │
│ vnd()        │ vnd()        │ vnd()        │                  │
└──────────────┴──────────────┴──────────────┴──────────────────┘
    ↓                ↓                ↓
Solution → is_feasible() → CSV + Plots
```

### Responsabilités des modules

- **`instance.py`** : Représentation du problème avec précalculs des voisinages de couverture et communication
- **`solution.py`** : Représentation des solutions, vérification de faisabilité, réparation de connectivité
- **`constructive.py`** : Génération de solutions initiales (gloutonne déterministe et randomisée)
- **`local_search.py`** : Amélioration locale via différents voisinages
- **`vns.py`** : Métaheuristique VNS complète
- **`genetic.py`** : Algorithme génétique mémétique
- **`simulated_annealing.py`** : Recuit simulé avec voisinages multi-niveaux
- **`visualization.py`** : Génération de graphiques montrant les solutions
- **`comparison.py`** : Analyse comparative des résultats (statistiques, tableaux, graphiques)

---

## Installation et utilisation

### Installation des dépendances

```bash
pip install -r requirements.txt
```

Les dépendances incluent :
- `numpy` : Calculs numériques
- `matplotlib` : Visualisation
- `tabulate` : Tableaux formatés dans la console
- `pandas` : Manipulation de données (optionnel, améliore l'affichage)
- `seaborn` : Graphiques améliorés (optionnel)

### Format des instances

Le projet supporte deux formats d'instances :

1. **Instances aléatoires** (`captANOR*.dat`) :
   - Liste de points avec coordonnées (x, y)
   - Le premier point est généralement le puits (0,0) et est ignoré

2. **Grilles tronquées** (`grille*.dat`) :
   - Grille carrée avec certaines cellules supprimées
   - Format : `grilleNNMM_k.dat` où NN×MM est la taille de la grille
   - Le fichier liste les cellules supprimées au format `(i, j)`

### Commandes principales

#### 1. Traitement batch

```bash
python main.py --folder "./instances" --algo sa --time 2.0 --csv results.csv
```

Options :
- `--folder` : Dossier contenant les fichiers `.dat`
- `--algo` : Algorithme à utiliser (`vns`, `ga`, `sa`, `annealing`)
- `--time` : Limite de temps par instance (secondes)
- `--csv` : Fichier de sortie CSV

#### 2. Exécution avec un seul algorithme

```bash
# VNS uniquement
python main.py --folder "./instances" --algo vns --time 2.0 --kmax 4

# Algorithme génétique
python main.py --folder "./instances" --algo ga --time 2.0

# Recuit simulé
python main.py --folder "./instances" --algo sa --time 2.0
```

#### 3. Exécution sur une paire de rayons spécifique

```bash
python main.py --folder "./instances" --rcapt 2 --rcom 3 --algo sa
```

#### 4. Visualisation d'une solution unique

```bash
python main.py --plot --file "./instances/captANOR150_7_4.dat" --rcapt 1 --rcom 2
```

#### 5. Comparaison des résultats

```bash
python main.py --compare --csv-in results.csv
```

Cette commande génère :
- Tableaux récapitulatifs dans la console
- Statistiques comparatives (moyennes, médianes, écarts-types)
- Graphiques sauvegardés dans `results/comparisons/` :
  - `comparison_sensors.png` : Comparaison du nombre de capteurs
  - `comparison_time.png` : Comparaison des temps de résolution
  - `comparison_by_radius.png` : Performance par paire de rayons

### Paramètres avancés

```bash
python main.py \
    --folder "./instances" \
    --algo sa \
    --time 3.0 \
    --restarts 5 \
    --seed 42 \
    --kmax 4 \
    --csv results.csv
```

- `--seed` : Graine pour la reproductibilité
- `--restarts` : Nombre de restarts (multi-départ)
- `--kmax` : Intensité maximale de shake VNS (pour VNS uniquement)
- `--alpha` : Paramètre pour construction randomisée (0=déterministe, 1=aléatoire)

---

## Résultats et comparaisons

### Format du fichier CSV

Le fichier `results.csv` contient les colonnes suivantes :

- `file` : Nom du fichier d'instance
- `rcapt` : Rayon de captation
- `rcom` : Rayon de communication
- `algo` : Algorithme utilisé (`vns`, `ga`, `sa`)
- `per_run_s` : Temps par restart (secondes)
- `restarts` : Nombre de restarts
- `time_total_s` : Temps total de résolution en secondes
- `sensors` : Nombre de capteurs dans la solution
- `feasible` : `True` si la solution est réalisable
- `uncovered` : Nombre de cibles non couvertes (devrait être 0)
- `disconnected` : Nombre de capteurs déconnectés (devrait être 0)
- `plot_path` : Chemin vers le graphique de la solution

### Utilisation du module de comparaison

Le module `meta.comparison` fournit plusieurs fonctions :

#### Tableaux récapitulatifs

```python
from meta.comparison import load_results_csv, print_summary_table, print_statistics

results = load_results_csv("results.csv")
print_summary_table(results)  # Tableau comparatif GRASP vs VNS
print_statistics(results)     # Statistiques détaillées
```

#### Génération de graphiques

```python
from meta.comparison import generate_all_comparisons

# Génère tous les tableaux et graphiques
generate_all_comparisons("results.csv", output_dir="results/comparisons", show=False)
```

### Interprétation des résultats

#### Tableaux

- **Tableau récapitulatif** : Compare les algorithmes pour chaque instance et paire de rayons
- **Colonne "Meilleur"** : Indique quel algorithme a trouvé la meilleure solution
- **Statistiques** : Moyennes, médianes, écarts-types pour comparer la robustesse

#### Graphiques

1. **Comparaison capteurs** : Visualise la différence de qualité entre les algorithmes
2. **Comparaison temps** : Montre les performances temporelles
3. **Performance par rayons** : Analyse l'impact des paramètres `Rcapt` et `Rcom`

### Analyse des résultats

Pour une analyse complète :

1. **Exécuter les algorithmes** sur toutes les instances :
   ```bash
   python main.py --folder "./instances" --algo sa --time 2.0 --restarts 5
   ```

2. **Examiner** :
   - Les résultats dans le CSV
   - Les visualisations individuelles dans `results/plots/` (organisées par timestamp)
   - Les statistiques affichées en fin d'exécution

---

## Notes de développement

- Le code est commenté en français
- Les algorithmes suivent les principes vus en cours de métaheuristiques
- La structure modulaire permet d'ajouter facilement de nouveaux voisinages ou métaheuristiques
- Les solutions sont toujours vérifiées pour la faisabilité avant d'être retournées
- Les plots sont organisés dans des sous-dossiers avec timestamp (format: `jour_heure_minute`)
- Les paramètres du recuit simulé ont été optimisés via grid search (voir `benchmark_sa_params.py`)

---

## Auteurs

Projet réalisé dans le cadre du cours de Métaheuristiques (M2 SOD24).

---

## Références

- Mladenović, N., & Hansen, P. (1997). Variable neighborhood search. *Computers & operations research*, 24(11), 1097-1100.
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671-680.
- Goldberg, D. E. (1989). *Genetic algorithms in search, optimization, and machine learning*. Addison-Wesley.
