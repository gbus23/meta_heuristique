
# Metaheuristics Project — GRASP + VNS (Course-style)

This project solves the sensor placement problem:
- Place the minimum number of sensors on target locations
- All targets must be covered within `Rcapt`
- All selected sensors must be connected to the sink within `Rcom` hops

## Structure

- `metaheuristics/io_instances.py` : load professor .dat instances (captANOR / grille trunc)
- `metaheuristics/instance.py`     : Instance + precomputations (coverage + communication)
- `metaheuristics/solution.py`     : feasibility, repair connectivity, BFS utilities
- `metaheuristics/constructive.py` : greedy and randomized greedy (RCL) constructions
- `metaheuristics/local_search.py` : descent (remove) + swap descent (VND-like)
- `metaheuristics/grasp.py`        : GRASP metaheuristic
- `metaheuristics/vns.py`          : VNS metaheuristic
- `main.py`                        : batch runner + CSV export

## Run

Extract a zip and run on all `.dat` inside:

```bash
python main.py --zip "Instances.zip" --algo both --time 2 --csv results.csv
```

Or run a folder:

```bash
python main.py --folder "./instances" --algo grasp --time 3
```

Control radii pairs:
- Default pairs: (1,1), (1,2), (2,2), (2,3)
- Or run a single pair:
```bash
python main.py --folder "./instances" --rcapt 1 --rcom 2
```

## Notes

- This is intentionally "course-first": greedy + repair + descent, then GRASP and VNS.
- Improvements can be added later (smarter scoring, better repair, etc.).
