from __future__ import annotations

import os
import csv
import glob
import time
import zipfile
import argparse
from datetime import datetime
from typing import List, Tuple, Optional

from meta.io_instances import load_targets
from meta.instance import Instance
from meta.solution import is_feasible, covered_targets, connected_to_sink
from meta.vns import vns
from meta.visualization import plot_solution_auto
from meta.genetic import genetic_algorithm
from meta.simulated_annealing import simulated_annealing


DEFAULT_PAIRS: List[Tuple[int, int]] = [(1, 1), (1, 2), (2, 2), (2, 3)]


def _safe_stem(s: str) -> str:
    s = s.replace(" ", "_")
    s = s.replace(":", "-").replace("/", "-").replace("\\", "-")
    s = "".join(ch for ch in s if ch.isalnum() or ch in "._-")
    return s


def ensure_extracted(zip_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return out_dir


def iter_dat_files(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "**", "*.dat"), recursive=True))


def solve_one(inst: Instance, algo: str, time_limit_s: float, seed: int, alpha: float, kmax: int):
    algo = algo.lower()
    if algo == "vns":
        return vns(inst, time_limit_s=time_limit_s, kmax=kmax, seed=seed)
    if algo == "ga":
        return genetic_algorithm(inst, time_limit_s=time_limit_s, pop_size=50, seed=seed)
    if algo == "sa" or algo == "annealing":
        return simulated_annealing(inst, time_limit_s=time_limit_s, seed=seed)
    raise ValueError(f"Algorithme inconnu: {algo}")


def solve_multi(inst: Instance,
                algo: str,
                per_run_s: float,
                restarts: int,
                seed: int,
                alpha: float,
                kmax: int,
                verbose: bool = False):
    """Multi-restart : exécute l'algorithme 'restarts' fois avec des seeds différentes."""
    best_sol = None
    best_size = 10**18
    all_sizes = []

    for r in range(restarts):
        sol = solve_one(
            inst,
            algo=algo,
            time_limit_s=per_run_s,
            seed=seed + r,
            alpha=alpha,
            kmax=kmax,
        )
        sol_size = sol.size() if is_feasible(inst, sol) else float('inf')
        all_sizes.append(sol_size)
        
        if is_feasible(inst, sol) and sol.size() < best_size:
            best_sol = sol
            best_size = sol.size()
        
        if verbose and restarts > 1:
            status = "OK" if is_feasible(inst, sol) else "FAIL"
            best_mark = " [BEST]" if sol_size == best_size and sol_size != float('inf') else ""
            print(f"  Restart {r+1}/{restarts} (seed={seed+r}): {sol.size()} sensors {status}{best_mark}")

    # Fallback (should not happen if instances are valid)
    if best_sol is None:
        best_sol = solve_one(
            inst,
            algo=algo,
            time_limit_s=per_run_s,
            seed=seed,
            alpha=alpha,
            kmax=kmax,
        )
    
    if verbose and restarts > 1:
        feasible_sizes = [s for s in all_sizes if s != float('inf')]
        if feasible_sizes:
            print(f"  Multi-départ: best={min(feasible_sizes)}, worst={max(feasible_sizes)}, "
                  f"avg={sum(feasible_sizes)/len(feasible_sizes):.1f}, "
                  f"feasible={len(feasible_sizes)}/{restarts}")

    return best_sol


def run_batch(paths: List[str],
              rcapt: Optional[float],
              rcom: Optional[float],
              pairs: List[Tuple[int, int]],
              algo: str,
              per_run_s: float,
              restarts: int,
              seed: int,
              alpha: float,
              kmax: int,
              csv_out: str) -> None:
    rows = []
    
    timestamp = datetime.now().strftime("%d_%H_%M")
    plots_dir = os.path.join("results", "plots", timestamp)
    os.makedirs(plots_dir, exist_ok=True)

    used_pairs = [(int(rcapt), int(rcom))] if (rcapt is not None and rcom is not None) else pairs
    total_tasks = len(paths) * len(used_pairs)
    current_task = 0

    print(f"\n{'='*70}")
    print(f"Batch: {len(paths)} instance(s) x {len(used_pairs)} paire(s) = {total_tasks} tache(s)")
    print(f"Algo: {algo.upper()}, Restarts: {restarts}, Temps/run: {per_run_s}s")
    print(f"Plots: {plots_dir}")
    print(f"{'='*70}\n")

    for idx, p in enumerate(paths):
        file_name = os.path.basename(p)
        print(f"[Instance {idx+1}/{len(paths)}] {file_name}")

        for pair_idx, (rc, rco) in enumerate(used_pairs):
            current_task += 1
            print(f"  [{current_task}/{total_tasks}] Paire R=({rc},{rco}) ... ", end="", flush=True)

            targets = load_targets(p, sink=(0.0, 0.0))
            inst = Instance.build(targets, sink=(0.0, 0.0), rcapt=float(rc), rcom=float(rco))

            t0 = time.time()
            sol = solve_multi(
                inst,
                algo=algo,
                per_run_s=per_run_s,
                restarts=restarts,
                seed=seed,
                alpha=alpha,
                kmax=kmax,
                verbose=False,  # No verbose in batch mode
            )
            dt = time.time() - t0

            feas = is_feasible(inst, sol)
            uncovered = inst.n - len(covered_targets(inst, sol))
            disconnected = sol.size() - len(connected_to_sink(inst, sol))

            status = "OK" if feas else "FAIL"
            print(f"Done: {sol.size()} capteurs, {dt:.2f}s, {status}")

            base = _safe_stem(os.path.splitext(file_name)[0])
            tag = f"R{int(inst.rcapt)}-{int(inst.rcom)}"
            out_name = f"{base}__{tag}__{algo}__S{sol.size()}__RR{restarts}__T{per_run_s}.png"
            save_path = os.path.join(plots_dir, out_name)

            title = f"{file_name} | R=({inst.rcapt},{inst.rcom}) | {algo} | sensors={sol.size()} | uncov={uncovered} | disc={disconnected}"

            plot_solution_auto(
                inst,
                sol,
                title=title,
                save_path=save_path,
                show=False,
            )

            rows.append({
                "file": file_name,
                "rcapt": rc,
                "rcom": rco,
                "algo": algo,
                "per_run_s": per_run_s,
                "restarts": restarts,
                "time_total_s": round(dt, 4),
                "sensors": sol.size(),
                "feasible": feas,
                "uncovered": uncovered,
                "disconnected": disconnected,
                "plot_path": save_path,
            })

    if not rows:
        raise RuntimeError("No results produced (no instances / pairs).")

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    feasible_count = sum(1 for r in rows if r["feasible"])
    avg_sensors = sum(r["sensors"] for r in rows) / len(rows) if rows else 0
    total_time = sum(r["time_total_s"] for r in rows)
    
    print(f"\n{'='*70}")
    print(f"Termine: {len(rows)} resultat(s) -> {csv_out}")
    print(f"  Faisables: {feasible_count}/{len(rows)}, Moyenne: {avg_sensors:.2f} capteurs")
    print(f"  Temps: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"{'='*70}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=str, default=None, help="Path to a zip of instances (optional).")
    ap.add_argument("--folder", type=str, default=None, help="Folder containing .dat instances (optional).")
    ap.add_argument("--outdir", type=str, default="./_instances_extract", help="Where to extract zip (if used).")
    ap.add_argument("--csv", type=str, default="results.csv", help="Output CSV file.")

    ap.add_argument("--algo", type=str, default="sa", choices=["vns", "ga", "sa", "annealing"])
    ap.add_argument("--time", type=float, default=2.0, help="Temps par run (secondes).")
    ap.add_argument("--per-run", type=float, default=None, help="Temps par restart (secondes).")
    ap.add_argument("--restarts", type=int, default=1, help="Nombre de restarts par instance+paire.")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.3, help="Alpha pour construction randomisée (0..1).")
    ap.add_argument("--kmax", type=int, default=4, help="VNS kmax.")

    ap.add_argument("--rcapt", type=float, default=None, help="If set with --rcom, run only this pair.")
    ap.add_argument("--rcom", type=float, default=None, help="If set with --rcapt, run only this pair.")

    # Plot mode (single instance)
    ap.add_argument("--plot", action="store_true", help="Plot the solution for a single instance.")
    ap.add_argument("--file", type=str, default=None, help="Path to one .dat instance (for plotting).")

    args = ap.parse_args()

    per_run_s = args.per_run if args.per_run is not None else args.time
    restarts = max(1, int(args.restarts))

    # Mode plot
    if args.plot:
        if args.file is None:
            raise SystemExit("--plot requires --file <path_to_dat>")
        if args.rcapt is None or args.rcom is None:
            raise SystemExit("--plot requires --rcapt and --rcom")

        targets = load_targets(args.file, sink=(0.0, 0.0))
        inst = Instance.build(
            targets,
            sink=(0.0, 0.0),
            rcapt=float(args.rcapt),
            rcom=float(args.rcom),
        )

        print(f"Multi-départ: {restarts} restart(s) avec {per_run_s}s par run")
        sol = solve_multi(
            inst,
            algo=args.algo,
            per_run_s=per_run_s,
            restarts=restarts,
            seed=args.seed,
            alpha=args.alpha,
            kmax=args.kmax,
            verbose=True,  # Show multi-start progress in plot mode
        )

        os.makedirs(os.path.join("results", "plots"), exist_ok=True)
        base = _safe_stem(os.path.splitext(os.path.basename(args.file))[0])
        tag = f"R{int(inst.rcapt)}-{int(inst.rcom)}"
        out_name = f"{base}__{tag}__{args.algo}__S{sol.size()}__RR{restarts}__T{per_run_s}.png"
        save_path = os.path.join("results", "plots", out_name)

        uncovered = inst.n - len(covered_targets(inst, sol))
        disconnected = sol.size() - len(connected_to_sink(inst, sol))
        title = (
            f"{os.path.basename(args.file)} | R=({inst.rcapt},{inst.rcom}) | "
            f"{args.algo} | sensors={sol.size()} | uncov={uncovered} | disc={disconnected}"
        )

        plot_solution_auto(
            inst,
            sol,
            title=title,
            save_path=save_path,
            show=True,
        )

        print(f"Saved plot -> {save_path}")
        return

    # Mode batch
    if args.zip is None and args.folder is None:
        raise SystemExit("Provide either --zip or --folder.")

    if args.zip is not None:
        folder = ensure_extracted(args.zip, args.outdir)
    else:
        folder = args.folder

    paths = iter_dat_files(folder)
    if not paths:
        raise SystemExit(f"No .dat files found in {folder}")

    run_batch(
        paths=paths,
        rcapt=args.rcapt,
        rcom=args.rcom,
        pairs=DEFAULT_PAIRS,
        algo=args.algo,
        per_run_s=per_run_s,
        restarts=restarts,
        seed=args.seed,
        alpha=args.alpha,
        kmax=args.kmax,
        csv_out=args.csv
    )

    print(f"Done. Wrote: {args.csv}")
    print(f"Instances processed: {len(paths)}")


if __name__ == "__main__":
    main()
