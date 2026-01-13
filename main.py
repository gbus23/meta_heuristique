from __future__ import annotations

import os
import csv
import glob
import time
import zipfile
import argparse
from typing import List, Tuple, Optional

from meta.io_instances import load_targets
from meta.instance import Instance
from meta.solution import is_feasible, covered_targets, connected_to_sink
from meta.grasp import grasp
from meta.vns import vns
from meta.visualization import plot_solution_auto
from meta.comparison import generate_all_comparisons


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
    if algo == "grasp":
        return grasp(inst, time_limit_s=time_limit_s, alpha=alpha, seed=seed)
    if algo == "vns":
        return vns(inst, time_limit_s=time_limit_s, kmax=kmax, seed=seed)
    if algo == "both":
        s1 = grasp(inst, time_limit_s=time_limit_s, alpha=alpha, seed=seed)
        s2 = vns(inst, time_limit_s=time_limit_s, kmax=kmax, seed=seed)
        if is_feasible(inst, s1) and is_feasible(inst, s2):
            return s1 if s1.size() <= s2.size() else s2
        return s1 if is_feasible(inst, s1) else s2
    raise ValueError(f"Unknown algo: {algo}")


def solve_both_separately(inst: Instance, time_limit_s: float, seed: int, alpha: float, kmax: int):
    """Résout avec GRASP et VNS séparément et retourne les deux solutions."""
    s1 = grasp(inst, time_limit_s=time_limit_s, alpha=alpha, seed=seed)
    s2 = vns(inst, time_limit_s=time_limit_s, kmax=kmax, seed=seed)
    return s1, s2


def run_batch(paths: List[str],
              rcapt: Optional[float],
              rcom: Optional[float],
              pairs: List[Tuple[int, int]],
              algo: str,
              time_limit_s: float,
              seed: int,
              alpha: float,
              kmax: int,
              csv_out: str) -> None:
    rows = []
    os.makedirs(os.path.join("results", "plots"), exist_ok=True)

    total_tasks = len(paths) * (len(pairs) if (rcapt is None or rcom is None) else 1)
    if algo == "both":
        total_tasks *= 2  # GRASP et VNS séparément
    current_task = 0

    print(f"\n{'='*80}")
    print(f"Démarrage du traitement batch")
    print(f"  Instances: {len(paths)}")
    print(f"  Paires (Rcapt, Rcom): {len(pairs) if (rcapt is None or rcom is None) else 1}")
    print(f"  Algorithme(s): {algo}")
    print(f"  Limite de temps: {time_limit_s}s par instance")
    print(f"{'='*80}\n")

    for p in paths:
        file_name = os.path.basename(p)
        used_pairs = [(int(rcapt), int(rcom))] if (rcapt is not None and rcom is not None) else pairs

        for (rc, rco) in used_pairs:
            targets = load_targets(p, sink=(0.0, 0.0))
            inst = Instance.build(targets, sink=(0.0, 0.0), rcapt=float(rc), rcom=float(rco))

            if algo == "both":
                # Exécuter GRASP et VNS séparément pour avoir des résultats comparables
                current_task += 1
                print(f"[{current_task}/{total_tasks}] {file_name} R=({rc},{rco}) - GRASP...", end=" ", flush=True)
                t0 = time.time()
                sol_grasp = grasp(inst, time_limit_s=time_limit_s, alpha=alpha, seed=seed)
                dt_grasp = time.time() - t0
                
                feas_grasp = is_feasible(inst, sol_grasp)
                uncovered_grasp = inst.n - len(covered_targets(inst, sol_grasp))
                disconnected_grasp = sol_grasp.size() - len(connected_to_sink(inst, sol_grasp))
                
                base = _safe_stem(os.path.splitext(file_name)[0])
                tag = f"R{int(inst.rcapt)}-{int(inst.rcom)}"
                out_name_grasp = f"{base}__{tag}__grasp__S{sol_grasp.size()}.png"
                save_path_grasp = os.path.join("results", "plots", out_name_grasp)
                
                title_grasp = (
                    f"{file_name} | R=({inst.rcapt},{inst.rcom}) | grasp | "
                    f"sensors={sol_grasp.size()} | uncov={uncovered_grasp} | disc={disconnected_grasp}"
                )
                
                plot_solution_auto(inst, sol_grasp, title=title_grasp, save_path=save_path_grasp, show=False)
                print(f"✓ {sol_grasp.size()} capteurs ({dt_grasp:.2f}s)")
                
                rows.append({
                    "file": file_name,
                    "rcapt": rc,
                    "rcom": rco,
                    "algo": "grasp",
                    "time_s": round(dt_grasp, 4),
                    "sensors": sol_grasp.size(),
                    "feasible": feas_grasp,
                    "uncovered": uncovered_grasp,
                    "disconnected": disconnected_grasp,
                    "plot_path": save_path_grasp,
                })
                
                current_task += 1
                print(f"[{current_task}/{total_tasks}] {file_name} R=({rc},{rco}) - VNS...", end=" ", flush=True)
                t0 = time.time()
                sol_vns = vns(inst, time_limit_s=time_limit_s, kmax=kmax, seed=seed)
                dt_vns = time.time() - t0
                
                feas_vns = is_feasible(inst, sol_vns)
                uncovered_vns = inst.n - len(covered_targets(inst, sol_vns))
                disconnected_vns = sol_vns.size() - len(connected_to_sink(inst, sol_vns))
                
                out_name_vns = f"{base}__{tag}__vns__S{sol_vns.size()}.png"
                save_path_vns = os.path.join("results", "plots", out_name_vns)
                
                title_vns = (
                    f"{file_name} | R=({inst.rcapt},{inst.rcom}) | vns | "
                    f"sensors={sol_vns.size()} | uncov={uncovered_vns} | disc={disconnected_vns}"
                )
                
                plot_solution_auto(inst, sol_vns, title=title_vns, save_path=save_path_vns, show=False)
                print(f"✓ {sol_vns.size()} capteurs ({dt_vns:.2f}s)")
                
                rows.append({
                    "file": file_name,
                    "rcapt": rc,
                    "rcom": rco,
                    "algo": "vns",
                    "time_s": round(dt_vns, 4),
                    "sensors": sol_vns.size(),
                    "feasible": feas_vns,
                    "uncovered": uncovered_vns,
                    "disconnected": disconnected_vns,
                    "plot_path": save_path_vns,
                })
            else:
                current_task += 1
                print(f"[{current_task}/{total_tasks}] {file_name} R=({rc},{rco}) - {algo.upper()}...", end=" ", flush=True)
                
                t0 = time.time()
                sol = solve_one(inst, algo=algo, time_limit_s=time_limit_s, seed=seed, alpha=alpha, kmax=kmax)
                dt = time.time() - t0

                feas = is_feasible(inst, sol)
                uncovered = inst.n - len(covered_targets(inst, sol))
                disconnected = sol.size() - len(connected_to_sink(inst, sol))

                base = _safe_stem(os.path.splitext(file_name)[0])
                tag = f"R{int(inst.rcapt)}-{int(inst.rcom)}"
                out_name = f"{base}__{tag}__{algo}__S{sol.size()}.png"
                save_path = os.path.join("results", "plots", out_name)

                title = (
                    f"{file_name} | R=({inst.rcapt},{inst.rcom}) | {algo} | "
                    f"sensors={sol.size()} | uncov={uncovered} | disc={disconnected}"
                )

                plot_solution_auto(inst, sol, title=title, save_path=save_path, show=False)
                print(f"✓ {sol.size()} capteurs ({dt:.2f}s)")

                rows.append({
                    "file": file_name,
                    "rcapt": rc,
                    "rcom": rco,
                    "algo": algo,
                    "time_s": round(dt, 4),
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
    
    print(f"\n{'='*80}")
    print(f"Traitement terminé!")
    print(f"  Résultats sauvegardés dans: {csv_out}")
    print(f"  Total de résultats: {len(rows)}")
    print(f"{'='*80}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=str, default=None, help="Path to a zip of instances (optional).")
    ap.add_argument("--folder", type=str, default=None, help="Folder containing .dat instances (optional).")
    ap.add_argument("--outdir", type=str, default="./_instances_extract", help="Where to extract zip (if used).")
    ap.add_argument("--csv", type=str, default="results.csv", help="Output CSV file.")

    ap.add_argument("--algo", type=str, default="both", choices=["grasp", "vns", "both"])
    ap.add_argument("--time", type=float, default=2.0, help="Time limit per instance+pair (seconds).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.3, help="GRASP alpha for RCL (0..1).")
    ap.add_argument("--kmax", type=int, default=4, help="VNS kmax.")

    ap.add_argument("--rcapt", type=float, default=None, help="If set with --rcom, run only this pair.")
    ap.add_argument("--rcom", type=float, default=None, help="If set with --rcapt, run only this pair.")

    # Plot mode (single instance)
    ap.add_argument("--plot", action="store_true", help="Plot the solution for a single instance.")
    ap.add_argument("--file", type=str, default=None, help="Path to one .dat instance (for plotting).")
    
    # Comparison mode
    ap.add_argument("--compare", action="store_true", help="Generate comparison tables and graphs from CSV results.")
    ap.add_argument("--csv-in", type=str, default="results.csv", help="Input CSV file for comparison (default: results.csv).")

    args = ap.parse_args()

    # ---------- COMPARISON MODE ----------
    if args.compare:
        if not os.path.exists(args.csv_in):
            raise SystemExit(f"Fichier CSV introuvable: {args.csv_in}")
        generate_all_comparisons(args.csv_in, show=False)
        return

    # ---------- PLOT MODE ----------
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

        sol = solve_one(
            inst,
            algo=args.algo,
            time_limit_s=args.time,
            seed=args.seed,
            alpha=args.alpha,
            kmax=args.kmax,
        )

        # Auto-save into results/plots/
        os.makedirs(os.path.join("results", "plots"), exist_ok=True)
        base = _safe_stem(os.path.splitext(os.path.basename(args.file))[0])
        tag = f"R{int(inst.rcapt)}-{int(inst.rcom)}"
        out_name = f"{base}__{tag}__{args.algo}__S{sol.size()}.png"
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

    # ---------- BATCH MODE ----------
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
        time_limit_s=args.time,
        seed=args.seed,
        alpha=args.alpha,
        kmax=args.kmax,
        csv_out=args.csv
    )

    print(f"Done. Wrote: {args.csv}")
    print(f"Instances processed: {len(paths)}")


if __name__ == "__main__":
    main()
