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
from meta.visualization import plot_solution


DEFAULT_PAIRS: List[Tuple[int, int]] = [(1, 1), (1, 2), (2, 2), (2, 3)]


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

    for p in paths:
        file_name = os.path.basename(p)
        used_pairs = [(int(rcapt), int(rcom))] if (rcapt is not None and rcom is not None) else pairs

        for (rc, rco) in used_pairs:
            targets = load_targets(p, sink=(0.0, 0.0))
            inst = Instance.build(targets, sink=(0.0, 0.0), rcapt=float(rc), rcom=float(rco))

            t0 = time.time()
            sol = solve_one(inst, algo=algo, time_limit_s=time_limit_s, seed=seed, alpha=alpha, kmax=kmax)
            dt = time.time() - t0

            feas = is_feasible(inst, sol)
            uncovered = inst.n - len(covered_targets(inst, sol))
            disconnected = sol.size() - len(connected_to_sink(inst, sol))

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
            })

    if not rows:
        raise RuntimeError("No results produced (no instances / pairs).")

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


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
    ap.add_argument("--savefig", type=str, default=None, help="If set, save plot to this path (png).")
    ap.add_argument("--draw_coverage", action="store_true", help="Draw coverage circles (may be slow).")
    ap.add_argument("--draw_edges", action="store_true", help="Draw some comm edges (may be slow).")

    args = ap.parse_args()

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

        title = f"{os.path.basename(args.file)} | R=({inst.rcapt},{inst.rcom}) | {args.algo} | sensors={sol.size()}"
        plot_solution(
            inst,
            sol,
            title=title,
            save_path=args.savefig,
            show=(args.savefig is None),
            draw_coverage=args.draw_coverage,
            draw_comm_edges=args.draw_edges,
        )
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
