
from __future__ import annotations

import os
import re
from typing import List, Tuple, Set

Point = Tuple[float, float]


def _parse_random_dat(path: str) -> List[Point]:
    """
    Reads a .dat file containing lines like:
      - "Nombre de cibles : 151"   (header, ignored)
      - "id  x  y"                 (kept)
    """
    pts: List[Point] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Skip French headers
            if "Nombre" in line or "cibles" in line or "Cibles" in line or "Nb" in line:
                continue

            line = line.replace(";", " ").replace(",", " ")
            parts = [p for p in line.split() if p]
            if len(parts) >= 2:
                # Either "x y" or "id x y" (or more tokens)
                try:
                    if len(parts) == 2:
                        x, y = float(parts[0]), float(parts[1])
                    else:
                        x, y = float(parts[-2]), float(parts[-1])
                    pts.append((x, y))
                except ValueError:
                    continue
    return pts


def _parse_grid_truncated(path: str) -> Tuple[int, int, Set[Tuple[int, int]]]:
    """
    Truncated grid file typically contains removed cells:
      (i,j)
    The grid dimensions are inferred from filename like:
      grille2020_1.dat  -> 20x20
      grille1015_2.dat  -> 10x15
    """
    base = os.path.basename(path)

    m = re.search(r"grille(\d{2})(\d{2})_", base, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        m_ = int(m.group(2))
    else:
        m2 = re.search(r"grille(\d+)_", base, flags=re.IGNORECASE)
        if not m2:
            raise ValueError(f"Cannot infer grid dims from filename: {base}")
        s = m2.group(1)
        half = len(s) // 2
        n = int(s[:half])
        m_ = int(s[half:])

    removed: Set[Tuple[int, int]] = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mm = re.search(r"\((\d+)\s*,\s*(\d+)\)", line)
            if mm:
                removed.add((int(mm.group(1)), int(mm.group(2))))
    return n, m_, removed


def load_targets(path: str, sink: Point = (0.0, 0.0)) -> List[Point]:
    """
    Auto-detects professor formats and returns target positions suitable as candidate sensor sites.

    - captANOR*.dat  : list of points, usually first point is sink (0,0) -> removed
    - grille*.dat    : truncated grid (file lists removed cells) -> reconstruct full grid

    For unknown .dat formats, tries to parse as a list of points.
    """
    base = os.path.basename(path).lower()

    if base.startswith("captanor"):
        pts = _parse_random_dat(path)
        if pts and abs(pts[0][0] - sink[0]) < 1e-9 and abs(pts[0][1] - sink[1]) < 1e-9:
            pts = pts[1:]
        return pts

    if base.startswith("grille"):
        n, m, removed = _parse_grid_truncated(path)
        pts: List[Point] = []
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if (i, j) in removed:
                    continue
                pts.append((float(i), float(j)))
        return pts

    # fallback: treat as point list
    pts = _parse_random_dat(path)
    if pts and abs(pts[0][0] - sink[0]) < 1e-9 and abs(pts[0][1] - sink[1]) < 1e-9:
        pts = pts[1:]
    if not pts:
        raise ValueError(f"Unknown/empty instance format: {path}")
    return pts
