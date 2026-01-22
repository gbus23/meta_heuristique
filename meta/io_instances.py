from __future__ import annotations

import os
import re
from typing import List, Tuple, Set

Point = Tuple[float, float]


def _parse_random_dat(path: str) -> List[Point]:
    """Parse un fichier .dat avec liste de points."""
    pts: List[Point] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if "Nombre" in line or "cibles" in line or "Cibles" in line or "Nb" in line:
                continue

            line = line.replace(";", " ").replace(",", " ")
            parts = [p for p in line.split() if p]
            if len(parts) >= 2:
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
    """Parse une grille tronquée : extrait dimensions depuis le nom de fichier et liste des cellules retirées."""
    base = os.path.basename(path)

    m = re.search(r"grille(\d{2})(\d{2})_", base, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        m_ = int(m.group(2))
    else:
        m2 = re.search(r"grille(\d+)_", base, flags=re.IGNORECASE)
        if not m2:
            raise ValueError(f"Impossible d'extraire les dimensions depuis: {base}")
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
                i_file = int(mm.group(1)) - 1
                j_file = int(mm.group(2)) - 1
                removed.add((i_file, j_file))
    return n, m_, removed


def load_targets(path: str, sink: Point = (0.0, 0.0)) -> List[Point]:
    """Charge les cibles depuis un fichier .dat (détection automatique du format)."""
    base = os.path.basename(path).lower()

    if base.startswith("captanor"):
        pts = _parse_random_dat(path)
        if pts and abs(pts[0][0] - sink[0]) < 1e-9 and abs(pts[0][1] - sink[1]) < 1e-9:
            pts = pts[1:]
        return pts

    if base.startswith("grille"):
        n, m, removed = _parse_grid_truncated(path)
        pts: List[Point] = []
        for i in range(0, n + 1):
            for j in range(0, m + 1):
                if (i, j) in removed:
                    continue
                pts.append((float(i), float(j)))
        return pts

    pts = _parse_random_dat(path)
    if pts and abs(pts[0][0] - sink[0]) < 1e-9 and abs(pts[0][1] - sink[1]) < 1e-9:
        pts = pts[1:]
    if not pts:
        raise ValueError(f"Format d'instance inconnu/vide: {path}")
    return pts
