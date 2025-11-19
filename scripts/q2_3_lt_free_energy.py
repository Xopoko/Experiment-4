#!/usr/bin/env python3
"""Enumerate low-temperature corrections for the 3D Ising free energy."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sympy as sp


def build_bonds(size: int) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]]]:
    """Return lattice coordinates and oriented nearest-neighbour bonds with PBC."""

    coords: List[Tuple[int, int, int]] = []
    index_map = {}
    for idx, (x, y, z) in enumerate(itertools.product(range(size), repeat=3)):
        coords.append((x, y, z))
        index_map[(x, y, z)] = idx

    bonds: List[Tuple[int, int]] = []
    # Only take positive directions to avoid double-counting bonds.
    directions = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for x, y, z in coords:
        for dx, dy, dz in directions:
            nb = ((x + dx) % size, (y + dy) % size, (z + dz) % size)
            bonds.append((index_map[(x, y, z)], index_map[nb]))
    return coords, bonds


def enumerate_counts(size: int) -> Dict[int, int]:
    coords, bonds = build_bonds(size)
    num_sites = len(coords)
    counts: Dict[int, int] = {}
    for state in range(1 << num_sites):
        disagree = 0
        for i, j in bonds:
            spin_i = 1 if (state >> i) & 1 else -1
            spin_j = 1 if (state >> j) & 1 else -1
            if spin_i != spin_j:
                disagree += 1
        counts[disagree] = counts.get(disagree, 0) + 1
    return counts


def build_series(counts: Dict[int, int], max_order: int, num_sites: int) -> Tuple[sp.Expr, Dict[int, sp.Expr], Dict[int, sp.Expr]]:
    y = sp.symbols("y")
    poly = sp.Integer(0)
    for disagree, multiplicity in sorted(counts.items()):
        poly += multiplicity * y**disagree
    log_series = sp.series(sp.log(poly), y, 0, max_order + 1).removeO()
    log_per_site = sp.expand(log_series / num_sites)
    free_energy_series = sp.expand(-log_per_site)
    coeffs_log = {}
    coeffs_free = {}
    for power in range(max_order + 1):
        coeffs_log[power] = sp.simplify(log_per_site.coeff(y, power))
        coeffs_free[power] = sp.simplify(free_energy_series.coeff(y, power))
    return log_per_site, coeffs_log, coeffs_free


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LT series via brute-force enumeration")
    parser.add_argument("--size", type=int, default=2, help="Linear lattice size with PBC")
    parser.add_argument("--max-order", type=int, default=6, help="Max power of y to keep")
    parser.add_argument("--output", type=Path, default=Path("artifacts/q2_3_lt_free_energy/results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    counts = enumerate_counts(args.size)
    coords, bonds = build_bonds(args.size)
    log_series, coeffs_log, coeffs_free = build_series(counts, args.max_order, len(coords))
    elapsed = time.time() - start
    payload = {
        "lattice": "sc",
        "size": args.size,
        "num_sites": len(coords),
        "num_bonds": len(bonds),
        "max_order": args.max_order,
        "counts": counts,
        "log_series_per_site": sp.srepr(log_series),
        "coeffs_per_site_log": {str(k): sp.srepr(v) for k, v in coeffs_log.items()},
        "coeffs_free_energy": {str(k): sp.srepr(v) for k, v in coeffs_free.items()},
        "elapsed_s": elapsed,
    }
    args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
