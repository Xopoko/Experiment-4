#!/usr/bin/env python3
"""Q1.1: enumerate the 2×2×1 Ising lattice for free/PBC boundaries."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SpinConfig = Tuple[int, ...]
Edge = Tuple[int, int]


@dataclass(frozen=True)
class Term:
    bonds_sum: int  # Σ_{<ij>} σ_i σ_j
    magnetization: int  # Σ_i σ_i
    degeneracy: int

    def weight(self, k: float, h: float) -> float:
        return self.degeneracy * math.exp(k * self.bonds_sum + h * self.magnetization)

    def to_json(self) -> dict:
        return {
            "bond_sum": self.bonds_sum,
            "magnetization": self.magnetization,
            "degeneracy": self.degeneracy,
        }


def lattice_sites(lx: int, ly: int, lz: int) -> List[Tuple[int, int, int]]:
    return [(x, y, z) for x in range(lx) for y in range(ly) for z in range(lz)]


def build_edges(
    lx: int,
    ly: int,
    lz: int,
    bc_x: str = "free",
    bc_y: str = "free",
    bc_z: str = "free",
) -> List[Edge]:
    coords = lattice_sites(lx, ly, lz)
    index = {coord: idx for idx, coord in enumerate(coords)}
    edges: List[Edge] = []

    for (x, y, z) in coords:
        for axis, size, bc in (
            ("x", lx, bc_x),
            ("y", ly, bc_y),
            ("z", lz, bc_z),
        ):
            if size == 1:
                continue
            coord_list = [x, y, z]
            axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
            if coord_list[axis_idx] == size - 1:
                if bc != "pbc":
                    continue
                coord_list[axis_idx] = 0
            else:
                coord_list[axis_idx] += 1
            neighbor = tuple(coord_list)
            i, j = index[(x, y, z)], index[neighbor]
            edge = (min(i, j), max(i, j))
            edges.append(edge)
    return edges


def enumerate_terms(edges: Sequence[Edge], n_sites: int) -> Tuple[List[Term], Dict[int, int]]:
    terms_counter: Dict[Tuple[int, int], int] = defaultdict(int)
    energy_hist: Dict[int, int] = defaultdict(int)

    for spins in itertools.product((-1, 1), repeat=n_sites):
        magnetization = sum(spins)
        bond_sum = sum(spins[i] * spins[j] for i, j in edges)
        terms_counter[(bond_sum, magnetization)] += 1
        energy_hist[bond_sum] += 1

    terms = [
        Term(bonds_sum=bond_sum, magnetization=magnetization, degeneracy=count)
        for (bond_sum, magnetization), count in sorted(terms_counter.items())
    ]
    return terms, dict(sorted(energy_hist.items()))


def partition_from_terms(terms: Sequence[Term], k: float, h: float) -> float:
    return sum(term.weight(k, h) for term in terms)


def brute_force_partition(edges: Sequence[Edge], n_sites: int, k: float, h: float) -> float:
    total = 0.0
    for spins in itertools.product((-1, 1), repeat=n_sites):
        bond_sum = sum(spins[i] * spins[j] for i, j in edges)
        magnetization = sum(spins)
        total += math.exp(k * bond_sum + h * magnetization)
    return total


def verify_terms(
    terms: Sequence[Term], edges: Sequence[Edge], n_sites: int, k: float, h: float
) -> float:
    z_terms = partition_from_terms(terms, k, h)
    z_brute = brute_force_partition(edges, n_sites, k, h)
    return abs(z_terms - z_brute)


def run_analysis(output: Path | None = None) -> dict:
    lx, ly, lz = 2, 2, 1
    bc_specs = {
        "free": dict(bc_x="free", bc_y="free", bc_z="free"),
        "pbc_xy": dict(bc_x="pbc", bc_y="pbc", bc_z="free"),
    }

    payload = {"lattice": [lx, ly, lz], "results": {}}
    check_points = [(0.3, -0.1), (0.0, 0.5), (1.0, 0.0)]

    for label, bc in bc_specs.items():
        edges = build_edges(lx, ly, lz, **bc)
        n_sites = lx * ly * lz
        terms, energy_hist = enumerate_terms(edges, n_sites)
        max_err = max(verify_terms(terms, edges, n_sites, k, h) for k, h in check_points)
        payload["results"][label] = {
            "boundary_conditions": bc,
            "num_bonds": len(edges),
            "terms": [term.to_json() for term in terms],
            "energy_histogram": energy_hist,
            "validation": {
                "check_points": check_points,
                "max_abs_partition_error": max_err,
            },
        }

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enumerate Q1.1 (2×2×1) lattice.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/q1_1_2x2x1/results.json"),
        help="Path to store JSON with enumeration results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_analysis(args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
