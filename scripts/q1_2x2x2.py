#!/usr/bin/env python3
"""Enumerate the 2×2×2 Ising lattice for Q1.2 (free vs PBC)."""

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
class LatticeSpec:
    name: str
    shape: Tuple[int, int, int]
    boundary_conditions: Tuple[str, str, str]

    @property
    def n_sites(self) -> int:
        lx, ly, lz = self.shape
        return lx * ly * lz

    def to_json_meta(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "boundary_conditions": list(self.boundary_conditions),
            "n_sites": self.n_sites,
        }


def lattice_sites(shape: Tuple[int, int, int]) -> Iterable[Tuple[int, int, int]]:
    lx, ly, lz = shape
    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                yield x, y, z


def linear_index(coord: Tuple[int, int, int], shape: Tuple[int, int, int]) -> int:
    lx, ly, _ = shape
    x, y, z = coord
    return x + lx * (y + ly * z)


def build_edges(spec: LatticeSpec) -> List[Edge]:
    lx, ly, lz = spec.shape
    bc = {
        "x": spec.boundary_conditions[0],
        "y": spec.boundary_conditions[1],
        "z": spec.boundary_conditions[2],
    }
    size = {"x": lx, "y": ly, "z": lz}
    edges: set[Edge] = set()
    for coord in lattice_sites(spec.shape):
        for axis in ("x", "y", "z"):
            if size[axis] == 1:
                continue
            next_coord = list(coord)
            axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
            if next_coord[axis_idx] == size[axis] - 1:
                if bc[axis] != "pbc":
                    continue
                next_coord[axis_idx] = 0
            else:
                next_coord[axis_idx] += 1
            i = linear_index(coord, spec.shape)
            j = linear_index(tuple(next_coord), spec.shape)
            if i != j:
                edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def enumerate_hist(edges: Sequence[Edge], n_sites: int) -> Dict[int, int]:
    hist: Dict[int, int] = defaultdict(int)
    for spins in itertools.product((-1, 1), repeat=n_sites):
        bond_sum = sum(spins[i] * spins[j] for i, j in edges)
        hist[bond_sum] += 1
    return dict(sorted(hist.items()))


def partition_from_hist(hist: Dict[int, int], K: float) -> float:
    return sum(count * math.exp(K * bond_sum) for bond_sum, count in hist.items())


def brute_force_partition(edges: Sequence[Edge], n_sites: int, K: float) -> float:
    total = 0.0
    for spins in itertools.product((-1, 1), repeat=n_sites):
        bond_sum = sum(spins[i] * spins[j] for i, j in edges)
        total += math.exp(K * bond_sum)
    return total


def verify_hist(hist: Dict[int, int], edges: Sequence[Edge], n_sites: int, K_values: Sequence[float]) -> float:
    max_err = 0.0
    for K in K_values:
        z_hist = partition_from_hist(hist, K)
        z_exact = brute_force_partition(edges, n_sites, K)
        max_err = max(max_err, abs(z_hist - z_exact))
    return max_err


def run_analysis(output: Path | None = None) -> dict:
    shape = (2, 2, 2)
    specs = [
        LatticeSpec("cube_free", shape, ("free", "free", "free")),
        LatticeSpec("cube_pbc", shape, ("pbc", "pbc", "pbc")),
    ]
    K_values = [-1.0, -0.4, 0.0, 0.6, 1.2]

    summary = {"shape": list(shape), "cases": []}
    for spec in specs:
        edges = build_edges(spec)
        hist = enumerate_hist(edges, spec.n_sites)
        max_err = verify_hist(hist, edges, spec.n_sites, K_values)
        summary["cases"].append(
            {
                "spec": spec.to_json_meta(),
                "num_edges": len(edges),
                "energy_histogram": hist,
                "validation": {
                    "K_values": K_values,
                    "max_abs_error": max_err,
                },
            }
        )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enumerate the 2×2×2 lattice (Q1.2).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/q1_2x2x2/results.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_analysis(args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
