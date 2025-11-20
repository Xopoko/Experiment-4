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


def build_edges(spec: LatticeSpec, directed: bool = False) -> List[Edge]:
    """Build edge list for the lattice.

    - directed=False: undirected edges counted once (default, previous behavior).
    - directed=True: include both +/- directions per axis (duplicate pairs allowed),
      yielding 6 neighbors per site under PBC even for L=2.
    """

    lx, ly, lz = spec.shape
    bc = {
        "x": spec.boundary_conditions[0],
        "y": spec.boundary_conditions[1],
        "z": spec.boundary_conditions[2],
    }
    size = {"x": lx, "y": ly, "z": lz}
    axis_idx_map = {"x": 0, "y": 1, "z": 2}

    if not directed:
        edges: set[Edge] = set()
        for coord in lattice_sites(spec.shape):
            for axis in ("x", "y", "z"):
                if size[axis] == 1:
                    continue
                next_coord = list(coord)
                idx = axis_idx_map[axis]
                if next_coord[idx] == size[axis] - 1:
                    if bc[axis] != "pbc":
                        continue
                    next_coord[idx] = 0
                else:
                    next_coord[idx] += 1
                i = linear_index(coord, spec.shape)
                j = linear_index(tuple(next_coord), spec.shape)
                if i != j:
                    edges.add((min(i, j), max(i, j)))
        return sorted(edges)

    # directed mode: walk both +/- directions without deduplication
    edges_dir: List[Edge] = []
    for coord in lattice_sites(spec.shape):
        for axis in ("x", "y", "z"):
            if size[axis] == 1:
                continue
            idx = axis_idx_map[axis]
            for step in (-1, 1):
                next_coord = list(coord)
                next_coord[idx] += step
                if next_coord[idx] < 0:
                    if bc[axis] != "pbc":
                        continue
                    next_coord[idx] = size[axis] - 1
                elif next_coord[idx] >= size[axis]:
                    if bc[axis] != "pbc":
                        continue
                    next_coord[idx] = 0
                i = linear_index(coord, spec.shape)
                j = linear_index(tuple(next_coord), spec.shape)
                if i != j:
                    edges_dir.append((i, j))
    return edges_dir


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


def verify_hist(
    hist: Dict[int, int],
    edges: Sequence[Edge],
    n_sites: int,
    K_values: Sequence[float],
    csv_path: Path | None = None,
) -> Dict[str, float]:
    """Return max absolute/relative errors; optionally write a CSV trace for K grid."""

    rows = []
    max_abs = 0.0
    max_rel = 0.0
    for K in K_values:
        z_hist = partition_from_hist(hist, K)
        z_exact = brute_force_partition(edges, n_sites, K)
        diff = abs(z_hist - z_exact)
        rel = diff / abs(z_exact) if z_exact != 0 else 0.0
        rows.append((K, z_hist, z_exact, diff, rel))
        max_abs = max(max_abs, diff)
        max_rel = max(max_rel, rel)

    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("K,Z_hist,Z_exact,abs_error,rel_error\n")
            for K, zh, ze, diff, rel in rows:
                f.write(f"{K},{zh},{ze},{diff},{rel}\n")

    return {"max_abs_error": max_abs, "max_rel_error": max_rel}


def run_analysis(
    output: Path | None = None,
    edge_mode: str = "undirected",
    csv: Path | None = None,
) -> dict:
    shape = (2, 2, 2)
    directed = edge_mode == "directed"
    specs = [
        LatticeSpec("cube_free", shape, ("free", "free", "free")),
        LatticeSpec("cube_pbc", shape, ("pbc", "pbc", "pbc")),
    ]
    K_values = [-1.0, -0.4, 0.0, 0.6, 1.2]

    summary = {"shape": list(shape), "edge_mode": edge_mode, "cases": []}
    for spec in specs:
        edges = build_edges(spec, directed=directed)
        hist = enumerate_hist(edges, spec.n_sites)
        csv_path = None
        if csv and spec.name == "cube_free":
            csv_path = csv.with_stem(csv.stem + f"_{edge_mode}") if csv.suffix else csv
        errors = verify_hist(hist, edges, spec.n_sites, K_values, csv_path)
        summary["cases"].append(
            {
                "spec": spec.to_json_meta(),
                "num_edges": len(edges),
                "energy_histogram": hist,
                "validation": {
                    "K_values": K_values,
                    **errors,
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
    parser.add_argument(
        "--edge-mode",
        choices=["undirected", "directed"],
        default="undirected",
        help="Edge counting convention: undirected (default) counts each pair once; directed counts both ± directions (6 neighbors under PBC).",
    )
    parser.add_argument(
        "--validation-csv",
        type=Path,
        help="Optional path to write a CSV with K, Z_hist, Z_exact, abs_error, rel_error for the first case (free).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_analysis(args.output, edge_mode=args.edge_mode, csv=args.validation_csv)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
