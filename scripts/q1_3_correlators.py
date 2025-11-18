#!/usr/bin/env python3
"""Q1.3: nearest-neighbour correlators on 2×2×1 and 2×2×2 at H=0."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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


def correlate(edges: Sequence[Edge], n_sites: int, K_values: Sequence[float]) -> Dict[float, float]:
    results: Dict[float, float] = {}
    configs = list(itertools.product((-1, 1), repeat=n_sites))
    for K in K_values:
        weights: List[float] = []
        corr_num = 0.0
        for spins in configs:
            bond_sum = sum(spins[i] * spins[j] for i, j in edges)
            w = math.exp(K * bond_sum)
            weights.append(w)
            corr_num += bond_sum * w
        Z = sum(weights)
        avg_bond = corr_num / (Z * len(edges))
        results[K] = avg_bond
    return results


def run_analysis(output: Path | None = None) -> dict:
    specs = [
        LatticeSpec("layer_2x2x1", (2, 2, 1), ("free", "free", "free")),
        LatticeSpec("cube_2x2x2", (2, 2, 2), ("free", "free", "free")),
    ]
    K_values = [-1.2, -0.6, -0.2, 0.0, 0.4, 0.8, 1.2]

    cases = []
    for spec in specs:
        edges = build_edges(spec)
        corr = correlate(edges, spec.n_sites, K_values)
        cases.append(
            {
                "spec": spec.to_json_meta(),
                "num_edges": len(edges),
                "K_values": K_values,
                "correlator": corr,
            }
        )

    payload = {"question_id": "Q1.3", "cases": cases}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nearest-neighbour correlators for Q1.3.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/q1_3_correlators/results.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_analysis(args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
