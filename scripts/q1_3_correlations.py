#!/usr/bin/env python3
r"""Q1.3: nearest-neighbour correlations for 2×2×1 and 2×2×2 at H=0.

The script enumerates all spin configurations, builds the degeneracy table
over bond sums S=\sum_{<ij>} sigma_i sigma_j, and derives
⟨sigma_i sigma_j⟩(K)= (1/|E|) (\sum S g_S e^{KS})/(\sum g_S e^{KS}).

Outputs a JSON artifact with degeneracies, closed-form polynomials in
x=e^{2K}, and numeric checks vs brute force.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SpinConfig = Tuple[int, ...]
Edge = Tuple[int, int]


def lattice_sites(lx: int, ly: int, lz: int) -> List[Tuple[int, int, int]]:
    return [(x, y, z) for x in range(lx) for y in range(ly) for z in range(lz)]


def build_edges(lx: int, ly: int, lz: int) -> List[Edge]:
    """Build nearest-neighbour edges with free boundaries."""

    coords = lattice_sites(lx, ly, lz)
    index = {coord: idx for idx, coord in enumerate(coords)}
    edges: List[Edge] = []

    for (x, y, z) in coords:
        for axis, size in ("x", lx), ("y", ly), ("z", lz):
            if size == 1:
                continue
            coord_list = [x, y, z]
            axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
            if coord_list[axis_idx] == size - 1:
                continue  # free BC
            coord_list[axis_idx] += 1
            neighbour = tuple(coord_list)
            i, j = index[(x, y, z)], index[neighbour]
            edge = (min(i, j), max(i, j))
            edges.append(edge)
    return edges


def enumerate_bond_sums(edges: List[Edge], n_sites: int) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for spins in itertools.product((-1, 1), repeat=n_sites):
        bond_sum = sum(spins[i] * spins[j] for i, j in edges)
        counts[bond_sum] += 1
    return dict(sorted(counts.items()))


def correlation_from_counts(counts: Dict[int, int], edges_count: int, k: float) -> float:
    """Compute ⟨σ_i σ_j⟩ from degeneracies at a given K."""

    num = sum(s * g * math.exp(k * s) for s, g in counts.items())
    den = sum(g * math.exp(k * s) for s, g in counts.items())
    return num / (edges_count * den)


def brute_force_corr(edges: List[Edge], n_sites: int, k: float) -> float:
    num = 0.0
    den = 0.0
    for spins in itertools.product((-1, 1), repeat=n_sites):
        weight = math.exp(k * sum(spins[i] * spins[j] for i, j in edges))
        den += weight
        num += weight * sum(spins[i] * spins[j] for i, j in edges)
    return num / (len(edges) * den)


def poly_str(coeffs: Dict[int, int], var: str = "x") -> str:
    terms = []
    for exp in sorted(coeffs):
        coef = coeffs[exp]
        if coef == 0:
            continue
        if exp == 0:
            terms.append(f"{coef}")
        elif exp == 1:
            terms.append(f"{coef} {var}")
        else:
            terms.append(f"{coef} {var}^{exp}")
    if not terms:
        return "0"
    return " + ".join(terms)


def ratio_polynomials(counts: Dict[int, int]) -> dict:
    """Return numerator/denominator polynomials in x=e^{2K} with nonnegative exponents."""

    exps = [s // 2 for s in counts]
    min_e = min(exps)
    shift = -min_e
    num_coeffs: Dict[int, int] = defaultdict(int)
    den_coeffs: Dict[int, int] = defaultdict(int)

    for s, g in counts.items():
        e = s // 2
        exp = e + shift  # ensures >=0
        num_coeffs[exp] += s * g
        den_coeffs[exp] += g

    return {
        "variable": "x = exp(2K)",
        "min_power": min_e,
        "numerator": num_coeffs,
        "denominator": den_coeffs,
        "numerator_str": poly_str(num_coeffs),
        "denominator_str": poly_str(den_coeffs),
        "note": "P_num/P_den already absorbs x^{-min_power}; correlation = (1/|E|) * P_num(x)/P_den(x)",
    }


@dataclass
class LatticeResult:
    label: str
    dims: Tuple[int, int, int]
    edges: List[Edge]
    counts: Dict[int, int]
    samples: Dict[float, Dict[str, float]]
    max_abs_diff: float
    ratio: dict

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "dims": list(self.dims),
            "num_sites": self.dims[0] * self.dims[1] * self.dims[2],
            "num_edges": len(self.edges),
            "degeneracy_by_S": self.counts,
            "ratio": self.ratio,
            "samples": self.samples,
            "max_abs_diff": self.max_abs_diff,
        }


def analyze_lattice(label: str, dims: Tuple[int, int, int], k_grid: Iterable[float]) -> LatticeResult:
    lx, ly, lz = dims
    edges = build_edges(lx, ly, lz)
    counts = enumerate_bond_sums(edges, lx * ly * lz)
    samples: Dict[float, Dict[str, float]] = {}
    max_abs_diff = 0.0

    for k in k_grid:
        corr_counts = correlation_from_counts(counts, len(edges), k)
        corr_brute = brute_force_corr(edges, lx * ly * lz, k)
        diff = abs(corr_counts - corr_brute)
        max_abs_diff = max(max_abs_diff, diff)
        samples[k] = {
            "corr_from_counts": corr_counts,
            "corr_bruteforce": corr_brute,
            "abs_diff": diff,
        }

    ratio = ratio_polynomials(counts)
    return LatticeResult(label, dims, edges, counts, samples, max_abs_diff, ratio)


def run(output: Path | None = None) -> dict:
    k_grid = (-1.0, -0.4, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0)
    lattices = [
        ("2x2x1_free", (2, 2, 1)),
        ("2x2x2_free", (2, 2, 2)),
    ]

    results = [analyze_lattice(label, dims, k_grid) for label, dims in lattices]
    payload = {
        "task": "Q1.3 correlations for finite free lattices at H=0",
        "k_grid": k_grid,
        "results": {res.label: res.to_json() for res in results},
    }

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enumerate correlations for Q1.3 lattices.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/q1_3_correlations/results.json"),
        help="Path to JSON artifact with degeneracies and correlation formulas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
