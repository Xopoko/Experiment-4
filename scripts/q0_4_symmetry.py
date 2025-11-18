#!/usr/bin/env python3
"""Q0.4: numerical verification of Z(K, H) = Z(K, -H)."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Edge = Tuple[int, int]
TermKey = Tuple[int, int]


@dataclass(frozen=True)
class LatticeSpec:
    """Finite simple-cubic lattice specification."""

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


@dataclass
class SymmetryRecord:
    K: float
    H: float
    Z_plus: float
    Z_minus: float

    def to_json(self) -> dict:
        abs_diff = abs(self.Z_plus - self.Z_minus)
        rel_diff = abs_diff / max(abs(self.Z_plus), abs(self.Z_minus))
        return {
            "K": self.K,
            "H": self.H,
            "Z(K,H)": self.Z_plus,
            "Z(K,-H)": self.Z_minus,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
        }


def lattice_sites(lx: int, ly: int, lz: int) -> Iterable[Tuple[int, int, int]]:
    for x in range(lx):
        for y in range(ly):
            for z in range(lz):
                yield x, y, z


def linear_index(x: int, y: int, z: int, shape: Tuple[int, int, int]) -> int:
    lx, ly, lz = shape
    return x + lx * (y + ly * z)


def build_edges(spec: LatticeSpec) -> List[Edge]:
    lx, ly, lz = spec.shape
    bc_x, bc_y, bc_z = spec.boundary_conditions
    bc_map = {"x": bc_x, "y": bc_y, "z": bc_z}
    size_map = {"x": lx, "y": ly, "z": lz}

    edges: set[Edge] = set()
    for (x, y, z) in lattice_sites(lx, ly, lz):
        for axis in ("x", "y", "z"):
            size = size_map[axis]
            if size == 1:
                continue
            bc = bc_map[axis]
            coord = {"x": x, "y": y, "z": z}
            if coord[axis] == size - 1:
                if bc != "pbc":
                    continue
                coord[axis] = 0
            else:
                coord[axis] += 1
            neighbor = (coord["x"], coord["y"], coord["z"])
            i = linear_index(x, y, z, spec.shape)
            j = linear_index(*neighbor, spec.shape)
            if i == j:
                continue
            edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def enumerate_terms(edges: Sequence[Edge], n_sites: int) -> Dict[TermKey, int]:
    counts: Dict[TermKey, int] = defaultdict(int)
    for spins in itertools.product((-1, 1), repeat=n_sites):
        bond_sum = sum(spins[i] * spins[j] for i, j in edges)
        magnetization = sum(spins)
        counts[(bond_sum, magnetization)] += 1
    return counts


def partition_sum(terms: Dict[TermKey, int], K: float, H: float) -> float:
    total = 0.0
    for (bond_sum, magnetization), count in terms.items():
        total += count * math.exp(K * bond_sum + H * magnetization)
    return total


def analyze_case(spec: LatticeSpec, Ks: Sequence[float], Hs: Sequence[float]) -> dict:
    edges = build_edges(spec)
    terms = enumerate_terms(edges, spec.n_sites)
    checks: List[dict] = []
    for K in Ks:
        for H in Hs:
            z_pos = partition_sum(terms, K, H)
            z_neg = partition_sum(terms, K, -H)
            record = SymmetryRecord(K=K, H=H, Z_plus=z_pos, Z_minus=z_neg)
            checks.append(record.to_json())

    max_abs = max(check["abs_diff"] for check in checks) if checks else 0.0
    max_rel = max(check["rel_diff"] for check in checks) if checks else 0.0
    return {
        "spec": spec.to_json_meta(),
        "num_edges": len(edges),
        "num_terms": len(terms),
        "checks": checks,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
    }


def default_cases() -> List[LatticeSpec]:
    return [
        LatticeSpec("single_spin", (1, 1, 1), ("free", "free", "free")),
        LatticeSpec("chain_1x1x2", (1, 1, 2), ("free", "free", "free")),
        LatticeSpec("chain_1x1x3", (1, 1, 3), ("free", "free", "free")),
        LatticeSpec("plaquette_2x2_free", (2, 2, 1), ("free", "free", "free")),
        LatticeSpec("torus_2x2_pbc", (2, 2, 1), ("pbc", "pbc", "free")),
        LatticeSpec("cube_2_free", (2, 2, 2), ("free", "free", "free")),
        LatticeSpec("cube_2_pbcxy", (2, 2, 2), ("pbc", "pbc", "free")),
    ]


def parse_csv_floats(raw: str) -> List[float]:
    return [float(part) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the Q0.4 symmetry Z(K,H)=Z(K,-H) on small lattices."
    )
    parser.add_argument(
        "--K-values",
        type=parse_csv_floats,
        default=parse_csv_floats("-1.2,-0.4,0.0,0.6,1.2"),
        help="Comma-separated list of K values (default: %(default)s)",
    )
    parser.add_argument(
        "--H-values",
        type=parse_csv_floats,
        default=parse_csv_floats("-0.9,-0.3,0.0,0.45,1.1"),
        help="Comma-separated list of H values (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/q0_4_symmetry/results.json"),
        help="Path to store JSON output with checks and metrics.",
    )
    return parser.parse_args()


def run_analysis(Ks: Sequence[float], Hs: Sequence[float]) -> dict:
    cases = default_cases()
    case_results = [analyze_case(spec, Ks, Hs) for spec in cases]
    max_abs = max(case["max_abs_diff"] for case in case_results)
    max_rel = max(case["max_rel_diff"] for case in case_results)
    total_checks = sum(len(case["checks"]) for case in case_results)
    return {
        "question_id": "Q0.4",
        "description": "Numerical verification that Z(K,H) = Z(K,-H) for finite lattices.",
        "K_values": list(Ks),
        "H_values": list(Hs),
        "cases": case_results,
        "metrics": {
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "num_checks": total_checks,
        },
    }


def main() -> None:
    args = parse_args()
    payload = run_analysis(args.K_values, args.H_values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
