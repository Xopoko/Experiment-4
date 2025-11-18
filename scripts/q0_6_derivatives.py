#!/usr/bin/env python3
"""Q0.6: Verify derivative identities for finite lattices."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


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


def build_edges(spec: LatticeSpec) -> List[Tuple[int, int]]:
    lx, ly, lz = spec.shape
    bc_map = {"x": spec.boundary_conditions[0], "y": spec.boundary_conditions[1], "z": spec.boundary_conditions[2]}
    size_map = {"x": lx, "y": ly, "z": lz}
    edges: set[Tuple[int, int]] = set()
    for coord in lattice_sites(spec.shape):
        x, y, z = coord
        for axis in ("x", "y", "z"):
            size = size_map[axis]
            if size == 1:
                continue
            bc = bc_map[axis]
            next_coord = list(coord)
            axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
            if next_coord[axis_idx] == size - 1:
                if bc != "pbc":
                    continue
                next_coord[axis_idx] = 0
            else:
                next_coord[axis_idx] += 1
            i = linear_index(coord, spec.shape)
            j = linear_index(tuple(next_coord), spec.shape)
            if i != j:
                edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def enumerate_states(spec: LatticeSpec) -> List[Tuple[int, int]]:
    edges = build_edges(spec)
    n = spec.n_sites
    state_data: List[Tuple[int, int]] = []
    for spins in itertools.product((-1, 1), repeat=n):
        magnetization = sum(spins)
        bond_sum = sum(spins[i] * spins[j] for i, j in edges)
        state_data.append((bond_sum, magnetization))
    return state_data


def partition_stats(state_data: List[Tuple[int, int]], K: float, H: float) -> Tuple[float, float, float]:
    weights: List[float] = []
    for S, M in state_data:
        weights.append(math.exp(K * S + H * M))
    Z = sum(weights)
    exp_M = sum(M * w for (S, M), w in zip(state_data, weights)) / Z
    exp_S = sum(S * w for (S, M), w in zip(state_data, weights)) / Z
    return Z, exp_M, exp_S


def numerical_derivative(state_data: List[Tuple[int, int]], K: float, H: float, eps: float = 1e-6) -> Tuple[float, float]:
    Z_hp, _, _ = partition_stats(state_data, K, H + eps)
    Z_hm, _, _ = partition_stats(state_data, K, H - eps)
    d_logZ_dH = (math.log(Z_hp) - math.log(Z_hm)) / (2 * eps)

    Z_kp, _, _ = partition_stats(state_data, K + eps, H)
    Z_km, _, _ = partition_stats(state_data, K - eps, H)
    d_logZ_dK = (math.log(Z_kp) - math.log(Z_km)) / (2 * eps)
    return d_logZ_dH, d_logZ_dK


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
        description="Verify derivative identities for finite Ising lattices."
    )
    parser.add_argument(
        "--K-values",
        type=parse_csv_floats,
        default=parse_csv_floats("-0.8,-0.2,0.0,0.5,1.0"),
        help="Comma-separated K values (default: %(default)s)",
    )
    parser.add_argument(
        "--H-values",
        type=parse_csv_floats,
        default=parse_csv_floats("-0.6,-0.15,0.0,0.4,0.9"),
        help="Comma-separated H values (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/q0_6_derivatives/results.json"),
        help="Path to store JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = default_cases()
    payload_cases = []
    eps = 1e-6
    all_abs_err_H: List[float] = []
    all_rel_err_H: List[float] = []
    all_abs_err_K: List[float] = []
    all_rel_err_K: List[float] = []

    for spec in cases:
        state_data = enumerate_states(spec)
        entries = []
        for K in args.K_values:
            for H in args.H_values:
                Z, exp_M, exp_S = partition_stats(state_data, K, H)
                d_logZ_dH, d_logZ_dK = numerical_derivative(state_data, K, H, eps)
                abs_err_H = abs(exp_M - d_logZ_dH)
                rel_err_H = abs_err_H / (abs(exp_M) + 1e-12)
                abs_err_K = abs(exp_S - d_logZ_dK)
                rel_err_K = abs_err_K / (abs(exp_S) + 1e-12)
                all_abs_err_H.append(abs_err_H)
                all_rel_err_H.append(rel_err_H)
                all_abs_err_K.append(abs_err_K)
                all_rel_err_K.append(rel_err_K)
                entries.append(
                    {
                        "K": K,
                        "H": H,
                        "logZ": math.log(Z),
                        "sum_sigma": exp_M,
                        "sum_bonds": exp_S,
                        "d_logZ_dH": d_logZ_dH,
                        "d_logZ_dK": d_logZ_dK,
                        "abs_err_H": abs_err_H,
                        "rel_err_H": rel_err_H,
                        "abs_err_K": abs_err_K,
                        "rel_err_K": rel_err_K,
                    }
                )
        payload_cases.append(
            {
                "spec": spec.to_json_meta(),
                "num_states": len(state_data),
                "results": entries,
            }
        )

    metrics = {
        "max_abs_err_H": max(all_abs_err_H),
        "max_rel_err_H": max(all_rel_err_H),
        "max_abs_err_K": max(all_abs_err_K),
        "max_rel_err_K": max(all_rel_err_K),
        "num_checks": len(all_abs_err_H),
        "epsilon": eps,
    }

    payload = {
        "question_id": "Q0.6",
        "description": "Compare ⟨Σσ_i⟩ and ⟨Σσ_iσ_j⟩ with derivatives of log Z via finite differences.",
        "K_values": args.K_values,
        "H_values": args.H_values,
        "cases": payload_cases,
        "metrics": metrics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
