#!/usr/bin/env python3
"""Q1.4: проверка формул для K=0 и H=0 на малых решётках."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

Edge = Tuple[int, int]
SpinConfig = Tuple[int, ...]


def build_edges(lx: int, ly: int, lz: int) -> List[Edge]:
    """Построить список рёбер для решётки lx×ly×lz с свободными границами."""

    coords = [(x, y, z) for x in range(lx) for y in range(ly) for z in range(lz)]
    index = {coord: idx for idx, coord in enumerate(coords)}
    edges: List[Edge] = []
    for (x, y, z) in coords:
        here = (x, y, z)
        if x + 1 < lx:
            edges.append((index[here], index[(x + 1, y, z)]))
        if y + 1 < ly:
            edges.append((index[here], index[(x, y + 1, z)]))
        if z + 1 < lz:
            edges.append((index[here], index[(x, y, z + 1)]))
    return edges


def brute_observables(
    edges: Sequence[Edge],
    n_sites: int,
    k: float,
    h: float,
    pair: Edge,
) -> Tuple[float, float]:
    total_weight = 0.0
    corr_weight = 0.0
    for spins in itertools.product((-1, 1), repeat=n_sites):
        bond_sum = sum(spins[i] * spins[j] for i, j in edges)
        magnetization = sum(spins)
        weight = math.exp(k * bond_sum + h * magnetization)
        total_weight += weight
        corr_weight += weight * spins[pair[0]] * spins[pair[1]]
    corr = corr_weight / total_weight if total_weight else float("nan")
    return total_weight, corr


def high_temp_partition_and_corr(
    edges: Sequence[Edge], n_sites: int, pair: Edge, k: float
) -> Tuple[float, float]:
    tanh_k = math.tanh(k)
    cosh_k = math.cosh(k)
    prefactor = (2.0 ** n_sites) * (cosh_k ** len(edges))
    even_sum = 0.0
    pair_sum = 0.0
    m = len(edges)
    pair_set = {pair[0], pair[1]}

    for mask in range(1 << m):
        parity = [0] * n_sites
        edges_used = 0
        for bit in range(m):
            if mask & (1 << bit):
                i, j = edges[bit]
                parity[i] ^= 1
                parity[j] ^= 1
                edges_used += 1
        if edges_used == 0:
            weight = 1.0
        else:
            weight = tanh_k ** edges_used
        if any(parity):
            odd_vertices = [idx for idx, val in enumerate(parity) if val]
            if len(odd_vertices) == 2 and pair_set == set(odd_vertices):
                pair_sum += weight
        else:
            even_sum += weight

    z_formula = prefactor * even_sum
    corr_formula = pair_sum / even_sum if even_sum else float("nan")
    return z_formula, corr_formula


def run_experiment(output: Path) -> dict:
    lattices = [
        {"name": "2x2x1_free", "dims": (2, 2, 1)},
        {"name": "2x2x2_free", "dims": (2, 2, 2)},
    ]
    h_values = [-1.2, -0.6, -0.1, 0.4, 0.9]
    k_values = [-1.0, -0.5, -0.2, 0.0, 0.4, 0.9]

    payload = {
        "task": "Q1.4",
        "description": "Проверяем формулы для K=0 (независимые спины) и H=0 (HT-разложение)",
        "results": {},
    }

    for spec in lattices:
        lx, ly, lz = spec["dims"]
        n_sites = lx * ly * lz
        edges = build_edges(lx, ly, lz)
        pair = edges[0]
        lattice_entry = {
            "dims": spec["dims"],
            "num_sites": n_sites,
            "num_edges": len(edges),
            "test_edge": pair,
        }

        # Case 1: K = 0
        k0_rows = []
        max_z_err = 0.0
        max_corr_err = 0.0
        for h_val in h_values:
            z_brute, corr_brute = brute_observables(edges, n_sites, k=0.0, h=h_val, pair=pair)
            z_formula = (2.0 * math.cosh(h_val)) ** n_sites
            corr_formula = math.tanh(h_val) ** 2
            err_z = abs(z_brute - z_formula)
            err_corr = abs(corr_brute - corr_formula)
            max_z_err = max(max_z_err, err_z)
            max_corr_err = max(max_corr_err, err_corr)
            k0_rows.append(
                {
                    "H": h_val,
                    "Z_brute": z_brute,
                    "Z_formula": z_formula,
                    "corr_brute": corr_brute,
                    "corr_formula": corr_formula,
                    "abs_err_Z": err_z,
                    "abs_err_corr": err_corr,
                }
            )
        lattice_entry["k_zero"] = {
            "H_values": h_values,
            "rows": k0_rows,
            "max_abs_err_Z": max_z_err,
            "max_abs_err_corr": max_corr_err,
        }

        # Case 2: H = 0
        h0_rows = []
        max_z_err = 0.0
        max_corr_err = 0.0
        for k_val in k_values:
            z_brute, corr_brute = brute_observables(edges, n_sites, k=k_val, h=0.0, pair=pair)
            z_formula, corr_formula = high_temp_partition_and_corr(
                edges, n_sites, pair, k=k_val
            )
            err_z = abs(z_brute - z_formula)
            err_corr = abs(corr_brute - corr_formula)
            max_z_err = max(max_z_err, err_z)
            max_corr_err = max(max_corr_err, err_corr)
            h0_rows.append(
                {
                    "K": k_val,
                    "Z_brute": z_brute,
                    "Z_formula": z_formula,
                    "corr_brute": corr_brute,
                    "corr_formula": corr_formula,
                    "abs_err_Z": err_z,
                    "abs_err_corr": err_corr,
                }
            )
        lattice_entry["h_zero"] = {
            "K_values": k_values,
            "rows": h0_rows,
            "max_abs_err_Z": max_z_err,
            "max_abs_err_corr": max_corr_err,
        }

        payload["results"][spec["name"]] = lattice_entry

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q1.4 проверка K=0 и H=0 формул")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/q1_4_k0_h0/results.json"),
        help="Путь для сохранения JSON отчёта",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_experiment(args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
