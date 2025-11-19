#!/usr/bin/env python3
"""Q2.2: High-temperature susceptibility series χ(K) via connected cluster enumeration."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import sympy as sp

Vec3 = Tuple[int, int, int]
Edge = Tuple[Vec3, Vec3]

DIRS: Tuple[Vec3, ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)
ORIGIN: Vec3 = (0, 0, 0)


def add_vec(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def within_bounds(v: Vec3, bound: int) -> bool:
    return all(-bound <= coord <= bound for coord in v)


def sorted_edge(a: Vec3, b: Vec3) -> Edge:
    return (a, b) if a <= b else (b, a)


def record_cluster(
    counts_by_length: Dict[int, int],
    target_hist: Dict[Vec3, int],
    target: Vec3,
    num_edges: int,
) -> None:
    counts_by_length[num_edges] += 1
    target_hist[target] += 1


def enumerate_clusters(
    max_edges: int,
    bound: int,
    first_direction_idx: Optional[int] = None,
) -> Tuple[Dict[int, int], Dict[Vec3, int], int]:
    counts_by_length: Dict[int, int] = defaultdict(int)
    target_hist: Dict[Vec3, int] = defaultdict(int)
    seen_states: Set[Tuple[Edge, ...]] = set()
    span = 2 * bound + 1
    vertex_stride = span ** 3

    def encode_vertex(v: Vec3) -> int:
        return ((v[0] + bound) * span + (v[1] + bound)) * span + (v[2] + bound)

    def decode_vertex(idx: int) -> Vec3:
        z = idx % span - bound
        y = (idx // span) % span - bound
        x = idx // (span * span) - bound
        return (x, y, z)

    def encode_edge(a: int, b: int) -> int:
        lo, hi = (a, b) if a <= b else (b, a)
        return lo * vertex_stride + hi

    origin_id = encode_vertex(ORIGIN)

    def dfs(
        edges: Tuple[int, ...],
        edges_set: Set[int],
        parity: Dict[int, bool],
        coords: Dict[int, Vec3],
        min_origin_dir: int,
    ) -> None:
        key = edges
        if key in seen_states:
            return
        seen_states.add(key)

        num_edges = len(edges)
        odd_vertices = [vertex for vertex, is_odd in parity.items() if is_odd]
        if num_edges > 0 and len(odd_vertices) == 2 and origin_id in odd_vertices:
            target_id = odd_vertices[0] if odd_vertices[1] == origin_id else odd_vertices[1]
            target = coords[target_id]
            record_cluster(counts_by_length, target_hist, target, num_edges)

        remaining_edges = max_edges - num_edges
        if remaining_edges <= 0:
            return
        if not odd_vertices:
            min_needed = 1
        else:
            min_needed = max(0, (len(odd_vertices) - 2) // 2)
        if origin_id not in odd_vertices:
            min_needed = max(min_needed, 1)
        if min_needed > remaining_edges:
            return

        for vertex_id, vertex in coords.items():
            for dir_idx, direction in enumerate(DIRS):
                nxt = add_vec(vertex, direction)
                if not within_bounds(nxt, bound):
                    continue
                nxt_id = encode_vertex(nxt)
                if vertex_id != origin_id and nxt_id == origin_id:
                    continue  # enforce canonical handling of origin edges
                edge = encode_edge(vertex_id, nxt_id)
                if edge in edges_set:
                    continue
                if vertex_id == origin_id:
                    if not edges:
                        if first_direction_idx is not None and dir_idx != first_direction_idx:
                            continue
                        new_min_dir = dir_idx
                    else:
                        if dir_idx < min_origin_dir:
                            continue
                        new_min_dir = min(min_origin_dir, dir_idx)
                else:
                    new_min_dir = min_origin_dir
                new_parity = parity.copy()
                if nxt_id not in new_parity:
                    new_parity[nxt_id] = False
                new_parity[vertex_id] = not new_parity[vertex_id]
                new_parity[nxt_id] = not new_parity[nxt_id]

                if nxt_id in coords:
                    new_coords = coords
                else:
                    new_coords = coords.copy()
                    new_coords[nxt_id] = nxt

                new_edges = list(edges)
                new_edges.append(edge)
                new_edges.sort()
                new_edges_set = set(edges_set)
                new_edges_set.add(edge)
                dfs(tuple(new_edges), new_edges_set, new_parity, new_coords, new_min_dir)

    initial_min_dir = len(DIRS) if first_direction_idx is None else first_direction_idx
    dfs(tuple(), set(), {origin_id: False}, {origin_id: ORIGIN}, initial_min_dir)
    return counts_by_length, target_hist, len(seen_states)


def _enumerate_direction(args: Tuple[int, int, int]) -> Tuple[Dict[int, int], Dict[Vec3, int], int]:
    max_edges, bound, direction_idx = args
    return enumerate_clusters(max_edges, bound, direction_idx)


def build_series(counts_by_length: Dict[int, int], max_edges: int) -> Tuple[str, Dict[str, str]]:
    K = sp.symbols("K")
    tau = sp.tanh(K)
    chi_tau = sp.Integer(1)
    for length, count in sorted(counts_by_length.items()):
        chi_tau += count * tau ** length
    series = sp.series(chi_tau, K, 0, max_edges + 1).removeO()
    expanded = sp.expand(series)
    coeffs = {}
    for n in range((max_edges // 2) + 1):
        coeffs[f"c_{n}"] = str(sp.simplify(expanded.coeff(K, 2 * n)))
    return str(series), coeffs


def run(max_edges: int, output: Path, workers: int = 1) -> Dict[str, object]:
    bound = max_edges
    if workers <= 1:
        counts_by_length, target_hist_raw, visited_states = enumerate_clusters(max_edges, bound)
    else:
        from multiprocessing import Pool

        direction_indices = list(range(len(DIRS)))
        actual_workers = min(workers, len(direction_indices))
        task_args = [(max_edges, bound, idx) for idx in direction_indices]

        counts_by_length = defaultdict(int)
        target_hist_raw = defaultdict(int)
        visited_states = 0
        with Pool(actual_workers) as pool:
            for partial_counts, partial_hist, partial_states in pool.map(_enumerate_direction, task_args):
                visited_states += partial_states
                for length, count in partial_counts.items():
                    counts_by_length[length] += count
                for target, count in partial_hist.items():
                    target_hist_raw[target] += count
    series_expr, coeffs = build_series(counts_by_length, max_edges)
    target_hist_list = [
        {"target": list(vec), "count": count}
        for vec, count in sorted(target_hist_raw.items())
    ]
    payload = {
        "max_edges": max_edges,
        "counts_by_length": {int(k): int(v) for k, v in sorted(counts_by_length.items())},
        "target_hist": target_hist_list,
        "series_expr": series_expr,
        "coefficients": coeffs,
        "state_count": visited_states,
        "notes": "Connected clusters enumerated directly via parity-constrained DFS.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q2.2 susceptibility HT series")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/q2_2_ht_susceptibility/results.json"),
        help="Where to store JSON report",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=6,
        help="Maximum number of bonds in clusters (controls truncation order)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="How many parallel workers to split first-step directions across",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args.max_edges, args.output, args.workers)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
