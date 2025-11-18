#!/usr/bin/env python3
"""Q2.2: High-temperature susceptibility series χ(K) via path+loop enumeration."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

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


def enumerate_loops(max_loop_edges: int, loop_bound: int) -> List[Dict[str, object]]:
    loops: List[Dict[str, object]] = []
    seen: Set[frozenset[Edge]] = set()

    def dfs(path: List[Vec3]) -> None:
        if len(path) > max_loop_edges + 1:
            return
        current = path[-1]
        for direction in DIRS:
            nxt = add_vec(current, direction)
            if not within_bounds(nxt, loop_bound):
                continue
            edge = sorted_edge(current, nxt)
            if nxt == ORIGIN:
                if len(path) >= 4:
                    edges: List[Edge] = []
                    vertices = set(path)
                    for i in range(len(path) - 1):
                        edges.append(sorted_edge(path[i], path[i + 1]))
                    edges.append(sorted_edge(path[-1], ORIGIN))
                    key = frozenset(edges)
                    if key not in seen:
                        seen.add(key)
                        loops.append(
                            {
                                "edges": tuple(edges),
                                "vertices": tuple(vertices | {ORIGIN}),
                                "length": len(edges),
                            }
                        )
                continue
            if nxt in path:
                continue
            path.append(nxt)
            dfs(path)
            path.pop()

    dfs([ORIGIN])
    return loops


def record_cluster(
    counts_by_length: Dict[int, int],
    target_hist: Dict[Vec3, int],
    target: Vec3,
    edges: Set[Edge],
) -> None:
    counts_by_length[len(edges)] += 1
    target_hist[target] += 1


def attach_loops(
    base_edges: Set[Edge],
    base_vertices: Set[Vec3],
    target: Vec3,
    counts_by_length: Dict[int, int],
    target_hist: Dict[Vec3, int],
    loops: Sequence[Dict[str, object]],
    path_bound: int,
    max_edges: int,
) -> None:
    seen_states: Set[frozenset[Edge]] = set()

    @lru_cache(maxsize=None)
    def translate_loop(loop_idx: int, anchor: Vec3) -> Tuple[Tuple[Edge, ...], Tuple[Vec3, ...]]:
        loop = loops[loop_idx]
        translated_edges = []
        translated_vertices = []
        for vertex in loop["vertices"]:
            shifted = add_vec(vertex, anchor)
            if not within_bounds(shifted, path_bound):
                raise ValueError
            translated_vertices.append(shifted)
        for a, b in loop["edges"]:
            sa = add_vec(a, anchor)
            sb = add_vec(b, anchor)
            if not within_bounds(sa, path_bound) or not within_bounds(sb, path_bound):
                raise ValueError
            translated_edges.append(sorted_edge(sa, sb))
        return tuple(translated_edges), tuple(translated_vertices)

    def dfs(edges: Set[Edge], vertices: Set[Vec3]) -> None:
        key = frozenset(edges)
        if key in seen_states:
            return
        seen_states.add(key)
        record_cluster(counts_by_length, target_hist, target, edges)
        if len(edges) >= max_edges:
            return
        vertices_list = sorted(vertices)
        for anchor in vertices_list:
            for idx in range(len(loops)):
                try:
                    loop_edges, loop_vertices = translate_loop(idx, anchor)
                except ValueError:
                    continue
                if any(edge in edges for edge in loop_edges):
                    continue
                new_length = len(edges) + len(loop_edges)
                if new_length > max_edges:
                    continue
                new_edges = set(edges)
                new_edges.update(loop_edges)
                new_vertices = set(vertices)
                new_vertices.update(loop_vertices)
                dfs(new_edges, new_vertices)

    dfs(set(base_edges), set(base_vertices))


def enumerate_paths(
    counts_by_length: Dict[int, int],
    target_hist: Dict[Vec3, int],
    loops: Sequence[Dict[str, object]],
    path_bound: int,
    max_edges: int,
) -> None:
    def dfs(current: Vec3, visited: List[Vec3], edges: List[Edge]) -> None:
        if edges:
            attach_loops(
                set(edges),
                set(visited),
                current,
                counts_by_length,
                target_hist,
                loops,
                path_bound,
                max_edges,
            )
        if len(edges) >= max_edges:
            return
        for direction in DIRS:
            nxt = add_vec(current, direction)
            if not within_bounds(nxt, path_bound):
                continue
            if nxt in visited:
                continue
            visited.append(nxt)
            edges.append(sorted_edge(current, nxt))
            dfs(nxt, visited, edges)
            edges.pop()
            visited.pop()

    dfs(ORIGIN, [ORIGIN], [])


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


def run(max_edges: int, output: Path) -> Dict[str, object]:
    path_bound = max_edges
    loop_bound = min(4, max_edges)
    max_loop_edges = min(8, max_edges)
    loops = enumerate_loops(max_loop_edges, loop_bound if loop_bound > 0 else 1)
    counts_by_length: Dict[int, int] = defaultdict(int)
    target_hist: Dict[Vec3, int] = defaultdict(int)
    enumerate_paths(counts_by_length, target_hist, loops, path_bound, max_edges)
    series_expr, coeffs = build_series(counts_by_length, max_edges)
    payload = {
        "max_edges": max_edges,
        "counts_by_length": {int(k): int(v) for k, v in sorted(counts_by_length.items())},
        "target_hist": [
            {"target": list(t), "count": c}
            for t, c in sorted(target_hist.items())
        ],
        "series_expr": series_expr,
        "coefficients": coeffs,
        "num_loop_templates": len(loops),
        "notes": "Enumeration truncated at max_edges; clusters built from paths plus attached loops.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args.max_edges, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
