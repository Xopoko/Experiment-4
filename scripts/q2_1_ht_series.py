#!/usr/bin/env python3
"""Q2.1: High-temperature expansion of 3D SC Ising free energy up to K^8."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sympy as sp

Vec3 = Tuple[int, int, int]
Edge = Tuple[Vec3, Vec3]

MAX_COORD = 2  # bounds for embeddings (enough for loops with ≤8 edges)
MAX_EDGES = 8
ORIGIN: Vec3 = (0, 0, 0)

DIRS: Tuple[Vec3, ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def add_vec(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def within_bounds(v: Vec3) -> bool:
    return all(-MAX_COORD <= c <= MAX_COORD for c in v)


def sorted_edge(a: Vec3, b: Vec3) -> Edge:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def canonicalize_edges(edges: Iterable[Edge]) -> Tuple[Edge, ...]:
    vertices = {v for edge in edges for v in edge}
    min_vertex = min(vertices)

    def shift(v: Vec3) -> Vec3:
        return (v[0] - min_vertex[0], v[1] - min_vertex[1], v[2] - min_vertex[2])

    shifted = [sorted_edge(shift(a), shift(b)) for a, b in edges]
    return tuple(sorted(shifted))


def enumerate_simple_cycles() -> Dict[Tuple[Edge, ...], Dict[str, object]]:
    """Enumerate self-avoiding loops (length ≤ 8) containing the origin."""

    loops: Dict[Tuple[Edge, ...], Dict[str, object]] = {}

    def dfs(path: List[Vec3], visited: set[Vec3]) -> None:
        current = path[-1]
        for d in DIRS:
            nxt = add_vec(current, d)
            if not within_bounds(nxt):
                continue
            if nxt == ORIGIN:
                edges_used = len(path)
                if 4 <= edges_used <= MAX_EDGES:
                    loop_vertices = path + [ORIGIN]
                    edges = [sorted_edge(loop_vertices[i], loop_vertices[i + 1]) for i in range(len(loop_vertices) - 1)]
                    key = canonicalize_edges(edges)
                    entry = loops.setdefault(
                        key,
                        {
                            "count": 0,
                            "num_edges": len(edges),
                            "num_vertices": len({v for e in edges for v in e}),
                        },
                    )
                    entry["count"] = entry.get("count", 0) + 1
                continue
            if nxt in visited:
                continue
            if len(path) >= MAX_EDGES:
                continue
            visited.add(nxt)
            path.append(nxt)
            dfs(path, visited)
            path.pop()
            visited.remove(nxt)

    dfs([ORIGIN], {ORIGIN})
    loops = {k: v for k, v in loops.items() if v["num_edges"] <= MAX_EDGES}
    return loops


def generate_axis_aligned_squares() -> List[Dict[str, object]]:
    squares: List[Dict[str, object]] = []
    axes_pairs = [(0, 1), (0, 2), (1, 2)]  # (i, j) axes vary, the remaining axis is fixed
    for ax1, ax2 in axes_pairs:
        fixed_axis = 3 - ax1 - ax2  # since 0+1+2 = 3
        ranges = {axis: range(-MAX_COORD, MAX_COORD + 1) for axis in (0, 1, 2)}
        for fixed in ranges[fixed_axis]:
            for base1 in range(-MAX_COORD, MAX_COORD):
                if base1 + 1 > MAX_COORD:
                    continue
                for base2 in range(-MAX_COORD, MAX_COORD):
                    if base2 + 1 > MAX_COORD:
                        continue
                    coords = {0: 0, 1: 0, 2: 0}
                    coords[fixed_axis] = fixed
                    coords[ax1] = base1
                    coords[ax2] = base2
                    v0 = (coords[0], coords[1], coords[2])
                    incr1 = [0, 0, 0]
                    incr1[ax1] = 1
                    incr2 = [0, 0, 0]
                    incr2[ax2] = 1
                    v1 = add_vec(v0, tuple(incr1))
                    v2 = add_vec(v1, tuple(incr2))
                    v3 = add_vec(v0, tuple(incr2))
                    verts = [v0, v1, v2, v3]
                    edges = [sorted_edge(verts[i], verts[(i + 1) % 4]) for i in range(4)]
                    squares.append(
                        {
                            "edges": set(edges),
                            "vertices": set(verts),
                        }
                    )
    return squares


def enumerate_figure_eights(squares: Sequence[Dict[str, object]]) -> Dict[Tuple[Edge, ...], Dict[str, object]]:
    figures: Dict[Tuple[Edge, ...], Dict[str, object]] = {}
    n = len(squares)
    for i in range(n):
        sq_i = squares[i]
        for j in range(i + 1, n):
            sq_j = squares[j]
            if sq_i["edges"] & sq_j["edges"]:
                continue  # sharing an edge would violate even-degree condition
            edges_union = sq_i["edges"].union(sq_j["edges"])
            if len(edges_union) > MAX_EDGES:
                continue
            vertices_union = sq_i["vertices"].union(sq_j["vertices"])
            if ORIGIN not in vertices_union:
                continue
            # connectivity check
            adjacency: Dict[Vec3, List[Vec3]] = defaultdict(list)
            for a, b in edges_union:
                adjacency[a].append(b)
                adjacency[b].append(a)
            seen = {next(iter(vertices_union))}
            queue = deque(seen)
            while queue:
                node = queue.popleft()
                for nb in adjacency[node]:
                    if nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
            if seen != vertices_union:
                continue
            key = canonicalize_edges(edges_union)
            entry = figures.setdefault(
                key,
                {
                    "count": 0,
                    "num_edges": len(edges_union),
                    "num_vertices": len(vertices_union),
                },
            )
            entry["count"] = entry.get("count", 0) + 1
    return figures


def merge_clusters(*clusters: Dict[Tuple[Edge, ...], Dict[str, object]]) -> Dict[Tuple[Edge, ...], Dict[str, object]]:
    merged: Dict[Tuple[Edge, ...], Dict[str, object]] = {}
    for cluster_dict in clusters:
        for key, info in cluster_dict.items():
            entry = merged.setdefault(
                key,
                {
                    "count": 0,
                    "num_edges": info["num_edges"],
                    "num_vertices": info["num_vertices"],
                },
            )
            entry["count"] += info["count"]
    return merged


def build_series(clusters: Dict[Tuple[Edge, ...], Dict[str, object]]) -> Dict[str, object]:
    density_by_edges = defaultdict(lambda: sp.Rational(0, 1))
    detailed = []
    for key, info in clusters.items():
        count = info["count"]
        num_vertices = info["num_vertices"]
        num_edges = info["num_edges"]
        density = sp.Rational(count, num_vertices)
        density_by_edges[num_edges] += density
        detailed.append(
            {
                "num_edges": num_edges,
                "num_vertices": num_vertices,
                "count_origin_embeddings": int(count),
                "density_per_site": str(sp.simplify(density)),
            }
        )
    K = sp.symbols("K")
    tau = sp.tanh(K)
    loop_poly = sum(density_by_edges[m] * tau ** m for m in sorted(density_by_edges))
    free_energy = sp.log(2) + 3 * sp.log(sp.cosh(K)) + loop_poly
    series = sp.series(free_energy, K, 0, MAX_EDGES + 2).removeO()
    coeffs = {}
    for n in range(5):
        coeffs[f"a_{n}"] = sp.simplify(sp.expand(series).coeff(K, 2 * n))
    return {
        "density_by_edges": {int(k): str(sp.simplify(v)) for k, v in density_by_edges.items()},
        "clusters": detailed,
        "series_expr": str(series),
        "coefficients": {k: str(v) for k, v in coeffs.items()},
    }


def run(output: Path) -> Dict[str, object]:
    loops = enumerate_simple_cycles()
    squares = generate_axis_aligned_squares()
    figures = enumerate_figure_eights(squares)
    clusters = merge_clusters(loops, figures)
    payload = build_series(clusters)
    payload["num_clusters"] = len(clusters)
    payload["max_coord"] = MAX_COORD
    payload["notes"] = "Counts include simple loops (length 4/6/8) and two-plaquette figure-eights."
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q2.1 HT-series generator (K^8)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/q2_1_ht_series/results.json"),
        help="Path to store JSON payload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
