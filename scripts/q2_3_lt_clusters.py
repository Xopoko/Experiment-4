#!/usr/bin/env python3
"""Enumerate rooted flipped-spin clusters for the LT series."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

DIRS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


Cell = Tuple[int, int, int]
Cluster = Tuple[Cell, ...]


def canonicalize(cells: Sequence[Cell]) -> Cluster:
    return tuple(sorted(cells))


def neighbors(cluster: Cluster) -> List[Cell]:
    cluster_set = set(cluster)
    neigh: set[Cell] = set()
    for x, y, z in cluster:
        for dx, dy, dz in DIRS:
            cand = (x + dx, y + dy, z + dz)
            if cand not in cluster_set:
                neigh.add(cand)
    return sorted(neigh)


def surface_area(cluster: Cluster) -> int:
    cluster_set = set(cluster)
    area = 0
    for x, y, z in cluster:
        for dx, dy, dz in DIRS:
            cand = (x + dx, y + dy, z + dz)
            if cand not in cluster_set:
                area += 1
    return area


def enumerate_clusters(max_cells: int) -> List[Cluster]:
    start = canonicalize(((0, 0, 0),))
    seen = {start}
    ordered: List[Cluster] = [start]

    def dfs(cluster: Cluster) -> None:
        if len(cluster) >= max_cells:
            return
        for cand in neighbors(cluster):
            new_cluster = canonicalize(cluster + (cand,))
            if new_cluster in seen:
                continue
            seen.add(new_cluster)
            ordered.append(new_cluster)
            dfs(new_cluster)

    dfs(start)
    return ordered


def compute_counts(clusters: Iterable[Cluster], max_area: int) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for cluster in clusters:
        area = surface_area(cluster)
        if area > max_area:
            continue
        counts[area] = counts.get(area, 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rooted cluster enumerator for LT series")
    parser.add_argument("--max-cells", type=int, default=4, help="Enumerate clusters up to this size")
    parser.add_argument("--max-area", type=int, default=10, help="Only retain clusters with area ≤ value")
    parser.add_argument("--output", type=Path, default=Path("artifacts/q2_3_lt_free_energy/cluster_counts.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    clusters = enumerate_clusters(args.max_cells)
    counts = compute_counts(clusters, args.max_area)
    elapsed = time.time() - start
    result = {
        "max_cells": args.max_cells,
        "max_area": args.max_area,
        "cluster_count": len(clusters),
        "counts_by_area": counts,
        "free_energy_coeffs": {str(area): -cnt for area, cnt in sorted(counts.items())},
        "elapsed_s": elapsed,
    }
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
