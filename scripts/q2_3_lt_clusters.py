#!/usr/bin/env python3
"""Enumerate rooted flipped-spin clusters for the LT series."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def add_cell_area(cluster: Cluster, cluster_set: set[Cell], cur_area: int, cell: Cell) -> int:
    """Return updated surface area after adding a cell to the cluster."""

    shared = 0
    for dx, dy, dz in DIRS:
        if (cell[0] + dx, cell[1] + dy, cell[2] + dz) in cluster_set:
            shared += 1
    return cur_area + 6 - 2 * shared


def enumerate_counts(max_cells: int, max_area: int, use_pruned: bool) -> Tuple[Dict[int, int], int]:
    """Enumerate rooted clusters and accumulate counts on the fly.

    The previous implementation enumerated all clusters, then recomputed areas in a
    second pass. Streaming the counts saves both time and memory, which matters once
    cluster counts exceed millions.
    """

    start = canonicalize(((0, 0, 0),))
    start_area = 6  # single cube surface
    seen = {start}
    counts: Dict[int, int] = {}
    if start_area <= max_area:
        counts[start_area] = 1
    stack: List[Tuple[Cluster, int]] = [(start, start_area)]
    cluster_count = 1

    while stack:
        cluster, area = stack.pop()
        if len(cluster) >= max_cells:
            continue

        cluster_set = set(cluster)
        remaining = max_cells - len(cluster)

        for cand in neighbors(cluster):
            new_cluster = canonicalize(cluster + (cand,))
            if new_cluster in seen:
                continue

            new_area = add_cell_area(cluster, cluster_set, area, cand)
            remaining_after = remaining - 1
            if use_pruned:
                # Each new cell can reduce area by at most 6 (full contact).
                min_possible_area = new_area - 6 * remaining_after
                if min_possible_area > max_area:
                    continue

            seen.add(new_cluster)
            cluster_count += 1
            if new_area <= max_area:
                counts[new_area] = counts.get(new_area, 0) + 1
            stack.append((new_cluster, new_area))

    return counts, cluster_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rooted cluster enumerator for LT series")
    parser.add_argument("--max-cells", type=int, default=4, help="Enumerate clusters up to this size")
    parser.add_argument("--max-area", type=int, default=10, help="Only retain clusters with area ≤ value")
    parser.add_argument(
        "--use-pruned",
        action="store_true",
        help="Use area-bound pruning (safe lower bound) to cut the search space",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/q2_3_lt_free_energy/cluster_counts.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    counts, cluster_count = enumerate_counts(args.max_cells, args.max_area, args.use_pruned)
    elapsed = time.time() - start
    result = {
        "max_cells": args.max_cells,
        "max_area": args.max_area,
        "cluster_count": cluster_count,
        "use_pruned": args.use_pruned,
        "counts_by_area": counts,
        "free_energy_coeffs": {str(area): -cnt for area, cnt in sorted(counts.items())},
        "elapsed_s": elapsed,
    }
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
