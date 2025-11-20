#!/usr/bin/env python3
"""Probe: enumerate distinct cluster shapes with rotation/translation canonicalization.

This is an *exploratory* tool to estimate how many unique shapes (up to rotations and
translations) exist for a given max_cells threshold. It does **not** attempt to
preserve rooted/oriented multiplicities used in the LT series; the goal is to measure
how much symmetry-reduction could shrink the search space.
"""

from __future__ import annotations

import argparse
import json
import time
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


def _parity(perm: Sequence[int]) -> int:
    """Return +1 for even permutations and -1 for odd."""
    inversions = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inversions += 1
    return 1 if inversions % 2 == 0 else -1


def _rotations() -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Generate the 24 proper rotations of the cube (determinant +1)."""
    rots: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
    axes = (0, 1, 2)
    signs = (-1, 1)
    for perm in (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ):
        perm_parity = _parity(perm)
        for sx in signs:
            for sy in signs:
                for sz in signs:
                    det = perm_parity * sx * sy * sz
                    if det == 1:
                        rots.append((perm, (sx, sy, sz)))
    return rots


ROTATIONS = _rotations()


def apply_rotation(cell: Cell, rot: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> Cell:
    perm, (sx, sy, sz) = rot
    coords = (cell[0], cell[1], cell[2])
    return (
        sx * coords[perm[0]],
        sy * coords[perm[1]],
        sz * coords[perm[2]],
    )


def normalize_translation(cells: Iterable[Cell]) -> Cluster:
    cells_list = list(cells)
    xs, ys, zs = zip(*cells_list)
    min_x, min_y, min_z = min(xs), min(ys), min(zs)
    return tuple(sorted((x - min_x, y - min_y, z - min_z) for x, y, z in cells_list))


def canonicalize(cells: Sequence[Cell]) -> Cluster:
    """Return lexicographically minimal rotation+translation of the cluster."""
    best: Cluster | None = None
    for rot in ROTATIONS:
        rotated = (apply_rotation(c, rot) for c in cells)
        normalized = normalize_translation(rotated)
        if best is None or normalized < best:
            best = normalized
    assert best is not None
    return best


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


def enumerate_shapes(max_cells: int, max_area: int, use_pruned: bool):
    """Enumerate distinct shapes up to rotations/translations.

    Returns:
      counts_by_area: number of shapes whose area ≤ max_area
      shapes_by_size: number of shapes per cell count
      shapes_total: total distinct shapes visited (all areas)
    """

    start = canonicalize(((0, 0, 0),))
    start_area = 6
    seen = {start}
    stack: List[Tuple[Cluster, int]] = [(start, start_area)]

    counts_by_area: Dict[int, int] = {start_area: 1} if start_area <= max_area else {}
    shapes_by_size: Dict[int, int] = {1: 1}
    shapes_total = 1

    while stack:
        cluster, area = stack.pop()
        size = len(cluster)
        if size >= max_cells:
            continue

        cluster_set = set(cluster)
        remaining = max_cells - size

        for cand in neighbors(cluster):
            new_area = add_cell_area(cluster, cluster_set, area, cand)
            remaining_after = remaining - 1
            if use_pruned:
                min_possible_area = new_area - 6 * remaining_after
                if min_possible_area > max_area:
                    continue

            new_cluster = canonicalize(cluster + (cand,))
            if new_cluster in seen:
                continue

            seen.add(new_cluster)
            shapes_total += 1
            new_size = size + 1
            shapes_by_size[new_size] = shapes_by_size.get(new_size, 0) + 1
            if new_area <= max_area:
                counts_by_area[new_area] = counts_by_area.get(new_area, 0) + 1
            stack.append((new_cluster, new_area))

    return counts_by_area, shapes_by_size, shapes_total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symmetry-reduced shape enumerator (probe)")
    parser.add_argument("--max-cells", type=int, default=7, help="Enumerate shapes up to this size")
    parser.add_argument("--max-area", type=int, default=30, help="Only retain shapes with area ≤ value")
    parser.add_argument(
        "--use-pruned",
        action="store_true",
        help="Enable safe lower bound pruning on area to trim the search tree",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/q2_3_lt_free_energy/shape_counts.json"),
        help="Where to store the JSON results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    counts_by_area, shapes_by_size, shapes_total = enumerate_shapes(
        args.max_cells, args.max_area, args.use_pruned
    )
    elapsed = time.time() - t0

    result = {
        "max_cells": args.max_cells,
        "max_area": args.max_area,
        "use_pruned": args.use_pruned,
        "shapes_total": shapes_total,
        "counts_by_area": counts_by_area,
        "shapes_by_size": shapes_by_size,
        "elapsed_s": elapsed,
        "note": "Shapes are canonicalized by 24 rotations + translation; rooted multiplicities are not included.",
    }
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
