#!/usr/bin/env python3
"""Enumerate rooted flipped-spin clusters for the LT series."""

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
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


def canonicalize_rooted(cells: Sequence[Cell]) -> Cluster:
    """Canonical form for rooted clusters (origin fixed)."""
    return tuple(sorted(cells))


def _parity(perm: Sequence[int]) -> int:
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inv += 1
    return 1 if inv % 2 == 0 else -1


def _rotations() -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """24 proper rotations of the cube (det=+1)."""
    rots: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
    signs = (-1, 1)
    for perm in (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ):
        perm_par = _parity(perm)
        for sx in signs:
            for sy in signs:
                for sz in signs:
                    if perm_par * sx * sy * sz == 1:
                        rots.append((perm, (sx, sy, sz)))
    return rots


ROTATIONS: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = _rotations()


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


def canonicalize_shape(cells: Sequence[Cell]) -> Cluster:
    """Canonical rotation+translation representative."""
    best: Cluster | None = None
    for rot in ROTATIONS:
        rotated = (apply_rotation(c, rot) for c in cells)
        normalized = normalize_translation(rotated)
        if best is None or normalized < best:
            best = normalized
    assert best is not None
    return best


def compute_area(cluster: Cluster) -> int:
    """Surface area (exposed faces)."""
    cluster_set = set(cluster)
    area = 0
    for x, y, z in cluster:
        for dx, dy, dz in DIRS:
            if (x + dx, y + dy, z + dz) not in cluster_set:
                area += 1
    return area


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


@lru_cache(maxsize=None)
def stabilizer_size(shape: Cluster) -> int:
    """Number of rotations leaving shape invariant up to translation."""
    count = 0
    for rot in ROTATIONS:
        rotated = normalize_translation(apply_rotation(c, rot) for c in shape)
        if rotated == shape:
            count += 1
    return count


@lru_cache(maxsize=None)
def rooted_orientations(shape: Cluster) -> Tuple[int, Dict[int, int]]:
    """Count distinct (root, rotation) embeddings with origin fixed; area histogram."""
    oriented: set[Cluster] = set()
    by_area: Dict[int, int] = {}
    for root in shape:
        shift = (-root[0], -root[1], -root[2])
        shifted = tuple((x + shift[0], y + shift[1], z + shift[2]) for x, y, z in shape)
        for rot in ROTATIONS:
            rotated = tuple(apply_rotation(c, rot) for c in shifted)
            rooted = canonicalize_rooted(rotated)
            if rooted in oriented:
                continue
            oriented.add(rooted)
            area = compute_area(rooted)
            by_area[area] = by_area.get(area, 0) + 1
    return len(oriented), by_area


def enumerate_counts(
    max_cells: int, max_area: int, use_pruned: bool, mode: str
) -> Tuple[Dict[str, Dict[int, int]], Dict[str, int]]:
    """Enumerate clusters in two modes:

    - oriented: baseline rooted enumeration (legacy behavior).
    - orbit: symmetry-reduced shapes with (root, rotation) weighting.
    """

    if mode == "oriented":
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

        return {"counts_by_area": counts}, {"cluster_count": cluster_count}

    # orbit mode (symmetry-reduced shapes)
    start = canonicalize_shape(((0, 0, 0),))
    start_area = 6
    seen_shapes = {start}
    stack: List[Tuple[Cluster, int]] = [(start, start_area)]

    counts_by_area: Dict[int, int] = {}
    oriented_by_area: Dict[int, int] = {}
    rooted_oriented_by_area: Dict[int, int] = {}

    shapes_total = 0
    oriented_total = 0
    rooted_oriented_total = 0

    while stack:
        shape, area = stack.pop()
        size = len(shape)

        stab = stabilizer_size(shape)
        orbit = len(ROTATIONS) // stab
        oriented_total += orbit
        root_count, root_area_hist = rooted_orientations(shape)
        rooted_oriented_total += root_count
        shapes_total += 1

        if area <= max_area:
            counts_by_area[area] = counts_by_area.get(area, 0) + 1
            oriented_by_area[area] = oriented_by_area.get(area, 0) + orbit
            for a, cnt in root_area_hist.items():
                if a <= max_area:
                    rooted_oriented_by_area[a] = rooted_oriented_by_area.get(a, 0) + cnt

        if size >= max_cells:
            continue

        remaining = max_cells - size

        for cand in neighbors(shape):
            new_shape = canonicalize_shape(shape + (cand,))
            if new_shape in seen_shapes:
                continue

            new_area = compute_area(new_shape)
            remaining_after = remaining - 1
            if use_pruned:
                min_possible_area = new_area - 6 * remaining_after
                if min_possible_area > max_area:
                    continue

            seen_shapes.add(new_shape)
            stack.append((new_shape, new_area))

    return (
        {
            "counts_by_area": counts_by_area,
            "oriented_by_area": oriented_by_area,
            "rooted_oriented_by_area": rooted_oriented_by_area,
        },
        {
            "shapes_total": shapes_total,
            "oriented_total_reconstructed": oriented_total,
            "rooted_oriented_total": rooted_oriented_total,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rooted cluster enumerator for LT series")
    parser.add_argument("--max-cells", type=int, default=4, help="Enumerate clusters up to this size")
    parser.add_argument("--max-area", type=int, default=10, help="Only retain clusters with area ≤ value")
    parser.add_argument(
        "--use-pruned",
        action="store_true",
        help="Use area-bound pruning (safe lower bound) to cut the search space",
    )
    parser.add_argument(
        "--mode",
        choices=["oriented", "orbit"],
        default="oriented",
        help="oriented = legacy rooted enumeration; orbit = symmetry-reduced shapes with (root, rotation) weights",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/q2_3_lt_free_energy/cluster_counts.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    counts, meta = enumerate_counts(args.max_cells, args.max_area, args.use_pruned, args.mode)
    elapsed = time.time() - start
    coeff_source = counts.get("rooted_oriented_by_area") or counts.get("counts_by_area") or {}
    result = {
        "max_cells": args.max_cells,
        "max_area": args.max_area,
        "mode": args.mode,
        "use_pruned": args.use_pruned,
        "elapsed_s": elapsed,
        **counts,
        **meta,
        "free_energy_coeffs": {str(area): -cnt for area, cnt in sorted(coeff_source.items())},
    }
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
