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


def canonicalize_rooted(cells: Sequence[Cell]) -> Cluster:
    """Canonical form for rooted clusters: sort coordinates (origin is fixed)."""
    return tuple(sorted(cells))


def compute_area(cluster: Cluster) -> int:
    """Surface area (number of exposed faces)."""
    cluster_set = set(cluster)
    area = 0
    for x, y, z in cluster:
        for dx, dy, dz in DIRS:
            if (x + dx, y + dy, z + dz) not in cluster_set:
                area += 1
    return area


def stabilizer_size(shape: Cluster) -> int:
    """Number of cube rotations that keep the shape invariant up to translation."""
    count = 0
    for rot in ROTATIONS:
        rotated = normalize_translation(apply_rotation(c, rot) for c in shape)
        if rotated == shape:
            count += 1
    return count


def rooted_orientations(shape: Cluster) -> Tuple[int, Dict[int, int]]:
    """Count unique rooted+oriented embeddings with origin fixed.

    For each cell as root: shift it to origin, rotate around origin, dedup by sorted
    coordinates (no translation). Returns total count and histogram by area.
    """
    oriented: set[Cluster] = set()
    by_area: Dict[int, int] = {}
    for root in shape:
        shift = tuple(-c for c in root)
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
      oriented_by_area: area histogram reconstructed via orbit sizes (24/|stab|)
      oriented_by_size: size histogram reconstructed via orbit sizes
      rooted_oriented_by_area: histogram for rooted+rotated embeddings (origin fixed)
      rooted_oriented_by_size: histogram for rooted+rotated embeddings (origin fixed)
      shapes_total: total distinct shapes visited (all areas)
      oriented_total: total reconstructed oriented (orbit) count
      rooted_oriented_total: total rooted+oriented embeddings (no translations)
    """

    start = canonicalize(((0, 0, 0),))
    start_area = 6
    seen = {start}
    stack: List[Tuple[Cluster, int]] = [(start, start_area)]

    counts_by_area: Dict[int, int] = {start_area: 1} if start_area <= max_area else {}
    shapes_by_size: Dict[int, int] = {1: 1}
    oriented_by_area: Dict[int, int] = {}
    oriented_by_size: Dict[int, int] = {1: 24}  # single cube has stab=24, orbit=1
    rooted_oriented_by_area: Dict[int, int] = {start_area: 1}
    rooted_oriented_by_size: Dict[int, int] = {1: 1}  # only one rooted orientation

    # For the seed shape, stab=24 -> orbit size = 1
    if start_area <= max_area:
        oriented_by_area[start_area] = 1

    shapes_total = 1
    oriented_total = 1  # already counted seed
    rooted_oriented_total = 1

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

            stab = stabilizer_size(new_cluster)
            orbit = len(ROTATIONS) // stab

            rooted_count, rooted_area_hist = rooted_orientations(new_cluster)
            new_size = size + 1
            shapes_by_size[new_size] = shapes_by_size.get(new_size, 0) + 1
            if new_area <= max_area:
                counts_by_area[new_area] = counts_by_area.get(new_area, 0) + 1
                oriented_by_area[new_area] = oriented_by_area.get(new_area, 0) + orbit
            oriented_by_size[new_size] = oriented_by_size.get(new_size, 0) + orbit
            rooted_oriented_total += rooted_count
            rooted_oriented_by_size[new_size] = rooted_oriented_by_size.get(new_size, 0) + rooted_count
            for a, cnt in rooted_area_hist.items():
                if a <= max_area:
                    rooted_oriented_by_area[a] = rooted_oriented_by_area.get(a, 0) + cnt
            oriented_total += orbit
            stack.append((new_cluster, new_area))

    return (
        counts_by_area,
        shapes_by_size,
        oriented_by_area,
        oriented_by_size,
        rooted_oriented_by_area,
        rooted_oriented_by_size,
        shapes_total,
        oriented_total,
        rooted_oriented_total,
    )


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
    parser.add_argument(
        "--backend",
        choices=["python", "cpp"],
        default="python",
        help="Which canonicalize implementation to use (python or C++ via ctypes)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    canonicalize_impl = canonicalize
    if args.backend == "cpp":
        # Lazy import to avoid loading the shared library when not requested.
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))
        import canonicalize_cpp  # type: ignore

        canonicalize_impl = canonicalize_cpp.canonicalize_shape_cpp

    t0 = time.time()
    # Swap the global canonicalize reference during the run to reuse the existing code path.
    orig_canonicalize = globals()["canonicalize"]
    globals()["canonicalize"] = canonicalize_impl
    try:
        (
            counts_by_area,
            shapes_by_size,
            oriented_by_area,
            oriented_by_size,
            rooted_oriented_by_area,
            rooted_oriented_by_size,
            shapes_total,
            oriented_total,
            rooted_oriented_total,
        ) = enumerate_shapes(args.max_cells, args.max_area, args.use_pruned)
    finally:
        globals()["canonicalize"] = orig_canonicalize
    elapsed = time.time() - t0

    result = {
        "max_cells": args.max_cells,
        "max_area": args.max_area,
        "use_pruned": args.use_pruned,
        "backend": args.backend,
        "shapes_total": shapes_total,
        "oriented_total_reconstructed": oriented_total,
        "rooted_oriented_total": rooted_oriented_total,
        "counts_by_area": counts_by_area,
        "shapes_by_size": shapes_by_size,
        "oriented_by_area": oriented_by_area,
        "oriented_by_size": oriented_by_size,
        "rooted_oriented_by_area": rooted_oriented_by_area,
        "rooted_oriented_by_size": rooted_oriented_by_size,
        "elapsed_s": elapsed,
        "note": "Shapes are canonicalized by 24 rotations + translation; oriented counts reconstructed via orbit size = 24/|stab|. rooted_oriented_* counts enumerate (root, rotation) embeddings with origin fixed; translations still not applied.",
    }
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
