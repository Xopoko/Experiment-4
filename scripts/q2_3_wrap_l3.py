#!/usr/bin/env python3
"""Compute wrap-around LT contributions on a 3x3x3 torus up to given area."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from scripts import q2_3_lt_clusters as lt

Cell = Tuple[int, int, int]
Cluster = Tuple[Cell, ...]


def embed_count_on_torus(shape: Cluster, L: int) -> int:
    """Count distinct embeddings of `shape` into (Z/LZ)^3 with injective cells."""

    shape_list = list(shape)
    embeddings: set[Tuple[Cell, ...]] = set()
    for rot in lt.ROTATIONS:
        rotated = [lt.apply_rotation(c, rot) for c in shape_list]
        xs, ys, zs = zip(*rotated)
        # Translate by all offsets mod L
        for tx in range(L):
            for ty in range(L):
                for tz in range(L):
                    embedded = tuple(((x + tx) % L, (y + ty) % L, (z + tz) % L) for x, y, z in rotated)
                    # Require injectivity (no collisions)
                    if len(set(embedded)) != len(shape_list):
                        continue
                    embeddings.add(tuple(sorted(embedded)))
    return len(embeddings)


def enumerate_shapes(max_cells: int, max_area: int, use_pruned: bool) -> List[Cluster]:
    """Enumerate canonical shapes (orbit mode) reusing cluster generator."""

    start = lt.canonicalize_shape(((0, 0, 0),))
    start_area = 6
    seen_shapes = {start}
    stack: List[Tuple[Cluster, int]] = [(start, start_area)]
    shapes: List[Cluster] = [start]

    while stack:
        shape, area = stack.pop()
        size = len(shape)
        if size >= max_cells:
            continue

        remaining = max_cells - size
        for cand in lt.neighbors(shape):
            new_shape = lt.canonicalize_shape(shape + (cand,))
            if new_shape in seen_shapes:
                continue
            new_area = lt.compute_area(new_shape)
            remaining_after = remaining - 1
            if use_pruned:
                min_possible_area = new_area - 6 * remaining_after
                if min_possible_area > max_area:
                    continue
            seen_shapes.add(new_shape)
            shapes.append(new_shape)
            stack.append((new_shape, new_area))
    return shapes


def compute_wrap_counts(L: int, max_cells: int, max_area: int, use_pruned: bool) -> Dict[str, Dict[int, int]]:
    shapes = enumerate_shapes(max_cells, max_area, use_pruned)
    counts: Dict[int, int] = {}
    oriented: Dict[int, int] = {}
    for shape in shapes:
        area = lt.compute_area(shape)
        if area > max_area:
            continue
        embeds = embed_count_on_torus(shape, L)
        if embeds == 0:
            continue
        counts[area] = counts.get(area, 0) + embeds
        oriented[area] = oriented.get(area, 0) + embeds
    return {
        "embeddings_by_area": counts,
        "free_energy_coeffs": {str(a): -cnt / (L**3) for a, cnt in sorted(counts.items())},
        "oriented_counts": oriented,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrap-around LT coefficients on L=3 torus")
    parser.add_argument("--L", type=int, default=3, help="Linear size of torus")
    parser.add_argument("--max-cells", type=int, default=9, help="Max cluster size")
    parser.add_argument("--max-area", type=int, default=18, help="Max surface area to keep")
    parser.add_argument(
        "--use-pruned",
        action="store_true",
        help="Use safe pruning bound on area during enumeration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/q2_3_lt_free_energy/lt_wrap_L3_y18.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lt.NEIGH_USE_SORT = False
    lt.CANONICAL_MODE = "sorted"
    start = time.time()
    result = compute_wrap_counts(args.L, args.max_cells, args.max_area, args.use_pruned)
    result.update(
        {
            "L": args.L,
            "max_cells": args.max_cells,
            "max_area": args.max_area,
            "use_pruned": args.use_pruned,
            "elapsed_s": time.time() - start,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
