#!/usr/bin/env python3
"""Canonicalize prefix states (sets of edges) up to cubic rotations."""

from __future__ import annotations

import argparse
import itertools
import json
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


def generate_rotations() -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Return list of (perm, signs) describing all 48 cube symmetries."""

    rotations = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            det = signs[0] * signs[1] * signs[2]
            # Accept both proper and improper rotations (48 elements)
            rotations.append((perm, signs))
    return rotations


ROTATIONS = generate_rotations()


def rotate_vec(vec: Tuple[int, int, int], perm: Sequence[int], signs: Sequence[int]) -> Tuple[int, int, int]:
    coords = [vec[0], vec[1], vec[2]]
    return (
        signs[0] * coords[perm[0]],
        signs[1] * coords[perm[1]],
        signs[2] * coords[perm[2]],
    )


def path_vertices(seq: Sequence[int]) -> List[Tuple[int, int, int]]:
    pos = (0, 0, 0)
    vertices = [pos]
    for idx in seq:
        step = DIRS[idx]
        pos = (pos[0] + step[0], pos[1] + step[1], pos[2] + step[2])
        vertices.append(pos)
    return vertices


def path_edges(vertices: Sequence[Tuple[int, int, int]]) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    edges = []
    for a, b in zip(vertices[:-1], vertices[1:]):
        edges.append((a, b))
    return edges


def canonical_edges(edges: Sequence[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]):
    best = None
    best_meta = None
    for perm, signs in ROTATIONS:
        rotated = []
        for a, b in edges:
            ra = rotate_vec(a, perm, signs)
            rb = rotate_vec(b, perm, signs)
            if ra <= rb:
                rotated.append((ra, rb))
            else:
                rotated.append((rb, ra))
        rotated.sort()
        tuple_repr = tuple(rotated)
        if best is None or tuple_repr < best:
            best = tuple_repr
            best_meta = (perm, signs)
    assert best is not None and best_meta is not None
    return best, best_meta


@dataclass
class CanonState:
    key: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...]
    sequences: List[List[int]]
    endpoints: Tuple[int, int, int]


def analyze_prefixes(prefix_length: int) -> Dict[str, object]:
    grouped: Dict[Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...], CanonState] = {}
    for seq in itertools.product(range(6), repeat=prefix_length):
        vertices = path_vertices(seq)
        edges = path_edges(vertices)
        key, _ = canonical_edges(edges)
        key_state = grouped.get(key)
        if key_state is None:
            grouped[key] = CanonState(key=key, sequences=[list(seq)], endpoints=vertices[-1])
        else:
            key_state.sequences.append(list(seq))
    states = []
    for idx, state in enumerate(grouped.values()):
        states.append(
            {
                "id": idx,
                "orbit_size": len(state.sequences),
                "sample_sequence": state.sequences[0],
                "sequences": state.sequences,
                "edges": [
                    {
                        "a": list(edge[0]),
                        "b": list(edge[1]),
                    }
                    for edge in state.key
                ],
                "endpoint": list(state.endpoints),
            }
        )
    return {
        "prefix_length": prefix_length,
        "total_sequences": 6**prefix_length,
        "unique_states": len(states),
        "states": states,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize prefix states up to cubic rotations")
    parser.add_argument("--prefix-length", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze_prefixes(args.prefix_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
