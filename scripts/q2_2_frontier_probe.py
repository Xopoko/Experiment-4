#!/usr/bin/env python3
"""Frontier-state prototype with connectivity check via final bitmask."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

SOURCE_VERTEX = 0


@dataclass(frozen=True)
class FrontierEntry:
    vertex: int
    parity: int
    remaining_future: int


StateKey = Tuple[Tuple[FrontierEntry, ...], int, int, int]


def enumerate_vertices(nx: int, ny: int, nz: int) -> List[Tuple[int, int, int]]:
    return [(x, y, z) for z in range(nz) for y in range(ny) for x in range(nx)]


def neighbours(x: int, y: int, z: int, nx: int, ny: int, nz: int) -> List[Tuple[int, int, int]]:
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    out = []
    for dx, dy, dz in dirs:
        nx_, ny_, nz_ = x + dx, y + dy, z + dz
        if 0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz:
            out.append((nx_, ny_, nz_))
    return out


def build_graph(nx: int, ny: int, nz: int):
    vertices = enumerate_vertices(nx, ny, nz)
    index = {coord: idx for idx, coord in enumerate(vertices)}
    past = [[] for _ in vertices]
    future = [[] for _ in vertices]
    adj = [[] for _ in vertices]
    for idx, (x, y, z) in enumerate(vertices):
        for nb in neighbours(x, y, z, nx, ny, nz):
            j = index[nb]
            adj[idx].append(j)
            if j < idx:
                past[idx].append(j)
            elif j > idx:
                future[idx].append(j)
    return vertices, past, future, adj


def encode_frontier(entries: Dict[int, Tuple[int, int]]) -> Tuple[FrontierEntry, ...]:
    return tuple(FrontierEntry(v, parity, rem) for v, (parity, rem) in sorted(entries.items()))


def frontier_probe(nx: int, ny: int, nz: int, max_edges: int):
    vertices, past_neighbors, future_neighbors, adjacency = build_graph(nx, ny, nz)
    n_vertices = len(vertices)

    states: Dict[StateKey, int] = {
        (tuple(), 0, 0, 0): 1
    }
    states_per_step: List[int] = []

    for idx in range(n_vertices):
        past = past_neighbors[idx]
        fut = future_neighbors[idx]
        new_states: Dict[StateKey, int] = defaultdict(int)
        for (frontier_tuple, odd_past, edges_used, mask), ways in states.items():
            frontier = {entry.vertex: [entry.parity, entry.remaining_future] for entry in frontier_tuple}
            processed = [n for n in past if n in frontier]
            for bits in range(1 << len(processed)):
                add_edges = [processed[k] for k in range(len(processed)) if bits & (1 << k)]
                new_edges = edges_used + len(add_edges)
                if new_edges > max_edges:
                    continue
                frontier2 = {v: info[:] for v, info in frontier.items()}
                odd_past2 = odd_past
                mask2 = mask
                valid = True
                for n in processed:
                    info = frontier2[n]
                    parity, rem = info
                    rem -= 1
                    if n in add_edges:
                        parity ^= 1
                        mask2 |= (1 << n)
                        mask2 |= (1 << idx)
                    info[0] = parity
                    info[1] = rem
                    frontier2[n] = info
                    if rem == 0:
                        if parity % 2 == 1:
                            odd_past2 += 1
                        frontier2.pop(n)
                if not valid:
                    continue
                parity_current = len(add_edges) % 2
                if fut:
                    frontier2[idx] = [parity_current, len(fut)]
                else:
                    if parity_current % 2 == 1:
                        odd_past2 += 1
                key = (encode_frontier(frontier2), odd_past2, new_edges, mask2)
                new_states[key] += ways
        states = new_states
        states_per_step.append(len(states))
        if not states:
            break

    counts_by_length: Dict[int, int] = defaultdict(int)
    for (frontier_tuple, odd_past, edges_used, mask), ways in states.items():
        if frontier_tuple or edges_used == 0:
            continue
        if odd_past != 2:
            continue
        if not (mask & (1 << SOURCE_VERTEX)):
            continue
        if not is_connected(mask, adjacency):
            continue
        counts_by_length[edges_used] += ways

    return {
        "dims": [nx, ny, nz],
        "max_edges": max_edges,
        "counts_by_length": {int(k): int(v) for k, v in sorted(counts_by_length.items())},
        "states_per_step": states_per_step,
        "final_state_count": len(states),
        "notes": "Frontier prototype with connectivity ensured via final BFS."
    }


def is_connected(mask: int, adjacency: List[List[int]]) -> bool:
    vertices = [idx for idx in range(len(adjacency)) if mask & (1 << idx)]
    if not vertices:
        return False
    if not (mask & (1 << SOURCE_VERTEX)):
        return False
    visited = set()
    stack = [SOURCE_VERTEX]
    if not (mask & (1 << SOURCE_VERTEX)):
        return False
    while stack:
        v = stack.pop()
        if v in visited or not (mask & (1 << v)):
            continue
        visited.add(v)
        for nb in adjacency[v]:
            if mask & (1 << nb):
                stack.append(nb)
    return visited == set(vertices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--nz", type=int, default=5)
    parser.add_argument("--max-edges", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/q2_2_frontier_probe/results.json"))
    args = parser.parse_args()
    payload = frontier_probe(args.nx, args.ny, args.nz, args.max_edges)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
