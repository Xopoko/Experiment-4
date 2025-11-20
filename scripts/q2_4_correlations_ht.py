"""High-temperature series for 3D Ising correlations to O(K^4).

Counts self-avoiding paths of length ≤4 between two sites on the SC lattice:
- nearest neighbors (distance 1)
- body-diagonal neighbors (distance (1,1,1))

At H=0 the two-point function in the HT expansion is a sum over graphs with
odd degree at the endpoints and even elsewhere; up to four edges this reduces
exactly to counting self-avoiding open paths that end at the target. We work in
v = tanh(K) and then re-expand in K.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Tuple

import sympy as sp

DIRS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

Point = Tuple[int, int, int]


def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def count_paths_dfs(start: Point, target: Point, max_len: int) -> Dict[int, int]:
    """Count self-avoiding paths of exact length ≤ max_len by DFS.

    Stops expansion when the target is reached to keep endpoints of degree 1.
    """

    counts: Dict[int, int] = {l: 0 for l in range(1, max_len + 1)}
    visited = {start}

    def dfs(pos: Point, length: int) -> None:
        if pos == target:
            if length:
                counts[length] += 1
            return
        if length == max_len:
            return
        for d in DIRS:
            nxt = add(pos, d)
            if nxt in visited:
                continue
            visited.add(nxt)
            dfs(nxt, length + 1)
            visited.remove(nxt)

    dfs(start, 0)
    return counts


def count_paths_sequences(start: Point, target: Point, max_len: int) -> Dict[int, int]:
    """Independent checker using brute-force step sequences (6^L each)."""

    counts: Dict[int, int] = defaultdict(int)
    for length in range(1, max_len + 1):
        for seq in product(DIRS, repeat=length):
            pos = start
            visited = {start}
            valid = True
            for step in seq:
                pos = add(pos, step)
                if pos in visited:
                    valid = False
                    break
                visited.add(pos)
            if valid and pos == target:
                counts[length] += 1
    # Fill missing lengths with zeros for consistency.
    for l in range(1, max_len + 1):
        counts.setdefault(l, 0)
    return dict(counts)


def counts_to_series(counts: Dict[int, int], max_k_order: int = 5) -> Dict[str, object]:
    K = sp.symbols("K")
    t = sp.tanh(K)
    expr_t = sum(sp.Integer(c) * t ** l for l, c in counts.items() if c)
    expr_k = sp.series(expr_t, K, 0, max_k_order + 1).removeO().expand()
    poly = sp.Poly(expr_k, K)
    coeffs_k = {int(exp[0]): str(sp.nsimplify(coef)) for exp, coef in poly.terms()}
    coeffs_t = {int(l): int(c) for l, c in counts.items() if c}
    return {
        "expr_t": sp.simplify(expr_t),
        "expr_k": expr_k,
        "coeffs_t": coeffs_t,
        "coeffs_k": coeffs_k,
    }


def evaluate(expr, K_values: Iterable[float]) -> Dict[str, float]:
    K = sp.symbols("K")
    f = sp.lambdify(K, expr, "numpy")
    return {str(k): float(f(k)) for k in K_values}


def main() -> None:
    max_len = 4
    start = (0, 0, 0)
    targets = {
        "nearest_neighbor": (1, 0, 0),
        "body_diagonal": (1, 1, 1),
    }

    dfs_counts = {
        name: count_paths_dfs(start, target, max_len) for name, target in targets.items()
    }
    seq_counts = {
        name: count_paths_sequences(start, target, max_len)
        for name, target in targets.items()
    }

    # Checks: two counting routes must match.
    assert dfs_counts == seq_counts, f"Mismatch between DFS and seq counts: {dfs_counts} vs {seq_counts}"

    series = {
        name: counts_to_series(counts, max_k_order=5) for name, counts in dfs_counts.items()
    }

    sample_K = [0.05, 0.1]
    evaluations = {
        name: evaluate(data["expr_k"], sample_K) for name, data in series.items()
    }

    out = {
        "counts": dfs_counts,
        "series_v": {name: str(data["expr_t"]) for name, data in series.items()},
        "series_K": {name: {"expr": str(data["expr_k"]), "coeffs": data["coeffs_k"]} for name, data in series.items()},
        "samples": evaluations,
        "meta": {
            "variable": "K",
            "t": "tanh(K)",
            "order_K": 4,
            "max_path_len": max_len,
            "note": "No contributions beyond length 4 at O(K^4); denominator corrections start at K^4 via plaquettes but do not affect coefficients up to this order.",
        },
    }

    out_path = Path("artifacts/q2_4_correlations/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"Saved {out_path}")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
