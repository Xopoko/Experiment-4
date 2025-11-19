#!/usr/bin/env python3
"""Driver for Q2.2 susceptibility enumeration using the C++ backend."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sympy as sp

Vec3 = Tuple[int, int, int]


def build_series(counts_by_length: Dict[int, int], max_edges: int) -> Tuple[str, Dict[str, str]]:
    K = sp.symbols("K")
    tau = sp.tanh(K)
    chi_tau = sp.Integer(1)
    for length, count in sorted(counts_by_length.items()):
        chi_tau += count * tau ** length
    series = sp.series(chi_tau, K, 0, max_edges + 1).removeO()
    expanded = sp.expand(series)
    coeffs = {}
    for n in range((max_edges // 2) + 1):
        coeffs[f"c_{n}"] = str(sp.simplify(expanded.coeff(K, 2 * n)))
    return str(series), coeffs


def run_backend(
    binary: Path,
    max_edges: int,
    bound: int,
    first_direction: int | None = None,
    prefix: Sequence[int] | None = None,
    collect_targets: bool = True,
) -> Dict[str, object]:
    cmd = [str(binary), f"--max-edges={max_edges}", f"--bound={bound}"]
    if first_direction is not None:
        cmd.append(f"--first-direction={first_direction}")
    if not collect_targets:
        cmd.append("--skip-targets")
    if prefix:
        cmd.append(f"--prefix={','.join(str(p) for p in prefix)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to parse C++ backend output: {result.stdout}\n{result.stderr}") from exc


def worker_run_task(params: Tuple[Path, int, int, Sequence[int] | None, int | None, bool]) -> Dict[str, object]:
    binary, max_edges, bound, prefix, first_dir, collect_targets = params
    return run_backend(binary, max_edges, bound, first_dir, prefix, collect_targets)


def aggregate_results(rows: List[Dict[str, object]], min_length: int = 1) -> Dict[str, object]:
    counts = defaultdict(int)
    targets = defaultdict(int)
    state_count = 0
    for row in rows:
        state_count += int(row["state_count"])
        for length, count in row["counts_by_length"].items():
            length_int = int(length)
            if length_int >= min_length:
                counts[length_int] += int(count)
        for entry in row.get("target_hist", []):
            target = tuple(entry["target"])
            targets[target] += int(entry["count"])
    return {
        "counts_by_length": {int(k): int(v) for k, v in sorted(counts.items())},
        "target_hist": [{"target": list(vec), "count": targets[vec]} for vec in sorted(targets)],
        "state_count": state_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q2.2 susceptibility via C++ backend")
    parser.add_argument("--output", type=Path, default=Path("artifacts/q2_2_ht_cpp/results.json"))
    parser.add_argument("--max-edges", type=int, default=9)
    parser.add_argument("--bound", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--binary", type=Path, default=Path("build/cluster_enum"))
    parser.add_argument("--prefix-length", type=int, default=0, help="Split recursion by first L edges (for max_edges≥10)")
    parser.add_argument(
        "--base-counts",
        type=Path,
        default=Path("artifacts/q2_2_ht_cpp/results_max9.json"),
        help="JSON with counts for lower orders (used when prefix-length>0)",
    )
    parser.add_argument("--prefix-chunks", type=int, default=1, help="Split prefix tasks into this many chunks")
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="Chunk index to run (0-based, used with --prefix-chunks)",
    )
    parser.add_argument(
        "--prefix-seq",
        type=str,
        default="",
        help="Explicit comma-separated prefix directions (overrides chunk slicing)",
    )
    parser.add_argument(
        "--prefix-weight",
        type=int,
        default=1,
        help="Multiplicity weight associated with explicit prefix (metadata only)",
    )
    parser.add_argument(
        "--prefix-state-id",
        type=str,
        default="",
        help="Identifier of canonical prefix state (for logging)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.binary.exists():
        raise FileNotFoundError(f"C++ backend not found at {args.binary}")
    bound = args.bound if args.bound > 0 else args.max_edges
    explicit_prefix = [
        int(token)
        for token in args.prefix_seq.split(",")
        if token.strip() != ""
    ]
    prefix_len = max(0, args.prefix_length)
    if explicit_prefix:
        if prefix_len not in (0, len(explicit_prefix)):
            raise ValueError("--prefix-length must match explicit --prefix-seq length")
        prefix_len = len(explicit_prefix)
    tasks_params: List[Tuple[Path, int, int, Sequence[int] | None, int | None, bool]] = []
    total_chunks = max(1, args.prefix_chunks)
    chunk_index = min(max(0, args.chunk_index), total_chunks - 1)
    if explicit_prefix:
        tasks_params.append((args.binary, args.max_edges, bound, explicit_prefix, None, False))
    elif prefix_len == 0:
        for idx in range(6):
            tasks_params.append((args.binary, args.max_edges, bound, None, idx, True))
    else:
        import itertools

        for position, seq in enumerate(itertools.product(range(6), repeat=prefix_len)):
            if position % total_chunks != chunk_index:
                continue
            tasks_params.append((args.binary, args.max_edges, bound, list(seq), None, False))

    results: List[Dict[str, object]] = []
    if args.workers <= 1:
        for params in tasks_params:
            results.append(worker_run_task(params))
    else:
        import multiprocessing as mp

        with mp.Pool(min(args.workers, len(tasks_params))) as pool:
            jobs = [pool.apply_async(worker_run_task, (params,)) for params in tasks_params]
            for job in jobs:
                results.append(job.get())

    min_len = 1 if prefix_len == 0 else prefix_len
    aggregate = aggregate_results(results, min_len)

    final_counts = aggregate["counts_by_length"].copy()
    target_hist = aggregate["target_hist"]
    if prefix_len > 0 and args.base_counts.exists():
        base = json.loads(args.base_counts.read_text())
        for length, count in base.get("counts_by_length", {}).items():
            length_int = int(length)
            if length_int < prefix_len:
                final_counts[length_int] = int(count)

    # Ensure we have entries for all lengths up to max_edges (fill zeros if missing)
    for ell in range(1, args.max_edges + 1):
        final_counts.setdefault(ell, 0)

    series_expr, coeffs = build_series(final_counts, args.max_edges)
    payload = {
        "max_edges": args.max_edges,
        "counts_by_length": {int(k): int(v) for k, v in sorted(final_counts.items())},
        "target_hist": target_hist,
        "series_expr": series_expr,
        "coefficients": coeffs,
        "state_count": aggregate["state_count"],
        "notes": (
            "Counts aggregated from C++ DFS backend with prefix splitting"
            if prefix_len > 0
            else "Counts aggregated from C++ DFS backend"
        ),
        "prefix_length": prefix_len,
        "prefix_sequence": explicit_prefix,
        "prefix_weight": int(args.prefix_weight),
        "prefix_state_id": args.prefix_state_id,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
