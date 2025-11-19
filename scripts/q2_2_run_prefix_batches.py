#!/usr/bin/env python3
"""Batch runner that aggregates prefix-split susceptibility chunks."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sympy as sp

DIR_VECTORS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def _build_symmetry_maps() -> Tuple[Tuple[int, ...], ...]:
    """Enumerate sign/permutation maps that act on direction indices."""

    def dir_index(vec: Tuple[int, int, int]) -> int:
        for idx, base in enumerate(DIR_VECTORS):
            if vec == base:
                return idx
        raise ValueError(f"Unknown direction vector {vec}")

    maps: List[Tuple[int, ...]] = []
    seen = set()
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            mapping = []
            for vec in DIR_VECTORS:
                coords = (
                    signs[0] * vec[perm[0]],
                    signs[1] * vec[perm[1]],
                    signs[2] * vec[perm[2]],
                )
                mapping.append(dir_index(coords))
            key = tuple(mapping)
            if key not in seen:
                seen.add(key)
                maps.append(key)
    return tuple(maps)


SYMMETRY_MAPS = _build_symmetry_maps()


def canonicalize_prefix(seq: Sequence[int]) -> Tuple[int, ...]:
    if not seq:
        return tuple()
    best: Tuple[int, ...] | None = None
    for mapping in SYMMETRY_MAPS:
        candidate = tuple(mapping[idx] for idx in seq)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


@dataclass(frozen=True)
class PrefixJob:
    sequence: Tuple[int, ...]
    weight: int


def build_series(counts_by_length: Dict[int, int], max_edges: int) -> tuple[str, Dict[str, str]]:
    """Reuse the SymPy expansion logic without importing the other script as a module."""

    K = sp.symbols("K")
    tau = sp.tanh(K)
    chi_tau = sp.Integer(1)
    for length, count in sorted(counts_by_length.items()):
        chi_tau += count * tau**length
    series = sp.series(chi_tau, K, 0, max_edges + 1).removeO()
    expanded = sp.expand(series)
    coeffs: Dict[str, str] = {}
    for n in range((max_edges // 2) + 1):
        coeffs[f"c_{n}"] = str(sp.simplify(expanded.coeff(K, 2 * n)))
    return str(series), coeffs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate prefix-based susceptibility chunks")
    parser.add_argument("--max-edges", type=int, default=10)
    parser.add_argument("--prefix-length", type=int, default=3)
    parser.add_argument(
        "--prefix-chunks",
        type=int,
        default=0,
        help="Limit how many canonical prefix jobs to run (0 = all)",
    )
    parser.add_argument("--binary", type=Path, default=Path("build/cluster_enum"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--driver",
        type=Path,
        default=Path("scripts/q2_2_ht_susceptibility_cpp.py"),
        help="Driver script that handles a single prefix chunk",
    )
    parser.add_argument(
        "--base-counts",
        type=Path,
        default=Path("artifacts/q2_2_ht_cpp/results_max9.json"),
        help="JSON with counts for orders below prefix_length",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=Path("artifacts/q2_2_ht_cpp/chunks"),
        help="Directory to store per-chunk JSON outputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/q2_2_ht_cpp/results_max10_prefix_batches.json"),
    )
    parser.add_argument(
        "--batch-concurrency",
        type=int,
        default=2,
        help="How many chunk jobs to launch in parallel",
    )
    parser.add_argument(
        "--enable-symmetry",
        action="store_true",
        help="Enable prefix symmetry grouping (requires validated weight map)",
    )
    return parser.parse_args()


def build_prefix_jobs(prefix_length: int, use_symmetry: bool) -> List[PrefixJob]:
    if prefix_length <= 0:
        return [PrefixJob(sequence=tuple(), weight=1)]

    counts: Dict[Tuple[int, ...], int] = {}
    for seq in itertools.product(range(6), repeat=prefix_length):
        key = canonicalize_prefix(seq) if use_symmetry else tuple(seq)
        counts[key] = counts.get(key, 0) + 1
    return [PrefixJob(sequence=key, weight=weight) for key, weight in counts.items()]


def format_prefix_arg(seq: Tuple[int, ...]) -> str:
    if not seq:
        return "--prefix-seq="
    return "--prefix-seq=" + ",".join(str(val) for val in seq)


def main() -> None:
    args = parse_args()
    args.chunk_dir.mkdir(parents=True, exist_ok=True)

    base_data = json.loads(args.base_counts.read_text(encoding="utf-8"))
    base_counts = {int(k): int(v) for k, v in base_data.get("counts_by_length", {}).items()}

    use_symmetry = bool(args.enable_symmetry)
    prefix_jobs_all = build_prefix_jobs(args.prefix_length, use_symmetry)
    prefix_job_classes = len(prefix_jobs_all)
    prefix_jobs = sorted(prefix_jobs_all, key=lambda job: (-job.weight, job.sequence))
    total_sequences = 6 ** args.prefix_length if args.prefix_length > 0 else 1
    if args.prefix_chunks and args.prefix_chunks > 0:
        prefix_jobs = prefix_jobs[: args.prefix_chunks]

    accum_counts: Dict[int, int] = {}
    chunk_meta: List[Dict[str, float]] = []
    total_state_count = 0

    def run_chunk(chunk_index: int, job: PrefixJob) -> tuple[int, float, Dict[str, object]]:
        chunk_path = args.chunk_dir / f"chunk_{chunk_index:03d}.json"
        cmd = [
            "python3",
            str(args.driver),
            f"--max-edges={args.max_edges}",
            f"--workers={args.workers}",
            f"--binary={args.binary}",
            f"--output={chunk_path}",
            f"--prefix-length={args.prefix_length}",
            format_prefix_arg(job.sequence),
            f"--prefix-weight={job.weight}",
        ]
        start = time.time()
        subprocess.run(cmd, check=True)
        runtime = time.time() - start
        chunk_data = json.loads(chunk_path.read_text(encoding="utf-8"))
        return chunk_index, runtime, chunk_data

    with ThreadPoolExecutor(max_workers=max(1, args.batch_concurrency)) as pool:
        futures = [pool.submit(run_chunk, idx, job) for idx, job in enumerate(prefix_jobs)]
        for future in as_completed(futures):
            chunk_index, runtime, chunk_data = future.result()
            total_state_count += int(chunk_data["state_count"])
            weight = int(chunk_data.get("prefix_weight", 1))
            for length_str, count in chunk_data["counts_by_length"].items():
                length = int(length_str)
                if length < args.prefix_length:
                    continue
                accum_counts[length] = accum_counts.get(length, 0) + int(count) * weight
            chunk_meta.append(
                {
                    "chunk": chunk_index,
                    "runtime_s": runtime,
                    "state_count": int(chunk_data["state_count"]),
                    "prefix_weight": weight,
                    "prefix_sequence": chunk_data.get("prefix_sequence", []),
                }
            )

    final_counts: Dict[int, int] = {}
    for length in range(1, args.max_edges + 1):
        if length < args.prefix_length:
            final_counts[length] = base_counts.get(length, 0)
        else:
            final_counts[length] = accum_counts.get(length, 0)

    series_expr, coeffs = build_series(final_counts, args.max_edges)
    counts_payload = {str(k): int(v) for k, v in sorted(final_counts.items())}
    payload = {
        "max_edges": args.max_edges,
        "counts_by_length": counts_payload,
        "series_expr": series_expr,
        "coefficients": coeffs,
        "state_count": total_state_count,
        "prefix_length": args.prefix_length,
        "prefix_jobs": len(prefix_jobs),
        "prefix_job_classes": prefix_job_classes,
        "prefix_total_sequences": total_sequences,
        "symmetry_used": use_symmetry,
        "chunk_runs": chunk_meta,
        "notes": "Aggregated from multiple prefix-split runs via q2_2_run_prefix_batches.py",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
