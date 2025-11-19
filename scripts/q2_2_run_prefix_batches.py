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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    label: str | None = None
    endpoint: Tuple[int, int, int] | None = None
    stats: Dict[str, float] | None = None


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


def parse_bounds(triple: Optional[str]) -> Optional[Tuple[Optional[int], Optional[int], Optional[int]]]:
    if triple is None:
        return None
    parts = triple.split(",")
    if len(parts) != 3:
        raise ValueError("Bounds must have exactly three comma-separated entries")
    parsed: List[Optional[int]] = []
    for token in parts:
        token = token.strip()
        if token in ("", "*"):
            parsed.append(None)
        else:
            parsed.append(int(token))
    return tuple(parsed)  # type: ignore[return-value]


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
    parser.add_argument(
        "--state-library",
        type=Path,
        help="Path to JSON with canonical prefix states (overrides --enable-symmetry)",
    )
    parser.add_argument(
        "--endpoint-min",
        type=str,
        help="Comma-separated minimum endpoint bounds (x,y,z); use '*' to skip a coordinate",
    )
    parser.add_argument(
        "--endpoint-max",
        type=str,
        help="Comma-separated maximum endpoint bounds (x,y,z); use '*' to skip a coordinate",
    )
    parser.add_argument(
        "--bridge-leaf-threshold",
        type=int,
        help="Skip state classes that have >= this number of non-terminal leaf vertices",
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


def analyze_state_geometry(edges: Iterable[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]) -> Dict[str, float]:
    degrees: Dict[Tuple[int, int, int], int] = {}
    for a, b in edges:
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1
    leaf_count = 0
    endpoint = None
    if degrees:
        for vertex, degree in degrees.items():
            if vertex == (0, 0, 0):
                continue
            if degree % 2 == 1:
                endpoint = vertex
        if endpoint is None:
            endpoint = max(degrees.keys(), key=lambda v: degrees[v])
        for vertex, degree in degrees.items():
            if vertex in ((0, 0, 0), endpoint):
                continue
            if degree == 1:
                leaf_count += 1
    return {"leaf_count": float(leaf_count), "endpoint": endpoint or (0, 0, 0)}


def endpoint_within_bounds(
    endpoint: Tuple[int, int, int] | None,
    min_bounds: Optional[Tuple[Optional[int], Optional[int], Optional[int]]],
    max_bounds: Optional[Tuple[Optional[int], Optional[int], Optional[int]]],
) -> bool:
    if endpoint is None:
        return True
    coords = endpoint
    if min_bounds:
        for idx, bound in enumerate(min_bounds):
            if bound is not None and coords[idx] < bound:
                return False
    if max_bounds:
        for idx, bound in enumerate(max_bounds):
            if bound is not None and coords[idx] > bound:
                return False
    return True


def load_state_library(
    path: Path,
    prefix_length: int,
    endpoint_min: Optional[Tuple[Optional[int], Optional[int], Optional[int]]],
    endpoint_max: Optional[Tuple[Optional[int], Optional[int], Optional[int]]],
    bridge_leaf_threshold: Optional[int],
) -> List[PrefixJob]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lib_len = int(data.get("prefix_length", -1))
    if lib_len != prefix_length:
        raise ValueError(f"State library length {lib_len} != requested {prefix_length}")
    jobs: List[PrefixJob] = []
    filtered_endpoint = 0
    filtered_bridge = 0
    for state in data.get("states", []):
        sequence = tuple(int(x) for x in state.get("sample_sequence", []))
        if len(sequence) != prefix_length:
            raise ValueError(f"State {state.get(id)} has invalid sequence length {len(sequence)}")
        weight = int(state.get("orbit_size", len(state.get("sequences", [])) or 1))
        state_id = state.get("id")
        label = f"state_{state_id}" if state_id is not None else None
        edges = []
        for edge in state.get("edges", []):
            a = tuple(edge.get("a", [0, 0, 0]))
            b = tuple(edge.get("b", [0, 0, 0]))
            edges.append((a, b))
        geometry = analyze_state_geometry(edges)
        endpoint = geometry["endpoint"] if isinstance(geometry["endpoint"], tuple) else tuple(geometry["endpoint"])  # type: ignore[arg-type]
        if not endpoint_within_bounds(endpoint, endpoint_min, endpoint_max):
            filtered_endpoint += 1
            continue
        if bridge_leaf_threshold is not None and geometry.get("leaf_count", 0.0) >= bridge_leaf_threshold:
            filtered_bridge += 1
            continue
        jobs.append(
            PrefixJob(
                sequence=sequence,
                weight=weight,
                label=label,
                endpoint=endpoint,
                stats={"leaf_count": geometry.get("leaf_count", 0.0)},
            )
        )
    if not jobs:
        raise ValueError(f"State library {path} contains no states")
    if filtered_endpoint or filtered_bridge:
        print(
            json.dumps(
                {
                    "state_library": str(path),
                    "filtered_by_endpoint": filtered_endpoint,
                    "filtered_by_bridge": filtered_bridge,
                    "states_kept": len(jobs),
                },
                ensure_ascii=False,
            )
        )
    return jobs


def format_prefix_arg(seq: Tuple[int, ...]) -> str:
    if not seq:
        return "--prefix-seq="
    return "--prefix-seq=" + ",".join(str(val) for val in seq)


def main() -> None:
    args = parse_args()
    args.chunk_dir.mkdir(parents=True, exist_ok=True)

    base_data = json.loads(args.base_counts.read_text(encoding="utf-8"))
    base_counts = {int(k): int(v) for k, v in base_data.get("counts_by_length", {}).items()}

    endpoint_min = parse_bounds(args.endpoint_min)
    endpoint_max = parse_bounds(args.endpoint_max)
    if args.state_library:
        prefix_jobs_all = load_state_library(
            args.state_library,
            args.prefix_length,
            endpoint_min,
            endpoint_max,
            args.bridge_leaf_threshold,
        )
        use_symmetry = True
        state_library_path = str(args.state_library)
    else:
        state_library_path = None
        use_symmetry = bool(args.enable_symmetry)
        prefix_jobs_all = build_prefix_jobs(args.prefix_length, use_symmetry)
        if endpoint_min or endpoint_max or args.bridge_leaf_threshold is not None:
            print("Warning: endpoint/bridge filters require --state-library")
    prefix_job_classes = len(prefix_jobs_all)
    prefix_jobs = sorted(prefix_jobs_all, key=lambda job: (-job.weight, job.sequence))
    total_sequences = 6 ** args.prefix_length if args.prefix_length > 0 else 1
    if args.prefix_chunks and args.prefix_chunks > 0:
        prefix_jobs = prefix_jobs[: args.prefix_chunks]

    accum_counts: Dict[int, int] = {}
    chunk_meta: List[Dict[str, float]] = []
    total_state_count = 0

    def run_chunk(chunk_index: int, job: PrefixJob) -> tuple[int, float, Dict[str, object], PrefixJob]:
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
        if job.label:
            cmd.append(f"--prefix-state-id={job.label}")
        start = time.time()
        subprocess.run(cmd, check=True)
        runtime = time.time() - start
        chunk_data = json.loads(chunk_path.read_text(encoding="utf-8"))
        return chunk_index, runtime, chunk_data, job

    with ThreadPoolExecutor(max_workers=max(1, args.batch_concurrency)) as pool:
        futures = [pool.submit(run_chunk, idx, job) for idx, job in enumerate(prefix_jobs)]
        for future in as_completed(futures):
            chunk_index, runtime, chunk_data, job = future.result()
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
                    "prefix_state_id": job.label,
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
        "state_library": state_library_path,
        "chunk_runs": chunk_meta,
        "notes": "Aggregated from multiple prefix-split runs via q2_2_run_prefix_batches.py",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
