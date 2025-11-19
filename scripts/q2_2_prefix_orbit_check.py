#!/usr/bin/env python3
"""Check directional symmetry orbits for prefix chunks and quantify mismatches."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DIR_VECTORS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def build_direction_maps() -> List[Tuple[int, ...]]:
    maps: List[Tuple[int, ...]] = []
    seen = set()

    def dir_index(vec: Tuple[int, int, int]) -> int:
        for idx, base in enumerate(DIR_VECTORS):
            if vec == base:
                return idx
        raise ValueError(f"Unknown direction vector {vec}")

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
    return maps


DIRECTION_MAPS: Tuple[Tuple[int, ...], ...] = tuple(build_direction_maps())


def canonicalize_seq(seq: Sequence[int]) -> Tuple[int, ...]:
    if not seq:
        return tuple()
    best: Tuple[int, ...] | None = None
    for mapping in DIRECTION_MAPS:
        candidate = tuple(mapping[idx] for idx in seq)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prefix chunks by directional orbits")
    parser.add_argument("--chunks-dir", type=Path, required=True, help="Directory with chunk_*.json files")
    parser.add_argument("--prefix-length", type=int, required=True, help="Length of prefixes in chunks")
    parser.add_argument("--max-edges", type=int, required=True, help="Max edges for counts tuple")
    parser.add_argument("--output", type=Path, required=True, help="Where to store JSON summary")
    return parser.parse_args()


def load_chunks(chunks_dir: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for path in sorted(chunks_dir.glob("chunk_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        seq = data.get("prefix_sequence")
        if seq is None:
            continue
        entries.append({
            "path": str(path),
            "sequence": tuple(int(x) for x in seq),
            "counts_by_length": {int(k): int(v) for k, v in data["counts_by_length"].items()},
            "state_count": int(data.get("state_count", 0)),
        })
    return entries


def main() -> None:
    args = parse_args()
    entries = load_chunks(args.chunks_dir)
    prefix_len = args.prefix_length
    lengths = list(range(prefix_len, args.max_edges + 1))
    grouped: Dict[Tuple[int, ...], List[Dict[str, object]]] = {}
    for entry in entries:
        if len(entry["sequence"]) != prefix_len:
            continue
        key = canonicalize_seq(entry["sequence"])
        counts_tuple = tuple(entry["counts_by_length"].get(length, 0) for length in lengths)
        entry["counts_tuple"] = counts_tuple
        grouped.setdefault(key, []).append(entry)

    mismatches = []
    max_relative_error = 0.0
    for key, entries_list in grouped.items():
        reference = entries_list[0]["counts_tuple"]
        if not all(e["counts_tuple"] == reference for e in entries_list[1:]):
            mismatches.append({
                "key": list(key),
                "representatives": [
                    {
                        "sequence": list(e["sequence"]),
                        "counts_tuple": list(e["counts_tuple"]),
                        "path": e["path"],
                    }
                    for e in entries_list[: min(6, len(entries_list))]
                ],
            })
            for e in entries_list:
                rel_err = 0.0
                for ref, val in zip(reference, e["counts_tuple"]):
                    denom = max(1, ref)
                    rel_err = max(rel_err, abs(ref - val) / denom)
                max_relative_error = max(max_relative_error, rel_err)

    summary = {
        "chunks_dir": str(args.chunks_dir),
        "prefix_length": prefix_len,
        "max_edges": args.max_edges,
        "total_prefixes": sum(len(entries) for entries in grouped.values()),
        "canonical_classes": len(grouped),
        "mismatched_classes": len(mismatches),
        "max_relative_error": max_relative_error,
        "lengths": lengths,
        "mismatches": mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
