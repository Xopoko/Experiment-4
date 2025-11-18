#!/usr/bin/env python3
"""Q0.5: Verify Z_Λ(K=0,H) = (2 cosh H)^{|Λ|} and m = tanh H."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class CheckResult:
    n_sites: int
    H: float
    Z_exact: float
    Z_formula: float
    magnetization_exact: float
    magnetization_formula: float

    def to_json(self) -> dict:
        abs_err_Z = abs(self.Z_exact - self.Z_formula)
        rel_err_Z = abs_err_Z / self.Z_formula if self.Z_formula else 0.0
        abs_err_m = abs(self.magnetization_exact - self.magnetization_formula)
        rel_err_m = abs_err_m / (abs(self.magnetization_formula) or 1.0)
        return {
            "n_sites": self.n_sites,
            "H": self.H,
            "Z_exact": self.Z_exact,
            "Z_formula": self.Z_formula,
            "abs_err_Z": abs_err_Z,
            "rel_err_Z": rel_err_Z,
            "m_exact": self.magnetization_exact,
            "m_formula": self.magnetization_formula,
            "abs_err_m": abs_err_m,
            "rel_err_m": rel_err_m,
        }


def enumerate_partition(n_sites: int, H: float) -> tuple[float, float]:
    """Enumerate 2^n configurations; return (Z, average magnetization per site)."""
    configs = itertools.product((-1, 1), repeat=n_sites)
    weights: List[float] = []
    total_magnetization = 0.0
    for spins in configs:
        magnetization = sum(spins)
        weight = math.exp(H * magnetization)
        weights.append(weight)
        total_magnetization += magnetization * weight

    Z = sum(weights)
    avg_magnetization = total_magnetization / (Z * n_sites)
    return Z, avg_magnetization


def run_checks(n_values: Sequence[int], H_values: Sequence[float]) -> List[CheckResult]:
    results: List[CheckResult] = []
    for n in n_values:
        for H in H_values:
            Z_exact, m_exact = enumerate_partition(n, H)
            Z_formula = (2.0 * math.cosh(H)) ** n
            m_formula = math.tanh(H)
            results.append(
                CheckResult(
                    n_sites=n,
                    H=H,
                    Z_exact=Z_exact,
                    Z_formula=Z_formula,
                    magnetization_exact=m_exact,
                    magnetization_formula=m_formula,
                )
            )
    return results


def parse_csv_floats(raw: str) -> List[float]:
    return [float(part) for part in raw.split(",") if part.strip()]


def parse_csv_ints(raw: str) -> List[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the independent-spin partition function at K=0."
    )
    parser.add_argument(
        "--site-counts",
        type=parse_csv_ints,
        default=parse_csv_ints("1,2,3,4,5,6,8,10"),
        help="Comma-separated site counts to test (default: %(default)s)",
    )
    parser.add_argument(
        "--H-values",
        type=parse_csv_floats,
        default=parse_csv_floats("-1.5,-0.75,-0.2,0.0,0.35,1.1"),
        help="Comma-separated list of H values (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/q0_5_independent/results.json"),
        help="Path for JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_checks(args.site_counts, args.H_values)
    payload = {
        "question_id": "Q0.5",
        "description": "Check Z_Λ(K=0,H)=(2 cosh H)^{|Λ|} and m=tanh H by enumeration.",
        "site_counts": args.site_counts,
        "H_values": args.H_values,
        "metrics": {
            "max_abs_err_Z": max(r.to_json()["abs_err_Z"] for r in results),
            "max_rel_err_Z": max(r.to_json()["rel_err_Z"] for r in results),
            "max_abs_err_m": max(r.to_json()["abs_err_m"] for r in results),
            "max_rel_err_m": max(r.to_json()["rel_err_m"] for r in results),
            "num_checks": len(results),
        },
        "checks": [r.to_json() for r in results],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
