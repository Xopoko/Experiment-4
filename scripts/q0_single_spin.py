#!/usr/bin/env python3
"""Utilities for verifying Q0.2 (single-spin partition function and magnetization)."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List


def analytic_partition(h: float) -> float:
    """Return analytic partition function Z = 2 cosh(H) for a single spin."""
    return 2.0 * math.cosh(h)


def analytic_magnetization(h: float) -> float:
    """Return analytic magnetization m = tanh(H) for a single spin."""
    return math.tanh(h)


def brute_force_stats(h: float) -> tuple[float, float]:
    """Enumerate the two spin states and compute Z and m directly."""
    weights = []
    spins = (-1.0, 1.0)
    for sigma in spins:
        # With dimensionless couplings, weights are exp(H * sigma).
        weights.append(math.exp(h * sigma))

    z = sum(weights)
    m = sum(sigma * weight for sigma, weight in zip(spins, weights)) / z
    return z, m


@dataclass
class CheckResult:
    H: float
    Z_bruteforce: float
    Z_analytic: float
    m_bruteforce: float
    m_analytic: float

    @property
    def abs_err_Z(self) -> float:
        return abs(self.Z_bruteforce - self.Z_analytic)

    @property
    def abs_err_m(self) -> float:
        return abs(self.m_bruteforce - self.m_analytic)

    def to_json(self) -> dict:
        data = asdict(self)
        data["abs_err_Z"] = self.abs_err_Z
        data["abs_err_m"] = self.abs_err_m
        return data


def run_checks(values: Iterable[float]) -> List[CheckResult]:
    results: List[CheckResult] = []
    for h in values:
        z_b, m_b = brute_force_stats(h)
        result = CheckResult(
            H=h,
            Z_bruteforce=z_b,
            Z_analytic=analytic_partition(h),
            m_bruteforce=m_b,
            m_analytic=analytic_magnetization(h),
        )
        results.append(result)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify analytic expressions for the 1x1x1 Ising partition function "
            "and magnetization (Q0.2)."
        )
    )
    parser.add_argument(
        "--values",
        type=str,
        default="-3,-1,-0.5,0,0.25,0.5,1,2.5,5",
        help="Comma-separated list of H values to check (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = [float(part) for part in args.values.split(",")]
    results = run_checks(values)
    payload = [res.to_json() for res in results]

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
