#!/usr/bin/env python3
"""Verification utilities for Q0.3 (two spins connected by one bond)."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def analytic_partition(k: float, h: float) -> float:
    """Closed-form partition function for 1×1×2 with free boundaries."""
    # Z = 2 e^{K} cosh(2H) + 2 e^{-K}
    return 2.0 * math.exp(k) * math.cosh(2.0 * h) + 2.0 * math.exp(-k)


def analytic_corr(k: float, h: float) -> float:
    """Analytic correlator <σ1σ2> = ∂ log Z / ∂K."""
    numerator = math.exp(k) * math.cosh(2.0 * h) - math.exp(-k)
    denominator = math.exp(k) * math.cosh(2.0 * h) + math.exp(-k)
    return numerator / denominator


def brute_force_stats(k: float, h: float) -> Tuple[float, float]:
    """Enumerate four configurations and return (Z, <σ1σ2>)."""
    configs = list(itertools.product((-1.0, 1.0), repeat=2))
    weights = []
    corr_num = 0.0
    for sigma1, sigma2 in configs:
        exponent = k * sigma1 * sigma2 + h * (sigma1 + sigma2)
        weight = math.exp(exponent)
        weights.append(weight)
        corr_num += sigma1 * sigma2 * weight

    z = sum(weights)
    corr = corr_num / z
    return z, corr


@dataclass
class CheckResult:
    K: float
    H: float
    Z_bruteforce: float
    Z_analytic: float
    corr_bruteforce: float
    corr_analytic: float

    @property
    def abs_err_Z(self) -> float:
        return abs(self.Z_bruteforce - self.Z_analytic)

    @property
    def abs_err_corr(self) -> float:
        return abs(self.corr_bruteforce - self.corr_analytic)

    def to_json(self) -> dict:
        data = asdict(self)
        data["abs_err_Z"] = self.abs_err_Z
        data["abs_err_corr"] = self.abs_err_corr
        return data


def run_checks(k_values: Sequence[float], h_values: Sequence[float]) -> List[CheckResult]:
    results: List[CheckResult] = []
    for k, h in itertools.product(k_values, h_values):
        z_b, corr_b = brute_force_stats(k, h)
        result = CheckResult(
            K=k,
            H=h,
            Z_bruteforce=z_b,
            Z_analytic=analytic_partition(k, h),
            corr_bruteforce=corr_b,
            corr_analytic=analytic_corr(k, h),
        )
        results.append(result)
    return results


def parse_csv_floats(raw: str) -> List[float]:
    return [float(part) for part in raw.split(",") if part]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify analytic formulas for Q0.3 (Z and correlator on 1×1×2)."
    )
    parser.add_argument(
        "--K-values",
        type=parse_csv_floats,
        default=parse_csv_floats("-1.5,-0.5,0,0.5,1.5"),
        help="Comma-separated list of K values (default: %(default)s)",
    )
    parser.add_argument(
        "--H-values",
        type=parse_csv_floats,
        default=parse_csv_floats("-0.75,-0.25,0,0.5,1.0"),
        help="Comma-separated list of H values (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_checks(args.K_values, args.H_values)
    payload = [res.to_json() for res in results]
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
