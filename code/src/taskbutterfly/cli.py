"""Command-line verification demo."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .compressibility import complementary_rank_audit
from .losses import randomized_probe_loss
from .operator import ParametricButterflyOperator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the matrix-free parametric butterfly prototype."
    )
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--rhs", type=int, default=8)
    parser.add_argument("--frequency", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_demo(n: int, rhs: int, frequency: float, seed: int) -> dict:
    operator = ParametricButterflyOperator.from_seed(
        n,
        [2.0, 8.0, 16.0],
        seed=seed,
        scale=0.08,
        complex_valued=True,
    )
    fixed = operator.operator_at(frequency)
    rng = np.random.default_rng(seed + 1)
    x = rng.standard_normal((n, rhs)) + 1j * rng.standard_normal((n, rhs))
    y = rng.standard_normal((n, rhs)) + 1j * rng.standard_normal((n, rhs))

    start = time.perf_counter()
    matrix_free_result = fixed.apply(x)
    matrix_free_seconds = time.perf_counter() - start
    dense = fixed.to_dense()
    start = time.perf_counter()
    dense_result = dense @ x
    dense_seconds = time.perf_counter() - start

    relative_apply_error = np.linalg.norm(matrix_free_result - dense_result) / np.linalg.norm(
        dense_result
    )
    left = np.vdot(fixed.apply(x), y)
    right = np.vdot(x, fixed.adjoint(y))
    adjoint_error = abs(left - right) / max(abs(left), abs(right), np.finfo(float).eps)
    probe_error = randomized_probe_loss(
        fixed.apply,
        lambda block: dense @ block,
        n=n,
        probes=min(rhs, 8),
        seed=seed + 2,
    )
    ranks = complementary_rank_audit(
        dense,
        relative_tolerance=1e-8,
        max_blocks_per_level=32,
        seed=seed,
    )
    return {
        "n": n,
        "rhs": rhs,
        "frequency": frequency,
        "relative_apply_error": float(relative_apply_error),
        "relative_adjoint_identity_error": float(adjoint_error),
        "randomized_probe_relative_mse": float(probe_error),
        "prototype_matrix_free_seconds": matrix_free_seconds,
        "dense_blas_seconds": dense_seconds,
        "factor_parameters": fixed.parameter_count,
        "dense_entries": int(n * n),
        "complementary_rank_audit": ranks,
        "timing_note": (
            "The transparent Python prototype is for correctness, not performance. "
            "The proposal calls for compiled OpenMP/BLAS kernels."
        ),
    }


def main() -> None:
    args = _parse_args()
    result = run_demo(args.n, args.rhs, args.frequency, args.seed)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

