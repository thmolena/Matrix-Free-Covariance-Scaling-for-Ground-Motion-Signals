"""Empirical complementary-low-rank gate for candidate field matrices."""

from __future__ import annotations

from collections.abc import Callable
from math import log2

import numpy as np


def _numerical_rank(block: np.ndarray, relative_tolerance: float) -> int:
    singular_values = np.linalg.svd(block, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0
    return int(np.count_nonzero(singular_values > relative_tolerance * singular_values[0]))


def complementary_rank_audit(
    matrix: np.ndarray,
    *,
    relative_tolerance: float = 1e-6,
    max_blocks_per_level: int = 64,
    seed: int = 0,
) -> list[dict[str, float | int]]:
    """Sample ranks of complementary row/column blocks at every tree level.

    A butterfly claim should be made only if these ranks remain moderate as the
    matrix size and frequency increase.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    n = matrix.shape[0]
    levels = int(log2(n))
    if n < 2 or 2**levels != n:
        raise ValueError("matrix dimension must be a power of two")
    if not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must lie in (0, 1)")

    rng = np.random.default_rng(seed)
    results: list[dict[str, float | int]] = []
    for level in range(levels + 1):
        row_blocks = 1 << level
        col_blocks = 1 << (levels - level)
        row_size = n // row_blocks
        col_size = n // col_blocks
        candidates = [(i, j) for i in range(row_blocks) for j in range(col_blocks)]
        if len(candidates) > max_blocks_per_level:
            selection = rng.choice(
                len(candidates), size=max_blocks_per_level, replace=False
            )
            candidates = [candidates[int(index)] for index in selection]
        ranks = []
        for row_block, col_block in candidates:
            rows = slice(row_block * row_size, (row_block + 1) * row_size)
            cols = slice(col_block * col_size, (col_block + 1) * col_size)
            ranks.append(_numerical_rank(matrix[rows, cols], relative_tolerance))
        results.append(
            {
                "level": level,
                "row_block_size": row_size,
                "column_block_size": col_size,
                "sampled_blocks": len(ranks),
                "maximum_rank": int(max(ranks, default=0)),
                "median_rank": float(np.median(ranks)) if ranks else 0.0,
            }
        )
    return results


def _restricted_action(
    action: Callable[[np.ndarray], np.ndarray],
    *,
    n: int,
    input_slice: slice,
    output_slice: slice,
    block: np.ndarray,
) -> np.ndarray:
    """Apply one restricted block without requesting any target entry."""

    embedded = np.zeros((n, block.shape[1]), dtype=float)
    embedded[input_slice] = block
    return np.asarray(action(embedded))[output_slice]


def _tested_range_rank(
    training_action: np.ndarray,
    test_action: np.ndarray,
    *,
    relative_tolerance: float,
) -> tuple[int, float, bool]:
    """Smallest sampled range whose independent action residual meets a tolerance."""

    u, _, _ = np.linalg.svd(training_action, full_matrices=False)
    denominator = max(float(np.linalg.norm(test_action, ord="fro")), np.finfo(float).eps)
    for rank in range(u.shape[1] + 1):
        if rank == 0:
            residual = test_action
        else:
            basis = u[:, :rank]
            residual = test_action - basis @ (basis.T @ test_action)
        ratio = float(np.linalg.norm(residual, ord="fro") / denominator)
        if ratio <= relative_tolerance:
            return rank, ratio, False
    return u.shape[1], ratio, True


def complementary_action_probe_audit(
    apply: Callable[[np.ndarray], np.ndarray],
    adjoint: Callable[[np.ndarray], np.ndarray],
    *,
    n: int,
    relative_tolerance: float = 0.05,
    max_blocks_per_level: int = 8,
    range_probes: int = 20,
    residual_probes: int = 8,
    seed: int = 0,
) -> list[dict[str, float | int | bool]]:
    """Probe complementary blocks using only operator and adjoint actions.

    This is a finite randomized diagnostic, not a rank certificate.  For each
    sampled complementary block, independent Gaussian range and residual
    probes test both its column and row spaces.  The audit never calls a dense
    materializer or requests individual entries.
    """

    levels = int(log2(n))
    if n < 2 or 2**levels != n:
        raise ValueError("operator dimension must be a power of two")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie in (0, 1)")
    if min(max_blocks_per_level, range_probes, residual_probes) < 1:
        raise ValueError("block and probe counts must be positive")

    rng = np.random.default_rng(seed)
    results: list[dict[str, float | int | bool]] = []
    for level in range(levels + 1):
        row_blocks = 1 << level
        col_blocks = 1 << (levels - level)
        row_size = n // row_blocks
        col_size = n // col_blocks
        candidates = [(i, j) for i in range(row_blocks) for j in range(col_blocks)]
        if len(candidates) > max_blocks_per_level:
            selection = rng.choice(
                len(candidates), size=max_blocks_per_level, replace=False
            )
            candidates = [candidates[int(index)] for index in selection]

        tested_ranks = []
        forward_residuals = []
        adjoint_residuals = []
        saturated_blocks = 0
        for row_block, col_block in candidates:
            rows = slice(row_block * row_size, (row_block + 1) * row_size)
            cols = slice(col_block * col_size, (col_block + 1) * col_size)
            forward_range_count = min(range_probes, row_size, col_size)
            adjoint_range_count = forward_range_count
            forward_training = _restricted_action(
                apply,
                n=n,
                input_slice=cols,
                output_slice=rows,
                block=rng.standard_normal((col_size, forward_range_count)),
            )
            forward_test = _restricted_action(
                apply,
                n=n,
                input_slice=cols,
                output_slice=rows,
                block=rng.standard_normal((col_size, residual_probes)),
            )
            adjoint_training = _restricted_action(
                adjoint,
                n=n,
                input_slice=rows,
                output_slice=cols,
                block=rng.standard_normal((row_size, adjoint_range_count)),
            )
            adjoint_test = _restricted_action(
                adjoint,
                n=n,
                input_slice=rows,
                output_slice=cols,
                block=rng.standard_normal((row_size, residual_probes)),
            )
            forward_rank, forward_residual, forward_saturated = _tested_range_rank(
                forward_training,
                forward_test,
                relative_tolerance=relative_tolerance,
            )
            adjoint_rank, adjoint_residual, adjoint_saturated = _tested_range_rank(
                adjoint_training,
                adjoint_test,
                relative_tolerance=relative_tolerance,
            )
            tested_ranks.append(max(forward_rank, adjoint_rank))
            forward_residuals.append(forward_residual)
            adjoint_residuals.append(adjoint_residual)
            saturated_blocks += int(forward_saturated or adjoint_saturated)

        results.append(
            {
                "level": level,
                "row_block_size": row_size,
                "column_block_size": col_size,
                "sampled_blocks": len(candidates),
                "maximum_tested_rank": int(max(tested_ranks, default=0)),
                "median_tested_rank": float(np.median(tested_ranks)) if tested_ranks else 0.0,
                "maximum_forward_residual": float(max(forward_residuals, default=0.0)),
                "maximum_adjoint_residual": float(max(adjoint_residuals, default=0.0)),
                "saturated_blocks": saturated_blocks,
                "diagnostic_only": True,
            }
        )
    return results
