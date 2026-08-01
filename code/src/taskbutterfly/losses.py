"""Loss terms for matrix-free scientific-operator learning."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def normalized_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    denominator = max(float(np.vdot(target, target).real), np.finfo(float).eps)
    return float(np.vdot(prediction - target, prediction - target).real / denominator)


def complex_huber(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    delta: float = 1.0,
    mask: np.ndarray | None = None,
) -> float:
    """Huber loss applied to complex residual magnitudes."""
    residual = np.abs(np.asarray(prediction) - np.asarray(target))
    if mask is not None:
        residual = residual[np.asarray(mask, dtype=bool)]
    quadratic = np.minimum(residual, delta)
    linear = residual - quadratic
    return float(np.mean(0.5 * quadratic**2 + delta * linear))


def reciprocity_loss(matrix: np.ndarray) -> float:
    """Normalized violation of complex transpose reciprocity."""
    matrix = np.asarray(matrix)
    return normalized_mse(matrix, matrix.T)


def conjugate_symmetry_loss(
    positive_frequency: np.ndarray, negative_frequency: np.ndarray
) -> float:
    """Normalized violation of D(-omega) = conjugate(D(omega))."""
    return normalized_mse(negative_frequency, np.conjugate(positive_frequency))


def randomized_probe_loss(
    approximate_apply: Callable[[np.ndarray], np.ndarray],
    reference_apply: Callable[[np.ndarray], np.ndarray],
    *,
    n: int,
    probes: int = 8,
    seed: int = 0,
    complex_valued: bool = True,
) -> float:
    """Estimate relative operator-action error without assembling either matrix."""
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n, probes))
    if complex_valued:
        omega = omega + 1j * rng.standard_normal((n, probes))
        omega /= np.sqrt(2.0)
    return normalized_mse(approximate_apply(omega), reference_apply(omega))


def certificate_gated_action_loss(
    action_loss: float,
    *,
    certificate_upper_bounds: np.ndarray | list[float] | tuple[float, ...],
    certificate_tolerance: float,
) -> float:
    r"""Return the selective action risk used by the deployment gate.

    The value is the independently measured action loss only when every fixed
    certified task passes.  Structural action probes are reported separately
    as diagnostics and cannot turn a failed error certificate into deployment.
    Certification data must be disjoint from optimization data; this
    non-differentiable functional is not used for gradient updates.
    """
    bounds = np.asarray(certificate_upper_bounds, dtype=float)
    if bounds.ndim != 1 or bounds.size == 0:
        raise ValueError("certificate_upper_bounds must be a nonempty vector")
    values = np.concatenate(([action_loss, certificate_tolerance], bounds))
    if not np.all(np.isfinite(values)):
        raise ValueError("loss, bound, and tolerance must be finite")
    if action_loss < 0.0 or np.any(bounds < 0.0):
        raise ValueError("losses and certificate bounds must be nonnegative")
    if certificate_tolerance <= 0.0:
        raise ValueError("certificate_tolerance must be positive")
    if np.any(bounds > certificate_tolerance):
        return float("inf")
    return float(action_loss)
