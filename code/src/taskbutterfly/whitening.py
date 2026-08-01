"""Empirical whitening targets, certificates, baselines, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Callable

import numpy as np


def empirical_covariance(
    quiet_windows: np.ndarray,
    *,
    physical_dimension: int = 192,
) -> np.ndarray:
    """Estimate covariance and give padded coordinates a neutral variance."""

    samples = np.asarray(quiet_windows, dtype=float)
    if samples.ndim != 2:
        raise ValueError("quiet_windows must have shape (dimension, samples)")
    with np.errstate(all="ignore"):
        covariance = samples @ samples.T / samples.shape[1]
    if not 0 < physical_dimension <= samples.shape[0]:
        raise ValueError("physical_dimension is outside the sample dimension")
    physical_trace = float(np.trace(covariance[:physical_dimension, :physical_dimension]))
    neutral_variance = physical_trace / physical_dimension
    if physical_dimension < samples.shape[0]:
        covariance[physical_dimension:, :] = 0.0
        covariance[:, physical_dimension:] = 0.0
        covariance[physical_dimension:, physical_dimension:] = (
            neutral_variance * np.eye(samples.shape[0] - physical_dimension)
        )
    return 0.5 * (covariance + covariance.T)


@dataclass(frozen=True)
class WhiteningFamily:
    """Inverse-square-root family derived from an empirical covariance."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    mean_variance: float

    @classmethod
    def from_covariance(cls, covariance: np.ndarray) -> "WhiteningFamily":
        matrix = np.asarray(covariance, dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
        eigenvalues = np.maximum(eigenvalues, 0.0)
        return cls(
            eigenvalues,
            eigenvectors,
            float(np.trace(matrix) / matrix.shape[0]),
        )

    @property
    def n(self) -> int:
        return self.eigenvalues.size

    def matrix(self, log10_relative_ridge: float) -> np.ndarray:
        ridge = (10.0 ** float(log10_relative_ridge)) * self.mean_variance
        weights = 1.0 / np.sqrt(self.eigenvalues + ridge)
        with np.errstate(all="ignore"):
            matrix = (self.eigenvectors * weights) @ self.eigenvectors.T
        # A fixed Frobenius normalization makes optimization comparable across
        # ridge values and has no effect on the downstream ordering score.
        return matrix * np.sqrt(self.n) / np.linalg.norm(matrix, ord="fro")

    def apply(self, block: np.ndarray, log10_relative_ridge: float) -> np.ndarray:
        """Apply the normalized whitener without materializing its dense matrix.

        The eigensystem defines the small experimental reference family, but
        training consumes only this action.  Dense assembly is reserved for
        explicitly labeled oracle controls and independent audits.
        """

        value = np.asarray(block)
        was_vector = value.ndim == 1
        if was_vector:
            value = value[:, None]
        if value.ndim != 2 or value.shape[0] != self.n:
            raise ValueError(f"block must have shape ({self.n},) or ({self.n}, b)")
        ridge = (10.0 ** float(log10_relative_ridge)) * self.mean_variance
        weights = 1.0 / np.sqrt(self.eigenvalues + ridge)
        weights *= np.sqrt(self.n) / np.linalg.norm(weights)
        with np.errstate(all="ignore"):
            result = self.eigenvectors @ (
                weights[:, None] * (self.eigenvectors.T @ value)
            )
        return result[:, 0] if was_vector else result


def relative_action_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = max(float(np.sum(np.abs(np.asarray(target)) ** 2)), np.finfo(float).eps)
    return float(
        np.sum(np.abs(np.asarray(prediction) - np.asarray(target)) ** 2)
        / denominator
    )


def gaussian_frobenius_certificate(
    approximate_apply: Callable[[np.ndarray], np.ndarray],
    reference_apply: Callable[[np.ndarray], np.ndarray],
    *,
    n: int,
    probes: int = 512,
    failure_probability: float = 0.05,
    familywise_hypotheses: int = 1,
    seed: int = 0,
) -> dict[str, float | int]:
    """Independent relative-Frobenius estimate with a family-wise upper bound.

    For Gaussian probes, the squared action norm divided by the probe count is
    unbiased for the squared Frobenius norm.  Gaussian quadratic-form tails
    bound the residual from below and the reference from above.  A union bound
    covers both tails and every member of a fixed candidate/task/seed family.

    ``failure_probability`` is the failure budget for the *entire* fixed
    family, not for one inspected hypothesis.  The same global value and
    ``familywise_hypotheses`` must therefore be supplied for every member.
    """

    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    if n < 1 or probes < 1:
        raise ValueError("n and probes must be positive")
    if familywise_hypotheses < 1:
        raise ValueError("familywise_hypotheses must be positive")
    per_hypothesis = failure_probability / familywise_hypotheses
    per_tail = per_hypothesis / 2.0
    concentration_t = log(1.0 / per_tail)
    lower_tail_radius = 2.0 * sqrt(concentration_t / probes)
    upper_tail_radius = lower_tail_radius + 2.0 * concentration_t / probes
    if lower_tail_radius >= 1.0:
        raise ValueError(
            "too few probes for a finite family-wise Gaussian confidence bound"
        )
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n, probes))
    with np.errstate(all="ignore"):
        reference = reference_apply(omega)
        residual = approximate_apply(omega) - reference
    residual_estimate = float(np.sum(np.abs(residual) ** 2) / probes)
    reference_estimate = float(np.sum(np.abs(reference) ** 2) / probes)
    if not np.isfinite(residual_estimate) or not np.isfinite(reference_estimate):
        raise ValueError("operator actions produced a nonfinite probe norm")
    if reference_estimate <= 0.0:
        raise ValueError("reference action has zero sampled Frobenius norm")
    relative_estimate = residual_estimate / reference_estimate
    upper = (
        relative_estimate
        * (1.0 + upper_tail_radius)
        / (1.0 - lower_tail_radius)
    )
    return {
        "probes": probes,
        "seed": seed,
        "familywise_failure_probability": failure_probability,
        "familywise_hypotheses": familywise_hypotheses,
        "per_hypothesis_failure_probability": per_hypothesis,
        "per_tail_failure_probability": per_tail,
        "concentration_t": concentration_t,
        "lower_tail_radius": lower_tail_radius,
        "upper_tail_radius": upper_tail_radius,
        "relative_frobenius_mse_estimate": relative_estimate,
        "relative_frobenius_mse_upper_bound": upper,
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware binary ROC area without a machine-learning dependency."""

    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    positive = values[y == 1]
    negative = values[y == 0]
    if not len(positive) or not len(negative):
        raise ValueError("both classes are required")
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0.0) + 0.5 * (comparisons == 0.0)))


def paired_bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    pair_ids: np.ndarray,
    *,
    repetitions: int = 2000,
    confidence: float = 0.95,
    seed: int = 260726,
) -> dict[str, float | int]:
    """Bootstrap event--quiet station/event pairs as the independent units."""

    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    groups = np.asarray(pair_ids)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("at least two independent pairs are required")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        sampled_labels = []
        sampled_scores = []
        for synthetic_group, source_group in enumerate(sampled):
            mask = groups == source_group
            sampled_labels.extend(y[mask])
            sampled_scores.extend(values[mask])
        estimates.append(roc_auc(np.asarray(sampled_labels), np.asarray(sampled_scores)))
    alpha = (1.0 - confidence) / 2.0
    return {
        "auc": roc_auc(y, values),
        "confidence": confidence,
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "independent_pairs": int(len(unique)),
    }


def trace_coherence_score(transformed_windows: np.ndarray) -> float:
    """Negative upper-tail whitened energy; larger means more coherent."""

    energies = np.sum(np.asarray(transformed_windows) ** 2, axis=0)
    return -float(np.quantile(energies, 0.90))


def operator_energy_score_error_bound(
    *,
    operator_error_norm: float,
    reference_operator_norm: float,
    maximum_input_norm: float,
) -> float:
    """Deterministic operator-to-energy-quantile score bound.

    If ``approximate = reference + error`` and the supplied values bound
    ``||error||_2``, ``||reference||_2``, and every input-window norm, then the
    returned value bounds the absolute change of any empirical energy
    quantile, including the negative 90th-percentile score used here.
    """

    values = np.asarray(
        [operator_error_norm, reference_operator_norm, maximum_input_norm],
        dtype=float,
    )
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("operator and input norm bounds must be finite and nonnegative")
    epsilon, reference_norm, radius = map(float, values)
    return epsilon * (2.0 * reference_norm + epsilon) * radius**2


def energy_score_perturbation_bound(
    reference_transformed: np.ndarray,
    approximate_transformed: np.ndarray,
) -> dict[str, float | int]:
    """A posteriori bound for the negative energy-quantile score.

    For each column, ``e = approximate - reference`` gives
    ``| ||reference+e||^2 - ||reference||^2 |
    <= ||e|| (2 ||reference|| + ||e||)``.  Every empirical quantile is
    one-Lipschitz in the maximum perturbation of its samples.
    """

    reference = np.asarray(reference_transformed, dtype=float)
    approximate = np.asarray(approximate_transformed, dtype=float)
    if reference.shape != approximate.shape or reference.ndim != 2:
        raise ValueError("transformed windows must be same-shaped matrices")
    error = approximate - reference
    error_norms = np.linalg.norm(error, axis=0)
    reference_norms = np.linalg.norm(reference, axis=0)
    per_window_bounds = error_norms * (2.0 * reference_norms + error_norms)
    score_error = abs(
        trace_coherence_score(approximate) - trace_coherence_score(reference)
    )
    upper = float(np.max(per_window_bounds, initial=0.0))
    return {
        "windows": int(reference.shape[1]),
        "score_absolute_error": float(score_error),
        "score_absolute_error_upper_bound": upper,
        "covered": bool(score_error <= upper + 32.0 * np.finfo(float).eps),
    }


def tie_aware_auc_perturbation_bound(
    labels: np.ndarray,
    reference_scores: np.ndarray,
    approximate_scores: np.ndarray,
    score_error_bounds: np.ndarray,
) -> dict[str, float | int | list[float]]:
    """Tie-aware AUC lower and absolute perturbation bounds.

    If trace ``i`` has score error at most ``eta_i``, a positive/negative
    margin can change by at most ``eta_i + eta_j``.  Strictly positive margins
    above that budget retain their AUC contribution.  Exact ties can lose at
    most one half, which is accounted for separately.
    """

    y = np.asarray(labels, dtype=int)
    reference = np.asarray(reference_scores, dtype=float)
    approximate = np.asarray(approximate_scores, dtype=float)
    eta = np.asarray(score_error_bounds, dtype=float)
    if not (y.shape == reference.shape == approximate.shape == eta.shape):
        raise ValueError("labels, scores, and error bounds must have one shape")
    if np.any(eta < 0.0):
        raise ValueError("score error bounds must be nonnegative")
    if np.any(np.abs(approximate - reference) > eta + 64.0 * np.finfo(float).eps):
        raise ValueError("a supplied score error bound is violated")
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("both classes are required")
    margins = reference[positive, None] - reference[negative][None, :]
    budgets = eta[positive, None] + eta[negative][None, :]
    pair_count = margins.size
    positive_at_risk = (margins > 0.0) & (margins <= budgets)
    exact_ties = margins == 0.0
    absolute_at_risk = np.abs(margins) <= budgets
    one_sided_loss = (
        float(np.count_nonzero(positive_at_risk))
        + 0.5 * float(np.count_nonzero(exact_ties))
    ) / pair_count
    absolute_change = float(np.count_nonzero(absolute_at_risk)) / pair_count
    reference_auc = roc_auc(y, reference)
    approximate_auc = roc_auc(y, approximate)
    return {
        "positive_negative_pairs": int(pair_count),
        "reference_auc": reference_auc,
        "approximate_auc": approximate_auc,
        "auc_lower_bound": max(0.0, reference_auc - one_sided_loss),
        "one_sided_auc_loss_bound": one_sided_loss,
        "absolute_auc_change_bound": absolute_change,
        "actual_auc_change": approximate_auc - reference_auc,
        "positive_margins_certified_stable": int(
            np.count_nonzero((margins > 0.0) & (margins > budgets))
        ),
        "positive_margins_at_risk": int(np.count_nonzero(positive_at_risk)),
        "exact_reference_ties": int(np.count_nonzero(exact_ties)),
        "pair_margins": margins.ravel().tolist(),
        "pair_error_budgets": budgets.ravel().tolist(),
    }


def diagonal_whitener(covariance: np.ndarray, relative_ridge: float) -> np.ndarray:
    matrix = np.asarray(covariance)
    ridge = relative_ridge * float(np.trace(matrix) / matrix.shape[0])
    diagonal = 1.0 / np.sqrt(np.maximum(np.diag(matrix), 0.0) + ridge)
    result = np.diag(diagonal)
    return result * np.sqrt(matrix.shape[0]) / np.linalg.norm(result, ord="fro")


def per_component_fft_whitener(
    quiet_windows: np.ndarray,
    *,
    samples_per_component: int = 64,
    physical_components: int = 3,
    relative_ridge: float = 1.0e-3,
) -> np.ndarray:
    """Stationary per-component spectral-whitening matrix baseline."""

    values = np.asarray(quiet_windows)
    n = values.shape[0]
    result = np.eye(n)
    for component in range(physical_components):
        segment = values[
            component * samples_per_component : (component + 1) * samples_per_component
        ]
        power = np.mean(np.abs(np.fft.rfft(segment, axis=0)) ** 2, axis=1)
        ridge = relative_ridge * float(np.mean(power))
        gain = 1.0 / np.sqrt(power + ridge)
        basis_action = np.fft.irfft(
            np.fft.rfft(np.eye(samples_per_component), axis=0) * gain[:, None],
            n=samples_per_component,
            axis=0,
        )
        selection = slice(
            component * samples_per_component,
            (component + 1) * samples_per_component,
        )
        result[selection, selection] = basis_action
    return result * np.sqrt(n) / np.linalg.norm(result, ord="fro")


def best_rank_approximation(matrix: np.ndarray, rank: int) -> np.ndarray:
    u, singular_values, vh = np.linalg.svd(np.asarray(matrix), full_matrices=False)
    retained = min(int(rank), len(singular_values))
    with np.errstate(all="ignore"):
        return (u[:, :retained] * singular_values[:retained]) @ vh[:retained]


@dataclass(frozen=True)
class SpectralLowRankOperator:
    """Symmetric low-rank baseline applied from its stored spectral factors."""

    basis: np.ndarray
    eigenvalues: np.ndarray

    @classmethod
    def from_dense(
        cls,
        matrix: np.ndarray,
        rank: int,
    ) -> "SpectralLowRankOperator":
        value = np.asarray(matrix, dtype=float)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("matrix must be square")
        if not 1 <= rank <= value.shape[0]:
            raise ValueError("rank must lie between 1 and the matrix dimension")
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (value + value.T))
        indices = np.argsort(np.abs(eigenvalues))[::-1][: int(rank)]
        return cls(eigenvectors[:, indices], eigenvalues[indices])

    @property
    def n(self) -> int:
        return int(self.basis.shape[0])

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    @property
    def parameter_count(self) -> int:
        return int(self.basis.size + self.eigenvalues.size)

    def apply(self, block: np.ndarray) -> np.ndarray:
        value = np.asarray(block)
        was_vector = value.ndim == 1
        if was_vector:
            value = value[:, None]
        if value.ndim != 2 or value.shape[0] != self.n:
            raise ValueError(f"block must have shape ({self.n},) or ({self.n}, b)")
        with np.errstate(all="ignore"):
            result = self.basis @ (
                self.eigenvalues[:, None] * (self.basis.T @ value)
            )
        return result[:, 0] if was_vector else result

    def adjoint(self, block: np.ndarray) -> np.ndarray:
        return self.apply(block)

    def to_dense(self) -> np.ndarray:
        return self.apply(np.eye(self.n))


@dataclass
class _HODLRNode:
    size: int
    leaf: np.ndarray | None
    left: "_HODLRNode | None"
    right: "_HODLRNode | None"
    upper_u: np.ndarray | None
    upper_vh: np.ndarray | None
    lower_u: np.ndarray | None
    lower_vh: np.ndarray | None

    @property
    def storage(self) -> int:
        if self.leaf is not None:
            return int(self.leaf.size)
        arrays = (self.upper_u, self.upper_vh, self.lower_u, self.lower_vh)
        return int(sum(array.size for array in arrays if array is not None)) + int(
            self.left.storage + self.right.storage  # type: ignore[union-attr]
        )

    def apply(self, block: np.ndarray) -> np.ndarray:
        if self.leaf is not None:
            with np.errstate(all="ignore"):
                return self.leaf @ block
        midpoint = self.size // 2
        left_input = block[:midpoint]
        right_input = block[midpoint:]
        left_output = self.left.apply(left_input)  # type: ignore[union-attr]
        right_output = self.right.apply(right_input)  # type: ignore[union-attr]
        with np.errstate(all="ignore"):
            left_output += self.upper_u @ (self.upper_vh @ right_input)
            right_output += self.lower_u @ (self.lower_vh @ left_input)
        return np.vstack((left_output, right_output))

    def adjoint(self, block: np.ndarray) -> np.ndarray:
        if self.leaf is not None:
            with np.errstate(all="ignore"):
                return self.leaf.T @ block
        midpoint = self.size // 2
        left_input = block[:midpoint]
        right_input = block[midpoint:]
        left_output = self.left.adjoint(left_input)  # type: ignore[union-attr]
        right_output = self.right.adjoint(right_input)  # type: ignore[union-attr]
        with np.errstate(all="ignore"):
            left_output += self.lower_vh.T @ (self.lower_u.T @ right_input)
            right_output += self.upper_vh.T @ (self.upper_u.T @ left_input)
        return np.vstack((left_output, right_output))


def _truncated_factors(block: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    u, singular_values, vh = np.linalg.svd(block, full_matrices=False)
    retained = min(rank, len(singular_values))
    return u[:, :retained] * singular_values[:retained], vh[:retained]


def _build_hodlr(matrix: np.ndarray, leaf_size: int, rank: int) -> _HODLRNode:
    n = matrix.shape[0]
    if n <= leaf_size:
        return _HODLRNode(n, matrix.copy(), None, None, None, None, None, None)
    midpoint = n // 2
    upper_u, upper_vh = _truncated_factors(matrix[:midpoint, midpoint:], rank)
    lower_u, lower_vh = _truncated_factors(matrix[midpoint:, :midpoint], rank)
    return _HODLRNode(
        n,
        None,
        _build_hodlr(matrix[:midpoint, :midpoint], leaf_size, rank),
        _build_hodlr(matrix[midpoint:, midpoint:], leaf_size, rank),
        upper_u,
        upper_vh,
        lower_u,
        lower_vh,
    )


@dataclass(frozen=True)
class HODLROperator:
    """Small transparent HODLR fallback used by the acceptance gate."""

    root: _HODLRNode

    @classmethod
    def from_dense(
        cls,
        matrix: np.ndarray,
        *,
        leaf_size: int = 32,
        off_diagonal_rank: int = 8,
    ) -> "HODLROperator":
        value = np.asarray(matrix, dtype=float)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("matrix must be square")
        return cls(_build_hodlr(value, leaf_size, off_diagonal_rank))

    @property
    def n(self) -> int:
        return self.root.size

    @property
    def parameter_count(self) -> int:
        return self.root.storage

    def apply(self, block: np.ndarray) -> np.ndarray:
        value = np.asarray(block)
        was_vector = value.ndim == 1
        if was_vector:
            value = value[:, None]
        result = self.root.apply(value)
        return result[:, 0] if was_vector else result

    def adjoint(self, block: np.ndarray) -> np.ndarray:
        value = np.asarray(block)
        was_vector = value.ndim == 1
        if was_vector:
            value = value[:, None]
        result = self.root.adjoint(value)
        return result[:, 0] if was_vector else result

    def to_dense(self) -> np.ndarray:
        return self.apply(np.eye(self.n))
