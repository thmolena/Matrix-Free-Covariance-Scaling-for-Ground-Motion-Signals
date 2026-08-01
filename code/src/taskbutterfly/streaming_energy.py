"""Streaming station-referenced energy statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def centered_block_energies(
    components: np.ndarray,
    *,
    block_size: int = 64,
) -> np.ndarray:
    """Return three-component, componentwise-centered block energies."""

    values = np.asarray(components, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("components must have shape (channels, samples)")
    if block_size < 2:
        raise ValueError("block_size must be at least two")
    blocks = values.shape[1] // block_size
    if blocks < 1:
        raise ValueError("the trace must contain one complete block")
    work = values[:, : blocks * block_size].reshape(
        values.shape[0], blocks, block_size
    )
    work = work - np.mean(work, axis=2, keepdims=True)
    return np.sum(work * work, axis=(0, 2))


def log_reference_ratio(
    candidate_energies: np.ndarray,
    reference_energies: np.ndarray,
    *,
    prefix_blocks: int,
) -> float:
    """Log10 prefix mean energy divided by the reference mean energy."""

    candidate = np.asarray(candidate_energies, dtype=float)
    reference = np.asarray(reference_energies, dtype=float)
    if prefix_blocks < 1 or prefix_blocks > candidate.size:
        raise ValueError("prefix_blocks is outside the candidate trace")
    if np.any(candidate < 0.0) or np.any(reference < 0.0):
        raise ValueError("energies must be nonnegative")
    denominator = float(np.mean(reference))
    numerator = float(np.mean(candidate[:prefix_blocks]))
    if numerator <= 0.0 or denominator <= 0.0:
        raise ValueError("candidate and reference energies must be positive")
    return float(np.log10(numerator / denominator))


def raw_log_mean(candidate_energies: np.ndarray, *, prefix_blocks: int) -> float:
    candidate = np.asarray(candidate_energies, dtype=float)
    if prefix_blocks < 1 or prefix_blocks > candidate.size:
        raise ValueError("prefix_blocks is outside the candidate trace")
    value = float(np.mean(candidate[:prefix_blocks]))
    if value <= 0.0:
        raise ValueError("candidate energy must be positive")
    return float(np.log10(value))


def raw_log_quantile(
    candidate_energies: np.ndarray,
    *,
    prefix_blocks: int,
    quantile: float = 0.9,
) -> float:
    candidate = np.asarray(candidate_energies, dtype=float)
    if prefix_blocks < 1 or prefix_blocks > candidate.size:
        raise ValueError("prefix_blocks is outside the candidate trace")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    value = float(np.quantile(candidate[:prefix_blocks], quantile))
    if value <= 0.0:
        raise ValueError("candidate energy must be positive")
    return float(np.log10(value))


def best_balanced_threshold(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> tuple[float, float]:
    """Choose the first threshold attaining maximum balanced accuracy."""

    positive = np.asarray(positive_scores, dtype=float)
    negative = np.asarray(negative_scores, dtype=float)
    if positive.size < 1 or negative.size < 1:
        raise ValueError("both classes must be nonempty")
    unique = np.unique(np.concatenate((positive, negative)))
    thresholds = np.concatenate(
        (
            [np.nextafter(unique[0], -np.inf)],
            0.5 * (unique[:-1] + unique[1:]),
            [np.nextafter(unique[-1], np.inf)],
        )
    )
    values = np.asarray(
        [
            0.5
            * (
                np.mean(positive >= threshold)
                + np.mean(negative < threshold)
            )
            for threshold in thresholds
        ]
    )
    index = int(np.argmax(values))
    return float(thresholds[index]), float(values[index])


def balanced_accuracy(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    positive = np.asarray(positive_scores, dtype=float)
    negative = np.asarray(negative_scores, dtype=float)
    sensitivity = float(np.mean(positive >= threshold))
    specificity = float(np.mean(negative < threshold))
    return 0.5 * (sensitivity + specificity), sensitivity, specificity


def log_ratio_perturbation_bound(relative_error: float) -> float:
    """Worst log-ratio change when both energies have relative error delta."""

    delta = float(relative_error)
    if not 0.0 <= delta < 1.0:
        raise ValueError("relative_error must lie in [0, 1)")
    return float(np.log10((1.0 + delta) / (1.0 - delta)))


@dataclass
class StreamingEnergy:
    """Constant-memory mean-energy accumulator for one multichannel block."""

    reference_mean_energy: float
    block_size: int = 64
    blocks_seen: int = 0
    accumulated_energy: float = 0.0

    def update(self, block: np.ndarray) -> float:
        values = np.asarray(block, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.block_size:
            raise ValueError("block has the wrong shape")
        centered = values - np.mean(values, axis=1, keepdims=True)
        self.accumulated_energy += float(np.sum(centered * centered))
        self.blocks_seen += 1
        if self.reference_mean_energy <= 0.0:
            raise ValueError("reference_mean_energy must be positive")
        return float(
            np.log10(
                self.accumulated_energy
                / (self.blocks_seen * self.reference_mean_energy)
            )
        )
