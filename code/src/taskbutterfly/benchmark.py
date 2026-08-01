"""End-to-end error-controlled covariance benchmark on frozen EarthScope data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .compressibility import complementary_action_probe_audit
from .losses import certificate_gated_action_loss
from .operator import (
    ButterflyEnsembleOperator,
    ParametricButterflyEnsembleOperator,
    identity_factors,
)
from .seismic_dataset import (
    EVENTS,
    STATIONS,
    load_record,
    multicomponent_windows,
    protocol_records,
)
from .training import train_parametric_ensemble
from .whitening import (
    HODLROperator,
    SpectralLowRankOperator,
    WhiteningFamily,
    diagonal_whitener,
    energy_score_perturbation_bound,
    empirical_covariance,
    gaussian_frobenius_certificate,
    paired_bootstrap_auc,
    per_component_fft_whitener,
    relative_action_mse,
    roc_auc,
    trace_coherence_score,
    tie_aware_auc_perturbation_bound,
)

DEFAULT_SEEDS = (260726, 260727, 260728, 260729, 260730)
PARAMETER_KNOTS = (-4.0, -2.0)
TRAINING_PARAMETERS = np.linspace(-4.0, -2.0, 5)
EVALUATION_PARAMETER = -3.0
COMPONENT_GRID = (1, 2, 4)
CERTIFICATE_TOLERANCE = 0.15
COMPLEMENTARY_RANK_TOLERANCE = 0.05
FAMILYWISE_FAILURE_PROBABILITY = 0.05


def _system_provenance() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not reported by platform",
        "cpu_count": __import__("os").cpu_count(),
    }


def _load_protocol(
    *,
    cache_dir: Path | None,
    offline: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prepared = []
    provenance = []
    for record in protocol_records():
        components, metadata = load_record(
            record,
            cache_dir=cache_dir,
            offline=offline,
        )
        shape_windows, amplitude_windows = multicomponent_windows(components)
        event = record["event"]
        station = record["station"]
        prepared.append(
            {
                "key": record["key"],
                "pair_id": f"{event.event_id}/{station.code}",
                "split": event.split,
                "label": record["label"],
                "shape_windows": shape_windows,
                "amplitude_windows": amplitude_windows,
                "components": components,
            }
        )
        provenance.append(
            {
                "key": metadata["key"],
                "url": metadata["url"],
                "sha256": metadata["sha256"],
                "bytes": metadata["bytes"],
                "selected_sids": metadata["selected_sids"],
                "native_sample_rates_hz": metadata["native_sample_rates_hz"],
                "target_sample_rate_hz": metadata["target_sample_rate_hz"],
                "common_samples": metadata["common_samples"],
                "maximum_gap_seconds": max(metadata["maximum_gaps_seconds"]),
                "scale_units": metadata["scale_units"],
                "instruments": metadata["instruments"],
            }
        )
    return prepared, provenance


def _scores(
    records: list[dict[str, object]],
    split: str,
    apply: Callable[[np.ndarray], np.ndarray] | None,
    *,
    amplitude: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = []
    scores = []
    pair_ids = []
    for record in records:
        if record["split"] != split:
            continue
        labels.append(1 if record["label"] == "event" else 0)
        pair_ids.append(record["pair_id"])
        if amplitude:
            windows = np.asarray(record["amplitude_windows"])
            scores.append(float(np.quantile(np.sum(windows**2, axis=0), 0.90)))
        elif apply is None:
            scores.append(0.0)
        else:
            windows = np.asarray(record["shape_windows"])
            with np.errstate(all="ignore"):
                transformed = apply(windows)
            scores.append(trace_coherence_score(transformed))
    return np.asarray(labels), np.asarray(scores), np.asarray(pair_ids)


def _threshold_metrics(
    train_labels: np.ndarray,
    train_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, float]:
    unique = np.unique(train_scores)
    candidates = np.concatenate(
        (
            [np.nextafter(unique[0], -np.inf)],
            0.5 * (unique[:-1] + unique[1:]),
            [np.nextafter(unique[-1], np.inf)],
        )
    )

    def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
        sensitivity = np.mean(predictions[labels == 1] == 1)
        specificity = np.mean(predictions[labels == 0] == 0)
        return float(0.5 * (sensitivity + specificity))

    objectives = [
        balanced_accuracy(train_labels, (train_scores >= threshold).astype(int))
        for threshold in candidates
    ]
    threshold = float(candidates[int(np.argmax(objectives))])
    predictions = (test_scores >= threshold).astype(int)
    true_positive = int(np.sum((test_labels == 1) & (predictions == 1)))
    false_positive = int(np.sum((test_labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((test_labels == 1) & (predictions == 0)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(float).eps)
    return {
        "training_threshold": threshold,
        "training_balanced_accuracy": float(max(objectives)),
        "test_balanced_accuracy": balanced_accuracy(test_labels, predictions),
        "test_f1": float(f1),
    }


def _evaluate_method(
    name: str,
    records: list[dict[str, object]],
    apply: Callable[[np.ndarray], np.ndarray] | None,
    *,
    amplitude: bool = False,
) -> dict[str, object]:
    train_labels, train_scores, _ = _scores(
        records, "train", apply, amplitude=amplitude
    )
    test_labels, test_scores, pair_ids = _scores(
        records, "test", apply, amplitude=amplitude
    )
    orientation = 1
    if roc_auc(train_labels, train_scores) < 0.5:
        orientation = -1
        train_scores = -train_scores
        test_scores = -test_scores
    return {
        "name": name,
        "orientation_selected_on_training": orientation,
        **paired_bootstrap_auc(test_labels, test_scores, pair_ids),
        **_threshold_metrics(
            train_labels, train_scores, test_labels, test_scores
        ),
        "test_scores": test_scores.tolist(),
        "test_labels": test_labels.tolist(),
        "test_pair_ids": pair_ids.tolist(),
    }


def _downstream_perturbation_audit(
    records: list[dict[str, object]],
    approximate_apply: Callable[[np.ndarray], np.ndarray],
    reference_apply: Callable[[np.ndarray], np.ndarray],
) -> dict[str, object]:
    """Post-hoc theorem audit using the fixed test traces and reference action."""

    train_labels, reference_train_scores, _ = _scores(
        records, "train", reference_apply
    )
    orientation = -1 if roc_auc(train_labels, reference_train_scores) < 0.5 else 1
    labels = []
    reference_scores = []
    approximate_scores = []
    score_bounds = []
    traces = []
    for record in records:
        if record["split"] != "test":
            continue
        windows = np.asarray(record["shape_windows"])
        reference = reference_apply(windows)
        approximate = approximate_apply(windows)
        bound = energy_score_perturbation_bound(reference, approximate)
        labels.append(1 if record["label"] == "event" else 0)
        reference_scores.append(orientation * trace_coherence_score(reference))
        approximate_scores.append(orientation * trace_coherence_score(approximate))
        score_bounds.append(bound["score_absolute_error_upper_bound"])
        traces.append(
            {
                "key": record["key"],
                "pair_id": record["pair_id"],
                "label": record["label"],
                **bound,
            }
        )
    auc_bound = tie_aware_auc_perturbation_bound(
        np.asarray(labels),
        np.asarray(reference_scores),
        np.asarray(approximate_scores),
        np.asarray(score_bounds),
    )
    return {
        "orientation_selected_on_dense_training_scores": orientation,
        "trace_score_bounds_covered": sum(bool(row["covered"]) for row in traces),
        "test_traces": len(traces),
        "maximum_actual_score_error": max(
            float(row["score_absolute_error"]) for row in traces
        ),
        "maximum_score_error_bound": max(
            float(row["score_absolute_error_upper_bound"]) for row in traces
        ),
        "per_trace": traces,
        "tie_aware_auc": auc_bound,
        "post_hoc_reference_access": True,
    }


def _timed(
    function: Callable[[], object],
    *,
    warmups: int = 3,
    repetitions: int = 21,
) -> dict[str, float | int]:
    for _ in range(warmups):
        function()
    values = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - start) / 1.0e6)
    return {
        "warmups": warmups,
        "repetitions": repetitions,
        "median_ms": float(np.median(values)),
        "q25_ms": float(np.quantile(values, 0.25)),
        "q75_ms": float(np.quantile(values, 0.75)),
    }


def _random_ensemble(n: int, components: int, seed: int) -> ButterflyEnsembleOperator:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(components):
        factors = identity_factors(n)
        factors[0] /= components
        factors += 1.0e-3 * rng.standard_normal(factors.shape) / np.sqrt(n)
        values.append(factors)
    return ButterflyEnsembleOperator(np.stack(values))


def _fft_kernel(block: np.ndarray) -> np.ndarray:
    return np.fft.irfft(np.fft.rfft(block, axis=0), n=block.shape[0], axis=0)


def _kernel_scaling() -> list[dict[str, object]]:
    results = []
    for n in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
        rng = np.random.default_rng(8000 + n)
        block = rng.standard_normal((n, 8))
        operator = _random_ensemble(n, 4, 9000 + n)
        row: dict[str, object] = {
            "n": n,
            "right_hand_sides": 8,
            "butterfly_parameters": operator.parameter_count,
            "butterfly_storage_bytes": 8 * operator.parameter_count,
            "butterfly_apply": _timed(lambda: operator.apply(block), repetitions=11),
            "butterfly_adjoint": _timed(
                lambda: operator.adjoint(block), repetitions=11
            ),
            "fft_apply": _timed(lambda: _fft_kernel(block), repetitions=11),
        }
        if n <= 2048:
            dense = rng.standard_normal((n, n)) / np.sqrt(n)
            row["dense_storage_bytes"] = int(dense.nbytes)
            with np.errstate(all="ignore"):
                row["dense_apply"] = _timed(lambda: dense @ block, repetitions=11)
                row["dense_adjoint"] = _timed(
                    lambda: dense.T @ block, repetitions=11
                )
        results.append(row)
    return results


def _compressibility_study(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    duration_results = []
    primary_family = None
    for samples_per_component in (16, 32, 64, 128):
        physical = 3 * samples_per_component
        n = 1 << (physical - 1).bit_length()
        quiet = []
        for record in records:
            if record["split"] == "train" and record["label"] == "quiet":
                windows, _ = multicomponent_windows(
                    np.asarray(record["components"]),
                    samples_per_component=samples_per_component,
                    stride=samples_per_component,
                    padded_dimension=n,
                )
                quiet.append(windows)
        covariance = empirical_covariance(
            np.column_stack(quiet), physical_dimension=physical
        )
        family = WhiteningFamily.from_covariance(covariance)
        if samples_per_component == 64:
            primary_family = family
        audit = complementary_action_probe_audit(
            lambda block, target=family: target.apply(block, EVALUATION_PARAMETER),
            lambda block, target=family: target.apply(block, EVALUATION_PARAMETER),
            n=n,
            relative_tolerance=COMPLEMENTARY_RANK_TOLERANCE,
            max_blocks_per_level=8,
            range_probes=20,
            residual_probes=8,
            seed=260726,
        )
        maximum = max(int(entry["maximum_tested_rank"]) for entry in audit)
        duration_results.append(
            {
                "samples_per_component": samples_per_component,
                "duration_seconds": samples_per_component / 20.0,
                "padded_dimension": n,
                "maximum_tested_rank": maximum,
                "saturated_blocks": sum(int(entry["saturated_blocks"]) for entry in audit),
                "diagnostic_only": True,
                "levels": audit,
            }
        )
    assert primary_family is not None
    ridge_results = []
    for parameter in TRAINING_PARAMETERS:
        audit = complementary_action_probe_audit(
            lambda block, target=primary_family, task=float(parameter): target.apply(block, task),
            lambda block, target=primary_family, task=float(parameter): target.apply(block, task),
            n=primary_family.n,
            relative_tolerance=COMPLEMENTARY_RANK_TOLERANCE,
            max_blocks_per_level=8,
            range_probes=20,
            residual_probes=8,
            seed=260726 + int(round(10.0 * (float(parameter) + 4.0))),
        )
        ridge_results.append(
            {
                "log10_relative_ridge": float(parameter),
                "maximum_tested_rank": max(
                    int(entry["maximum_tested_rank"]) for entry in audit
                ),
                "saturated_blocks": sum(int(entry["saturated_blocks"]) for entry in audit),
                "diagnostic_only": True,
            }
        )
    return duration_results, ridge_results


def run_benchmark(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    steps: int = 1500,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> dict[str, object]:
    total_start = time.perf_counter()
    records, data_provenance = _load_protocol(
        cache_dir=cache_dir,
        offline=offline,
    )
    quiet = np.column_stack(
        [
            np.asarray(record["shape_windows"])
            for record in records
            if record["split"] == "train" and record["label"] == "quiet"
        ]
    )
    construction_start = time.perf_counter()
    covariance = empirical_covariance(quiet)
    family = WhiteningFamily.from_covariance(covariance)
    dense_matrix = family.matrix(EVALUATION_PARAMETER)
    dense_lower = family.matrix(PARAMETER_KNOTS[0])
    dense_upper = family.matrix(PARAMETER_KNOTS[1])
    dense_endpoint_interpolant = 0.5 * (dense_lower + dense_upper)
    dense_construction_seconds = time.perf_counter() - construction_start

    duration_audit, ridge_audit = _compressibility_study(records)
    rng = np.random.default_rng(260725)
    gaussian_pool = rng.standard_normal((256, 1024)) / np.sqrt(256)
    training_inputs = np.column_stack((gaussian_pool, quiet))
    rank_runs = []
    selected_operators = []
    familywise_hypotheses = (
        len(seeds) * len(COMPONENT_GRID) * len(TRAINING_PARAMETERS)
    )
    for seed_index, seed in enumerate(seeds):
        selected = None
        for component_index, components in enumerate(COMPONENT_GRID):
            start = time.perf_counter()
            factors, history = train_parametric_ensemble(
                training_inputs,
                family.apply,
                parameter_knots=PARAMETER_KNOTS,
                training_parameters=TRAINING_PARAMETERS,
                components=components,
                steps=steps,
                batch_size=48,
                seed=seed,
                learning_rate=6.0e-3,
            )
            training_seconds = time.perf_counter() - start
            operator = ParametricButterflyEnsembleOperator(PARAMETER_KNOTS, factors)
            certificates = []
            for task_index, task in enumerate(TRAINING_PARAMETERS):
                certificate_seed = (
                    1_000_000
                    + 100 * seed_index
                    + 10 * component_index
                    + task_index
                )
                certificate = gaussian_frobenius_certificate(
                    lambda block, model=operator, value=float(task): model.apply(
                        value, block
                    ),
                    lambda block, target=family, value=float(task): target.apply(
                        block, value
                    ),
                    n=256,
                    probes=2048,
                    failure_probability=FAMILYWISE_FAILURE_PROBABILITY,
                    familywise_hypotheses=familywise_hypotheses,
                    seed=certificate_seed,
                )
                certificates.append(
                    {"log10_relative_ridge": float(task), **certificate}
                )
            worst_certificate = max(
                certificates,
                key=lambda item: item["relative_frobenius_mse_upper_bound"],
            )
            selective_risk = certificate_gated_action_loss(
                worst_certificate["relative_frobenius_mse_estimate"],
                certificate_upper_bounds=[
                    item["relative_frobenius_mse_upper_bound"]
                    for item in certificates
                ],
                certificate_tolerance=CERTIFICATE_TOLERANCE,
            )
            accepted = bool(np.isfinite(selective_risk))
            run = {
                "seed": seed,
                "components": components,
                "parameter_count_two_knots": operator.parameter_count,
                "training_seconds": training_seconds,
                "training": history,
                "certificates_by_task": certificates,
                "worst_task_certificate": worst_certificate,
                "accepted": accepted,
            }
            rank_runs.append(run)
            if selected is None and accepted:
                selected = (operator, run)
        if selected is None:
            selected_operators.append(("rejected", None, None))
        else:
            selected_operators.append(("butterfly", selected[0], selected[1]))

    hodlr_start = time.perf_counter()
    hodlr = HODLROperator.from_dense(
        dense_matrix, leaf_size=32, off_diagonal_rank=8
    )
    hodlr_construction_seconds = time.perf_counter() - hodlr_start
    endpoint_hodlr_start = time.perf_counter()
    hodlr_lower = HODLROperator.from_dense(
        dense_lower, leaf_size=32, off_diagonal_rank=5
    )
    hodlr_upper = HODLROperator.from_dense(
        dense_upper, leaf_size=32, off_diagonal_rank=5
    )
    endpoint_hodlr_construction_seconds = time.perf_counter() - endpoint_hodlr_start
    diagonal = diagonal_whitener(covariance, 10.0 ** EVALUATION_PARAMETER)
    fft = per_component_fft_whitener(
        quiet, relative_ridge=10.0 ** EVALUATION_PARAMETER
    )

    selected_parameter_count = 4 * 2 * 8 * (256 // 2) * 4
    pca_rank = min(255, selected_parameter_count // (256 + 1))
    pca = SpectralLowRankOperator.from_dense(dense_matrix, pca_rank)
    fixed_methods = {
        "shape_identity": _evaluate_method("shape_identity", records, None),
        "diagonal_whitening": _evaluate_method(
            "diagonal_whitening", records, lambda block: diagonal @ block
        ),
        "fft_whitening": _evaluate_method(
            "fft_whitening", records, lambda block: fft @ block
        ),
        "equal_memory_pca": _evaluate_method(
            "equal_memory_pca", records, pca.apply
        ),
        "hodlr_rank8": _evaluate_method("hodlr_rank8", records, hodlr.apply),
        "equal_memory_hodlr_two_knot": _evaluate_method(
            "equal_memory_hodlr_two_knot",
            records,
            lambda block: 0.5
            * (hodlr_lower.apply(block) + hodlr_upper.apply(block)),
        ),
        "dense_two_knot_interpolant": _evaluate_method(
            "dense_two_knot_interpolant",
            records,
            lambda block: dense_endpoint_interpolant @ block,
        ),
        "dense_whitening": _evaluate_method(
            "dense_whitening", records, lambda block: dense_matrix @ block
        ),
        "raw_amplitude": _evaluate_method(
            "raw_amplitude", records, None, amplitude=True
        ),
    }
    approximation = {
        "diagonal_relative_frobenius_mse": relative_action_mse(
            diagonal, dense_matrix
        ),
        "fft_relative_frobenius_mse": relative_action_mse(fft, dense_matrix),
        "equal_memory_pca_rank": pca_rank,
        "equal_memory_pca_parameters": pca.parameter_count,
        "equal_memory_pca_relative_frobenius_mse": relative_action_mse(
            pca.to_dense(), dense_matrix
        ),
        "hodlr_rank8_parameters": hodlr.parameter_count,
        "hodlr_rank8_relative_frobenius_mse": relative_action_mse(
            hodlr.to_dense(), dense_matrix
        ),
        "equal_memory_hodlr_two_knot_rank": 5,
        "equal_memory_hodlr_two_knot_parameters": (
            hodlr_lower.parameter_count + hodlr_upper.parameter_count
        ),
        "equal_memory_hodlr_two_knot_relative_frobenius_mse": (
            relative_action_mse(
                0.5 * (hodlr_lower.to_dense() + hodlr_upper.to_dense()),
                dense_matrix,
            )
        ),
        "dense_two_knot_parameters": int(2 * dense_matrix.size),
        "dense_two_knot_relative_frobenius_mse": relative_action_mse(
            dense_endpoint_interpolant, dense_matrix
        ),
        "dense_parameters": int(dense_matrix.size),
    }

    seed_metrics = []
    for seed, (selected_kind, operator, run) in zip(seeds, selected_operators):
        if selected_kind == "butterfly":
            apply = lambda block, model=operator: model.apply(
                EVALUATION_PARAMETER, block
            )
            parameter_count = operator.parameter_count
            components = run["components"]
            worst_certificate = run["worst_task_certificate"]
            metrics = _evaluate_method(f"selected_seed_{seed}", records, apply)
            seed_metrics.append(
                {
                    "seed": seed,
                    "selected_kind": selected_kind,
                    "components": components,
                    "parameter_count_two_knots_or_fallback": parameter_count,
                    "worst_task_certificate": worst_certificate,
                    "downstream_perturbation": _downstream_perturbation_audit(
                        records,
                        apply,
                        lambda block: family.apply(block, EVALUATION_PARAMETER),
                    ),
                    **metrics,
                }
            )
        else:
            seed_metrics.append(
                {
                    "seed": seed,
                    "selected_kind": "rejected",
                    "components": None,
                    "parameter_count_two_knots_or_fallback": None,
                    "reason": "no fixed candidate passed every task certificate",
                }
            )
    deployed_metrics = [
        row for row in seed_metrics if row["selected_kind"] == "butterfly"
    ]
    auc_values = np.asarray([row["auc"] for row in deployed_metrics], dtype=float)
    balanced_values = np.asarray(
        [row["test_balanced_accuracy"] for row in deployed_metrics], dtype=float
    )
    f1_values = np.asarray([row["test_f1"] for row in deployed_metrics], dtype=float)
    t_multiplier = 2.776 if len(deployed_metrics) == 5 else 1.96
    if len(deployed_metrics) > 1:
        auc_sd = float(np.std(auc_values, ddof=1))
        auc_half_width = t_multiplier * auc_sd / np.sqrt(len(deployed_metrics))
    elif len(deployed_metrics) == 1:
        auc_sd = 0.0
        auc_half_width = 0.0
    else:
        auc_sd = float("nan")
        auc_half_width = float("nan")
    selected_summary = {
        "optimization_seeds": list(seeds),
        "successful_butterfly_selections": sum(
            row["selected_kind"] == "butterfly" for row in seed_metrics
        ),
        "rejected_selections": len(seed_metrics) - len(deployed_metrics),
        "auc_mean": float(np.mean(auc_values)) if len(auc_values) else None,
        "auc_sample_sd": auc_sd if len(auc_values) else None,
        "auc_seed_mean_95ci_lower": (
            float(np.mean(auc_values) - auc_half_width) if len(auc_values) else None
        ),
        "auc_seed_mean_95ci_upper": (
            float(np.mean(auc_values) + auc_half_width) if len(auc_values) else None
        ),
        "balanced_accuracy_mean": (
            float(np.mean(balanced_values)) if len(balanced_values) else None
        ),
        "f1_mean": float(np.mean(f1_values)) if len(f1_values) else None,
        "per_seed": seed_metrics,
    }

    timing_rng = np.random.default_rng(260726)
    timing_block = timing_rng.standard_normal((256, 32))
    representative_tuple = next(
        (item for item in selected_operators if item[0] == "butterfly"),
        ("rejected", None, None),
    )
    representative_kind, representative, representative_run = representative_tuple
    if representative_kind == "butterfly" and representative is not None:
        representative_apply = lambda: representative.apply(
            EVALUATION_PARAMETER, timing_block
        )
        representative_adjoint = lambda: representative.adjoint(
            EVALUATION_PARAMETER, timing_block
        )
        representative_storage = 8 * representative.parameter_count
    with np.errstate(all="ignore"):
        primary_timing = {
            "right_hand_sides": 32,
            "selected_kind": representative_kind,
            "dense_apply": _timed(lambda: dense_matrix @ timing_block),
            "dense_adjoint": _timed(lambda: dense_matrix.T @ timing_block),
            "hodlr_apply": _timed(lambda: hodlr.apply(timing_block)),
            "hodlr_adjoint": _timed(lambda: hodlr.adjoint(timing_block)),
            "dense_storage_bytes": int(dense_matrix.nbytes),
            "hodlr_storage_bytes": int(8 * hodlr.parameter_count),
            "dense_covariance_eigendecomposition_seconds": dense_construction_seconds,
            "hodlr_construction_seconds": hodlr_construction_seconds,
            "equal_memory_hodlr_two_knot_construction_seconds": (
                endpoint_hodlr_construction_seconds
            ),
        }
        if representative_kind == "butterfly":
            primary_timing.update(
                {
                    "selected_apply": _timed(representative_apply),
                    "selected_adjoint": _timed(representative_adjoint),
                    "selected_storage_bytes": representative_storage,
                }
            )

    result: dict[str, object] = {
        "schema_version": 4,
        "study": {
            "name": "certificate-gated parametric butterfly whitening",
            "primary_task": (
                "shape-only event-versus-clock-matched-quiet discrimination "
                "after empirical three-component covariance whitening"
            ),
            "train_events": [
                event.event_id for event in EVENTS if event.split == "train"
            ],
            "test_events": [
                event.event_id for event in EVENTS if event.split == "test"
            ],
            "train_stations": [
                station.code for station in STATIONS if station.split == "train"
            ],
            "test_stations": [
                station.code for station in STATIONS if station.split == "test"
            ],
            "train_station_event_pairs": 12,
            "test_station_event_pairs": 12,
            "components_per_trace": 3,
            "samples_per_component": 64,
            "padded_dimension": 256,
            "event_window_offset_seconds": 900,
            "quiet_window_offset_seconds": -85500,
            "window_duration_seconds": 240,
            "evaluation_log10_relative_ridge": EVALUATION_PARAMETER,
        },
        "data": {
            "provider": "EarthScope FDSN, network IU",
            "catalog": "USGS ComCat event identifiers",
            "waveforms_redistributed": False,
            "request_count": len(data_provenance),
            "total_download_bytes": sum(row["bytes"] for row in data_provenance),
            "records": data_provenance,
        },
        "algorithm": {
            "component_grid": list(COMPONENT_GRID),
            "certificate_relative_mse_tolerance": CERTIFICATE_TOLERANCE,
            "familywise_failure_probability": FAMILYWISE_FAILURE_PROBABILITY,
            "familywise_hypotheses": familywise_hypotheses,
            "certified_tasks": TRAINING_PARAMETERS.tolist(),
            "certificate_probes": 2048,
            "complementary_rank_relative_tolerance": COMPLEMENTARY_RANK_TOLERANCE,
            "structural_probe_audit": "diagnostic only; not a deployment gate",
            "no_candidate_policy": "reject; do not substitute an entry-built fallback",
            "entry_based_control": "HODLR leaf_size=32 off_diagonal_rank=8",
            "parameter_knots": list(PARAMETER_KNOTS),
            "training_parameters": TRAINING_PARAMETERS.tolist(),
            "steps_per_model": steps,
            "rank_runs": rank_runs,
            "selected": selected_summary,
        },
        "complementary_rank": {
            "duration_and_dimension": duration_audit,
            "ridge_sweep_at_n256": ridge_audit,
        },
        "methods": fixed_methods,
        "approximation": approximation,
        "timing": {
            "primary_n256": primary_timing,
            "kernel_scaling": _kernel_scaling(),
            "timing_protocol": "3 warm-ups; median and interquartile range",
        },
        "provenance": {
            **_system_provenance(),
            "end_to_end_seconds": time.perf_counter() - total_start,
            "process_max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "scope": (
            "Retrospective M>=8 waveform study. The labels are catalog-window "
            "labels, not analyst phase picks; the amplitude control shows that "
            "this benchmark is not an operational detection claim."
        ),
    }
    return result


def canonical_bytes(result: dict[str, object]) -> bytes:
    return (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()


def _verify(result: dict[str, object], locked: dict[str, object]) -> None:
    """Check every non-timing result while allowing platform timing/BLAS drift."""

    volatile_keys = {
        "end_to_end_seconds",
        "process_max_rss_raw",
        "training_seconds",
        "median_ms",
        "q25_ms",
        "q75_ms",
        "dense_covariance_eigendecomposition_seconds",
        "hodlr_construction_seconds",
        "equal_memory_hodlr_two_knot_construction_seconds",
    }

    def without_volatile(value, *, parent: str = ""):
        if parent == "provenance":
            return None
        if isinstance(value, dict):
            return {
                key: without_volatile(item, parent=key)
                for key, item in value.items()
                if key not in volatile_keys
            }
        if isinstance(value, list):
            return [without_volatile(item, parent=parent) for item in value]
        return value

    def compare(current, reference, path: str = "result") -> None:
        if isinstance(reference, dict):
            if set(current) != set(reference):
                raise RuntimeError(f"{path} keys changed")
            for key in reference:
                compare(current[key], reference[key], f"{path}.{key}")
            return
        if isinstance(reference, list):
            if len(current) != len(reference):
                raise RuntimeError(f"{path} length changed")
            for index, (current_item, reference_item) in enumerate(
                zip(current, reference)
            ):
                compare(current_item, reference_item, f"{path}[{index}]")
            return
        if isinstance(reference, float):
            if not np.isclose(current, reference, rtol=1.0e-6, atol=1.0e-9):
                raise RuntimeError(
                    f"{path} changed: locked {reference}, recomputed {current}"
                )
            return
        if current != reference:
            raise RuntimeError(
                f"{path} changed: locked {reference!r}, recomputed {current!r}"
            )

    current_contract = without_volatile(result)
    locked_contract = without_volatile(locked)
    compare(current_contract, locked_contract)

    for run in result["algorithm"]["selected"]["per_seed"]:
        if run["selected_kind"] == "butterfly" and (
            run["worst_task_certificate"]["relative_frobenius_mse_upper_bound"]
            > CERTIFICATE_TOLERANCE
        ):
            raise RuntimeError("a selected model exceeds the certificate tolerance")
    expected_hypotheses = (
        len(DEFAULT_SEEDS) * len(COMPONENT_GRID) * len(TRAINING_PARAMETERS)
    )
    if result["algorithm"]["familywise_hypotheses"] != expected_hypotheses:
        raise RuntimeError("the family-wise certificate does not cover the fixed grid")
    certificates = [
        certificate
        for run in result["algorithm"]["rank_runs"]
        for certificate in run["certificates_by_task"]
    ]
    if len(certificates) != expected_hypotheses:
        raise RuntimeError("the number of certificates differs from the fixed family")
    if len({certificate["seed"] for certificate in certificates}) != len(
        certificates
    ):
        raise RuntimeError("certification streams are not unique across the fixed family")
    global_delta = result["algorithm"]["familywise_failure_probability"]
    expected_tail_budget = global_delta / (2.0 * expected_hypotheses)
    for certificate in certificates:
        if certificate["familywise_hypotheses"] != expected_hypotheses:
            raise RuntimeError("a certificate uses the wrong family size")
        if not np.isclose(
            certificate["familywise_failure_probability"], global_delta
        ):
            raise RuntimeError("a certificate uses the wrong global failure budget")
        if not np.isclose(
            certificate["per_tail_failure_probability"], expected_tail_budget
        ):
            raise RuntimeError("a certificate uses the wrong tail allocation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce certificate-gated whitening on station/event-disjoint "
            "EarthScope waveforms."
        )
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "waveform cache directory; defaults to "
            "~/.cache/taskbutterfly/earthscope-v1"
        ),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(
        steps=args.steps,
        cache_dir=args.cache_dir,
        offline=args.offline,
    )
    if args.verify:
        locked_path = Path(__file__).with_name("results").joinpath(
            "seismic_whitening_results.json"
        )
        _verify(result, json.loads(locked_path.read_text()))
        print("verification=passed", file=sys.stderr)
    rendered = canonical_bytes(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(rendered)
    sys.stdout.buffer.write(rendered)
    print(f"sha256={hashlib.sha256(rendered).hexdigest()}", file=sys.stderr)


if __name__ == "__main__":
    main()
