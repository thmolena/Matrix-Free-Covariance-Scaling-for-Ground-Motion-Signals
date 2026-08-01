import numpy as np

from taskbutterfly.streaming_energy import (
    StreamingEnergy,
    balanced_accuracy,
    best_balanced_threshold,
    centered_block_energies,
    log_reference_ratio,
    log_ratio_perturbation_bound,
)


def test_streaming_statistic_matches_batch_and_is_gain_invariant() -> None:
    rng = np.random.default_rng(260801)
    candidate = rng.standard_normal((3, 8 * 64))
    reference = rng.standard_normal((3, 11 * 64))
    candidate_energy = centered_block_energies(candidate)
    reference_energy = centered_block_energies(reference)
    expected = log_reference_ratio(
        candidate_energy, reference_energy, prefix_blocks=8
    )
    stream = StreamingEnergy(float(np.mean(reference_energy)))
    observed = np.nan
    for index in range(8):
        observed = stream.update(candidate[:, 64 * index : 64 * (index + 1)])
    np.testing.assert_allclose(observed, expected, atol=2e-15)
    np.testing.assert_allclose(
        log_reference_ratio(
            centered_block_energies(7.25 * candidate),
            centered_block_energies(7.25 * reference),
            prefix_blocks=8,
        ),
        expected,
        atol=2e-15,
    )


def test_log_ratio_perturbation_bound_is_sharp() -> None:
    delta = 0.07
    candidate = np.array([2.0, 3.0])
    reference = np.array([1.0, 4.0])
    exact = log_reference_ratio(candidate, reference, prefix_blocks=2)
    perturbed = log_reference_ratio(
        (1.0 + delta) * candidate,
        (1.0 - delta) * reference,
        prefix_blocks=2,
    )
    np.testing.assert_allclose(
        perturbed - exact, log_ratio_perturbation_bound(delta), atol=1e-15
    )


def test_threshold_tie_break_and_metrics() -> None:
    positive = np.array([2.0, 3.0, 4.0])
    negative = np.array([-1.0, 0.0, 1.0])
    threshold, training = best_balanced_threshold(positive, negative)
    assert training == 1.0
    observed, sensitivity, specificity = balanced_accuracy(
        positive, negative, threshold
    )
    assert observed == sensitivity == specificity == 1.0
