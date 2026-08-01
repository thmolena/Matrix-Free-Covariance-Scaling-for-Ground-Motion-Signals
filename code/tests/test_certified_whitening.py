import numpy as np

from taskbutterfly.operator import (
    ButterflyEnsembleOperator,
    ParametricButterflyEnsembleOperator,
    identity_factors,
)
from taskbutterfly.seismic_dataset import (
    EXPECTED_SHA256,
    parse_geocsv,
    protocol_records,
    resample_three_components,
)
from taskbutterfly.training import ensemble_loss_gradient
from taskbutterfly.whitening import (
    HODLROperator,
    SpectralLowRankOperator,
    WhiteningFamily,
    energy_score_perturbation_bound,
    empirical_covariance,
    operator_energy_score_error_bound,
    roc_auc,
    tie_aware_auc_perturbation_bound,
)


def test_frozen_protocol_is_station_and_event_disjoint() -> None:
    records = protocol_records()
    assert len(records) == 48
    assert len(EXPECTED_SHA256) == 48
    train_events = {record["event"].event_id for record in records if record["event"].split == "train"}
    test_events = {record["event"].event_id for record in records if record["event"].split == "test"}
    train_stations = {record["station"].code for record in records if record["event"].split == "train"}
    test_stations = {record["station"].code for record in records if record["event"].split == "test"}
    assert train_events.isdisjoint(test_events)
    assert train_stations.isdisjoint(test_stations)
    assert all(record["expected_sha256"] for record in records)


def test_segmented_geocsv_parser_and_resampling() -> None:
    rows = ["# dataset: GeoCSV 2.0"]
    for channel, offset in (("BH1", 0.0), ("BH2", 1.0), ("BHZ", 2.0)):
        rows.extend(
            [
                "# dataset: GeoCSV 2.0",
                f"# SID: IU_TEST_00_{channel}",
                "# sample_rate_hz: 20",
                "# instrument: synthetic",
                "# scale_units: m/s",
                "Time, Sample",
            ]
        )
        for index in range(8):
            rows.append(
                f"2020-01-01T00:00:00.{index * 50:03d}Z, {offset + index}"
            )
    streams = parse_geocsv(("\n".join(rows) + "\n").encode())
    components, metadata = resample_three_components(
        streams, maximum_gap_seconds=0.1, minimum_common_samples=7
    )
    assert components.shape == (3, 7)
    assert metadata["selected_sids"] == [
        "IU_TEST_00_BH1",
        "IU_TEST_00_BH2",
        "IU_TEST_00_BHZ",
    ]


def test_ensemble_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(22)
    factors = np.stack([identity_factors(8), identity_factors(8)])
    factors += 0.01 * rng.standard_normal(factors.shape)
    inputs = rng.standard_normal((8, 4))
    target = rng.standard_normal((8, 4))
    _, gradient = ensemble_loss_gradient(factors, inputs, target)
    index = (1, 1, 2, 0, 1)
    epsilon = 1.0e-6
    plus = factors.copy()
    minus = factors.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    loss_plus, _ = ensemble_loss_gradient(plus, inputs, target)
    loss_minus, _ = ensemble_loss_gradient(minus, inputs, target)
    np.testing.assert_allclose(
        gradient[index],
        (loss_plus - loss_minus) / (2.0 * epsilon),
        rtol=3.0e-6,
    )


def test_parametric_ensemble_and_hodlr_actions() -> None:
    knots = np.stack(
        [
            np.stack([identity_factors(16), identity_factors(16)]),
            np.stack([2.0 * identity_factors(16), identity_factors(16)]),
        ]
    )
    model = ParametricButterflyEnsembleOperator((0.0, 1.0), knots)
    block = np.arange(48, dtype=float).reshape(16, 3)
    np.testing.assert_allclose(
        model.apply(0.0, block),
        ButterflyEnsembleOperator(knots[0]).apply(block),
    )
    rng = np.random.default_rng(23)
    matrix = rng.standard_normal((32, 32))
    hodlr = HODLROperator.from_dense(matrix, leaf_size=8, off_diagonal_rank=32)
    np.testing.assert_allclose(hodlr.apply(block=np.eye(32)), matrix, atol=1e-12)
    np.testing.assert_allclose(hodlr.adjoint(np.eye(32)), matrix.T, atol=1e-12)


def test_spectral_low_rank_uses_its_factor_storage() -> None:
    diagonal = np.diag(np.arange(1.0, 9.0))
    spectral = SpectralLowRankOperator.from_dense(diagonal, rank=3)
    assert spectral.parameter_count == 27
    expected = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7.0, 8.0])
    np.testing.assert_allclose(spectral.to_dense(), expected, atol=1e-12)


def test_whitening_family_and_tie_aware_auc() -> None:
    rng = np.random.default_rng(24)
    samples = rng.standard_normal((16, 100))
    covariance = empirical_covariance(samples, physical_dimension=16)
    family = WhiteningFamily.from_covariance(covariance)
    matrix = family.matrix(-3.0)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    assert np.isclose(np.linalg.norm(matrix, ord="fro"), 4.0)
    block = rng.standard_normal((16, 5))
    np.testing.assert_allclose(
        family.apply(block, -3.0),
        matrix @ block,
        rtol=1e-13,
        atol=1e-13,
    )
    vector = block[:, 0]
    np.testing.assert_allclose(
        family.apply(vector, -3.0),
        matrix @ vector,
        rtol=1e-13,
        atol=1e-13,
    )
    assert roc_auc(np.array([0, 0, 1, 1]), np.ones(4)) == 0.5


def test_energy_quantile_score_bound_covers_observed_change() -> None:
    rng = np.random.default_rng(25)
    reference = rng.standard_normal((8, 11))
    approximate = reference + 0.03 * rng.standard_normal((8, 11))
    bound = energy_score_perturbation_bound(reference, approximate)
    assert bound["covered"]
    assert (
        bound["score_absolute_error"]
        <= bound["score_absolute_error_upper_bound"]
    )


def test_operator_norm_bound_covers_energy_quantile_score_change() -> None:
    rng = np.random.default_rng(26)
    reference_operator = rng.standard_normal((8, 8))
    error_operator = 0.02 * rng.standard_normal((8, 8))
    windows = rng.standard_normal((8, 13))
    reference = reference_operator @ windows
    approximate = (reference_operator + error_operator) @ windows
    upper = operator_energy_score_error_bound(
        operator_error_norm=float(np.linalg.norm(error_operator, ord=2)),
        reference_operator_norm=float(np.linalg.norm(reference_operator, ord=2)),
        maximum_input_norm=float(np.max(np.linalg.norm(windows, axis=0))),
    )
    reference_score = -float(np.quantile(np.sum(reference**2, axis=0), 0.90))
    approximate_score = -float(np.quantile(np.sum(approximate**2, axis=0), 0.90))
    assert abs(approximate_score - reference_score) <= upper


def test_tie_aware_auc_bound_counts_positive_margins_and_ties() -> None:
    labels = np.array([1, 1, 0, 0])
    reference = np.array([2.0, 1.0, 1.0, 0.0])
    approximate = np.array([1.8, 0.8, 1.2, 0.0])
    eta = np.full(4, 0.2)
    result = tie_aware_auc_perturbation_bound(
        labels, reference, approximate, eta
    )
    assert result["positive_negative_pairs"] == 4
    assert result["exact_reference_ties"] == 1
    assert result["reference_auc"] == 0.875
    assert result["approximate_auc"] >= result["auc_lower_bound"]
    assert abs(result["actual_auc_change"]) <= result["absolute_auc_change_bound"]
