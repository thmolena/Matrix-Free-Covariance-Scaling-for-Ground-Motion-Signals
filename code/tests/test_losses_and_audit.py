import numpy as np
import pytest

from taskbutterfly import (
    ButterflyOperator,
    certificate_gated_action_loss,
    complementary_action_probe_audit,
    complementary_rank_audit,
    gaussian_frobenius_certificate,
)


def test_probe_certificate_is_zero_for_same_operator() -> None:
    operator = ButterflyOperator.from_seed(32, seed=12)
    certificate = gaussian_frobenius_certificate(
        operator.apply,
        operator.apply,
        n=32,
        probes=128,
        failure_probability=0.05,
        seed=13,
    )
    assert certificate["relative_frobenius_mse_upper_bound"] < 1e-28


def test_probe_certificate_allocates_the_declared_familywise_budget() -> None:
    matrix = np.diag(np.linspace(1.0, 2.0, 16))
    certificate = gaussian_frobenius_certificate(
        lambda block: 0.9 * matrix @ block,
        lambda block: matrix @ block,
        n=16,
        probes=2048,
        failure_probability=0.05,
        familywise_hypotheses=75,
        seed=14,
    )
    assert certificate["familywise_hypotheses"] == 75
    assert np.isclose(
        certificate["per_hypothesis_failure_probability"], 0.05 / 75
    )
    assert np.isclose(certificate["per_tail_failure_probability"], 0.05 / 150)
    assert certificate["relative_frobenius_mse_upper_bound"] >= 0.01


def test_probe_certificate_rejects_a_zero_reference_action() -> None:
    with pytest.raises(ValueError, match="zero sampled Frobenius norm"):
        gaussian_frobenius_certificate(
            lambda block: block,
            lambda block: np.zeros_like(block),
            n=8,
            probes=64,
            seed=15,
        )


def test_complementary_rank_audit_shape() -> None:
    operator = ButterflyOperator.from_seed(32, seed=15)
    audit = complementary_rank_audit(
        operator.to_dense(), relative_tolerance=1e-8, max_blocks_per_level=8
    )
    assert len(audit) == 6
    assert [entry["level"] for entry in audit] == list(range(6))
    assert all(entry["sampled_blocks"] <= 8 for entry in audit)


def test_certificate_gated_action_loss_requires_every_fixed_task() -> None:
    accepted = certificate_gated_action_loss(
        0.08,
        certificate_upper_bounds=[0.11, 0.13, 0.12],
        certificate_tolerance=0.15,
    )
    rejected_error = certificate_gated_action_loss(
        0.08,
        certificate_upper_bounds=[0.13, 0.16, 0.12],
        certificate_tolerance=0.15,
    )
    assert accepted == 0.08
    assert np.isinf(rejected_error)


def test_action_probe_audit_never_needs_dense_materialization() -> None:
    rng = np.random.default_rng(19)
    left = rng.standard_normal((32, 2))
    right = rng.standard_normal((32, 2))
    audit = complementary_action_probe_audit(
        lambda block: left @ (right.T @ block),
        lambda block: right @ (left.T @ block),
        n=32,
        relative_tolerance=1.0e-10,
        max_blocks_per_level=4,
        range_probes=4,
        residual_probes=3,
        seed=20,
    )
    assert len(audit) == 6
    assert all(entry["maximum_tested_rank"] <= 2 for entry in audit)
    assert all(entry["saturated_blocks"] == 0 for entry in audit)
    assert all(entry["diagnostic_only"] for entry in audit)
