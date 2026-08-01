import numpy as np

from taskbutterfly import ButterflyOperator, ParametricButterflyOperator


def test_apply_matches_dense() -> None:
    operator = ButterflyOperator.from_seed(32, seed=2, scale=0.05)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((32, 5)) + 1j * rng.standard_normal((32, 5))
    np.testing.assert_allclose(operator.apply(x), operator.to_dense() @ x, rtol=1e-12)


def test_adjoint_identity() -> None:
    operator = ButterflyOperator.from_seed(64, seed=4, scale=0.07)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((64, 3)) + 1j * rng.standard_normal((64, 3))
    y = rng.standard_normal((64, 3)) + 1j * rng.standard_normal((64, 3))
    np.testing.assert_allclose(
        np.vdot(operator.apply(x), y),
        np.vdot(x, operator.adjoint(y)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_parametric_endpoints_and_midpoint() -> None:
    operator = ParametricButterflyOperator.from_seed(
        16, [2.0, 10.0], seed=10, complex_valued=False
    )
    np.testing.assert_allclose(operator.factors_at(1.0), operator.factor_knots[0])
    np.testing.assert_allclose(operator.factors_at(12.0), operator.factor_knots[-1])
    np.testing.assert_allclose(
        operator.factors_at(6.0),
        0.5 * (operator.factor_knots[0] + operator.factor_knots[1]),
    )

