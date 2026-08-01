import numpy as np

from taskbutterfly.operator import ButterflyOperator
from taskbutterfly.training import butterfly_loss_gradient, train_parametric_butterfly


def test_analytic_factor_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(4)
    factors = ButterflyOperator.from_seed(
        8, seed=2, scale=0.03, complex_valued=False
    ).factors
    inputs = rng.normal(size=(8, 3))
    target = rng.normal(size=(8, 3))
    _, gradient = butterfly_loss_gradient(factors, inputs, target)
    index = (1, 2, 0, 1)
    epsilon = 1e-6
    plus = factors.copy()
    minus = factors.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    loss_plus, _ = butterfly_loss_gradient(plus, inputs, target)
    loss_minus, _ = butterfly_loss_gradient(minus, inputs, target)
    np.testing.assert_allclose(
        gradient[index], (loss_plus - loss_minus) / (2 * epsilon), rtol=2e-6
    )


def test_numpy_training_reduces_action_loss() -> None:
    rng = np.random.default_rng(21)
    inputs = rng.normal(size=(16, 40))

    def target(block, cutoff):
        return (0.6 + cutoff) * block

    factors, result = train_parametric_butterfly(
        inputs,
        target,
        frequency_knots=(0.1, 0.3),
        steps=120,
        batch_size=10,
        seed=21,
        learning_rate=0.02,
    )
    assert factors.shape == (2, 4, 8, 2, 2)
    assert result["final_minibatch_relative_mse"] < result["initial_minibatch_relative_mse"]
    assert result["history"][0]["iteration"] == 1
    assert result["history"][-1]["iteration"] == 120
