"""NumPy optimization for products of sparse butterfly stages."""

from __future__ import annotations

import numpy as np

from .operator import ButterflyOperator, _pair_indices, identity_factors


def _forward_with_cache(factors: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    operator = ButterflyOperator(factors)
    work = np.asarray(x, dtype=float)
    cache = [work]
    for stage in range(operator.levels):
        left_index, right_index = _pair_indices(operator.n, stage)
        pair = factors[stage]
        output = np.empty_like(work)
        output[left_index] = (
            pair[:, 0, 0, None] * work[left_index]
            + pair[:, 0, 1, None] * work[right_index]
        )
        output[right_index] = (
            pair[:, 1, 0, None] * work[left_index]
            + pair[:, 1, 1, None] * work[right_index]
        )
        work = output
        cache.append(work)
    return work, cache


def butterfly_loss_gradient(
    factors: np.ndarray, x: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray]:
    """Relative squared action loss and exact factor gradient."""

    prediction, cache = _forward_with_cache(factors, x)
    denominator = max(float(np.sum(target**2)), np.finfo(float).eps)
    error = prediction - target
    loss = float(np.sum(error**2) / denominator)
    adjoint = 2.0 * error / denominator
    gradient = np.zeros_like(factors)
    n = x.shape[0]
    for stage in reversed(range(factors.shape[0])):
        left_index, right_index = _pair_indices(n, stage)
        stage_input = cache[stage]
        dl = adjoint[left_index]
        dr = adjoint[right_index]
        xl = stage_input[left_index]
        xr = stage_input[right_index]
        gradient[stage, :, 0, 0] = np.sum(dl * xl, axis=1)
        gradient[stage, :, 0, 1] = np.sum(dl * xr, axis=1)
        gradient[stage, :, 1, 0] = np.sum(dr * xl, axis=1)
        gradient[stage, :, 1, 1] = np.sum(dr * xr, axis=1)
        pair = factors[stage]
        previous = np.empty_like(adjoint)
        previous[left_index] = (
            pair[:, 0, 0, None] * dl + pair[:, 1, 0, None] * dr
        )
        previous[right_index] = (
            pair[:, 0, 1, None] * dl + pair[:, 1, 1, None] * dr
        )
        adjoint = previous
    return loss, gradient


def _factor_gradient_from_adjoint(
    factors: np.ndarray,
    cache: list[np.ndarray],
    output_adjoint: np.ndarray,
) -> np.ndarray:
    """Reverse one butterfly product for a supplied output adjoint."""

    adjoint = np.asarray(output_adjoint)
    gradient = np.zeros_like(factors)
    n = adjoint.shape[0]
    for stage in reversed(range(factors.shape[0])):
        left_index, right_index = _pair_indices(n, stage)
        stage_input = cache[stage]
        dl = adjoint[left_index]
        dr = adjoint[right_index]
        xl = stage_input[left_index]
        xr = stage_input[right_index]
        gradient[stage, :, 0, 0] = np.sum(dl * xl, axis=1)
        gradient[stage, :, 0, 1] = np.sum(dl * xr, axis=1)
        gradient[stage, :, 1, 0] = np.sum(dr * xl, axis=1)
        gradient[stage, :, 1, 1] = np.sum(dr * xr, axis=1)
        pair = factors[stage]
        previous = np.empty_like(adjoint)
        previous[left_index] = (
            pair[:, 0, 0, None] * dl + pair[:, 1, 0, None] * dr
        )
        previous[right_index] = (
            pair[:, 0, 1, None] * dl + pair[:, 1, 1, None] * dr
        )
        adjoint = previous
    return gradient


def ensemble_loss_gradient(
    factor_components: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Relative action loss and exact gradient for a sum of butterflies."""

    predictions = []
    caches = []
    for factors in factor_components:
        prediction, cache = _forward_with_cache(factors, x)
        predictions.append(prediction)
        caches.append(cache)
    total = np.sum(predictions, axis=0)
    denominator = max(float(np.sum(target**2)), np.finfo(float).eps)
    error = total - target
    loss = float(np.sum(error**2) / denominator)
    output_adjoint = 2.0 * error / denominator
    gradients = [
        _factor_gradient_from_adjoint(factors, cache, output_adjoint)
        for factors, cache in zip(factor_components, caches)
    ]
    return loss, np.stack(gradients)


def _ensemble_initial_factors(
    n: int,
    *,
    components: int,
    knots: int,
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    base = np.stack([identity_factors(n) for _ in range(components)])
    base[:, 0] /= components
    factors = np.stack([base.copy() for _ in range(knots)])
    factors += noise_scale * rng.standard_normal(factors.shape) / np.sqrt(n)
    return factors


def train_parametric_ensemble(
    inputs: np.ndarray,
    target_function,
    *,
    parameter_knots: tuple[float, float],
    training_parameters: np.ndarray,
    components: int = 2,
    steps: int = 800,
    batch_size: int = 32,
    seed: int = 0,
    learning_rate: float = 8.0e-3,
    history_interval: int = 50,
    initialization_noise: float = 2.0e-2,
) -> tuple[np.ndarray, dict[str, object]]:
    """Train a certificate-ready additive parametric butterfly.

    Target actions on the fixed input pool are precomputed at a small,
    preregistered parameter grid.  Optimization never requests matrix entries.
    """

    values = np.asarray(inputs, dtype=float)
    parameters = np.asarray(training_parameters, dtype=float)
    if values.ndim != 2:
        raise ValueError("inputs must have shape (n, samples)")
    lower, upper = map(float, parameter_knots)
    if not lower < upper:
        raise ValueError("parameter knots must increase")
    if parameters.ndim != 1 or np.any(parameters < lower) or np.any(parameters > upper):
        raise ValueError("training_parameters must lie inside the knot interval")
    if components < 1:
        raise ValueError("components must be positive")
    if history_interval < 1:
        raise ValueError("history_interval must be positive")

    targets = np.stack(
        [np.asarray(target_function(values, float(parameter))) for parameter in parameters]
    )
    rng = np.random.default_rng(seed)
    factors = _ensemble_initial_factors(
        values.shape[0],
        components=components,
        knots=2,
        rng=rng,
        noise_scale=initialization_noise,
    )
    first_moment = np.zeros_like(factors)
    second_moment = np.zeros_like(factors)
    history: list[dict[str, float | int]] = []
    initial_loss = 0.0
    final_loss = 0.0
    for iteration in range(1, steps + 1):
        parameter_index = int(rng.integers(0, len(parameters)))
        parameter = float(parameters[parameter_index])
        indices = rng.integers(
            0, values.shape[1], size=min(batch_size, values.shape[1])
        )
        batch = values[:, indices]
        target = targets[parameter_index][:, indices]
        weight = (parameter - lower) / (upper - lower)
        interpolated = (1.0 - weight) * factors[0] + weight * factors[1]
        loss, local_gradient = ensemble_loss_gradient(interpolated, batch, target)
        gradient = np.stack(
            [(1.0 - weight) * local_gradient, weight * local_gradient]
        )
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = first_moment / (1.0 - 0.9**iteration)
        corrected_second = second_moment / (1.0 - 0.999**iteration)
        factors -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + 1.0e-8
        )
        if iteration == 1:
            initial_loss = loss
        final_loss = loss
        if iteration == 1 or iteration % history_interval == 0 or iteration == steps:
            history.append(
                {
                    "iteration": iteration,
                    "parameter": parameter,
                    "minibatch_relative_mse": float(loss),
                }
            )
    return factors, {
        "steps": steps,
        "components": components,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "parameter_knots": [lower, upper],
        "training_parameters": parameters.tolist(),
        "initial_minibatch_relative_mse": float(initial_loss),
        "final_minibatch_relative_mse": float(final_loss),
        "history": history,
    }


def train_parametric_butterfly(
    inputs: np.ndarray,
    target_function,
    *,
    frequency_knots: tuple[float, float],
    steps: int = 600,
    batch_size: int = 24,
    seed: int = 7,
    learning_rate: float = 1.5e-2,
    history_interval: int = 50,
) -> tuple[np.ndarray, dict[str, object]]:
    """Train two factor knots using continuum action supervision."""

    values = np.asarray(inputs, dtype=float)
    if values.ndim != 2:
        raise ValueError("inputs must have shape (n, samples)")
    lower, upper = map(float, frequency_knots)
    if not lower < upper:
        raise ValueError("frequency knots must increase")
    if history_interval < 1:
        raise ValueError("history_interval must be positive")
    rng = np.random.default_rng(seed)
    factors = np.stack(
        [identity_factors(values.shape[0]), identity_factors(values.shape[0])]
    )
    first_moment = np.zeros_like(factors)
    second_moment = np.zeros_like(factors)
    initial_loss = None
    final_loss = None
    history: list[dict[str, float | int]] = []
    for iteration in range(1, steps + 1):
        indices = rng.integers(0, values.shape[1], size=min(batch_size, values.shape[1]))
        batch = values[:, indices]
        cutoff = rng.uniform(lower, upper)
        weight = (cutoff - lower) / (upper - lower)
        interpolated = (1.0 - weight) * factors[0] + weight * factors[1]
        target = target_function(batch, cutoff)
        loss, local_gradient = butterfly_loss_gradient(interpolated, batch, target)
        gradient = np.stack(
            [(1.0 - weight) * local_gradient, weight * local_gradient]
        )
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = first_moment / (1.0 - 0.9**iteration)
        corrected_second = second_moment / (1.0 - 0.999**iteration)
        factors -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + 1.0e-8
        )
        if iteration == 1:
            initial_loss = loss
        final_loss = loss
        if (
            iteration == 1
            or iteration % history_interval == 0
            or iteration == steps
        ):
            history.append(
                {
                    "iteration": iteration,
                    "minibatch_relative_mse": float(loss),
                }
            )
    return factors, {
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "initial_minibatch_relative_mse": float(initial_loss or 0.0),
        "final_minibatch_relative_mse": float(final_loss or 0.0),
        "history": history,
    }
