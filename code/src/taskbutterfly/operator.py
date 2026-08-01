"""Transparent NumPy implementation of matrix-free butterfly products."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import log2
from typing import Iterable

import numpy as np


def _require_power_of_two(n: int) -> int:
    if n < 2 or n & (n - 1):
        raise ValueError(f"n must be a power of two and at least 2; received {n}")
    return int(log2(n))


@lru_cache(maxsize=None)
def _pair_indices(n: int, stage: int) -> tuple[np.ndarray, np.ndarray]:
    stride = 1 << stage
    block = 2 * stride
    left: list[int] = []
    right: list[int] = []
    for base in range(0, n, block):
        for offset in range(stride):
            left.append(base + offset)
            right.append(base + offset + stride)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def identity_factors(n: int, dtype: np.dtype = np.float64) -> np.ndarray:
    levels = _require_power_of_two(n)
    factors = np.zeros((levels, n // 2, 2, 2), dtype=dtype)
    factors[..., 0, 0] = 1
    factors[..., 1, 1] = 1
    return factors


def random_near_identity_factors(
    n: int,
    *,
    seed: int = 0,
    scale: float = 0.1,
    complex_valued: bool = True,
) -> np.ndarray:
    """Create well-conditioned factors for tests and examples."""
    rng = np.random.default_rng(seed)
    dtype = np.complex128 if complex_valued else np.float64
    factors = identity_factors(n, dtype=dtype)
    noise = rng.standard_normal(factors.shape)
    if complex_valued:
        noise = noise + 1j * rng.standard_normal(factors.shape)
    return factors + scale * noise / np.sqrt(2.0 if complex_valued else 1.0)


@dataclass(frozen=True)
class ButterflyOperator:
    """Product of sparse two-by-two butterfly stages.

    The dense operator is never needed by :meth:`apply` or :meth:`adjoint`.
    Inputs may be a vector of shape ``(n,)`` or a block of right-hand sides of
    shape ``(n, b)``.
    """

    factors: np.ndarray

    def __post_init__(self) -> None:
        factors = np.asarray(self.factors)
        if factors.ndim != 4 or factors.shape[-2:] != (2, 2):
            raise ValueError("factors must have shape (levels, n/2, 2, 2)")
        n = 2 * factors.shape[1]
        levels = _require_power_of_two(n)
        if factors.shape[0] != levels:
            raise ValueError(
                f"expected {levels} levels for n={n}; received {factors.shape[0]}"
            )
        object.__setattr__(self, "factors", factors)

    @classmethod
    def from_seed(
        cls,
        n: int,
        *,
        seed: int = 0,
        scale: float = 0.1,
        complex_valued: bool = True,
    ) -> "ButterflyOperator":
        return cls(
            random_near_identity_factors(
                n,
                seed=seed,
                scale=scale,
                complex_valued=complex_valued,
            )
        )

    @property
    def n(self) -> int:
        return 2 * self.factors.shape[1]

    @property
    def levels(self) -> int:
        return self.factors.shape[0]

    @property
    def parameter_count(self) -> int:
        return int(self.factors.size)

    def _prepare(self, x: np.ndarray) -> tuple[np.ndarray, bool]:
        array = np.asarray(x)
        was_vector = array.ndim == 1
        if was_vector:
            array = array[:, None]
        if array.ndim != 2 or array.shape[0] != self.n:
            raise ValueError(f"x must have shape ({self.n},) or ({self.n}, b)")
        dtype = np.result_type(array.dtype, self.factors.dtype)
        return np.asarray(array, dtype=dtype), was_vector

    def apply(self, x: np.ndarray) -> np.ndarray:
        work, was_vector = self._prepare(x)
        for stage in range(self.levels):
            left, right = _pair_indices(self.n, stage)
            pair = self.factors[stage]
            x_left = work[left]
            x_right = work[right]
            out = np.empty_like(work)
            out[left] = pair[:, 0, 0, None] * x_left + pair[:, 0, 1, None] * x_right
            out[right] = pair[:, 1, 0, None] * x_left + pair[:, 1, 1, None] * x_right
            work = out
        return work[:, 0] if was_vector else work

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        work, was_vector = self._prepare(y)
        for stage in reversed(range(self.levels)):
            left, right = _pair_indices(self.n, stage)
            pair = self.factors[stage].conj()
            y_left = work[left]
            y_right = work[right]
            out = np.empty_like(work)
            out[left] = pair[:, 0, 0, None] * y_left + pair[:, 1, 0, None] * y_right
            out[right] = pair[:, 0, 1, None] * y_left + pair[:, 1, 1, None] * y_right
            work = out
        return work[:, 0] if was_vector else work

    def to_dense(self) -> np.ndarray:
        return self.apply(np.eye(self.n, dtype=self.factors.dtype))


class ParametricButterflyOperator:
    """Piecewise-linear interpolation of butterfly cores across frequency."""

    def __init__(self, frequency_knots: Iterable[float], factor_knots: np.ndarray):
        knots = np.asarray(list(frequency_knots), dtype=float)
        factors = np.asarray(factor_knots)
        if knots.ndim != 1 or len(knots) < 2 or np.any(np.diff(knots) <= 0):
            raise ValueError("frequency_knots must be a strictly increasing vector")
        if factors.ndim != 5 or factors.shape[0] != len(knots):
            raise ValueError(
                "factor_knots must have shape (knots, levels, n/2, 2, 2)"
            )
        ButterflyOperator(factors[0])
        self.frequency_knots = knots
        self.factor_knots = factors

    @classmethod
    def from_seed(
        cls,
        n: int,
        frequency_knots: Iterable[float],
        *,
        seed: int = 0,
        scale: float = 0.1,
        complex_valued: bool = True,
    ) -> "ParametricButterflyOperator":
        knots = np.asarray(list(frequency_knots), dtype=float)
        factors = np.stack(
            [
                random_near_identity_factors(
                    n,
                    seed=seed + index,
                    scale=scale,
                    complex_valued=complex_valued,
                )
                for index in range(len(knots))
            ]
        )
        return cls(knots, factors)

    @property
    def n(self) -> int:
        return 2 * self.factor_knots.shape[2]

    def factors_at(self, frequency: float) -> np.ndarray:
        f = float(frequency)
        if f <= self.frequency_knots[0]:
            return self.factor_knots[0]
        if f >= self.frequency_knots[-1]:
            return self.factor_knots[-1]
        upper = int(np.searchsorted(self.frequency_knots, f, side="right"))
        lower = upper - 1
        span = self.frequency_knots[upper] - self.frequency_knots[lower]
        weight = (f - self.frequency_knots[lower]) / span
        return (1.0 - weight) * self.factor_knots[lower] + weight * self.factor_knots[upper]

    def operator_at(self, frequency: float) -> ButterflyOperator:
        return ButterflyOperator(self.factors_at(frequency))

    def apply(self, frequency: float, x: np.ndarray) -> np.ndarray:
        return self.operator_at(frequency).apply(x)

    def adjoint(self, frequency: float, y: np.ndarray) -> np.ndarray:
        return self.operator_at(frequency).adjoint(y)


@dataclass(frozen=True)
class ButterflyEnsembleOperator:
    """Sum of butterfly products used for rank-adaptive approximation."""

    factor_components: np.ndarray

    def __post_init__(self) -> None:
        factors = np.asarray(self.factor_components)
        if factors.ndim != 5 or factors.shape[0] < 1:
            raise ValueError(
                "factor_components must have shape "
                "(components, levels, n/2, 2, 2)"
            )
        for component in factors:
            ButterflyOperator(component)
        object.__setattr__(self, "factor_components", factors)

    @property
    def n(self) -> int:
        return 2 * self.factor_components.shape[2]

    @property
    def components(self) -> int:
        return self.factor_components.shape[0]

    @property
    def parameter_count(self) -> int:
        return int(self.factor_components.size)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return sum(ButterflyOperator(factors).apply(x) for factors in self.factor_components)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return sum(
            ButterflyOperator(factors).adjoint(y)
            for factors in self.factor_components
        )

    def to_dense(self) -> np.ndarray:
        return self.apply(np.eye(self.n, dtype=self.factor_components.dtype))


class ParametricButterflyEnsembleOperator:
    """Affine factor interpolation for a sum of butterfly products."""

    def __init__(self, parameter_knots: Iterable[float], factor_knots: np.ndarray):
        knots = np.asarray(list(parameter_knots), dtype=float)
        factors = np.asarray(factor_knots)
        if knots.ndim != 1 or len(knots) < 2 or np.any(np.diff(knots) <= 0):
            raise ValueError("parameter_knots must be a strictly increasing vector")
        if factors.ndim != 6 or factors.shape[0] != len(knots):
            raise ValueError(
                "factor_knots must have shape "
                "(knots, components, levels, n/2, 2, 2)"
            )
        ButterflyEnsembleOperator(factors[0])
        self.parameter_knots = knots
        self.factor_knots = factors

    @property
    def n(self) -> int:
        return 2 * self.factor_knots.shape[3]

    @property
    def components(self) -> int:
        return self.factor_knots.shape[1]

    @property
    def parameter_count(self) -> int:
        return int(self.factor_knots.size)

    def factors_at(self, parameter: float) -> np.ndarray:
        value = float(parameter)
        if value <= self.parameter_knots[0]:
            return self.factor_knots[0]
        if value >= self.parameter_knots[-1]:
            return self.factor_knots[-1]
        upper = int(np.searchsorted(self.parameter_knots, value, side="right"))
        lower = upper - 1
        span = self.parameter_knots[upper] - self.parameter_knots[lower]
        weight = (value - self.parameter_knots[lower]) / span
        return (1.0 - weight) * self.factor_knots[lower] + weight * self.factor_knots[
            upper
        ]

    def operator_at(self, parameter: float) -> ButterflyEnsembleOperator:
        return ButterflyEnsembleOperator(self.factors_at(parameter))

    def apply(self, parameter: float, x: np.ndarray) -> np.ndarray:
        return self.operator_at(parameter).apply(x)

    def adjoint(self, parameter: float, y: np.ndarray) -> np.ndarray:
        return self.operator_at(parameter).adjoint(y)
