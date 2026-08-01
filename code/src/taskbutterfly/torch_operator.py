"""Optional differentiable butterfly module implemented with PyTorch."""

from __future__ import annotations

from math import log2
from typing import Iterable

import torch
from torch import nn


def _require_power_of_two(n: int) -> int:
    levels = int(log2(n))
    if n < 2 or 2**levels != n:
        raise ValueError("n must be a power of two and at least 2")
    return levels


def _identity_knots(
    knot_count: int, n: int, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    levels = _require_power_of_two(n)
    factors = torch.zeros(knot_count, levels, n // 2, 2, 2, dtype=dtype)
    factors[..., 0, 0] = 1
    factors[..., 1, 1] = 1
    return factors


class TorchParametricButterfly(nn.Module):
    """Differentiable frequency-interpolated product of butterfly stages."""

    def __init__(
        self,
        n: int,
        frequency_knots: Iterable[float],
        *,
        init_scale: float = 0.02,
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        levels = _require_power_of_two(n)
        knots = torch.as_tensor(list(frequency_knots), dtype=dtype)
        if knots.ndim != 1 or knots.numel() < 2 or not torch.all(knots[1:] > knots[:-1]):
            raise ValueError("frequency_knots must be strictly increasing")
        generator = torch.Generator().manual_seed(seed)
        factors = _identity_knots(int(knots.numel()), n, dtype=dtype)
        factors += init_scale * torch.randn(
            factors.shape, dtype=dtype, generator=generator
        )
        self.n = n
        self.levels = levels
        self.register_buffer("frequency_knots", knots)
        self.factors = nn.Parameter(factors)

    def factors_at(self, frequency: float | torch.Tensor) -> torch.Tensor:
        f = torch.as_tensor(
            frequency, dtype=self.frequency_knots.dtype, device=self.frequency_knots.device
        )
        f = torch.clamp(f, self.frequency_knots[0], self.frequency_knots[-1])
        upper = torch.searchsorted(self.frequency_knots, f, right=True)
        upper = torch.clamp(upper, 1, self.frequency_knots.numel() - 1)
        lower = upper - 1
        span = self.frequency_knots[upper] - self.frequency_knots[lower]
        weight = (f - self.frequency_knots[lower]) / span
        return (1 - weight) * self.factors[lower] + weight * self.factors[upper]

    def forward(self, frequency: float | torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        vector = x.ndim == 1
        if vector:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] != self.n:
            raise ValueError(f"x must have shape ({self.n},) or ({self.n}, b)")
        work = x
        factors = self.factors_at(frequency)
        for stage in range(self.levels):
            stride = 1 << stage
            base = torch.arange(0, self.n, 2 * stride, device=x.device)
            offset = torch.arange(stride, device=x.device)
            left = (base[:, None] + offset[None, :]).reshape(-1)
            right = left + stride
            pair = factors[stage]
            x_left = work.index_select(0, left)
            x_right = work.index_select(0, right)
            y_left = pair[:, 0, 0, None] * x_left + pair[:, 0, 1, None] * x_right
            y_right = pair[:, 1, 0, None] * x_left + pair[:, 1, 1, None] * x_right
            out = torch.empty_like(work)
            out = out.index_copy(0, left, y_left)
            out = out.index_copy(0, right, y_right)
            work = out
        return work[:, 0] if vector else work

    def dense(self, frequency: float | torch.Tensor) -> torch.Tensor:
        identity = torch.eye(
            self.n, dtype=self.factors.dtype, device=self.factors.device
        )
        return self(frequency, identity)

