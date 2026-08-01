import numpy as np

from taskbutterfly.field_experiment import (
    _dense_soft_lowpass,
    _fourier_low_rank,
    _interpolate_endpoints,
    soft_lowpass,
    waveform_windows,
)


def test_waveform_windows_are_normalized():
    samples = np.sin(np.linspace(0, 30, 256)) + 0.1 * np.arange(256)
    windows = waveform_windows(samples, n=32, stride=16)
    np.testing.assert_allclose(np.mean(windows, axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(windows, axis=0), 1.0, atol=1e-14)


def test_soft_lowpass_preserves_shape_and_reduces_roughness():
    block = np.zeros((64, 1))
    block[::2] = 1.0
    filtered = soft_lowpass(block, 0.2)
    assert filtered.shape == block.shape
    assert np.linalg.norm(np.diff(filtered[:, 0])) < np.linalg.norm(np.diff(block[:, 0]))


def test_small_baseline_helpers_are_consistent():
    matrix = _dense_soft_lowpass(16, 0.2)
    block = np.arange(48, dtype=float).reshape(16, 3)
    np.testing.assert_allclose(matrix @ block, soft_lowpass(block, 0.2), atol=1e-13)
    assert np.linalg.matrix_rank(_fourier_low_rank(16, 0.2, 3), tol=1e-10) <= 3
    np.testing.assert_allclose(
        _interpolate_endpoints(matrix, 2.0 * matrix, 0.2, 0.1, 0.3),
        1.5 * matrix,
    )
