"""Field-waveform experiment and command-line reproducibility entry point."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from .operator import ParametricButterflyOperator
from .training import train_parametric_butterfly

DATA_URL = (
    "https://service.earthscope.org/fdsnws/dataselect/1/query?"
    "net=IU&sta=ANMO&loc=00&cha=BHZ&starttime=2010-02-27T06:34:00&"
    "endtime=2010-02-27T06:44:00&format=geocsv"
)
EXPECTED_SHA256 = "e202fc350422a720d0bb3f5094c143c61b851b97f937b521dd3fa1d06391aba0"


def fetch_field_trace(
    url: str = DATA_URL,
    expected_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, str]]:
    """Download and verify one public EarthScope GeoCSV waveform."""

    expected = (
        EXPECTED_SHA256
        if url == DATA_URL and expected_sha256 is None
        else expected_sha256
    )
    if url != DATA_URL and expected is None:
        raise ValueError("a SHA-256 digest is required for a custom data URL")
    with urlopen(url, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if expected is not None and digest != expected:
        raise RuntimeError(
            f"EarthScope payload checksum changed: expected {expected}, got {digest}"
        )
    text = payload.decode("utf-8")
    metadata: dict[str, str] = {}
    data_lines = []
    for line in text.splitlines():
        if line.startswith("# "):
            key, _, value = line[2:].partition(":")
            metadata.setdefault(key.strip(), value.strip())
        elif line and not line.startswith("#"):
            data_lines.append(line)
    reader = csv.DictReader(io.StringIO("\n".join(data_lines)))
    samples_list = []
    for row in reader:
        try:
            samples_list.append(float(row[" Sample"]))
        except (TypeError, ValueError):
            continue
    samples = np.asarray(samples_list, dtype=float)
    return samples, metadata


def waveform_windows(samples: np.ndarray, n: int = 64, stride: int = 16) -> np.ndarray:
    """Create centered, unit-energy windows without crossing the trace."""

    windows = []
    for start in range(0, samples.size - n + 1, stride):
        window = np.asarray(samples[start : start + n], dtype=float)
        window = window - np.mean(window)
        norm = np.linalg.norm(window)
        if norm > 0.0:
            windows.append(window / norm)
    return np.column_stack(windows)


def soft_lowpass(block: np.ndarray, cutoff: float) -> np.ndarray:
    """Reference zero-phase low-pass task, with cutoff in Nyquist units."""

    frequencies = np.fft.rfftfreq(block.shape[0]) / 0.5
    gain = 1.0 / (1.0 + (frequencies / float(cutoff)) ** 8)
    return np.fft.irfft(np.fft.rfft(block, axis=0) * gain[:, None], n=block.shape[0], axis=0)


def relative_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sum((prediction - target) ** 2) / np.sum(target**2))


def _dense_soft_lowpass(n: int, cutoff: float) -> np.ndarray:
    """Assemble the reference action only for small baseline audits."""

    return soft_lowpass(np.eye(n), cutoff)


def _fourier_low_rank(n: int, cutoff: float, rank: int) -> np.ndarray:
    """Deterministic real-Fourier low-rank approximation of the test filter."""

    if not 1 <= rank <= n:
        raise ValueError("rank must lie between 1 and n")
    sample = np.arange(n, dtype=float)
    columns = [np.ones(n) / np.sqrt(n)]
    frequencies = [0.0]
    for index in range(1, n // 2):
        phase = 2.0 * np.pi * index * sample / n
        columns.extend(
            [
                np.sqrt(2.0 / n) * np.cos(phase),
                np.sqrt(2.0 / n) * np.sin(phase),
            ]
        )
        frequencies.extend([2.0 * index / n, 2.0 * index / n])
    columns.append((-1.0) ** sample / np.sqrt(n))
    frequencies.append(1.0)
    basis = np.column_stack(columns[:rank])
    normalized_frequencies = np.asarray(frequencies[:rank])
    gains = 1.0 / (1.0 + (normalized_frequencies / float(cutoff)) ** 8)
    return (basis * gains) @ basis.T


def _interpolate_endpoints(
    lower_matrix: np.ndarray,
    upper_matrix: np.ndarray,
    cutoff: float,
    lower_cutoff: float,
    upper_cutoff: float,
) -> np.ndarray:
    weight = (cutoff - lower_cutoff) / (upper_cutoff - lower_cutoff)
    return (1.0 - weight) * lower_matrix + weight * upper_matrix


def run_field_experiment(
    *,
    n: int = 64,
    stride: int = 16,
    steps: int = 900,
    seed: int = 260724,
    data_url: str = DATA_URL,
    data_sha256: str | None = None,
    lower_cutoff: float = 0.12,
    upper_cutoff: float = 0.28,
    evaluation_cutoff: float = 0.20,
    training_fraction: float = 0.70,
    batch_size: int = 24,
    learning_rate: float = 1.5e-2,
) -> dict:
    if not 0.0 < training_fraction < 1.0:
        raise ValueError("training_fraction must lie strictly between zero and one")
    if not 0.0 < lower_cutoff < upper_cutoff <= 1.0:
        raise ValueError("cutoff knots must satisfy 0 < lower < upper <= 1")
    if not lower_cutoff <= evaluation_cutoff <= upper_cutoff:
        raise ValueError("evaluation_cutoff must lie between the cutoff knots")
    expected = (
        EXPECTED_SHA256
        if data_url == DATA_URL and data_sha256 is None
        else data_sha256
    )
    samples, metadata = fetch_field_trace(data_url, expected)
    windows = waveform_windows(samples, n=n, stride=stride)
    split = int(training_fraction * windows.shape[1])
    train = windows[:, :split]
    test = windows[:, split:]
    knots = (float(lower_cutoff), float(upper_cutoff))
    factors, training = train_parametric_butterfly(
        train,
        soft_lowpass,
        frequency_knots=knots,
        steps=steps,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    model = ParametricButterflyOperator(knots, factors)
    held_out_cutoff = float(evaluation_cutoff)
    target = soft_lowpass(test, held_out_cutoff)
    prediction = model.apply(held_out_cutoff, test)
    identity_error = relative_mse(test, target)
    learned_error = relative_mse(prediction, target)

    # The comparison is intentionally unfavorable but necessary: the reference
    # task is known exactly in Fourier space, and a credible paper must show
    # that fact alongside the learned approximation.  The rank-six endpoint
    # family has the same 1,536 stored scalars as the two-knot butterfly at
    # n=64: 2 endpoints * 2 factors * n * rank.
    equal_budget_rank = max(1, int(factors.size // (4 * n)))
    dense_lower = _dense_soft_lowpass(n, lower_cutoff)
    dense_upper = _dense_soft_lowpass(n, upper_cutoff)
    low_rank_lower = _fourier_low_rank(n, lower_cutoff, equal_budget_rank)
    low_rank_upper = _fourier_low_rank(n, upper_cutoff, equal_budget_rank)
    cutoff_values = np.linspace(lower_cutoff, upper_cutoff, 9)
    cutoff_sweep = []
    for cutoff in cutoff_values:
        exact = soft_lowpass(test, float(cutoff))
        low_rank = _interpolate_endpoints(
            low_rank_lower,
            low_rank_upper,
            float(cutoff),
            lower_cutoff,
            upper_cutoff,
        )
        dense_interpolant = _interpolate_endpoints(
            dense_lower,
            dense_upper,
            float(cutoff),
            lower_cutoff,
            upper_cutoff,
        )
        cutoff_sweep.append(
            {
                "cutoff_nyquist": float(cutoff),
                "identity_relative_mse": relative_mse(test, exact),
                "butterfly_relative_mse": relative_mse(
                    model.apply(float(cutoff), test), exact
                ),
                "equal_budget_rank_relative_mse": relative_mse(
                    low_rank @ test, exact
                ),
                "dense_endpoint_interpolant_relative_mse": relative_mse(
                    dense_interpolant @ test, exact
                ),
                "exact_fft_relative_mse": relative_mse(
                    soft_lowpass(test, float(cutoff)), exact
                ),
            }
        )

    low_rank_held_out = _interpolate_endpoints(
        low_rank_lower,
        low_rank_upper,
        held_out_cutoff,
        lower_cutoff,
        upper_cutoff,
    )
    dense_held_out = _interpolate_endpoints(
        dense_lower,
        dense_upper,
        held_out_cutoff,
        lower_cutoff,
        upper_cutoff,
    )
    return {
        "schema_version": 2,
        "data": {
            "provider": "EarthScope FDSN",
            "url": data_url,
            "sha256": expected,
            "network_station_channel": "IU.ANMO.00.BHZ",
            "start_utc": metadata["start_time"],
            "sample_rate_hz": float(metadata["sample_rate_hz"]),
            "samples": int(samples.size),
            "instrument": metadata["instrument"],
        },
        "model": {
            "window_length": n,
            "stride": stride,
            "training_windows": int(train.shape[1]),
            "held_out_windows": int(test.shape[1]),
            "training_fraction": training_fraction,
            "frequency_knots_nyquist": list(knots),
            "held_out_cutoff_nyquist": held_out_cutoff,
            "factor_parameters": int(factors.size),
            "dense_entries_per_cutoff": int(n * n),
            "equal_budget_rank": equal_budget_rank,
            "equal_budget_rank_parameters": int(4 * n * equal_budget_rank),
            "dense_two_knot_parameters": int(2 * n * n),
            "matrix_free": True,
        },
        "training": training,
        "held_out": {
            "identity_relative_mse": identity_error,
            "learned_relative_mse": learned_error,
            "improvement_factor_over_identity": identity_error / learned_error,
            "equal_budget_rank_relative_mse": relative_mse(
                low_rank_held_out @ test, target
            ),
            "dense_endpoint_interpolant_relative_mse": relative_mse(
                dense_held_out @ test, target
            ),
            "exact_fft_relative_mse": relative_mse(
                soft_lowpass(test, held_out_cutoff), target
            ),
        },
        "cutoff_sweep": cutoff_sweep,
        "scope": (
            "Single-station field-waveform operator-action pilot; not a source-receiver "
            "response factorization, acquisition-design result, or production benchmark."
        ),
    }


def canonical_bytes(result: dict) -> bytes:
    return (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the EarthScope field-waveform butterfly pilot."
    )
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=260724)
    parser.add_argument("--data-url", default=DATA_URL)
    parser.add_argument("--data-sha256")
    parser.add_argument("--lower-cutoff", type=float, default=0.12)
    parser.add_argument("--upper-cutoff", type=float, default=0.28)
    parser.add_argument("--evaluation-cutoff", type=float, default=0.20)
    parser.add_argument("--training-fraction", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1.5e-2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = run_field_experiment(
        n=args.n,
        stride=args.stride,
        steps=args.steps,
        seed=args.seed,
        data_url=args.data_url,
        data_sha256=args.data_sha256,
        lower_cutoff=args.lower_cutoff,
        upper_cutoff=args.upper_cutoff,
        evaluation_cutoff=args.evaluation_cutoff,
        training_fraction=args.training_fraction,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    rendered = canonical_bytes(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(rendered)
    if args.verify:
        locked = Path(__file__).with_name("results").joinpath("field_results.json")
        if rendered != locked.read_bytes():
            raise RuntimeError("recomputed result differs from locked field_results.json")
    print(rendered.decode("utf-8"), end="")
    print(f"sha256={hashlib.sha256(rendered).hexdigest()}")


if __name__ == "__main__":
    main()
