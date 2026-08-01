#!/usr/bin/env python3
"""Regenerate the website certificate plot from the locked real-data replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = (
    REPOSITORY
    / "code"
    / "src"
    / "taskbutterfly"
    / "results"
    / "seismic_whitening_results.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY
    / "code"
    / "manuscript_assets"
    / "figures"
    / "web_overview.png"
)


def _load_four_product_curves(path: Path) -> tuple[list[dict], float, int, int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    algorithm = payload["algorithm"]
    runs = sorted(
        (run for run in algorithm["rank_runs"] if run["components"] == 4),
        key=lambda run: run["seed"],
    )
    if len(runs) != 5:
        raise ValueError(f"expected five four-product runs, found {len(runs)}")
    tasks = tuple(float(v) for v in algorithm["certified_tasks"])
    for run in runs:
        run_tasks = tuple(
            float(cert["log10_relative_ridge"])
            for cert in run["certificates_by_task"]
        )
        if run_tasks != tasks:
            raise ValueError("certificate task order differs from the registered task grid")
    return (
        runs,
        float(algorithm["certificate_relative_mse_tolerance"]),
        int(algorithm["familywise_hypotheses"]),
        int(algorithm["certificate_probes"]),
        float(algorithm["familywise_failure_probability"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            'Figure generation needs matplotlib. Install with: pip install "./code[figures]"'
        ) from exc

    runs, tolerance, hypotheses, probes, delta = _load_four_product_curves(
        args.result.resolve()
    )
    failed_runs = sum(
        any(
            float(cert["relative_frobenius_mse_upper_bound"]) > tolerance
            for cert in run["certificates_by_task"]
        )
        for run in runs
    )
    least_regularized = min(
        float(cert["log10_relative_ridge"])
        for run in runs
        for cert in run["certificates_by_task"]
    )
    least_regularized_failures = sum(
        any(
            float(cert["log10_relative_ridge"]) == least_regularized
            and float(cert["relative_frobenius_mse_upper_bound"]) > tolerance
            for cert in run["certificates_by_task"]
        )
        for run in runs
    )
    colors = ["#13231f", "#176b5b", "#4d8177", "#9d6b48", "#d2763c"]
    markers = ["o", "s", "^", "D", "P"]

    fig, ax = plt.subplots(figsize=(9.6, 5.6), constrained_layout=True)
    for run, color, marker in zip(runs, colors, markers, strict=True):
        certificates = run["certificates_by_task"]
        x = [float(cert["log10_relative_ridge"]) for cert in certificates]
        y = [float(cert["relative_frobenius_mse_upper_bound"]) for cert in certificates]
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5.5,
            label=f"seed {run['seed']}",
        )

    ax.axhline(tolerance, color="#1f1f1f", linestyle="--", linewidth=1.7)
    ax.text(
        -2.03,
        tolerance + 0.003,
        f"tolerance {tolerance:.2f}",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#1f1f1f",
    )
    ax.set_title("Simultaneous five-task certificate: four-product candidate")
    ax.set_xlabel(r"$\log_{10}$ relative ridge")
    ax.set_ylabel("relative Frobenius-MSE upper bound")
    ax.set_xticks([-4.0, -3.5, -3.0, -2.5, -2.0])
    ax.set_xlim(-4.08, -1.92)
    ax.set_ylim(0.04, 0.21)
    ax.grid(True, color="#d7d6cc", linewidth=0.8)
    ax.legend(ncol=2, frameon=False, fontsize=9, loc="upper right")
    ax.text(
        0.0,
        -0.19,
        f"Locked EarthScope replay: J={hypotheses}, m={probes:,}, global delta={delta:.2f}. "
        f"{failed_runs}/{len(runs)} runs fail; {least_regularized_failures}/{len(runs)} "
        f"fail at t={least_regularized:g}.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#53645f",
    )

    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.resolve(), dpi=120, facecolor="white")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
