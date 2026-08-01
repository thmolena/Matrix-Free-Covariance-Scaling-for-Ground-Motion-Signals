"""Registered development and confirmation study for streaming energy ratios."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from . import seismic_dataset as sd
from .benchmark import _load_protocol
from .streaming_energy import (
    StreamingEnergy,
    balanced_accuracy,
    best_balanced_threshold,
    centered_block_energies,
    log_reference_ratio,
    raw_log_mean,
    raw_log_quantile,
)
from .whitening import roc_auc

BLOCK_SIZE = 64
PREFIX_GRID = (1, 2, 4, 8, 16, 32, 64)
SAMPLE_RATE_HZ = 20.0
CONFIRMATION_STATIONS = ("TATO", "PAB", "CASY")


@dataclass(frozen=True)
class ConfirmationEvent:
    event_id: str
    name: str
    origin_epoch_seconds: float
    magnitude: float
    latitude: float
    longitude: float
    depth_km: float


CONFIRMATION_EVENTS = (
    ConfirmationEvent(
        "us7000dflf",
        "Kermadec Islands",
        1614886113.178,
        8.1,
        -29.7228,
        -177.2794,
        28.93,
    ),
    ConfirmationEvent(
        "ak0219neiszm",
        "Chignik, Alaska",
        1627539349.188,
        8.2,
        55.3635,
        -157.8876,
        35.0,
    ),
    ConfirmationEvent(
        "us6000f53e",
        "South Sandwich Islands",
        1628793317.231,
        8.1,
        -58.3753,
        -25.2637,
        22.79,
    ),
    ConfirmationEvent(
        "us6000qw60",
        "Kamchatka Peninsula",
        1753831492.483,
        8.8,
        52.4948,
        160.2395,
        35.0,
    ),
)

# A second quiet day is used only as an independent negative.  These digests
# were frozen before the confirmation records were opened.
DEVELOPMENT_NEGATIVE_SHA256 = {
    "official20100227063411530_30/ANMO": "0946c6b08505c9f18a85794523773b33425fb9a591a41b41dc8d491705c108c5",
    "official20100227063411530_30/COLA": "6905466eccaae8394fdc6726ea012eb3c6b9b39a660de84aaca885ebe57b391c",
    "official20100227063411530_30/MAJO": "56812764539321288eb9fcf31280610c949243e944425c0cf0fccda2a16e4e51",
    "official20110311054624120_30/ANMO": "168b0b2b75a5c4d7388655f16d672bed2e96f60fbb31d919de3a2e2798ef29db",
    "official20110311054624120_30/COLA": "ded7f583a2731efd4d3704fb29017c61598ed6397e9ff0139499fb38c53c438a",
    "official20110311054624120_30/MAJO": "4b1c3acdb415c61892d07caf8aa65fc23fec1f7891d09e8ce371135abd8e280b",
    "official20120411083836720_20/ANMO": "064c2e827caac8bafe547ff4b34bbc0bc0d203186b2cfe8b30331d9e7b62d99c",
    "official20120411083836720_20/COLA": "2333ab791a445cc590880e04509990028e2fd2e33a0d1c6ef35faaeb7ea996a8",
    "official20120411083836720_20/MAJO": "f004773be2d8fe6ee49d6d98743865a3299a4a382cd1849ef09c6d6de13a4d77",
    "usb000h4jh/ANMO": "cd9a6658c0f682bbdca69b5f67aab6ce0444b8170c5b566c97219567e76bab23",
    "usb000h4jh/COLA": "beaac2bbf6148ab4b7c62d0503093b16fe7e831f7bd2333c854716ec7a7a6f96",
    "usb000h4jh/MAJO": "fb0d480be4d0b00e84e6a7997d3d6cd91aa9d85c733bf285c2a333d2a1d273ef",
    "usc000nzvd/COR": "c088cdf473d7a6f9bbb729182daa3ef3bd4ca750b83bb354b1149ff5abac9723",
    "usc000nzvd/KONO": "601b939dc230ef6f81f4d8c1d01ebd193b816b683bf0bea3a98824b1528da5cb",
    "usc000nzvd/HRV": "c1a70a0c8c7b8a60dc8bf1db63c42fdf2ef5717a996383482197b8b9f8f552f8",
    "us20003k7a/COR": "2ffbb379bf3bdf0bea26f5b364307f2a5c3184a34a9b01b53312ae0be936351f",
    "us20003k7a/KONO": "e143584b8523067d17e35c217edfeb7defe283ccb994f54ccc33866b99ef9039",
    "us20003k7a/HRV": "b3631ba4bf61386e28af1dd056b8eafb405c77ddc725cc79ceea837ae24d10a8",
    "us2000ahv0/COR": "31a9237e1da9396407338eedbb14b5df5bbf3d554797a46e1b8c0160cd350cc0",
    "us2000ahv0/KONO": "a4476f2595510a99e1ef3cca3bcfef5bb957795bbf27c73ad9814817737c0ad1",
    "us2000ahv0/HRV": "f3043a29bfe8bd8262b0b0a6bb5cdef31133809008000520372c40c33388d4cb",
    "us60003sc0/COR": "3d4ffcc8f486efbfb61015a148c2cef0b24b807635aef14769e24255ada9636a",
    "us60003sc0/KONO": "115f7ac5e77cb5b4dca705135c60215f7a76fce456c3c640167c9cf6f00f4db5",
    "us60003sc0/HRV": "e5dfcd5265e956ca19a3786c3c0c3f5360f262345916b4cfc5e4c5dbd4f57fa7",
}

CONFIRMATION_SHA256 = {
    "us7000dflf/TATO/event": "13dd9b95f39bc8e8d2ca87e04548557e3e518591955a1964f119a307d8b1d75a",
    "us7000dflf/TATO/reference": "1f339d26a1d64f6d83a6ed380bc2a10f6affe068deaa21b00463e54b458ee45c",
    "us7000dflf/TATO/negative": "6104bc26fd6a9b30493bc0f1473596c79f5c30ab619ce6fe59bc2dd58167f7e1",
    "us7000dflf/PAB/event": "eebcb81b16a3652b8a843326a44d51440273f1cfe4b3360f211012218c66c64c",
    "us7000dflf/PAB/reference": "6db02a1d91aada5795107ba6abfe946a51772c655e90664a4f33511be37f174d",
    "us7000dflf/PAB/negative": "905d9290f4bfdeabb847964a44d68fbdac7b2ce184b076e675470060d654df32",
    "us7000dflf/CASY/event": "a883dbebeab63cbcb373288f1dde7676927b714e4eabc6d202ba23bd7cd399a2",
    "us7000dflf/CASY/reference": "b2c17b0930f95340cde7afdfbd91788591b8b9a97603c68b8962e81796a3b430",
    "us7000dflf/CASY/negative": "1648b27eb249228c7bc69717c9a408b5d1da5fadae1084ba9a0f0279767c007e",
    "ak0219neiszm/TATO/event": "e2e39832b2b81ee63f5b6a2dab9018dd054790c2b6d9a1aa6ba22573f4611356",
    "ak0219neiszm/TATO/reference": "b5eaf5f18264fdd19cbf4c5a37d03cbbe0cfb87c2e8e5196cb510bf93f855731",
    "ak0219neiszm/TATO/negative": "bb7e5ed2fc6b4a5325a2100e254493c8cc3744d5151eb2ba4ca606f65193bf5b",
    "ak0219neiszm/PAB/event": "9ac7ad9f2b7f66b5ef9e6998f50742c0df4642a6a601303a7d167e980c8bf504",
    "ak0219neiszm/PAB/reference": "9637d9f049e804c5f9ea77f98c183b89d3b954ef07b04d97431ae4542633cb4e",
    "ak0219neiszm/PAB/negative": "73b6000ccffe01ed19d234e08031e62c3369b40468ee82d08969b0db54450c0f",
    "ak0219neiszm/CASY/event": "2f6343b6ea5a3615fcbd441b1de63d9ab412e3b664f8d8f5e65fdbd4d6831e74",
    "ak0219neiszm/CASY/reference": "27faef5b8bcf6fd93712cb7d9674f66bbed05ad642d20fed27acb84fea55d590",
    "ak0219neiszm/CASY/negative": "a90f9a4b9e3f49cd1f89eb835111e99d9383f117e7396914d91f139b1d52c1de",
    "us6000f53e/TATO/event": "ef471d385997ece0ef6b38ecacfda1d3774cab8415c3de7d7f9806c10e676072",
    "us6000f53e/TATO/reference": "20b02088acef8712fc8b9d616ede0e9b3f8f1e1fc242b8dc122209078b89dc1e",
    "us6000f53e/TATO/negative": "21dba24d87bf10be54b1f792b4f6562bf0b5867f13d1e4d49e2a90c99901d21d",
    "us6000f53e/PAB/event": "13dafb9ee6282e420a89369c38f7e31dfc9a4aa64f6686281186474342fe6ca7",
    "us6000f53e/PAB/reference": "345a77c7b42d5ae5f05cbfa54c88931e939683e429fb3219cebabea05d024454",
    "us6000f53e/PAB/negative": "8f741eede9f176739a518c084eaade8fa424b231883d24bc07a4c0e22343be4e",
    "us6000f53e/CASY/event": "1cc11c065a1475fd623c1e62f5e97d38a9978404695fe5f3c9837d2339c48ba9",
    "us6000f53e/CASY/reference": "829b375e2a321acd30077a23ce8929d1013b7d13f6f522fe136dd601bac1b2b0",
    "us6000f53e/CASY/negative": "dbb11f758b13d6e5ce7fd785ae1659859fe1fd3a680148b14001de6640a8dac0",
    "us6000qw60/TATO/event": "b5c3ab0e76c7dfc9d964eb8607eafd857856c2a903e5e1e10ea4b719afb6964b",
    "us6000qw60/TATO/reference": "04fd70dac060f6284fe1e1c96bd82392204f5da532696d7c3d7d3aeb8013a6b1",
    "us6000qw60/TATO/negative": "5e9f68e1e0678ebb3059df97ad3cc45eb41021f4d3c468e0a6cb58a7eb73d725",
    "us6000qw60/PAB/event": "9e00fbd1509d15403b74566c05c55138d0ad83c57c323bedbe89438d0ff1e672",
    "us6000qw60/PAB/reference": "2d1a75beef644dd1e4b1264aa8645e9386180dfce103be6bf72b0a3c2ee2a7c0",
    "us6000qw60/PAB/negative": "56eaa02948fa4f3f6c1528936a5e6927afac443b1181a5992629b806c4368cb9",
    "us6000qw60/CASY/event": "2ac0e055fcc50bc9dfb110795dd9652face5b39115b4538190b8a61773259603",
    "us6000qw60/CASY/reference": "c2d8adee51df91fd129b35fdb30b6bbfb85a2aeeea248a616cd69f5eadc11a51",
    "us6000qw60/CASY/negative": "81e33ea6980771a1ee1404be641ac54e22f65e10e771f98293e1af28f0f98fd7",
}


def _request_url(origin_seconds: float, station: str, day_offset: int) -> str:
    start = datetime.fromtimestamp(origin_seconds, tz=timezone.utc) + timedelta(
        seconds=sd.WINDOW_OFFSET_SECONDS, days=-day_offset
    )
    query = urlencode(
        {
            "net": "IU",
            "sta": station,
            "loc": "00",
            "cha": "BH?",
            "starttime": sd._format_utc(start),
            "endtime": sd._format_utc(
                start + timedelta(seconds=sd.WINDOW_DURATION_SECONDS)
            ),
            "format": "geocsv",
            "scale": "AUTO",
            "nodata": 404,
        }
    )
    return f"{sd.EARTHSCOPE_DATASELECT}?{query}"


def _load_url(
    url: str,
    expected_sha256: str,
    *,
    cache_dir: Path | None,
    offline: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    payload, digest = sd.fetch_payload(
        url,
        expected_sha256=expected_sha256,
        cache_dir=cache_dir,
        offline=offline,
    )
    components, quality = sd.resample_three_components(sd.parse_geocsv(payload))
    return components, {
        "url": url,
        "sha256": digest,
        "bytes": len(payload),
        "common_samples": quality["common_samples"],
        "selected_sids": quality["selected_sids"],
        "scale_units": quality["scale_units"],
    }


def _development_rows(
    *, cache_dir: Path | None, offline: bool
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prepared, original_provenance = _load_protocol(
        cache_dir=cache_dir, offline=offline
    )
    by_key = {
        (str(record["pair_id"]), str(record["label"])): np.asarray(
            record["components"]
        )
        for record in prepared
    }
    rows: list[dict[str, object]] = []
    extra_provenance: list[dict[str, object]] = []
    for event in sd.EVENTS:
        for station in sd.STATIONS:
            if event.split != station.split:
                continue
            pair_id = f"{event.event_id}/{station.code}"
            negative_url = _request_url(
                sd._utc(event.origin_utc).timestamp(), station.code, 2
            )
            negative, provenance = _load_url(
                negative_url,
                DEVELOPMENT_NEGATIVE_SHA256[pair_id],
                cache_dir=cache_dir,
                offline=offline,
            )
            extra_provenance.append(
                {"key": f"{pair_id}/negative", **provenance}
            )
            rows.append(
                {
                    "event_id": event.event_id,
                    "event_name": event.name,
                    "station": station.code,
                    "event": centered_block_energies(
                        by_key[(pair_id, "event")], block_size=BLOCK_SIZE
                    ),
                    "reference": centered_block_energies(
                        by_key[(pair_id, "quiet")], block_size=BLOCK_SIZE
                    ),
                    "negative": centered_block_energies(
                        negative, block_size=BLOCK_SIZE
                    ),
                }
            )
    return rows, original_provenance + extra_provenance


def _confirmation_rows(
    *, cache_dir: Path | None, offline: bool
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    for event in CONFIRMATION_EVENTS:
        for station in CONFIRMATION_STATIONS:
            arrays: dict[str, np.ndarray] = {}
            event_components: np.ndarray | None = None
            for day_offset, label in (
                (0, "event"),
                (1, "reference"),
                (2, "negative"),
            ):
                key = f"{event.event_id}/{station}/{label}"
                url = _request_url(
                    event.origin_epoch_seconds, station, day_offset
                )
                components, details = _load_url(
                    url,
                    CONFIRMATION_SHA256[key],
                    cache_dir=cache_dir,
                    offline=offline,
                )
                arrays[label] = centered_block_energies(
                    components, block_size=BLOCK_SIZE
                )
                if label == "event":
                    event_components = components
                provenance.append({"key": key, **details})
            assert event_components is not None
            rows.append(
                {
                    "event_id": event.event_id,
                    "event_name": event.name,
                    "station": station,
                    "event_components": event_components,
                    **arrays,
                }
            )
    return rows, provenance


def _scores(
    rows: list[dict[str, object]], prefix_blocks: int, method: str
) -> tuple[np.ndarray, np.ndarray]:
    positive: list[float] = []
    negative: list[float] = []
    for row in rows:
        event = np.asarray(row["event"])
        reference = np.asarray(row["reference"])
        quiet = np.asarray(row["negative"])
        if method == "reference_ratio":
            positive.append(
                log_reference_ratio(
                    event, reference, prefix_blocks=prefix_blocks
                )
            )
            negative.append(
                log_reference_ratio(
                    quiet, reference, prefix_blocks=prefix_blocks
                )
            )
        elif method == "raw_mean":
            positive.append(raw_log_mean(event, prefix_blocks=prefix_blocks))
            negative.append(raw_log_mean(quiet, prefix_blocks=prefix_blocks))
        elif method == "raw_q90":
            positive.append(
                raw_log_quantile(event, prefix_blocks=prefix_blocks)
            )
            negative.append(
                raw_log_quantile(quiet, prefix_blocks=prefix_blocks)
            )
        else:
            raise ValueError(f"unknown method: {method}")
    return np.asarray(positive), np.asarray(negative)


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    labels = np.concatenate((np.ones(positive.size), np.zeros(negative.size)))
    scores = np.concatenate((positive, negative))
    return float(roc_auc(labels, scores))


def _leave_event_out(
    rows: list[dict[str, object]], prefix_blocks: int
) -> dict[str, object]:
    values: list[float] = []
    by_event: dict[str, float] = {}
    for event_id in sorted({str(row["event_id"]) for row in rows}):
        training = [row for row in rows if row["event_id"] != event_id]
        held = [row for row in rows if row["event_id"] == event_id]
        positive, negative = _scores(
            training, prefix_blocks, "reference_ratio"
        )
        threshold, _ = best_balanced_threshold(positive, negative)
        held_positive, held_negative = _scores(
            held, prefix_blocks, "reference_ratio"
        )
        value, _, _ = balanced_accuracy(
            held_positive, held_negative, threshold
        )
        values.append(value)
        by_event[event_id] = value
    return {
        "mean_balanced_accuracy": float(np.mean(values)),
        "worst_balanced_accuracy": float(np.min(values)),
        "by_event": by_event,
    }


def _cluster_bootstrap_difference(
    proposed_positive: np.ndarray,
    proposed_negative: np.ndarray,
    proposed_threshold: float,
    baseline_positive: np.ndarray,
    baseline_negative: np.ndarray,
    baseline_threshold: float,
    *,
    repetitions: int = 10000,
    seed: int = 260801,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    count = proposed_positive.size
    differences = np.empty(repetitions)
    for index in range(repetitions):
        sample = rng.integers(0, count, size=count)
        proposed = balanced_accuracy(
            proposed_positive[sample],
            proposed_negative[sample],
            proposed_threshold,
        )[0]
        baseline = balanced_accuracy(
            baseline_positive[sample],
            baseline_negative[sample],
            baseline_threshold,
        )[0]
        differences[index] = proposed - baseline
    return {
        "repetitions": repetitions,
        "seed": seed,
        "mean_difference": float(np.mean(differences)),
        "lower_95": float(np.quantile(differences, 0.025)),
        "upper_95": float(np.quantile(differences, 0.975)),
    }


def _runtime_study(row: dict[str, object]) -> list[dict[str, float | int]]:
    components = np.asarray(row["event_components"])
    reference = np.asarray(row["reference"])
    reference_mean = float(np.mean(reference))
    results: list[dict[str, float | int]] = []
    repeats = 1500
    for blocks in PREFIX_GRID:
        block_bank = np.moveaxis(
            components[:, : blocks * BLOCK_SIZE].reshape(
                3, blocks, BLOCK_SIZE
            ),
            1,
            0,
        )
        samples = []
        for _ in range(7):
            start = time.perf_counter_ns()
            for _ in range(repeats):
                stream = StreamingEnergy(reference_mean, block_size=BLOCK_SIZE)
                for block in block_bank:
                    stream.update(block)
            samples.append((time.perf_counter_ns() - start) / repeats / 1000.0)
        results.append(
            {
                "prefix_blocks": blocks,
                "prefix_seconds": blocks * BLOCK_SIZE / SAMPLE_RATE_HZ,
                "median_microseconds": float(np.median(samples)),
                "lower_microseconds": float(np.quantile(samples, 0.1)),
                "upper_microseconds": float(np.quantile(samples, 0.9)),
                "streaming_state_bytes": 24,
                "stored_energy_bytes": 8 * blocks,
            }
        )
    return results


def _make_figures(result: dict[str, object], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )
    curves = result["prefix_curves"]
    seconds = [entry["prefix_seconds"] for entry in curves]
    fig, ax = plt.subplots(figsize=(5.5, 3.25))
    ax.semilogx(
        seconds,
        [entry["confirmation"]["reference_ratio"]["balanced_accuracy"] for entry in curves],
        "o-",
        label="station-reference ratio",
    )
    ax.semilogx(
        seconds,
        [entry["confirmation"]["raw_mean"]["balanced_accuracy"] for entry in curves],
        "s--",
        label="raw mean energy",
    )
    ax.semilogx(
        seconds,
        [entry["confirmation"]["raw_q90"]["balanced_accuracy"] for entry in curves],
        "^--",
        label="raw 0.9-quantile",
    )
    ax.semilogx(
        seconds,
        [entry["development_leave_event_out"]["mean_balanced_accuracy"] for entry in curves],
        color="0.35",
        linestyle=":",
        label="development leave-one-event-out",
    )
    ax.axvline(result["selection"]["prefix_seconds"], color="#b22222", alpha=0.7)
    ax.set(xlabel="available waveform (s)", ylabel="balanced accuracy", ylim=(0.45, 1.02))
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"prefix_performance.{suffix}", bbox_inches="tight")
    plt.close(fig)

    rows = result["confirmation"]["pairs"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.4, 3.35))
    ax.plot(x, [row["event_score"] for row in rows], "o", label="earthquake window")
    ax.plot(x, [row["negative_score"] for row in rows], "s", label="independent quiet window")
    ax.axhline(result["selection"]["threshold"], color="#b22222", label="development threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['event_short']}\n{row['station']}" for row in rows], rotation=45, ha="right")
    ax.set(ylabel=r"$\log_{10}$ energy ratio")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"confirmation_scores.{suffix}", bbox_inches="tight")
    plt.close(fig)

    runtime = result["runtime"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75))
    sx = [entry["prefix_seconds"] for entry in runtime]
    med = [entry["median_microseconds"] for entry in runtime]
    lo = [entry["lower_microseconds"] for entry in runtime]
    hi = [entry["upper_microseconds"] for entry in runtime]
    axes[0].loglog(sx, med, "o-")
    axes[0].fill_between(sx, lo, hi, alpha=0.22)
    axes[0].set(xlabel="processed waveform (s)", ylabel="apply time (microseconds)")
    axes[1].loglog(sx, [entry["streaming_state_bytes"] for entry in runtime], "o-", label="streaming state")
    axes[1].loglog(sx, [entry["stored_energy_bytes"] for entry in runtime], "s--", label="stored block energies")
    axes[1].set(xlabel="processed waveform (s)", ylabel="working storage (bytes)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"runtime_memory.{suffix}", bbox_inches="tight")
    plt.close(fig)


def run_study(
    *, cache_dir: Path | None = None, offline: bool = False
) -> dict[str, object]:
    start = time.perf_counter()
    development, development_provenance = _development_rows(
        cache_dir=cache_dir, offline=offline
    )
    selection_rows = []
    for prefix in PREFIX_GRID:
        selection_rows.append(
            {"prefix_blocks": prefix, **_leave_event_out(development, prefix)}
        )
    selected = max(
        selection_rows,
        key=lambda item: (item["mean_balanced_accuracy"], -item["prefix_blocks"]),
    )
    selected_prefix = int(selected["prefix_blocks"])

    thresholds: dict[str, float] = {}
    development_metrics: dict[str, dict[str, float]] = {}
    for method in ("reference_ratio", "raw_mean", "raw_q90"):
        positive, negative = _scores(development, selected_prefix, method)
        threshold, training_ba = best_balanced_threshold(positive, negative)
        thresholds[method] = threshold
        development_metrics[method] = {
            "threshold": threshold,
            "balanced_accuracy": training_ba,
            "auc": _auc(positive, negative),
        }

    confirmation, confirmation_provenance = _confirmation_rows(
        cache_dir=cache_dir, offline=offline
    )
    confirmation_metrics: dict[str, dict[str, float]] = {}
    for method in ("reference_ratio", "raw_mean", "raw_q90"):
        positive, negative = _scores(confirmation, selected_prefix, method)
        ba, sensitivity, specificity = balanced_accuracy(
            positive, negative, thresholds[method]
        )
        confirmation_metrics[method] = {
            "balanced_accuracy": ba,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "auc": _auc(positive, negative),
        }

    prefix_curves = []
    for prefix in PREFIX_GRID:
        confirmation_at_prefix: dict[str, dict[str, float]] = {}
        for method in ("reference_ratio", "raw_mean", "raw_q90"):
            dev_positive, dev_negative = _scores(development, prefix, method)
            threshold, _ = best_balanced_threshold(dev_positive, dev_negative)
            positive, negative = _scores(confirmation, prefix, method)
            ba, sensitivity, specificity = balanced_accuracy(
                positive, negative, threshold
            )
            confirmation_at_prefix[method] = {
                "balanced_accuracy": ba,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "auc": _auc(positive, negative),
                "development_threshold": threshold,
            }
        prefix_curves.append(
            {
                "prefix_blocks": prefix,
                "prefix_seconds": prefix * BLOCK_SIZE / SAMPLE_RATE_HZ,
                "development_leave_event_out": _leave_event_out(
                    development, prefix
                ),
                "confirmation": confirmation_at_prefix,
            }
        )

    proposed_positive, proposed_negative = _scores(
        confirmation, selected_prefix, "reference_ratio"
    )
    baseline_positive, baseline_negative = _scores(
        confirmation, selected_prefix, "raw_q90"
    )
    pairs = []
    for row, event_score, negative_score in zip(
        confirmation, proposed_positive, proposed_negative
    ):
        pairs.append(
            {
                "event_id": row["event_id"],
                "event_short": {
                    "Kermadec Islands": "Kermadec",
                    "Chignik, Alaska": "Chignik",
                    "South Sandwich Islands": "Sandwich",
                    "Kamchatka Peninsula": "Kamchatka",
                }[str(row["event_name"])],
                "station": row["station"],
                "event_score": float(event_score),
                "negative_score": float(negative_score),
                "event_decision": bool(event_score >= thresholds["reference_ratio"]),
                "negative_decision": bool(negative_score < thresholds["reference_ratio"]),
            }
        )

    runtime = _runtime_study(confirmation[0])
    semantic_core: dict[str, object] = {
        "schema_version": "3.0",
        "data": {
            "provider": "EarthScope FDSN, IU network",
            "catalog": "USGS ComCat",
            "development_events": 8,
            "development_stations_per_split": 3,
            "development_pairs": 24,
            "confirmation_events": [event.__dict__ for event in CONFIRMATION_EVENTS],
            "confirmation_stations": list(CONFIRMATION_STATIONS),
            "confirmation_pairs": 12,
            "records": development_provenance + confirmation_provenance,
        },
        "protocol": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": 3,
            "block_size": BLOCK_SIZE,
            "prefix_grid": list(PREFIX_GRID),
            "reference_offset_days": -1,
            "negative_offset_days": -2,
            "event_window_offset_seconds": sd.WINDOW_OFFSET_SECONDS,
            "window_duration_seconds": sd.WINDOW_DURATION_SECONDS,
            "selection_rule": "maximize mean leave-one-earthquake-out balanced accuracy; break ties toward fewer blocks",
        },
        "selection": {
            "prefix_blocks": selected_prefix,
            "prefix_seconds": selected_prefix * BLOCK_SIZE / SAMPLE_RATE_HZ,
            "threshold": thresholds["reference_ratio"],
            "development_leave_event_out": selected,
        },
        "development": development_metrics,
        "confirmation": {
            "metrics": confirmation_metrics,
            "pairs": pairs,
            "paired_bootstrap_difference_vs_raw_q90": _cluster_bootstrap_difference(
                proposed_positive,
                proposed_negative,
                thresholds["reference_ratio"],
                baseline_positive,
                baseline_negative,
                thresholds["raw_q90"],
            ),
        },
        "prefix_curves": prefix_curves,
    }
    semantic_json = json.dumps(
        semantic_core, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        **semantic_core,
        "semantic_sha256": hashlib.sha256(semantic_json).hexdigest(),
        "runtime": runtime,
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "elapsed_seconds": time.perf_counter() - start,
        },
    }


def _write_tables(result: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "prefix_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "prefix_blocks",
                "prefix_seconds",
                "development_loeo_ba",
                "reference_ratio_ba",
                "raw_mean_ba",
                "raw_q90_ba",
            ]
        )
        for row in result["prefix_curves"]:
            writer.writerow(
                [
                    row["prefix_blocks"],
                    row["prefix_seconds"],
                    row["development_leave_event_out"]["mean_balanced_accuracy"],
                    row["confirmation"]["reference_ratio"]["balanced_accuracy"],
                    row["confirmation"]["raw_mean"]["balanced_accuracy"],
                    row["confirmation"]["raw_q90"]["balanced_accuracy"],
                ]
            )
    with (output_dir / "confirmation_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(result["confirmation"]["pairs"][0].keys())
        )
        writer.writeheader()
        writer.writerows(result["confirmation"]["pairs"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("code/results/streaming"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("code/manuscript_assets/figures"),
    )
    args = parser.parse_args()
    locked_path = args.output_dir / "streaming_study.json"
    locked_hash = None
    if args.verify and locked_path.exists():
        locked_hash = json.loads(locked_path.read_text())["semantic_sha256"]
    result = run_study(cache_dir=args.cache_dir, offline=args.offline)
    if locked_hash is not None and result["semantic_sha256"] != locked_hash:
        raise SystemExit(
            "verification failed: semantic hash changed from "
            f"{locked_hash} to {result['semantic_sha256']}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    locked_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_tables(result, args.output_dir)
    _make_figures(result, args.figure_dir)
    print(json.dumps({
        "semantic_sha256": result["semantic_sha256"],
        "selected_prefix_seconds": result["selection"]["prefix_seconds"],
        "confirmation": result["confirmation"]["metrics"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
