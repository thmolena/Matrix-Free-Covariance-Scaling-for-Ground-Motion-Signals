"""Frozen, station- and event-disjoint EarthScope waveform protocol."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

EARTHSCOPE_DATASELECT = (
    "https://service.earthscope.org/fdsnws/dataselect/1/query"
)
WINDOW_OFFSET_SECONDS = 15 * 60
WINDOW_DURATION_SECONDS = 4 * 60
QUIET_OFFSET_SECONDS = -24 * 60 * 60
TARGET_SAMPLE_RATE_HZ = 20.0


@dataclass(frozen=True)
class Event:
    event_id: str
    name: str
    origin_utc: str
    magnitude: float
    latitude: float
    longitude: float
    depth_km: float
    split: str


@dataclass(frozen=True)
class Station:
    code: str
    latitude: float
    longitude: float
    split: str


EVENTS = (
    Event(
        "official20100227063411530_30",
        "Maule",
        "2010-02-27T06:34:11.530Z",
        8.8,
        -36.122,
        -72.898,
        22.9,
        "train",
    ),
    Event(
        "official20110311054624120_30",
        "Tohoku",
        "2011-03-11T05:46:24.120Z",
        9.1,
        38.297,
        142.373,
        29.0,
        "train",
    ),
    Event(
        "official20120411083836720_20",
        "Wharton",
        "2012-04-11T08:38:36.720Z",
        8.6,
        2.327,
        93.063,
        20.0,
        "train",
    ),
    Event(
        "usb000h4jh",
        "Sea of Okhotsk",
        "2013-05-24T05:44:48.980Z",
        8.3,
        54.892,
        153.221,
        598.1,
        "train",
    ),
    Event(
        "usc000nzvd",
        "Iquique",
        "2014-04-01T23:46:47.260Z",
        8.2,
        -19.6097,
        -70.7691,
        25.0,
        "test",
    ),
    Event(
        "us20003k7a",
        "Illapel",
        "2015-09-16T22:54:32.860Z",
        8.3,
        -31.5729,
        -71.6744,
        22.44,
        "test",
    ),
    Event(
        "us2000ahv0",
        "Tehuantepec",
        "2017-09-08T04:49:19.180Z",
        8.2,
        15.0222,
        -93.8993,
        47.39,
        "test",
    ),
    Event(
        "us60003sc0",
        "Peru",
        "2019-05-26T07:41:15.073Z",
        8.0,
        -5.8119,
        -75.2697,
        122.57,
        "test",
    ),
)

STATIONS = (
    Station("ANMO", 34.9459, -106.4571, "train"),
    Station("COLA", 64.8736, -147.8616, "train"),
    Station("MAJO", 36.5457, 138.2041, "train"),
    Station("COR", 44.5855, -123.3046, "test"),
    Station("KONO", 59.6491, 9.5982, "test"),
    Station("HRV", 42.5064, -71.5583, "test"),
)

# Populated from exact EarthScope GeoCSV responses by ``freeze-dataset``.
# Keys are ``event_id/station/label``.  Waveforms are not redistributed.
EXPECTED_SHA256: dict[str, str] = {
    "official20100227063411530_30/ANMO/event": "67b36456c4b3b42d1fe42efa023f70a7fb758545b5c4e978517aa99d81339a06",
    "official20100227063411530_30/ANMO/quiet": "18b23976640493ff4fc489cb5cf3bf4106cdd67e1fff6982f9234f542d0b83a5",
    "official20100227063411530_30/COLA/event": "07a541d1f077f3170330d13f2b870c50aa3958e21bd9dc936506ac546a310a58",
    "official20100227063411530_30/COLA/quiet": "abf58a4d647151f294fe572c4dcf68208055f24530d9f2c256e524a9077fac81",
    "official20100227063411530_30/MAJO/event": "73453d6dc9c185756e47069eae79f34fc3f68884aca4fb5f70b73198e639293b",
    "official20100227063411530_30/MAJO/quiet": "343fc0011996f5192b79125ce80e0f7c3c99f96b0734bd5dedac83e2bae85c40",
    "official20110311054624120_30/ANMO/event": "bd3c2a92943ffe4e69208db90030ebb35051a49245e9f4754a55d5fcc596adb5",
    "official20110311054624120_30/ANMO/quiet": "b838209a56be485777d7eb3c1e005791bf36c1187a451b630284b4bdb2f28682",
    "official20110311054624120_30/COLA/event": "5528c2fc51c66b73afff60c580c7eef1ce33a888a06589b10aefd6ae60146f51",
    "official20110311054624120_30/COLA/quiet": "db62cbae7949e8162b33b9212a45cea25ce28d58d29f431e3874092496fbfd25",
    "official20110311054624120_30/MAJO/event": "1546f5e6f85d01ed17d49f954ed009147cf67acad5f148c90ae720ca8684d81a",
    "official20110311054624120_30/MAJO/quiet": "224ada967076ae8f7e97ae71f02267a1f84f848d2c6ec3495d5b36f26fb40201",
    "official20120411083836720_20/ANMO/event": "2c641143da583edbbc17de7708d449811fe3cfc9863cfb18811d3ab6860a37a1",
    "official20120411083836720_20/ANMO/quiet": "797820f2c7994c5e8e4cfc05720391f168fb83e4d9d7cb6223524f362f5b9bae",
    "official20120411083836720_20/COLA/event": "4deefcde571b9d67ef874aa683ae60273f87152a47fd377644d7fccec28813ba",
    "official20120411083836720_20/COLA/quiet": "35ea963d3f557ed515373a4112b011847062d59a63356757f5515c07dc812f04",
    "official20120411083836720_20/MAJO/event": "392a8a10095289299942ffc688f2ca52f44460996fde957f5b13e5b83a98fbbd",
    "official20120411083836720_20/MAJO/quiet": "01b5d3b4d0d03149db45cb86167023930f60fb115336fe5c5fad1b92a4828899",
    "usb000h4jh/ANMO/event": "9a000c0be7b99c27a3944024da9e4417b16af37dcd933af968522ba97902a509",
    "usb000h4jh/ANMO/quiet": "08e8fccf438558bbda0b296981d810f5637656cee932d87e5bc4ac2723e6db75",
    "usb000h4jh/COLA/event": "08cac235c46df6bb416f8bcf85a8d500a6a30eb12154fbfb29498c541b88168e",
    "usb000h4jh/COLA/quiet": "652bec85b7211c5b489be68a05dbf8a4c51e2cca51c1e1b3d6cdc0e441af93c9",
    "usb000h4jh/MAJO/event": "60cd3ed53b77a8f6484a81c868eb64807ae787076a6d817dfef747ec559a1227",
    "usb000h4jh/MAJO/quiet": "0f208f358c7122744ca930c16027d25369d43e7a17ad9a063cbcda86389f3959",
    "usc000nzvd/COR/event": "e8917bc76d977c45648961108b9f000ea157e40da2eab1961b6eb20ecf696d2f",
    "usc000nzvd/COR/quiet": "704021fe2525fb532af608f692c371dda8931e253b82bcc8de1aa3af080fd1c1",
    "usc000nzvd/KONO/event": "e6191afc1c8f830076b3270cefa7a8739e432a96a5bc9647698902949ea0cfb6",
    "usc000nzvd/KONO/quiet": "b21f2dbb4030262c8f95c13b5f97050c4589f9a66f4a317ca059933dd0367efa",
    "usc000nzvd/HRV/event": "6f98c88c58e36b4daadc552c74166ebb4f72fe55b2d99ccf6d2e4bde303aaf40",
    "usc000nzvd/HRV/quiet": "b08e14000e43fdb66f6fb86cb8cd60814c8cf77e15eaccdfd424a6f3f93d6137",
    "us20003k7a/COR/event": "55407c2c59d1b6484e0346928dbc1d3120b2da91c4c7a27e1ddfdf45aa045d1b",
    "us20003k7a/COR/quiet": "403f09a8e9c7f712c2d8017096e085d640c730dd7fc369cec4e323b93653e62c",
    "us20003k7a/KONO/event": "ce322023bc30768b42b3391faad7db22ae99bb467283760c499d631d7fc6df7a",
    "us20003k7a/KONO/quiet": "5ad911b181ef7a2991904bede319c1bf19fb2257f263da545d4740b3edf6b888",
    "us20003k7a/HRV/event": "291e1d26db3bc0a082a6fe07b03b05ec8184ffdf19a9f9b0a63c6a956e41c62c",
    "us20003k7a/HRV/quiet": "6adad3686e2b2dd5d211df640b0304ef9448d4c4fa071b421b2475ff3c5aa3dd",
    "us2000ahv0/COR/event": "a59b3f7fd29c4703401e90ca10e35c21270f820c7bcec9eb167afdfefcd8ac36",
    "us2000ahv0/COR/quiet": "144a531d34cf0926944d9ac804293551d6ac9fa9bb80fa136b8b2b9f6b2f4dd1",
    "us2000ahv0/KONO/event": "b81f1252d49f6cff89039de2275d6ac5038771dad2c2ca5528785eac23e2aaaa",
    "us2000ahv0/KONO/quiet": "72847cb35ff97191fb51f595f1d4769dd465d09df0e2e7b2c96633c30c1ce782",
    "us2000ahv0/HRV/event": "4e6756c9510f4c8069bf5b0e75aefb6632bdaff073fc72e87bd0bb628aa33b08",
    "us2000ahv0/HRV/quiet": "7341a8277325cb2dd8c284309449621d7cf719b2e407e4c8a120127d1adb4b4c",
    "us60003sc0/COR/event": "0731d9721d6729559776409245ff014d5a8914764b36770f675647bd1419d757",
    "us60003sc0/COR/quiet": "48d2a59238a5f56ba007aebf5e3fb3f4444694371d723bafd27ac9172543450c",
    "us60003sc0/KONO/event": "b4094910d38567ce186a1e4f5612350551f1e59304b87c2e2eeafc4207ba0a87",
    "us60003sc0/KONO/quiet": "7d3361f8f74dcfff7c1bbb9c2e5f2c7ed804bf3bbaae183f57ba4e0ab226a346",
    "us60003sc0/HRV/event": "bab5de3ca953a782af9677c253dac6515aa31428e045376c699e5392befc607d",
    "us60003sc0/HRV/quiet": "a2d521283660d46014d31fa21ea5eda22b0840108ed5951c53a303c6285241ed",
}


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def request_url(event: Event, station: Station, label: str) -> str:
    """Return the exact three-component EarthScope GeoCSV request."""

    if label not in {"event", "quiet"}:
        raise ValueError("label must be 'event' or 'quiet'")
    start = _utc(event.origin_utc) + timedelta(seconds=WINDOW_OFFSET_SECONDS)
    if label == "quiet":
        start += timedelta(seconds=QUIET_OFFSET_SECONDS)
    end = start + timedelta(seconds=WINDOW_DURATION_SECONDS)
    query = urlencode(
        {
            "net": "IU",
            "sta": station.code,
            "loc": "00",
            "cha": "BH?",
            "starttime": _format_utc(start),
            "endtime": _format_utc(end),
            "format": "geocsv",
            "scale": "AUTO",
            "nodata": "404",
        }
    )
    return f"{EARTHSCOPE_DATASELECT}?{query}"


def protocol_records(split: str | None = None) -> list[dict[str, object]]:
    """Return the preregistered Cartesian pairs within each split."""

    records: list[dict[str, object]] = []
    for event in EVENTS:
        if split is not None and event.split != split:
            continue
        for station in STATIONS:
            if station.split != event.split:
                continue
            for label in ("event", "quiet"):
                key = f"{event.event_id}/{station.code}/{label}"
                records.append(
                    {
                        "key": key,
                        "event": event,
                        "station": station,
                        "label": label,
                        "url": request_url(event, station, label),
                        "expected_sha256": EXPECTED_SHA256.get(key),
                    }
                )
    return records


def _cache_path(url: str, cache_dir: Path | None) -> Path:
    root = (
        cache_dir
        if cache_dir is not None
        else Path.home().joinpath(".cache", "taskbutterfly", "earthscope-v1")
    )
    return root.joinpath(f"{hashlib.sha256(url.encode()).hexdigest()}.csv")


def fetch_payload(
    url: str,
    *,
    expected_sha256: str | None,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> tuple[bytes, str]:
    """Fetch, cache, and checksum one immutable protocol response."""

    path = _cache_path(url, cache_dir)
    if path.exists():
        payload = path.read_bytes()
    elif offline:
        raise FileNotFoundError(f"waveform not cached for offline run: {url}")
    else:
        with urlopen(url, timeout=90) as response:
            payload = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(
            f"EarthScope checksum mismatch: expected {expected_sha256}, got {digest}"
        )
    return payload, digest


def parse_geocsv(payload: bytes) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, str]]]:
    """Parse segmented multi-channel GeoCSV into timestamped component streams."""

    streams: dict[str, list[tuple[float, float]]] = {}
    metadata: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}
    reading_data = False
    for line in payload.decode("utf-8").splitlines():
        if line.startswith("# dataset:"):
            current = {}
            reading_data = False
        elif line.startswith("# "):
            key, separator, value = line[2:].partition(":")
            if separator:
                current[key.strip()] = value.strip()
                sid = current.get("SID")
                if sid is not None:
                    metadata.setdefault(sid, {}).update(current)
        elif line.startswith("Time,"):
            reading_data = True
        elif reading_data and line:
            row = next(csv.reader(io.StringIO(line)))
            if len(row) < 2 or "SID" not in current:
                continue
            timestamp = _utc(row[0].strip()).timestamp()
            try:
                sample = float(row[1])
            except ValueError:
                continue
            streams.setdefault(current["SID"], []).append((timestamp, sample))

    parsed: dict[str, tuple[np.ndarray, np.ndarray, dict[str, str]]] = {}
    for sid, pairs in streams.items():
        ordered = sorted(set(pairs))
        parsed[sid] = (
            np.asarray([pair[0] for pair in ordered], dtype=float),
            np.asarray([pair[1] for pair in ordered], dtype=float),
            metadata[sid],
        )
    return parsed


def _component_order(sid: str) -> int:
    suffix = sid.rsplit("_", 1)[-1]
    order = {"BHE": 0, "BH1": 0, "BHN": 1, "BH2": 1, "BHZ": 2}
    return order.get(suffix, 99)


def resample_three_components(
    streams: dict[str, tuple[np.ndarray, np.ndarray, dict[str, str]]],
    *,
    sample_rate_hz: float = TARGET_SAMPLE_RATE_HZ,
    maximum_gap_seconds: float = 1.0,
    minimum_common_samples: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Interpolate three components to a common 20-Hz grid with a gap gate."""

    candidates = sorted(
        (sid for sid in streams if _component_order(sid) < 99),
        key=_component_order,
    )
    selected: list[str] = []
    used_orders: set[int] = set()
    for sid in candidates:
        order = _component_order(sid)
        if order not in used_orders:
            selected.append(sid)
            used_orders.add(order)
    if len(selected) != 3 or {_component_order(sid) for sid in selected} != {0, 1, 2}:
        raise RuntimeError(f"expected two horizontals and one vertical; found {candidates}")

    start = max(float(streams[sid][0][0]) for sid in selected)
    end = min(float(streams[sid][0][-1]) for sid in selected)
    count = int(np.floor((end - start) * sample_rate_hz)) + 1
    required_samples = (
        int(0.95 * WINDOW_DURATION_SECONDS * sample_rate_hz)
        if minimum_common_samples is None
        else int(minimum_common_samples)
    )
    if required_samples < 1:
        raise ValueError("minimum_common_samples must be positive")
    if count < required_samples:
        raise RuntimeError(f"insufficient common coverage: {count} samples")
    grid = start + np.arange(count, dtype=float) / sample_rate_hz
    components = []
    maximum_gaps = []
    native_rates = []
    instruments = []
    for sid in selected:
        times, values, meta = streams[sid]
        gaps = np.diff(times)
        maximum_gap = float(np.max(gaps, initial=0.0))
        if maximum_gap > maximum_gap_seconds:
            raise RuntimeError(f"{sid} contains a {maximum_gap:.3f}-s gap")
        components.append(np.interp(grid, times, values))
        maximum_gaps.append(maximum_gap)
        native_rates.append(float(meta["sample_rate_hz"]))
        instruments.append(meta.get("instrument", "unknown"))
    return np.vstack(components), {
        "selected_sids": selected,
        "common_samples": count,
        "target_sample_rate_hz": sample_rate_hz,
        "native_sample_rates_hz": native_rates,
        "maximum_gaps_seconds": maximum_gaps,
        "instruments": instruments,
        "scale_units": [streams[sid][2].get("scale_units") for sid in selected],
    }


def load_record(
    record: dict[str, object],
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    payload, digest = fetch_payload(
        str(record["url"]),
        expected_sha256=record.get("expected_sha256"),
        cache_dir=cache_dir,
        offline=offline,
    )
    components, quality = resample_three_components(parse_geocsv(payload))
    return components, {
        "key": record["key"],
        "label": record["label"],
        "url": record["url"],
        "sha256": digest,
        "bytes": len(payload),
        **quality,
    }


def multicomponent_windows(
    components: np.ndarray,
    *,
    samples_per_component: int = 64,
    stride: int = 64,
    padded_dimension: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return shape-only and physical-amplitude three-component windows."""

    values = np.asarray(components, dtype=float)
    if values.shape[0] != 3:
        raise ValueError("components must have shape (3, samples)")
    used = 3 * samples_per_component
    if padded_dimension < used or padded_dimension & (padded_dimension - 1):
        raise ValueError("padded_dimension must be a power of two covering 3 components")
    shape_windows = []
    amplitude_windows = []
    for start in range(0, values.shape[1] - samples_per_component + 1, stride):
        block = values[:, start : start + samples_per_component].copy()
        block -= np.mean(block, axis=1, keepdims=True)
        vector = np.zeros(padded_dimension, dtype=float)
        vector[:used] = block.ravel()
        amplitude_windows.append(vector)
        norm = np.linalg.norm(vector)
        if norm > np.finfo(float).eps:
            shape_windows.append(vector / norm)
    return np.column_stack(shape_windows), np.column_stack(amplitude_windows)
