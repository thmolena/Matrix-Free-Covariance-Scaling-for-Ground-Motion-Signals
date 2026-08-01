# Station-Referenced Streaming Energy Ratios for Early Ground-Motion Screening

This repository contains the manuscript, authenticated waveform protocol, executable method, theorem tests, locked numerical results, and publication figures for a station-referenced streaming energy statistic.

The earlier covariance-compression hypothesis failed its action-error certificate. The redesigned method preserves physical amplitude and normalizes it by a station-matched historical quiet trace. After development on eight earthquakes, the locked rule selects eight 3.2-second blocks. Confirmation uses four later magnitude-8.0 or larger earthquakes and three stations absent from development.

## Locked result

At the registered 25.6-second prefix, the station-reference statistic attains:

- balanced accuracy: 0.792;
- sensitivity: 0.667;
- specificity: 0.917;
- area under the ROC curve: 0.833;
- median apply time: 56.1 microseconds;
- numerical streaming state: three scalar values.

Raw mean energy and the raw 0.9-energy quantile each attain balanced accuracy 0.708 and area 0.833. The paired 95 percent bootstrap interval for the balanced-accuracy difference is `[-0.083, 0.250]`; the directional improvement is not statistically resolved.

The semantic result hash is:

```text
6121bb9eaa0affe7f9b3fb7e5973e552260f180f8c4a13e2b86042ebad93c655
```

## Data contract

The study references 108 original EarthScope FDSN responses totaling 66,981,395 bytes. USGS ComCat supplies the twelve event identifiers and metadata. Each locked record contains the source URL, SHA-256 digest, byte count, selected channel identifiers, scale units, and common sample count. Provider waveforms are not redistributed.

The partitions are disjoint by earthquake and station:

| Role | Earthquakes | Stations | Pairs |
|---|---:|---:|---:|
| Development | 8 | 6 across two historical station groups | 24 |
| Confirmation | 4 | TATO, PAB, CASY | 12 |

Each pair uses an event-aligned candidate, a one-day-offset reference, and an independent two-day-offset negative.

## Reproduction

Create an isolated environment and install the package:

```bash
cd code
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,figures]'
python -m pytest -q
```

An online run downloads or reuses the registered responses and verifies every digest:

```bash
cd ..
station-energy-study --verify
```

After the cache is populated, the scientific result can be reproduced without network access:

```bash
station-energy-study --offline --verify
```

The default cache is `~/.cache/taskbutterfly/earthscope-v1`. A different cache may be supplied with `--cache-dir PATH`.

## Artifact map

- `main.tex` and `main.pdf`: blank-page manuscript rewrite and compiled paper.
- `code/src/taskbutterfly/streaming_energy.py`: streaming statistic and perturbation bound.
- `code/src/taskbutterfly/streaming_study.py`: frozen development, confirmation, timing, tables, and figures.
- `code/tests/test_streaming_energy.py`: batch equivalence, gain invariance, sharp bound, and threshold tests.
- `code/results/streaming/streaming_study.json`: complete locked result.
- `code/results/streaming/prefix_metrics.csv`: latency-performance curve.
- `code/results/streaming/confirmation_pairs.csv`: every confirmation score and decision.
- `code/manuscript_assets/figures/`: PDF and PNG publication figures.

## Scope

The study concerns fixed catalog windows beginning 900 seconds after event origin. It is not an operational phase picker, event associator, or universal detector. Four confirmation event windows are missed and one independent quiet window produces a false alarm. These failures and the uncertainty interval remain visible in the manuscript and result files.

Repository code is licensed under BSD-3-Clause. EarthScope waveforms remain governed by their provider terms.
