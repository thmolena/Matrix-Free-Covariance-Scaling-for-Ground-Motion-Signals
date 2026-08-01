# Data provenance

**Dataset:** EarthScope FDSN GeoCSV waveforms: 48 requests from eight magnitude-at-least-eight earthquakes.

**Original source:** https://service.earthscope.org/fdsnws/dataselect/1/

**Split:** Event- and station-disjoint held-out comparisons.

**Integrity:** The reproduction verifies the frozen SHA-256 list before analysis.

Run from the repository root:

```bash
python -m pip install -e code
python code/scripts/download_data.py
```

Downloaded third-party files remain governed by the source terms documented in
`THIRD_PARTY.md`. When redistribution is not explicit, the package fetches
the data into an external cache instead of committing the source bytes.
