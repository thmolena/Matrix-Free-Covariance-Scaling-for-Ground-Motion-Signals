# Reproducibility package

The package implements station-referenced streaming energy ratios on checksum-verified EarthScope waveforms. The primary command is `station-energy-study`.

```bash
python -m pip install -e '.[test,figures]'
python -m pytest -q
cd ..
station-energy-study --offline --verify
```

The online form omits `--offline` and downloads missing registered responses. The default cache is `~/.cache/taskbutterfly/earthscope-v1`.

The semantic result hash is `6121bb9eaa0affe7f9b3fb7e5973e552260f180f8c4a13e2b86042ebad93c655`. Timing fields are recorded but excluded from semantic equality.

The retained `taskbutterfly-reproduce` command reproduces the earlier rejected covariance-compression experiment. It is preserved as a falsification record and is not the primary method of the rewritten manuscript.
