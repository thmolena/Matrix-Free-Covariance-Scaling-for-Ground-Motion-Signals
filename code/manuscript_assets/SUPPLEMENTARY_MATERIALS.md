# Supplementary-material index

This index is intended to accompany the files uploaded as supplementary
material in the SISC submission system.

| Item | File | Purpose |
|---|---|---|
| Supplementary note | `supplement.pdf` | Complete rejection-capable algorithm, frozen split, seed-level simultaneous-certificate evidence, apply/adjoint probe diagnostics, and falsification criteria |
| Reproducible source | `supplementary-code.zip` | Installable Python source, tests, licenses, dependency lock, and locked numerical outputs |
| Primary locked result | `code/src/taskbutterfly/results/seismic_whitening_results.json` | Full machine-readable real-data protocol, hashes, seeds, baselines, uncertainty, structural diagnostics, and timing |
| Legacy locked result | `code/src/taskbutterfly/results/field_results.json` | Historical analytic low-pass sanity check, explicitly excluded from the novelty claim |

The waveform bytes are not redistributed. The code retrieves them from
EarthScope's FDSN service and verifies the 48 frozen SHA-256 hashes. USGS
ComCat identifiers provide the earthquake catalog provenance.

The author retains responsibility for confirming that every uploaded item is
covered by the repository BSD-3-Clause code license or by an identified
third-party source term. No executable, data, or text should be described as
publicly archived until the final files have been deposited at a persistent
public URL or DOI.
