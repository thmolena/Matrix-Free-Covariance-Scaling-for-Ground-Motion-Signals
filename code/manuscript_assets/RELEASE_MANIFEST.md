# Release manifest

Release date: 2026-07-31  
Package version: 2.1.0

## Frozen SHA-256 digests

| Artifact | SHA-256 |
|---|---|
| `main.tex` | `03b92236d986b6a2f938f7fae2d3d47273ca5afc55269c0014d830e9d65d2793` |
| `main.pdf` | `a3c63621278391c3884b6a71132c40adfa596e5908463d6c60d73309ca6991a3` |
| `code/manuscript_assets/supplement.tex` | `9015042f8b2072586fd801cb78d4bfb34fc4166d390f773c9d7384209e53f066` |
| `code/manuscript_assets/supplement.pdf` | `8423e2cd16c54cf33e4e08a0b444888da6f5d9c5b865b9b3421721eb3c868736` |
| `code/manuscript_assets/main.bbl` | `b4128e5ae6232f8674a0d440afd929049d27e12e2e15c145623201acdc5f5024` |
| `code/manuscript_assets/references.bib` | `d7d8180ab3f456ab3f64d504ff040421406264b7ef38cb5379f7e540d5fd97dd` |
| `code/src/taskbutterfly/results/seismic_whitening_results.json` | `49d8f3d4b10b1da235735bf9fdacb2862400b4d69813c46ec5eded8050e997cc` |
| `code/results/locked_results.json` | `8849fb035acc4c13c3a3c55da0a251d52e2d0ce9de4f6828efbfad423780164b` |
| `code/manuscript_assets/figures/web_overview.png` | `9a81b28f6dba431c73e30d012b3ab6ae3b9a6e878f817c6ed11d902391440a56` |

Timing and machine-provenance fields in a newly reproduced JSON are expected
to differ. `--verify` compares all nonvolatile protocol and scientific fields,
including the 75-member family-wise allocation, rejection decisions,
certificates, baseline metrics, structural diagnostics, and approximation
errors.

## Validation record

- Test suite: 23 passed on CPython 3.13.12.
- Full cached/offline EarthScope replay with `--verify`: passed.
- Verification checks 75 distinct certification streams and the allocation
  `0.05/(2*75)` to every quadratic-form tail.
- Website certificate plot regenerated directly from the locked JSON.
- Public release-structure validator: passed.
- `main.pdf`: 25 US-letter pages; all pages rasterized and visually inspected.
- `supplement.pdf`: 3 US-letter pages; all pages rasterized and visually
  inspected.
- Both TeX sources compile with Tectonic; no unresolved citation or reference
  remains.

This manifest covers the live source package only. It does not claim that a
wheel, source distribution, submission ZIP, DOI deposit, remote push, or
journal upload was produced in this repair pass.
