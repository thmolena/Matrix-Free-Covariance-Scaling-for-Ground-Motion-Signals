# Data provenance and licenses

## Software

The source code in `code/` is released under the BSD 3-Clause License. The
full license text is in `LICENSE` and `code/LICENSE`.

## Waveforms

The experiment retrieves waveform responses from the EarthScope FDSN
Dataselect web service. Waveform bytes are not copied into this repository or
its release archives. Each request URL, returned channel metadata, byte count,
and SHA-256 digest is frozen in `seismic_dataset.py` and the primary locked
JSON. Users must comply with EarthScope and network-specific terms and cite
the IU Global Seismographic Network as required by the provider.

## Earthquake catalog

Earthquake identifiers and origin metadata come from the U.S. Geological
Survey ComCat/FDSN Event service. The source freezes the identifiers used to
form the train/test split.

## Generated results

The numerical JSON file, manuscript tables, and plots are generated from
the licensed code and provider-fetched inputs. Timing values are
machine-specific. Scientific-metric verification deliberately tolerates
minor floating-point variation and excludes timing from exact acceptance.

Before public release or journal submission, the author must recheck the
providers' current attribution, citation, and reuse terms.
