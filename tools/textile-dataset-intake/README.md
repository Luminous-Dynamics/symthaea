# Textile Dataset Intake v0.1

Standalone Rust CLI for offline, fail-closed verification of local textile benchmark datasets.

## Safety and claim boundaries

- No network access or dataset download is implemented. The tool never fetches a URL.
- It does not decide legal rights. A human review must set rights.status to reviewed, name the reviewer and date, record the exact license-text SHA-256, include the matching license file as a role=license_text inventory entry, bind its local path, and explicitly list permitted operations before a receipt can be issued.
- Pending and rejected rights states fail closed.
- The verifier checks local file path containment, source-provided SHA-256, file length, duplicate source/local paths and the exact manifest bytes.
- It rejects absolute paths, parent traversal and symlink escapes from the dataset root.
- Raw data belongs outside Git. Commit only manifests, schemas, scripts and benchmark results that may lawfully be published.
- A receipt proves only that the declared local files match the declared manifest at verification time. It does not prove upstream labels, specimen chemistry, sensor calibration, permission beyond the recorded review, or model accuracy.

## Manifest template

manifest.template.json is prefilled for NIST NIR-SORT 2.0 source metadata, but deliberately has rights.status = pending and no files. It cannot generate a receipt until a human completes the rights fields and inventories local files with source checksums.

NIST's public catalog points to NIR-SORT data at https://data.nist.gov/od/id/mds2-3325, identifies the NIST open-data license, and provides a Data Framework, Data Dictionary, sample information, source directories and per-file checksum sidecars. Read and preserve the dataset's actual license statement, its attribution requirement and its no-warranty terms before using or redistributing data.

## Commands

~~~sh
cargo test --manifest-path tools/textile-dataset-intake/Cargo.toml

cargo run --manifest-path tools/textile-dataset-intake/Cargo.toml -- \
  verify \
  --manifest /path/to/manifest.json \
  --root /path/to/local-dataset \
  --operation benchmark_reproduction \
  --out /path/to/intake-receipt.json

cargo run --manifest-path tools/textile-dataset-intake/Cargo.toml -- \
  verify-receipt \
  --manifest /path/to/manifest.json \
  --root /path/to/local-dataset \
  --receipt /path/to/intake-receipt.json
~~~

cargo run --manifest-path tools/textile-dataset-intake/Cargo.toml -- \
  inspect-nir-sort-name \
  --file /path/to/A01_a100x000x000_00_00_N_01A.csv
~~~

The receipt is deterministic for a fixed manifest, operation and file inventory. A change to any manifest byte or input file invalidates the receipt.

## NIR-SORT v2 intake recommendation

Use the NIST Data Framework naming convention and maintain distinct identifiers for:
- material source, specimen and batch;
- nominal fibre blend and fabric form factor;
- instrument (Benchtop_N, Handheld_F, Handheld_S, Standoff_Q);
- acquisition replicate and measurement mode (absorbance, reflectance, colour, mono, polarized);
- data dictionary and sample-information version.

Do not split replicates from the same specimen across train/test. A separate held-out-instrument evaluation is required. Treat the 61 custom-blend specimens as a ratio-generalization challenge, and keep the different device/wavelength profiles separate until transfer is measured.

## Status

Unit tests use tiny synthetic files created in temporary directories. No NIST or TexSortML raw data is checked in or claimed to have been downloaded.


## NIR-SORT filename parser

The `inspect-nir-sort-name` command decodes source class, nominal fibre slots,
batch/lot, fabric form, instrument, replicate, acquisition mode and standoff
distance. Its stable `specimen_group_key` excludes instrument and replicate so
all acquisitions for the same nominal specimen can remain in one partition.
The nominal fibre percentages are filename metadata only; use the NIST sample
information and independent reference measurements as composition truth.
