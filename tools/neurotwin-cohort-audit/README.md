# NeuroTwin cohort audit (v0.1)

A small Rust-only, local BIDS metadata preflight for NeuroTwin research intake.

## What it does

- Requires and parses the BIDS `dataset_description.json` file.
- Checks required `Name` and `BIDSVersion` metadata.
- Discovers subject directories and counts session directories.
- If `participants.tsv` is present, validates the `participant_id` column, duplicate IDs and subject-directory coverage.
- Counts files by recognized BIDS modality directory.
- Checks that a derivative dataset declares non-empty `GeneratedBy`.
- Emits a JSON receipt with aggregate counts and stable finding codes.

## Privacy properties

The tool never opens neuroimaging / EEG file contents. It reads the dataset description, directory metadata and the participant identifier column needed to compare subject directories. It does not emit subject labels, filenames, paths, participant-table values, age, sex, diagnosis, clinical outcomes, or file hashes.

Run it inside the approved data environment. Do not redirect output into a public repository if the aggregate cohort counts or modality breakdown are themselves restricted by the data-use agreement.

## Build and run

```bash
cargo test --manifest-path tools/neurotwin-cohort-audit/Cargo.toml
cargo run --release --manifest-path tools/neurotwin-cohort-audit/Cargo.toml -- /secure/path/to/bids-dataset
```

Exit codes: `0` for preliminary pass or review, `2` for a blocked receipt or execution error, and `64` for usage error. A `REVIEW` status is not a pass; it means one or more warning-level findings exist. An absent `participants.tsv` is a warning because BIDS 1.11.2 marks it recommended rather than mandatory.

## Status semantics

- `PRELIMINARY_PASS`: no structural issue found by this limited preflight.
- `REVIEW`: no blocking issue found, but at least one warning exists.
- `BLOCKED`: a blocking metadata/structure issue was found.

The receipt contract is published at `schema/cohort-receipt-v0.1.schema.json`.

This tool targets BIDS 1.11.2 as a supported stable baseline, while accepting and reporting the version declared by each dataset. See the [BIDS 1.11.2 specification](https://bids-specification.readthedocs.io/en/v1.11.2/), [dataset description requirements](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files.html), and [participants file requirements](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/data-summary-files.html).

## Explicit limitations

This tool is not the official BIDS Validator and does not validate all BIDS entity rules, JSON sidecars, TSV schemas, modality-specific metadata, image integrity, scanner quality, task labels, acquisition provenance, consent, ethics approval, licences, data-use agreements, cohort appropriateness, clinical endpoints, leakage, or statistical validity. Use the official BIDS Validator as a separate gate.

This receipt is not authorization for data use, model training, or clinical deployment.
