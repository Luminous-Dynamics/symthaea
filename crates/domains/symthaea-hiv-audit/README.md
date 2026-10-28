# Symthaea HIV Reservoir Audit

This Rust CLI adds a reproducible, fail-closed first step for the HIV Reservoir Discovery benchmark. It does **not** fetch data, fit models, interpret biology, or validate a treatment.

## Acquire the public Dryad deposit

From the Symthaea repository root, download and extract the current public deposit with a strict file allowlist (requires curl, unzip, sha256sum and standard shell utilities):

    bash scripts/research/acquire_hiv_reservoir_dryad.sh /path/to/dryad-files

The downloader records the source URL, retrieval time and downloaded archive SHA-256 beside the extracted files. It refuses to overwrite an existing archive/provenance file and extracts only the exact registered archive members. Official API documentation: https://github.com/datadryad/dryad-app/blob/main/documentation/apis/README.md

## Build and run

From the Symthaea repository root, after acquiring the registered Dryad files into a clean directory, run:

    cargo test -p symthaea-hiv-audit
    cargo run -p symthaea-hiv-audit -- capture \
      --manifest docs/research/hiv-reservoir-discovery/datasets.json \
      --dataset dryad_k3j9kd5dw \
      --data-dir /path/to/dryad-files \
      --output /path/to/receipts/dryad_k3j9kd5dw.json \
      --source-commit "$(git rev-parse HEAD)"

The registered filenames must match the source deposit. The receipt is deliberately written outside the dataset directory.

After storing the receipt in a controlled, versioned location, verify it later:

    cargo run -p symthaea-hiv-audit -- verify \
      --manifest docs/research/hiv-reservoir-discovery/datasets.json \
      --dataset dryad_k3j9kd5dw \
      --data-dir /path/to/dryad-files \
      --receipt /path/to/receipts/dryad_k3j9kd5dw.json

## What is checked

- The dataset ID is explicitly present in the manifest and has an HTTPS source URL.
- Every registered file exists; unregistered CSV files in the data directory fail closed.
- SHA-256 and byte length are recorded for every registered file.
- Every CSV parses with a fixed row shape, a unique/nonempty header, and uuid as the first column.
- Empty or duplicated participant UUIDs block receipt capture.
- The tool records row/column counts, hashes the sorted UUID set without storing raw IDs, and records pairwise joins/overlaps between files.
- Verification recomputes all checks and fails if the current files differ from the receipt.

The source publication reports 191 participants, but assay files can have different sample coverage. The tool therefore reports per-file counts and pairwise overlap without assuming every file has all 191 participants. Researchers must investigate missingness and joins rather than forcing them to match.

## Receipt semantics

A captured receipt is a local integrity snapshot, not an upstream-authoritative checksum. It cannot prove that the data were downloaded from the cited source, that the source publication is correct, or that a result reproduces the paper. Keep the original archive, source URL/version, retrieval date, and receipt under versioned provenance. The captured source commit is required and must be a hexadecimal Git commit identifier.

No participant identifiers are written to the receipt; only counts and a hash of the sorted UUID set are stored.

## Current limit

The current execution environment could not resolve external DNS, so no biological files were acquired or analysed during this change. The CLI is a prepared executable path to the M0 data-quality gate; compile/test results must be recorded from an actual Rust environment before it is called validated.
