# Connectomics Intake Audit

Small, standalone Rust gate for the ConnectomeBench2 V2 intake process described in the Symthaea neuroscience research plan.

## Scope

This tool validates the declared manifest, checks task/label/input-mode consistency, rejects known EM-view leakage fields, hashes every declared local artifact with SHA-256, checks byte sizes and split declarations, and writes a deterministic JSON report.

It does **not** parse Parquet schemas, prove label correctness, validate the upstream commit contents, prove samples are independent, or measure model accuracy. A passing report only means the manifest and local bytes satisfy these checks.

## Build

This directory is a standalone Cargo workspace and is intentionally not a member of the main Symthaea workspace.

```sh
cargo test --manifest-path tools/connectomics-intake-audit/Cargo.toml
cargo build --release --manifest-path tools/connectomics-intake-audit/Cargo.toml
```

For reproducible CI, generate and commit this crate's `Cargo.lock`, then add `--locked` to both commands.

## Run

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml -- \
  --manifest path/to/intake-manifest.json \
  --root path/to/local-dataset-artifacts \
  --output intake-audit-report.json
```

Every `input_files[].path` is relative to `--root`. Absolute paths, parent-directory traversal, symlink escapes, missing files, byte-count mismatches, and SHA-256 mismatches fail closed. Keep dataset license/source metadata with the manifest.

## Exit codes

- `0`: all declared checks pass.
- `1`: manifest/data integrity gate failed; a JSON report is written when possible.
- `2`: command/input parsing or report-writing error.

The tool deliberately does not auto-correct labels, task names, or file hashes. Fix the source or have the mapping reviewed, update the manifest, then rerun.
