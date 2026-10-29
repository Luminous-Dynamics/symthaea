# Connectomics Intake Audit

Small, standalone Rust gate for the ConnectomeBench2 V2 intake process described in the Symthaea neuroscience research plan.

## Scope

This tool validates the declared manifest, checks task/label/input-mode consistency, rejects known EM-view leakage fields, requires the correct single-/dual-segment geometry preprocessing profile, hashes every declared local artifact with SHA-256, checks byte sizes and split declarations, and writes a deterministic JSON report.

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

## Manifest contract

- Dataset ID is `jeffbbrown2/ConnectomeBench2`; dataset revision must be `v2` (not `main`). Record the exact dataset-revision commit SHA in the manifest.
- The upstream code commit is pinned to `5679b39aa2641d8d0afeed88b441e696f87a66df`. This pin was reviewed against `src/neurips_training/data.py::route_row` and `scripts/data_generation/neurips/postprocess_giga_parquet.py::derive_task_routing`. Change the pin only after auditing the replacement source commit and updating the code/schema contracts together.
- `false_split_correction` uses `false_split_correction_label = same_neuron` and dual-segment `geometry` for merge correction.
- `false_merge_identification` uses `false_merge_identification_label = !same_neuron` and single-segment `geometry_single` for split correction.
- Allowed model inputs are `geometry`, `geometry_single`, `em_xy`, `em_xz`, and `em_yz`. `em_best`, labels, IDs, routing, and metadata are not accepted as model inputs.
- Single-segment geometry declares `geometry_preprocessing_profile=single_segment_zero_mask_channels` to mirror the reference loader's defensive mask-zeroing step. Dual geometry declares `dual_segment_native`.
- If EM slices are inputs, the preprocessing profile must declare A/B fusion and zeroing B for single-segment input, or native dual masks for dual-segment input.

## Exit codes

- `0`: all declared checks pass.
- `1`: manifest/data integrity gate failed; a JSON report is written when possible.
- `2`: command/input parsing or report-writing error.

The tool deliberately does not auto-correct labels, task names, or file hashes. Fix the source or have the mapping reviewed, update the manifest, then rerun.
