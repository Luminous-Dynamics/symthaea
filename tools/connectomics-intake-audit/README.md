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

## Frozen prediction scorer

A second binary, `proofreading-score`, evaluates **frozen** predictions and independently adjudicated labels. It is deliberately separate from the runtime model/ranker. It currently consumes JSONL so metrics can be unit-tested without downloading raw EM or decoding Parquet.

Build and test:
```sh
cargo test --manifest-path tools/connectomics-intake-audit/Cargo.toml
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml --bin proofreading-score -- \
  --input predictions.jsonl \
  --mapping-reference https://example.org/reviewed-label-map \
  --dataset-revision-commit-sha <40-character-hub-commit-sha> \
  --policy-id specialist-v1 \
  --prediction-profile-sha256 <64-character-config-sha256> \
  --evaluation-partition development \
  --review-time-basis synthetic-proxy \
  --bootstrap-replicates 500 \
  --output score-report.json
```

The input JSONL contains one row per unique source operation:
```json
{"sample_id":"source-op-001","group":"mouse:merge","cluster_id":"mouse-cube-017","label_is_error":true,"p_error":0.93,"review_priority_score":0.94,"review_minutes":2.0,"impact_weight":4.0}
{"sample_id":"source-op-002","group":"fly:split","cluster_id":"fly-cube-204","label_is_error":false,"p_error":0.12,"review_priority_score":0.07,"review_minutes":1.5,"impact_weight":0.0}
```

- `cluster_id` is the declared dependency unit for uncertainty estimation (e.g. source volume/spatial cube/neuron); choose the unit that matches the sampling and intended generalization claim. Multiple rows from one cluster are resampled together. A cluster ID is not proof that clusters are independent.
- `label_is_error` is an **evaluation-only** adjudicated label, normalized so `true` means a genuine reconstruction error requiring correction. `mapping_reference` must point to the reviewed task/label contract.
- `p_error` is the model's error probability; the scorer reports balanced accuracy at 0.5, AUROC, tie-aware average precision, Brier score, log loss, and group-level metrics.
- `review_priority_score` is the frozen policy ranking for which candidate an expert should review next. The evaluator measures error yield and impact-weight capture at fixed item and review-minute budgets.
- `impact_weight` is ground-truth evaluator metadata. It must never be a runtime input to the ranker or the candidate model.
- `review_minutes` may be prospectively measured or a synthetic proxy. The required `--review-time-basis` makes that distinction explicit; synthetic proxies cannot support a real-world time-saving claim.
- The report includes dataset revision commit SHA, pinned upstream code commit, policy ID, frozen profile digest, evaluation partition, input SHA-256, group breakdowns, deterministic cluster-bootstrap 95% intervals for balanced accuracy, review-budget yield, and risk–coverage checkpoints. Bootstrap resamples caller-declared clusters, uses a seed derived from the frozen input digest, and marks intervals not estimable when too few valid replicates or fewer than two clusters exist. The status is always `SCORED_NOT_PROMOTED`; passing metrics do not by themselves qualify a model.
- Duplicate sample IDs fail closed. Aggregate multiple image views to the source-operation level before scoring.
- Cluster-bootstrap intervals cover balanced accuracy only; AUROC, average precision, and calibration metrics currently have point estimates only. The scorer does not validate Parquet labels or certify cluster independence. It is a metric calculator for a frozen output file; scientific claims still require the locked split protocol and independent grouped statistical analysis.

## Exit codes

- `0`: all declared checks pass.
- `1`: manifest/data integrity gate failed; a JSON report is written when possible.
- `2`: command/input parsing or report-writing error.

The tool deliberately does not auto-correct labels, task names, or file hashes. Fix the source or have the mapping reviewed, update the manifest, then rerun.
