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

## Query-specific stopping snapshot audit

The `connectome-stopping-audit` binary validates a JSONL series against the initial versioned snapshot contract in `docs/neuroscience/CONNECTOME_STOPPING_SNAPSHOT.schema.json`. It checks immutable dataset/query/profile identity, strictly increasing snapshot sequence, parent graph digest continuity, nondecreasing edit counts, operation-class totals, invariant names, query metric names/units, and digest formatting. It also emits a descriptive tail-range stability signal for graph invariants and declared query metrics.

```sh
cargo test --manifest-path tools/connectomics-intake-audit/Cargo.toml
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml --bin connectome_stopping_audit -- \
  --input snapshot-series.jsonl \
  --tail-window 3 \
  --relative-tolerance 0.01 \
  --absolute-tolerance 1e-9 \
  --output stopping-audit.json
```

Each JSONL record is one snapshot object conforming to `CONNECTOME_STOPPING_SNAPSHOT.schema.json`. Order records from oldest to newest. The first snapshot has sequence index 0 and no previous graph digest; each later snapshot names the immediately preceding snapshot's graph digest. Cumulative operation-class counts must sum to `cumulative_committed`. An illustrative three-snapshot fixture is at `docs/neuroscience/examples/CONNECTOME_STOPPING_SNAPSHOTS.example.jsonl`; it intentionally shows graph invariants looking stable while a query metric changes, and uses fake data/digests only.

**Important:** the tool always reports `CONTINUE_REVIEW`; its status is only `VALIDATED_NOT_STOPPED` when consistency checks pass. Tail stability is descriptive, not an asymptotic forecast or proof of reconstruction correctness. The schema contains no independently calibrated residual high-impact-error risk, and global graph invariants can stabilize while local scientific errors persist. A valid automatic stopping gate requires separate risk evidence and held-out evaluation; this binary does not make that decision.


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
- The schema-version-4 report includes dataset revision commit SHA, pinned upstream code commit, policy ID, frozen profile digest, evaluation partition, input SHA-256, deterministic cluster-bootstrap 95% intervals for balanced accuracy, review-budget yield, and risk–coverage checkpoints. Review-budget yield and risk–coverage are reported overall and per `group`, at the same item/minute budgets and coverage targets. Three matched-budget control policies are also scored on the same frozen rows: `uncertainty_only` (least confident first), `probability_only` (highest predicted error probability first), and `deterministic_random` (SHA-256 keyed order from the frozen input digest, group ID, and sample ID). Control metrics are computed overall and per group, and the random baseline is deterministic for replay.

For every control and budget, the report also estimates the paired difference **submitted priority minus control** using a deterministic cluster bootstrap over the same resampled clusters for both policies. It reports 95% intervals separately for impact-weighted error recall, ordinary error recall, and precision. Item budgets are recalculated at the same requested fraction in each bootstrap replicate; minute budgets use the fixed supplied minute cap. Each metric reports its valid replicate count and is marked not estimable unless at least 80% of requested replicates are valid and at least two dependency clusters exist. Intervals are conditional on the submitted frozen rankings and the deterministic-random seed; they do not account for re-training, policy selection on the lockbox, or incorrect cluster assignments.

Per-group deltas help stop strong pooled scores from hiding a weak species × error-family or a queue policy that fails to beat simple alternatives. Duplicate sample IDs fail closed before scoring; aggregate multiple image views at the source-operation level. Balanced-accuracy intervals and paired review-budget delta intervals are provided; AUROC, average precision, Brier score, and log loss currently have point estimates only. The scorer does not validate Parquet labels or certify cluster independence. It always emits `SCORED_NOT_PROMOTED`; model promotion still requires the locked split protocol and independent grouped statistical analysis.

## Exit codes

- `0`: all declared checks pass.
- `1`: manifest/data integrity gate failed; a JSON report is written when possible.
- `2`: command/input parsing or report-writing error.

The tool deliberately does not auto-correct labels, task names, or file hashes. Fix the source or have the mapping reviewed, update the manifest, then rerun.
