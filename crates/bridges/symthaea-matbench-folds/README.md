# symthaea-matbench-folds

Audit-oriented Matbench v0.1 fold binding for Energy Discovery Benchmark Zero.

This bridge sits on top of `symthaea-matbench-gap`. Its public benchmark-slice entry point accepts the **exact compressed `matbench_expt_gap` bytes** again and re-runs the pinned parser before applying any fold semantics. Callers cannot hand the bridge a fabricated parsed dataset and have it treated as qualified truth.

## Split provenance

The fold manifest is derived from the published Matbench v0.1 split procedure for regression tasks:

- upstream commit: `936176db18ca4cd7b38cbd957c017a5bac770c6b`
- validation file: `matbench/matbench_v0.1_validation.json`
- validation Git blob: `ca9d0d157b31cadb7d99ba7c404cd96cfa09cad9`
- folds: 5
- shuffle: true
- random state: `18012019`
- procedure: `KFold`
- dataset rows: 4,604

Matbench creates IDs from the original dataframe index before applying folds, so row 0 is `mb-expt-gap-0001`, row 4603 is `mb-expt-gap-4604`, and each row has exactly one test-fold assignment.

The compact per-row fold manifest has Symthaea SHA-256:

`03a37eb4876e836878507c09559fefc55b1ff5f08db0c2229ac7dd80c0bffd7c`

The digest binds the dataset ID, upstream commit, upstream validation blob identity, split parameters, and all 4,604 fold-assignment bytes.

## Leakage-clean test slices

`build_leakage_clean_test_fold_from_official_bytes(...)`:

1. validates the compact fold manifest;
2. parses only the exact pinned `matbench_expt_gap` artifact;
3. selects one published v0.1 test fold;
4. computes composition-level overlap against Symthaea's currently exposed band-gap training table;
5. removes those overlapping compositions from the evaluation truth;
6. content-addresses the exclusion mask;
7. content-addresses the retained truth slice;
8. returns a `BandgapTruthSet` suitable for the existing measurement-only Benchmark Zero protocol.

The exclusion list preserves the source row, Matbench ID, normalized candidate ID, and matching Symthaea training labels.

## Important non-claim

A leakage-clean fold is **not an official Matbench leaderboard test set** because rows have been removed. It is a derived audit slice for Symthaea Benchmark Zero.

Removing exact normalized-composition overlap also does **not** prove historical blindness. Public semiconductor knowledge may have influenced hand-written baselines, architecture choices, thresholds, or other prior model decisions.

## Qualification boundary

This crate adds no model training, fitting, candidate generation, experiment, solver execution, material certification, novelty claim, procurement, or deployment authority.

The PR should remain draft until the stacked workspace lock is reconciled under the pinned toolchain and exact-head check/test/strict-Clippy evidence exists.
