# symthaea-matbench-folds

Audit-oriented Matbench v0.1 positional fold binding for Energy Discovery Benchmark Zero.

This bridge sits on top of `symthaea-matbench-gap`. Its public benchmark-slice entry point accepts the **exact compressed `matbench_expt_gap` bytes** again and re-runs the pinned parser before applying any fold semantics. Callers cannot hand the bridge a fabricated parsed dataset and have it treated as qualified truth.

## Split provenance

The fold manifest reproduces the published Matbench v0.1 split procedure for regression tasks:

- upstream commit: `936176db18ca4cd7b38cbd957c017a5bac770c6b`
- validation file: `matbench/matbench_v0.1_validation.json`
- validation Git blob: `ca9d0d157b31cadb7d99ba7c404cd96cfa09cad9`
- folds: 5
- shuffle: true
- random state: `18012019`
- procedure: `KFold`
- dataset rows: 4,604

Matbench's KFold operates on dataframe **row positions**. Its public `mb-expt-gap-*` labels are created separately from the dataframe's source index. The checked-in dataset-construction script sorts a newly constructed dataframe and shows a later `reset_index(drop=True)` line commented out, so this bridge deliberately does **not** infer official Matbench IDs from row position. Exact source-index-to-ID parity remains a separate artifact-audit gate.

The compact per-position fold manifest has Symthaea SHA-256:

`03a37eb4876e836878507c09559fefc55b1ff5f08db0c2229ac7dd80c0bffd7c`

The digest binds the dataset ID, upstream commit, upstream validation blob identity, split parameters, and all 4,604 positional fold-assignment bytes. The Rust implementation also freezes a 20-index legacy-NumPy permutation prefix independently reproduced against NumPy `RandomState` semantics.

Direct byte-for-byte comparison of the reproduced assignments with the 46 MB upstream validation JSON is still a separate qualification gate; the bridge says so explicitly.

## Leakage-clean test slices

`build_leakage_clean_test_fold_from_official_bytes(...)`:

1. validates the compact positional fold manifest;
2. parses only the exact pinned `matbench_expt_gap` artifact;
3. selects one published-procedure test fold by row position;
4. content-addresses the **complete current Symthaea band-gap training table**;
5. computes composition-level overlap against that training table;
6. removes overlapping compositions from evaluation truth;
7. content-addresses the positional exclusion mask, including the training-table digest;
8. content-addresses the retained truth slice;
9. emits one qualification SHA-256 binding source artifact, parent row-order identity, fold manifest, training-table identity, exclusion mask, retained truth, and row counts;
10. returns a `BandgapTruthSet` suitable for the existing measurement-only Benchmark Zero protocol.

The exclusion list preserves the exact source row position, normalized candidate ID, and matching Symthaea training labels. It intentionally does not invent an upstream Matbench ID.

Binding the full training table matters even when an edit produces no new overlap: the qualification identity changes whenever the leakage reference table changes.

## Network-free audit command

Once the exact pinned `matbench_expt_gap.json.gz` artifact is available locally, the crate provides:

`cargo run -p symthaea-matbench-folds --bin matbench-fold-audit -- /path/to/matbench_expt_gap.json.gz`

An optional final argument `0` through `4` audits only one fold. With no fold argument, all five are audited.

The CLI performs no network access. It emits path-independent JSON receipts containing the source/fold/training/mask/truth/qualification identities, counts, exact exclusions, and all epistemic disclosures. The local filesystem path is deliberately omitted from the receipt so the same source bytes yield shareable machine-independent output.

The command still does not run the screening model or compute Benchmark Zero performance metrics; it only qualifies leakage-clean truth slices.

## Important non-claims

A leakage-clean fold is **not an official Matbench leaderboard test set** because rows have been removed. It is a derived audit slice for Symthaea Benchmark Zero.

Reproducing the published positional split algorithm is not yet the same claim as extracting the exact fold IDs from the upstream validation JSON. That direct parity check remains pending.

Removing exact normalized-composition overlap also does **not** prove historical blindness. Public semiconductor knowledge may have influenced hand-written baselines, architecture choices, thresholds, or other prior model decisions.

## Qualification boundary

This crate adds no model training, fitting, candidate generation, experiment, solver execution, material certification, novelty claim, procurement, or deployment authority.

The PR should remain draft until the stacked workspace lock is reconciled under the pinned toolchain and exact-head check/test/strict-Clippy evidence exists.
