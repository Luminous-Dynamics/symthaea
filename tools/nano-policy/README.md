# Nano Policy Prototype

A small, deterministic, Rust-first research utility for comparing candidate-selection policies under a fixed evaluation budget. It is a **standalone experiment prototype**, not yet a Symthaea cognitive subsystem, atomistic simulator, optical solver, or evidence of improved discovery.

## Scope

Version 0 supports two policies over numeric, pre-outcome candidate descriptors:

- **random** — seeded sampling without replacement.
- **maxmin** — seeded first candidate, then greedy farthest-point sampling using the distance to the nearest already selected candidate in standardized descriptor space.

Both policies start with the same seeded first candidate for a given input and seed. Exact duplicate feature rows are retained in the output audit but excluded from selection after the first occurrence. Constant feature columns contribute zero distance. Ties for maxmin are resolved by input order. A fixed SplitMix64 PRNG makes the random baseline independent of external crates.

This is a useful deterministic baseline and adapter seam. It does **not** implement HDC, active inference, spectral prediction, acquisition by hidden labels, calibrated uncertainty, or scientific physics. The benchmark must show that a more elaborate Symthaea policy beats these simple baselines before it deserves integration.

## Build and run

From the repository root:

    cargo test --locked --manifest-path tools/nano-policy/Cargo.toml
    cargo run --locked --manifest-path tools/nano-policy/Cargo.toml -- --input tools/nano-policy/examples/synthetic_candidates.tsv --policy maxmin --budget 4 --seed 42 > selection.tsv
    cargo run --locked --manifest-path tools/nano-policy/Cargo.toml -- --input tools/nano-policy/examples/synthetic_candidates.tsv --policy random --budget 4 --seed 42 > random.tsv

The example input is synthetic and is included to exercise parsing and selection plumbing only. It is not a material dataset and contains no physical labels.

## Input contract

Input is UTF-8 TSV with exactly one header row. The first header must be candidate_id; every remaining header is a numeric feature column using ASCII letters, digits, underscore, dot or hyphen. Each candidate ID must be unique, and feature values must be finite and have absolute value at most 1e12. Blank lines are ignored. Limits: 256 MiB input, 100,000 rows, 512 features, 20,000,000 feature cells and 100,000,000 estimated distance-feature operations.

Example header:

    candidate_id<TAB>period_nm<TAB>minimum_feature_nm<TAB>fill_fraction

Feature columns may contain only information that is legally and scientifically allowed **before** the candidate is selected for evaluation—for example design parameters or a frozen descriptor derived only from candidate geometry. Do not pass target spectra, output energies, outcomes, test-set labels, oracle observations not yet acquired, split membership that reveals outcomes, or features derived from them. In inverse design, the requested target can define the task, but candidate features must not smuggle the hidden measured response into the candidate matrix.

## Output and accounting

The program writes TSV to stdout with metadata comments and one row for every input candidate:

- selected rows include selection rank and nearest distance to a previously selected candidate at selection time;
- exact-feature duplicates are reported as duplicate_features with duplicate_of pointing to the first representative;
- eligible but unselected candidates remain present as not_selected.

Output order matches input order. Distances are standardized Euclidean descriptor-space diagnostics, **not** a physical distance, model confidence, predicted quality or probability of discovery. The first selection has no prior distance and is marked with a dash.

## Benchmark plan

Before using real data:

1. Audit source/data/checkpoint rights and freeze exact bytes and source revisions.
2. Reconcile candidate identity, duplicates/replicas and any data split; retain exclusions with reasons.
3. Run parsing and determinism tests on the pinned environment.
4. Compare random and maxmin under exactly the same pool, seed, budget and target-free input.
5. Add one Symthaea policy only behind an explicit allowed-input contract.
6. Evaluate returned candidates with a frozen scientific oracle, expose outcomes only after selection, and independently recalculate task metrics.

Report task utility versus oracle-query budget, diversity/coverage, duplicate fraction, valid-candidate rate, failure/abstention rate, wall time, peak memory and uncertainty across independent held-out groups. The prototype itself cannot claim utility or inverse-design success because it has no oracle labels.

## Status

Source code and unit tests are committed. Rust/cargo/rustfmt are unavailable in the authoring environment used for this commit, so compilation, tests, formatting, and Clippy are **not yet verified locally**. The dedicated GitHub Actions workflow is intended to perform those checks; queued/skipped runs must not be described as passing.
