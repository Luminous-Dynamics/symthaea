# Falsification power reference v0

A dependency-free Rust reference crate for the statistical power ceiling created by a fixed experimental design.

This crate is a narrow work item for Symthaea issue #12375. It does not add a model generator, experiment selector, posterior engine, model IR, evidence ledger, or production model-adequacy decision.

## What it computes

For each evaluator-declared alternative, provide its expected residual at every observation in the fixed design:

- residual = alternative-model prediction minus working-model prediction;
- sigma = known standard deviation of independent Gaussian observation noise;
- alpha = predeclared two-sided false-alarm level.

The crate computes the alternative-specific Gaussian matched-filter noncentrality and best-case power. It then reports every alternative separately, the least-detectable alternative, and whether all alternatives meet the requested power target.

The power envelope intentionally refuses alternatives with different observation counts. A policy cannot receive a better score by quietly spending a larger budget on one alternative.

## Assumptions and claim ceiling

This is an **evaluator-side oracle diagnostic**, not an operational adequacy test:

- independent observations;
- common, known Gaussian noise standard deviation;
- a fixed design and fixed stopping rule;
- known discrepancy direction for each declared alternative;
- two-sided matched-filter test;
- no nuisance-parameter estimation, sequential peeking, multiplicity correction, correlated noise, censoring, missingness or sensor uncertainty.

Those assumptions make the calculation optimistic. When the model parameters or noise process must be estimated, the actual test can have lower power. Adaptive policies require an appropriate sequentially valid test or predeclared alpha-spending rule. Correlated and heteroskedastic noise require a covariance-aware extension and independent validation.

A failed target is reported as **underpowered**, not as evidence for or against the working model. Meeting the target only refers to the finite, declared alternative envelope. It does not prove that the envelope contains every scientifically material alternative.

## API

- **known_direction_power** — best-case power for one declared discrepancy.
- **evaluate_power_envelope** — retain alternative-specific rows and compute worst-case power.
- **minimum_replicates_for_power** — solve the constant-effect known-variance sample-size reference.

## Run

From the repository root:

    cargo fmt --manifest-path research/falsification-power-reference/Cargo.toml -- --check
    cargo test --manifest-path research/falsification-power-reference/Cargo.toml
    cargo clippy --manifest-path research/falsification-power-reference/Cargo.toml --all-targets -- -D warnings

No external Rust crates are required.

## Known-answer controls

The test suite includes:
- null power equals nominal alpha;
- the 5% two-sided normal critical value;
- sign symmetry of two-sided power;
- monotonicity with additional informative observations;
- the minimum-replicate boundary for delta=0.24, sigma=0.15, alpha=0.05 and 80% target;
- alternative-by-alternative reporting and worst-case identification;
- rejection of mismatched budgets, duplicate identifiers and malformed inputs;
- no finite sample-size claim for a zero discrepancy.

These are authored Rust tests. They must be compiled and executed on the exact candidate head before any test-pass claim is made.
