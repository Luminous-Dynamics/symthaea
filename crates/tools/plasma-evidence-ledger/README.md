# Plasma trajectory evidence ledger

This standalone, dependency-free Rust crate qualifies recorded evidence for one bounded trajectory segment. It is a policy evaluator, not a solver-correctness proof and not a source of physical error bounds.

## Fail-closed metric qualification

Each metric record carries its equation cohort, units, time interval, applicability state, raw observable, error estimate and interval, acceptance tolerance, bound kind, and source/input/config/operator provenance.

- Missing observations, raw values, estimates, bounds or tolerances are unresolved.
- An unavailable metric is unresolved; it is never a zero.
- Not-applicable is explicit and reasoned, and is accepted only when the cohort-specific policy permits it.
- An upper error bound within tolerance can pass only for deterministic or probabilistic evidence.
- A lower error bound above tolerance fails.
- An interval that overlaps tolerance is unresolved.
- Empirical intervals remain descriptive and cannot be promoted into a pass.
- Cohort, segment, units, provenance and frozen-policy mismatches cannot pass.

There is deliberately no universal required-metric set. Each equation cohort and discretization must supply a versioned policy.

## Error-budget aggregation

Only bounds for the same cohort, target metric, units, trajectory segment and provenance may be aggregated.

- Deterministic upper bounds sum conservatively.
- Probabilistic upper bounds sum with the union-bound confidence lower bound 1 - sum(alpha_i), with no independence assumption.
- Empirical and unavailable components are rejected from qualified aggregation.
- Root-sum-square aggregation is not implemented.

The caller must prove that all component bounds refer to the same target norm and quantity. This crate cannot create the required physical assumptions or bounds.

## Run

    cargo fmt --manifest-path crates/tools/plasma-evidence-ledger/Cargo.toml -- --check
    cargo test --locked --manifest-path crates/tools/plasma-evidence-ledger/Cargo.toml
    cargo clippy --locked --manifest-path crates/tools/plasma-evidence-ledger/Cargo.toml --all-targets -- -D warnings
