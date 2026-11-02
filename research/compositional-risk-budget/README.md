# Compositional Risk Budget Ledger v0.1

Standalone Rust research reference for typed composition of prediction, action, temporal and compute-stopping risk claims. Tracking: issue #10066.

## Run

    GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo run --manifest-path research/compositional-risk-budget/Cargo.toml --bin risk-budget-replay
    GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo test --manifest-path research/compositional-risk-budget/Cargo.toml

Generate Cargo.lock after the first dependency resolution, commit it, then use --locked.

## What it does

- Freezes a complete family of risk contracts and budget allocations in a canonical SHA-256 plan digest.
- Separates operational risk alpha from statistical certification-failure delta; all arithmetic uses exact integer parts-per-billion.
- Verifies all claims share one target population, execution scope, subject/policy digest and joint certification experiment.
- Appends evidence receipts in preregistered order with contract digests, previous-receipt links and an aggregate ledger digest.
- Composes unconditional marginal claims by the union bound.
- Composes a complete action-conditional partition using max over action IDs of the within-action union-bound sum. It requires a complete action support digest and identical event-family membership for every action.
- Keeps empirical-only/unsupported claims in an explicit no-formal-credit group, which prevents a full formal composition report.
- Accepts an upstream receipt digest from the canonical anytime family ledger (#9465) without reimplementing that ledger.

## Mathematical contract

For marginal event group g, alpha_g = sum_i alpha_i.
For an action partition with complete support A, alpha_g = max_{a in A} sum_i alpha_{i|a}.
Across groups, alpha_system = sum_g alpha_g. Confidence delta_system = sum_i delta_i across every certificate, including every action-specific certificate.

The action-partition maximum is sound only when each conditional statement refers to the actual selected action, the declared support exhausts every possible action, and all claims share the same population/execution scope. It does not average away rare actions.

## Claim ceiling

The crate validates bookkeeping, compatibility and receipt integrity. It does not verify source theorems, calibration assumptions, sampling independence, exchangeability, external attestation signatures, preregistration chronology, or whether a real system obeys the declared scope. The report always sets formal_system_guarantee_claimed=false. The bundled fixture is synthetic topology only and its evidence hashes are deliberately synthetic.

## Reuse rather than duplication

The canonical anytime confidence family ledger (#9465) remains the owner of preregistered stream/look allocation and replay. This layer only binds its exported receipt digest to an upstream risk contract. The run-wide pruning risk ledger remains the owner of pruning-event alpha/delta. This crate composes typed component-level outputs; it does not replace either.
