# ADR-036: Freeze dependency intent during committed RSK Cargo.lock repair

**Status**: Accepted for qualification candidate

**Change Class**: A

## Context

The Replicator Safety Kernel (RSK) lockfile repair workflow distinguishes a non-admissible generation phase from a later committed-candidate verification phase. A committed Cargo.lock repair is only meaningful if it repairs the already-reviewed dependency intent rather than combining dependency-intent changes with the generated lockfile.

## Decision

In commit-verification mode, where the PR head Cargo.lock differs from the PR-base Cargo.lock, the following dependency-intent inputs MUST be byte-identical to the PR base:

- root `Cargo.toml`;
- `rust-toolchain.toml`;
- `crates/domains/symthaea-replicator-safety/Cargo.toml`;
- `crates/domains/symthaea-replicator-semantics/Cargo.toml`;
- `crates/domains/symthaea-replicator-ledger/Cargo.toml`.

The workflow records the base-to-head dependency-intent diff as evidence and fails if it is non-empty in commit-verification mode.

Generation mode remains allowed to evaluate an exact intended stack head before a lockfile candidate is committed. The generated lockfile remains non-admissible until separately reviewed and committed.

## Consequences

A repair PR cannot silently change dependency intent while presenting the resulting lockfile as a pure reproducibility repair. Legitimate manifest or toolchain changes require their own reviewed lineage before a new repair candidate is generated.

This gate establishes only lockfile repair lineage. It does not establish successful compilation, runtime admission, physical safety, or replication authority.

Production admission remains **DENIED / NOT YET ELIGIBLE**.
