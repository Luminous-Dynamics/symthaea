# Selective Memory Actions v0.1

Research-only Rust reference selector. It is intentionally outside the root workspace and has no external dependencies.

## Run

```bash
cargo test --manifest-path research/selective-memory-actions-v0-1/Cargo.toml
cargo fmt --manifest-path research/selective-memory-actions-v0-1/Cargo.toml -- --check
```

## Scope

- Separates ingestion actions from query-time information actions.
- Ranks precomputed probability-weighted loss distributions using integer fixed-point arithmetic.
- Applies hard admission gates before cost ranking.
- Uses deterministic tie-breaking from a frozen policy rank.
- Rejects mixed-stage options, duplicate actions, invalid probability mass and arithmetic overflow.

## Deliberate limitations

This crate does not estimate or calibrate probabilities, author the cost matrix, verify sources, authenticate `AdmissionSnapshot`, write memory, call tools, authorize environment actions, or compute benchmark labels. The caller supplies the loss distributions and must derive the admission snapshot from existing consent, provenance, scope, freshness, retraction, privacy and authority owners.

It is not a production memory manager, a benchmark result or a SOTA claim. No integration into the root workspace or production code is made by this research crate.
