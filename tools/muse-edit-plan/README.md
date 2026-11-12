# CQEP-001 — Capability-qualified edit-plan compiler

This crate compiles normalized audio/music edit requests into a deterministic execution/evidence DAG using a frozen registry of externally qualified operators and observers.

## Run

```sh
cargo fmt --manifest-path tools/muse-edit-plan/Cargo.toml --all -- --check
cargo test --manifest-path tools/muse-edit-plan/Cargo.toml --locked
cargo clippy --manifest-path tools/muse-edit-plan/Cargo.toml --all-targets --locked -- -D warnings
```

The crate is standard-library-only and declares its own nested Cargo workspace.

## Implemented contract

- Exact source artifact IDs, types, modalities and SHA-256 digests.
- Typed edit stages with explicit source/stage input bindings and topological dependencies.
- Operator matching on operation, input/output types, modality, capabilities, locality ceiling, preservation guarantees, pre/postconditions, receipts, controls, rights, reproducibility, risk and resource estimates.
- Deterministic candidate ranking; equal-cost/equal-risk top candidates return an explicit Ambiguous result unless the request names an exact preferred operator version.
- Independent observer selection by metric, artifact type, modality and independence group.
- Explicit Unsupported, Ambiguous, DependencyCycle, TypeMismatch, InvalidRequest and BudgetExceeded results.
- Stable topologically ordered plan nodes, per-node obligations and a deterministic canonical manifest.

## Claim boundary

- Profiles are declarations from an externally qualified registry; this compiler does not prove their claims true.
- It does not parse raw natural language, call a model, execute a tool, mutate audio, certify perceptual quality, or authorize a commit.
- The canonical manifest is stable input material for an external SHA-256 implementation; this crate does not claim a digest or cryptographic signature.
- Resource latency is conservatively summed, not parallel critical-path estimated.
- Static plan qualification is not a production execution or acoustic-success receipt.