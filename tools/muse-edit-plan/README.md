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

- Exact source artifact IDs, types, modalities and SHA-256 digests; source, operator and observer rights states with digest-bound licence/consent/authorization bases; canonical request contracts and a SHA-256 binding to the frozen registry snapshot.
- Typed edit stages with explicit source/stage input bindings and topological dependencies.
- Operator matching on operation, input/output types, modality, capabilities, locality ceiling, preservation guarantees, pre/postconditions, receipts, controls, rights, reproducibility, risk and resource estimates.
- Deterministic candidate ranking; equal-cost/equal-risk top candidates return an explicit Ambiguous result unless the request names an exact preferred operator version.
- Independent observer selection by metric, artifact type, modality and independence group.
- Explicit Unsupported, Ambiguous, DependencyCycle, TypeMismatch, InvalidRequest and BudgetExceeded results.
- Stable topologically ordered plan nodes, per-node obligations and a deterministic canonical manifest.
- A stage-qualification barrier after every mutation; downstream edits cannot consume a parent stage until its required independent observations have passed.
- The full normalized request (budgets, policy, target scope, controls, protections, observers, seeds and preferred operator) is bound into the canonical manifest.

## Claim boundary

- Profiles are declarations from an externally qualified registry; this compiler does not prove their claims true.
- It does not parse raw natural language, call a model, execute a tool, mutate audio, certify perceptual quality, or authorize a commit.
- The request's registry snapshot digest is supplied by the external registry publisher; the compiler validates and binds it but cannot prove the registry publisher's process trustworthy.
- The canonical plan manifest is stable input material for an external SHA-256 implementation; this crate does not claim its own digest or cryptographic signature.
- Resource latency is conservatively summed, not parallel critical-path estimated.
- Stage gates are static DAG barriers. A runtime executor must enforce them and evaluate the returned evidence states; the compiler does not observe actual execution.
- Static plan qualification is not a production execution or acoustic-success receipt.