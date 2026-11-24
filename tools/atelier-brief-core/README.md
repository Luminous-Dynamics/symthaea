# atelier-brief-core

Dependency-free Rust semantic guardrails for the Luminous Atelier design-brief boundary.

## What it does

- Checks brief identity, revision and SHA-256 digest shape.
- Checks unique hard-constraint, objective and suggestion-decision identifiers.
- Exposes `preset_stack`, a deterministic style/function preset composer with source-revision provenance, conflict detection and explicit user overrides.
- Rejects objective weights outside the declared range and all-zero objective sets.
- Detects simple contradictions among constraints on the same field: conflicting categorical requirements, required-and-forbidden values, incompatible units and inverted numeric bounds.
- Binds suggestion decisions to the exact brief revision/digest and requires an actor reference for accepted, rejected or edited suggestions.
- Provides fail-closed engineering-handoff gating: failed gates block; unknown/missing gates remain unresolved; passing gates require an evidence reference.

## What it deliberately does not do

- Parse or serialize JSON (the adjacent JSON Schema is the wire contract).
- Generate patterns, grade sizes, infer body measurements or evaluate aesthetic quality.
- Calculate fabric properties, fit, comfort, durability, environmental impacts or manufacturing cost.
- Verify referenced evidence artifacts. Each owning subsystem must verify its own receipts and calibration.
- Authorize physical production, machine actuation, publication or marketplace transactions.

## Integration boundary

Use the public JSON Schema as the API contract, decode it in the owning application, map the relevant fields to these typed values, and call `validate_brief` before handing objectives to the existing Symtropy textile Pareto engine. Do not create a second Pareto implementation here. The brief core protects the boundary between user intent and downstream optimization; Symtropy continues to own objective/frontier semantics.

## Run

```bash
cargo test --manifest-path tools/atelier-brief-core/Cargo.toml
cargo clippy --manifest-path tools/atelier-brief-core/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path tools/atelier-brief-core/Cargo.toml -- --check
```

These commands are provided for a Rust-enabled checkout. The authoring environment did not have `rustc`, `cargo` or `rustfmt`, so no compilation or test pass is claimed.
