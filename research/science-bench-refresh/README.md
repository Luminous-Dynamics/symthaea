# Science Benchmark Refresh

Protocol-first workstream for SCI-BENCH-REFRESH-001 (Issue #8771).

## Current contents

- `docs/research/SCI-BENCH-REFRESH-001.md`: benchmark-lifecycle protocol.
- `schema/generation-manifest.schema.json`: versioned JSON Schema for the declared v1 manifest shape.
- `fixtures/generation-manifest.example.json`: non-scoring placeholder fixture.
- `src/lib.rs` and `src/main.rs`: standalone Rust manifest validator and CLI.

## Validate a manifest

```sh
cargo fmt --manifest-path research/science-bench-refresh/Cargo.toml -- --check
cargo test --locked --manifest-path research/science-bench-refresh/Cargo.toml
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \
  research/science-bench-refresh/fixtures/generation-manifest.example.json
```

The example should report `valid: true` and `manifest_declares_scoring_eligible: false`. The fixture must fail when the CLI is invoked with `--require-scoring` or `--require-sealed`; the focused GitHub Actions workflow encodes this negative control.

## What the validator checks

- Strict versioned manifest shape, including rejection of unknown/oracle-only fields.
- Syntax of SHA-256 and Git commit digests; qualified runs require non-placeholder digests and a pinned 40-character lowercase commit SHA.
- Declared evaluator/oracle-separation status for a qualified run.
- Compatibility between the lane and its reveal policy.
- Unique task-family IDs, task-family weights summing to 1, positive per-family counts, and planned counts matching the declared expected denominator.
- A deterministic SHA-256 receipt of the exact raw manifest bytes.
- Separate flags for a valid manifest, manifest-declared scoring eligibility, and a manifest-declared sealed-refresh profile.

## Limits (important)

An all-zero digest is deliberately used by the example and does not identify any real artifact. The generator revision says `NOT_IMPLEMENTED_YET`. This fixture is never admissible for scoring.

The validator does **not** recompute the hidden seed commitment or task-set commitment because the reveal artifact/task generator is not implemented yet. It does not establish that a policy could not access the oracle, that a runner executed the declared subject, that a generated task is scientifically identifiable, or that a score is correct. `manifest_declares_sealed_generalization_profile` means only that the *manifest's declared shape* passed; external execution-boundary evidence is still required.

The current PR has 18 authored Rust tests. They must not be described as passing until exact-head CI actually runs them. This environment has no Rust toolchain available for a local compile/test run.