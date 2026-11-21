# Plasma phase-space invariant moments

This standalone Rust crate provides a small-N reference for weighted 1D1V
phase-space moment diagnostics on the periodic cylinder S^1_L x R.

It reports raw signed moments:

- zeroth moment: sum(w),
- first velocity moment: sum(w v),
- half second velocity moment: 0.5 sum(w v^2),
- total variation after co-located signed coefficients are aggregated.

It also reports dimensionless first/second moments under a declared velocity
scale, the unique aggregated support count, and whether the represented measure
is nonnegative. The half second moment is physical kinetic energy only for a
nonnegative particle measure with the appropriate mass normalization; for
signed delta-f inputs it remains a signed moment contribution.

## Important limitations

This is a diagnostic reference, not a kinetic solver, field solver, or
trajectory-acceptance certificate. It does not compute field energy, Gauss-law
or charge-continuity residuals, entropy, Casimirs, or time-integration error.
Those must be added in equation- and discretization-specific layers.

The frozen definitions and known-answer/metamorphic fixtures are recorded in `benchmarks/plasma/phase-space-invariants-v0.1.json`. Its status is explicitly reference-only and it does not assign solver-acceptance tolerances.

## Validation

Run with Rust 1.96:

```sh
cargo fmt --manifest-path crates/tools/plasma-phase-space-invariants/Cargo.toml -- --check
cargo test --locked --manifest-path crates/tools/plasma-phase-space-invariants/Cargo.toml
cargo clippy --locked --manifest-path crates/tools/plasma-phase-space-invariants/Cargo.toml --all-targets -- -D warnings
```
