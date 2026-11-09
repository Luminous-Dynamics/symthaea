# SAM Option Preservation v0.1

Implemented in `crates/domains/symthaea-vehicle/src/option_preservation.rs`.

See `docs/autonomy/SAM_OPTION_PRESERVATION_V0_1.md` for the measurement definition, ownership and claim limits.

Run:

```bash
cargo fmt --package symthaea-vehicle -- --check
cargo test --locked -p symthaea-vehicle --lib option_preservation
cargo clippy --locked -p symthaea-vehicle --lib -- -D warnings
```

This module is an offline finite-envelope evaluator. It does not simulate, predict, perform reachability analysis, or authorize commands. Upstream receipt identities are retained but not cryptographically verified by this evaluator.
