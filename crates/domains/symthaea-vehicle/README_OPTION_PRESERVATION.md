# SAM Option Preservation v0.1

Implemented in `crates/domains/symthaea-vehicle/src/option_preservation.rs`.

See `docs/autonomy/SAM_OPTION_PRESERVATION_V0_1.md` for the measurement definition, ownership and claim limits.

Run:

```bash
cargo fmt --package symthaea-vehicle -- --check
cargo test --locked -p symthaea-vehicle --lib option_preservation
cargo clippy --locked -p symthaea-vehicle --lib -- -D warnings
```

This module is an offline finite-envelope evaluator. It replays a persistent analytic clairvoyance-trap fixture from `docs/autonomy/fixtures/SAM_OPTION_PRESERVATION_CLAIRVOYANCE_TRAP_V0_1.json`. It does not simulate, predict, perform reachability analysis, or authorize commands. The expected-cell manifest is canonically re-digested with BLAKE3 and duplicate cell/execution receipts are rejected. Upstream qualification and execution receipt identities are still not authenticated by this evaluator.
