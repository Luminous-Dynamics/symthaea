# Mycelix CI harness

> **Status:** stub — full guide in Phase 1 of the [roadmap](../roadmap.md).

The `symtropy-sim-bridge` crate ships a headless test binary that runs Mycelix governance, economy, federated learning, and DKG ceremony simulations with anti-tyranny invariants checked every tick.

## Current scenarios (11)

- `tier-boundaries` — consciousness tier transitions
- `consciousness-weights` — vote weight scaling
- `veto-threshold` — veto power limits
- `byzantine-fl` — TrimmedMean against poisoned gradients
- `emergency-limits` — session caps on emergency powers
- `governance-invariants` — structural safety properties
- `epistemic-transitions` — Observer → Guardian → Sage progression
- `dkg-threshold` — Feldman DKG ceremony
- `tyranny-300-ticks` — 300-tick tyranny probe
- `multi-seed` — reproducibility across seeds
- `fl-overwhelm` — stress test on defences

## Running

```bash
cargo run -p symtropy-sim-bridge --bin headless_test -- --ticks 300 --seed 42
```

## In CI

Phase 1 wires this as the `symtropy-governance-verify` job. Every PR to the Mycelix clusters runs tyranny-300-ticks and the per-cluster invariant sets.

> **Licensing note:** `symtropy-sim-bridge` depends on Mycelix zomes and is **AGPL-3.0-or-later**.
