# Contributing to Symtropy

## Scope
Symtropy is a consciousness-driven survival game backed by an ND rigid-body physics engine (`symtropy-physics`) and a consciousness/thermodynamics coupling layer (`symtropy-consciousness-physics`).

If you’re contributing to core simulation: **determinism is a hard requirement**, not a “nice to have”.

## Quick start
- Physics engine tests: `cargo test --manifest-path crates/symtropy-physics/Cargo.toml`
- Consciousness coupling tests: `cargo test --manifest-path crates/symtropy-consciousness-physics/Cargo.toml`
- Game crate: `cargo run` (from this folder)

## Determinism rules (read before PRs)
- Prefer deterministic iteration order for any float summation/accumulation:
  - Use ordered maps (e.g., `BTreeMap`) or explicitly sort keys before iterating.
- Avoid non-replayable randomness:
  - No `thread_rng()` in simulation logic; use seeded RNG with seed recorded in replay.
- Keep simulation stepping explicit:
  - Fixed timestep, stable ordering of collision pairs/contacts/constraints.
- Add/extend tests that fail loudly on divergence:
  - `symtropy-physics` includes a record/replay harness that asserts bitwise-identical snapshots per tick.

## Code quality
- Run `cargo fmt` and keep changes focused.
- Prefer adding tests next to the code you change.

## Licensing

Symtropy uses a **dual-track license model** — see [LICENSING.md](LICENSING.md) for the full breakdown.

- Contributions to **core crates** (`symtropy-math`, `symtropy-physics`, `symtropy-render-bridge`, `symtropy-robotics-bridge`, `symtropy-net`, `symtropy-bevy`) are dual-licensed **Apache-2.0 OR MIT** (Rust ecosystem convention).
- Contributions to **research crates** (`symtropy-consciousness-physics`, `symtropy-sim-bridge`, `symtropy-world`, game crates) are licensed **AGPL-3.0-or-later**.

By submitting a contribution to a given crate, you agree your contribution is licensed under that crate's terms. This matches the convention used by the Rust compiler and Bevy itself.

Commercial licensing for the AGPL tier: see [../COMMERCIAL_LICENSE.md](../COMMERCIAL_LICENSE.md) or contact tristan.stoltz@evolvingresonantcocreationism.com.

