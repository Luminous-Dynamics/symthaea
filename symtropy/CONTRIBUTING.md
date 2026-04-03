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
Symtropy is AGPL-3.0-or-later (with commercial licensing available). By contributing, you agree your contributions are licensed under the repository’s licensing terms.

