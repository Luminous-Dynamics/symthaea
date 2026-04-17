# Contributing

See `../../CONTRIBUTING.md` at the repo root for the authoritative contributor guide, including the determinism rules and the licensing-per-crate agreement.

## The short version

1. **Determinism is a hard requirement.** No `HashMap` in simulation logic; seeded RNG only; fixed timestep; stable collision-pair ordering. See [Determinism contract](./core-concepts/determinism.md).
2. **Tests next to the code you change.** Prefer property tests (`proptest`) for anything that touches the solver.
3. **`cargo fmt` before PR.**
4. **Licensing:** contributions inherit the crate's license (Apache-2.0 OR MIT for core crates, AGPL for research crates). See [Licensing](./reference/licensing.md).

## High-leverage contributions

From the roadmap:

- **Determinism hardening** — cross-platform float strategies, invariant tests, ordering guarantees.
- **ND-first ergonomics** — debug rendering for D ≠ 3, 4D gizmos, tooling that makes 4D authoring tolerable.
- **New experiments** that test the formal specification's predictions — cite what's confirmed or refuted.
- **Ecosystem crates** — if you build a Symtropy-compatible GPU broadphase or soft-body solver, we'll link it from `crates-overview.md`.
- **Language bindings** — Rhai bridges for `PhysicsCallback`, Python via PyO3 for research.
- **Book chapters** — every stub in `book/src/` with "Status: stub" is an invitation.

## Communication

- **Issues**: `https://github.com/luminous-dynamics/symtropy/issues`
- **Commercial licensing**: `tristan.stoltz@evolvingresonantcocreationism.com`
