# Introduction

**Symtropy is a state-coupled, N-dimensional rigid-body physics engine written in Rust, with deterministic replay as a first-class physical law.**

Any metric you define — health, trust, skill, wealth, Φ (integrated information), or anything you can compute into a `[0, 1]` value — modulates forces, friction, and energy budgets in real-time through a thermodynamically-closed system.

It ships with two hero capabilities most engines don't try:

1. **Φ-coupled physics** — the world's first game engine where integrated information (Tononi 2004) is a first-class physics parameter. Not a UI overlay; Φ literally changes collision impulses, friction coefficients, and energy budgets via a `PhysicsCallback` trait, enforced every solver step.
2. **True N-dimensional physics** — const-generic `PhysicsWorld<D>`, `Sphere<D>`, `HyperBox<D>`, `Rotor<D>`, `HingeJoint<D>` all compile for any D. 2D, 3D, 4D all work with identical code; 4D cross-section slicing is first-class.

Both are gated on a third capability most engines don't have:

3. **Deterministic replay** — integer Morton codes, `BTreeMap` iteration, sorted collision pairs, `NetId`, `ReplayTape`. Bit-reproducible on a given CPU.

## Who is this for?

Symtropy is dual-track by design:

### Track A — Research hero

If you're working on:

- **Integrated Information Theory** — Symtropy is the only engine where Φ modulates forces at the solver level.
- **Active inference / FEP** — `symtropy-robotics-bridge::RoboticAgent` wraps Symthaea's `EmbodimentBridge` for consciousness-gated motor output.
- **Emergent cooperation studies** — 63 experiments in `symtropy-consciousness-physics/examples/` cover cooperation emergence, J/Φ convergence, tragedy of commons, Dunbar number, anesthesia transitions.
- **N-dimensional rigid body dynamics** — 4D Rubik's cubes, Miegakure-style hidden geometry, hyperdimensional simulation.

### Track B — Generalist adoption

If you're building:

- **A game with deterministic multiplayer** — the same features that enable replay give you rollback netcode for free.
- **A robotics simulation** — first-class humanoid / quadruped / manipulator / flight / AUV / helicopter platforms through `symtropy-robotics-bridge`.
- **A state-driven emergent system** — couple *any* per-body metric (health, trust, skill, wealth) to physics through the same `PhysicsCallback` interface Φ uses.

Symtropy is built on Bevy 0.18 but the core physics (`symtropy-math`, `symtropy-physics`) has zero Bevy dependency — you can use it as a standalone physics library.

## What Symtropy is NOT

Honesty matters more than marketing. Symtropy does not:

- **Replace Rapier for general 3D physics at scale.** Rapier is faster at >1000 bodies, has GPU broadphase, triangle meshes, and years of commercial battle-testing. Symtropy targets <1000 bodies with rich per-body state. For high-fidelity 3D robotics, the Phase 1 `symtropy-rapier3d-bridge` lets you mix both.
- **Ship soft-body, cloth, or fluid simulation.** These belong in optional ecosystem crates (`symtropy-soft`, `symtropy-fluid`). The core stays rigid-body.
- **Ship a level editor.** We layer on `bevy_inspector_egui` instead of building a bespoke editor.
- **Claim sentience or qualia.** Φ in this codebase is a formal information-theoretic quantity from IIT. The word "consciousness" refers exclusively to that mathematical object. No claims are made about subjective experience.

## Engine at a glance

- **61K LOC Rust** across 23 crates, 677 tests, 8 on crates.io (as of 2026-04)
- **5 collision shapes**, all ND: `Sphere`, `Capsule`, `HyperBox`, `HalfSpace`, `ConvexHull`
- **4 joint types**, all ND: `DistanceConstraint`, `BallJoint`, `FixedJoint`, `HingeJoint`
- **GJK + EPA + CCD + raycasting**, all ND-generic
- **Zero heap allocation** in the physics hot path
- **GJK sphere-sphere 3D in 156 ns**; 100-body step in 135 μs; 500-body step in 910 μs

## License at a glance

Symtropy is **dual-licensed**:

- **Permissive today** (`symtropy-math`, `symtropy-physics`, `symtropy-render-bridge`): **Apache-2.0 OR MIT** — zero AGPL deps, ship in proprietary products freely.
- **AGPL today, permissive `-core` variants planned** (`symtropy-bevy`, `symtropy-robotics-bridge`, `symtropy-net`): each currently requires an AGPL dep — not yet safe to ship in closed-source products; see [LICENSING.md](https://github.com/Luminous-Dynamics/symtropy/blob/main/docs/LICENSING.md) for the split plan.
- **Research layer** (`symtropy-consciousness-physics`, `symtropy-sim-bridge`, game crates): **AGPL-3.0-or-later** — modifications must be shared back, or negotiate a commercial license.

See [Licensing](./reference/licensing.md) for the full breakdown.

## Next up

- [Quickstart](./getting-started/quickstart.md) — drop a conscious sphere in 20 lines.
- [Generic state coupling](./core-concepts/generic-state-coupling.md) — make *your* metric modulate physics.
- [The 63 experiments](./research/experiments.md) — reproduce the emergent-cooperation results.
