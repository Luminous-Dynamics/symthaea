# Symtropy Roadmap

## North Star
Build the best **N-dimensional + consciousness-coupled simulation engine** with **deterministic replay** as a first-class physical law (for P2P / Mycelix-style decentralized simulation).

This project is not trying to out-box Box2D/Rapier/Jolt as general-purpose rigid-body physics. The niche is: **physics that can be meaningfully modulated by consciousness/agency constraints**, and that can be **proven replayable**.

## Now (0–8 weeks)
- Determinism harness (record/replay) for `symtropy-physics` (bitwise snapshots per tick).
- Deterministic iteration order in consciousness coupling (avoid `HashMap`-order float summations).
- Deterministic RNG policy for simulation/procgen (seeded, explicit, recorded).
- “Sim invariants” checklist: fixed timestep, ordered collision resolution, stable IDs/handles.

## Next (2–4 months)
- Expand shapes beyond spheres: capsule, box, convex polytope (ND-friendly).
- Improve broadphase (BVH / sweep-prune) while preserving deterministic ordering.
- Constraints/joints with deterministic solve order and stable warm-starting.
- Deterministic “math mode” (feature flag): constrained libm/sin-cos usage, optional quantization at tick boundaries.

## Later (6–12 months)
- Rollback/reconciliation layer (Lightyear or custom), backed by deterministic snapshots.
- Cross-platform determinism strategy (float policy, quantization, or fixed-point where required).
- Tooling: state hash visualizer, replay debugger, determinism CI gates.

## Contributions that move the needle
- Anything that strengthens determinism guarantees.
- ND-first features (debug rendering, ergonomics for D≠3).
- Tests that lock down invariants (especially around ordering and floating-point edge cases).

