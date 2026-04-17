# Deterministic replay

> **Status:** stub — full tutorial with lockstep netcode example in Phase 4.

See [Determinism contract](../core-concepts/determinism.md) for the guarantee and the four contributor rules. This tutorial will extend that into:

1. Recording a replay tape from a running simulation.
2. Diffing two tapes tick-by-tick to find divergence.
3. Lockstep networking with shared seed (Phase 4 Lightyear integration).
4. Cross-machine determinism (Phase 2 float quantisation).

## Quick API snapshot

```rust
use symtropy_physics::{PhysicsWorld, ReplayTape, WorldSnapshot};

let mut tape = ReplayTape::new();
let mut world = PhysicsWorld::<3>::new(gravity);

// record
for tick in 0..1000 {
    world.step(dt);
    tape.record(tick, WorldSnapshot::from(&world));
}

// save / load via serde (stable ordering guaranteed)
let bytes = bincode::serialize(&tape)?;
```
