# Determinism contract

**Symtropy guarantees bit-identical replay on the same CPU with a given seed.** This isn't a side effect — it's a first-class design goal, and the engine's architecture is shaped around it.

## What you get

| Guarantee | Mechanism |
|---|---|
| Same inputs → same outputs, same tick, same machine | Integer Morton broadphase, `BTreeMap` iteration, sorted contact pairs |
| Replay from a `ReplayTape` reproduces simulation exactly | `NetId` stable identity, `WorldSnapshot` serialisation |
| Network lockstep with shared seed | `step_with_callback` is deterministic for single-threaded physics |

## What you do NOT get (yet)

| Non-guarantee | Why |
|---|---|
| Cross-platform bitwise equality | Different CPUs may yield different float results (x87 vs SSE, ARM vs x86). |
| Determinism under multi-threaded physics stepping | The physics hot path is deterministic single-threaded; parallel solver is future work. |
| Determinism with GPU broadphase | Not in core scope; would live in an opt-in `symtropy-physics-gpu` crate. |

Cross-platform determinism is planned (Phase 2 of the roadmap): float quantisation (`deterministic-net` feature exists) or constrained-libm strategy. Until then, lockstep multiplayer works on same-CPU-class peers.

## The four rules contributors must follow

If you're sending a PR that touches physics:

### 1. No `HashMap` in simulation logic

`std::collections::HashMap` iterates in randomised order. Use `BTreeMap` or explicitly sort keys.

```rust
// ✗ WRONG
let mut contacts: HashMap<BodyHandle, ContactManifold<D>> = HashMap::new();
for (body, manifold) in contacts.iter() { /* non-deterministic order */ }

// ✓ RIGHT
let mut contacts: BTreeMap<BodyHandle, ContactManifold<D>> = BTreeMap::new();
for (body, manifold) in contacts.iter() { /* sorted order */ }
```

### 2. Seeded RNG only

`thread_rng()` is forbidden in simulation logic. Use `StdRng::from_seed()` with a seed that's recorded in the replay tape.

```rust
// ✗ WRONG
use rand::thread_rng;
let mut rng = thread_rng();

// ✓ RIGHT
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
let mut rng = ChaCha8Rng::seed_from_u64(replay_seed);
```

### 3. Fixed timestep

All physics stepping is via `FixedUpdate` at a constant `dt`. Never accumulate dt from frame times.

```rust
// ✗ WRONG
app.add_systems(Update, physics_step);  // variable dt

// ✓ RIGHT
app.insert_resource(Time::<Fixed>::from_hz(64.0));
app.add_systems(FixedUpdate, physics_step);  // 1/64 s every tick
```

### 4. Stable ordering of collision pairs

Collision broadphase output is sorted by `(BodyHandle, BodyHandle)` before narrowphase. If you add a new broadphase, preserve this.

## The replay system

`symtropy-physics` ships a record/replay harness that asserts bit-identical snapshots per tick:

```rust
use symtropy_physics::{PhysicsWorld, ReplayTape, WorldSnapshot};

// Record
let mut world = PhysicsWorld::<3>::new(gravity);
let mut tape = ReplayTape::new();

for tick in 0..1000 {
    world.step(dt);
    tape.record(tick, WorldSnapshot::from(&world));
}

// Replay — reproduces exactly
let mut replay = PhysicsWorld::<3>::new(gravity);
for (tick, snapshot) in tape.iter() {
    replay.step(dt);
    assert_eq!(snapshot, WorldSnapshot::from(&replay), "divergence at tick {}", tick);
}
```

The `replay-cli` binary runs this as an integration test. Every merge to `main` runs replay diffs on the 63 consciousness experiments — divergence breaks the build.

## Testing your changes

If your change *could* affect determinism, add a test that records 100 ticks, then replays and asserts bit-identity. Pattern:

```rust
#[test]
fn my_change_preserves_determinism() {
    let mut a = PhysicsWorld::<2>::new(gravity);
    let mut b = PhysicsWorld::<2>::new(gravity);

    // identical setup
    setup(&mut a);
    setup(&mut b);

    for _ in 0..100 {
        a.step(1.0 / 64.0);
        b.step(1.0 / 64.0);
    }

    assert_eq!(WorldSnapshot::from(&a), WorldSnapshot::from(&b));
}
```

## Why this matters

1. **Multiplayer** — deterministic lockstep avoids sending full state; clients just send inputs.
2. **Science** — reproducible experiments. The 63 consciousness examples are citable results *because* they reproduce exactly.
3. **Debugging** — "record once, replay N times" lets you attach a debugger to the exact failing tick.
4. **CI** — replay diffs catch regressions that unit tests miss.

## Further reading

- [Deterministic replay tutorial](../tutorials/replay.md) — worked example with lockstep networking.
- [`ReplayTape` reference](https://docs.rs/symtropy-physics) — API documentation.
