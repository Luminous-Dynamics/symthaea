# Quickstart

**Goal:** A 2D physics world with a sphere whose motor authority is gated by a custom `[0, 1]` metric you control. Runs in about 20 lines.

## Add dependencies

```toml
# Cargo.toml
[dependencies]
symtropy-math = "0.2"                    # Apache-2.0 OR MIT
symtropy-physics = "0.2"                 # Apache-2.0 OR MIT
symtropy-consciousness-physics = "0.1"   # AGPL-3.0-or-later (for SimpleCoupledField)
nalgebra = "0.34"
```

> **Licensing note:** `symtropy-math` and `symtropy-physics` are permissive (Apache-2.0 OR MIT) — drop them into proprietary projects. `symtropy-consciousness-physics` is AGPL because it carries the coupling primitives. If you don't want AGPL, implement [`PhysicsCallback`](../reference/physics-callback.md) directly on your own type — all the core traits are in `symtropy-physics` under the permissive licenses.

## Drop a coupled sphere

```rust
use symtropy_physics::PhysicsWorld;
use symtropy_consciousness_physics::SimpleCoupledField;
use symtropy_math::Point;
use nalgebra::SVector;

fn main() {
    // 1. A 2D physics world with Earth gravity
    let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, -9.81]));

    // 2. A sphere at (0, 10) with radius 1 and mass 1
    let agent = world.add_sphere(Point::new([0.0, 10.0]), 1.0, 1.0);

    // 3. A coupling field that accepts any metric you want
    let mut field = SimpleCoupledField::<2>::new();
    field.register(agent, 100.0, 10.0);       // energy budget, maintenance cost
    field.set_metric(agent, 0.8);             // YOUR metric: health, trust, skill, wealth

    // 4. Step for one second. Force and friction are modulated by your metric.
    for _ in 0..60 {
        world.step_with_callback(1.0 / 60.0, &mut field);
    }

    // 5. Query the final state
    let body = world.body(agent);
    println!("position after 1s: {:?}", body.transform.position);
    println!("energy remaining: {:.2}", field.energy(agent));
}
```

## What just happened

Your `metric = 0.8` (in Symtropy's 4-tier safety system, that's "Green") let the body keep **≈80%** of its motor authority. If you'd set `metric = 0.2`, the body would have been throttled to Orange tier, applying gravity but with suppressed horizontal response. At `metric = 0.0` the body becomes inert — physical but unresponsive — until you replenish its energy budget.

This mechanism works identically for:

- A health system (low HP → sluggish movement)
- A trust metric (social cooperation → friction reduction in shared zones)
- A skill metric (high skill → tighter motor precision)
- **Φ** (the default; see [Φ-coupling](../core-concepts/phi-coupling.md))

## Bevy integration (optional)

If you're using Bevy:

```toml
symtropy-bevy = "0.2"         # AGPL-3.0-or-later (Phi-coupled physics)
# Or, for permissive licensing without the Phi-coupling layer:
# symtropy-bevy-core = "0.1"  # Apache-2.0 OR MIT
```

```rust
use bevy::prelude::*;
use symtropy_bevy::SymtropyPhysicsPlugin;

App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(SymtropyPhysicsPlugin::<2>::with_gravity([0.0, -9.81]))
    .run();
```

See [Bevy integration](../getting-started/first-body.md) for the full setup,
or jump straight to a runnable demo:

```bash
cargo run -p symtropy-bevy --example pendulum_swarm --release
```

`pendulum_swarm` spawns a 10×10 grid of pendulums where Phi — computed from
neighborhood velocity variance — modulates each bob's damping. Click to
inject velocity into the nearest bob. Source + design notes live in
`crates/symtropy-bevy/examples/`.

## Next up

- [Your first coupled body](./first-body.md) — full example with Bevy rendering.
- [Generic state coupling](../core-concepts/generic-state-coupling.md) — how the coupling math works.
- [The five channels](../core-concepts/five-channels.md) — force, energy, impulse, friction, feedback.
