# symtropy-bevy

Drop-in Bevy plugin for Phi-coupled N-dimensional physics.

## Usage

```rust
use bevy::prelude::*;
use symtropy_bevy::{SymtropyPhysicsPlugin, SymtropyPhysics, PhysicsBody};
use symtropy_math::Point;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SymtropyPhysicsPlugin::<2>::with_gravity([0.0, -9.81]))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, mut physics: ResMut<SymtropyPhysics<2>>) {
    // Add a physics body
    let handle = physics.world.add_sphere(Point::new([0.0, 10.0]), 1.0, 1.0);
    
    // Register it with the Phi-coupling field
    physics.field.register(handle, 100.0, 10.0);
    
    // Spawn a Bevy entity linked to the physics body
    commands.spawn((
        Sprite::from_color(Color::WHITE, Vec2::new(32.0, 32.0)),
        Transform::from_xyz(0.0, 10.0, 0.0),
        PhysicsBody::new(handle, 16.0),
    ));
    
    // The plugin handles stepping and transform sync automatically
}
```

## What You Get

- `SymtropyPhysics<D>` resource (physics world + Phi field)
- Automatic `FixedUpdate` stepping with Phi-coupling
- Automatic `Transform` sync from physics to Bevy
- Debug gizmos: collider outlines (colored by safety tier), energy bars, contact points

## Debug Gizmos

Enabled by default. Disable with:

```toml
symtropy-bevy = { version = "0.1.0", default-features = false }
```

## Dimensions

```rust
SymtropyPhysicsPlugin::<2>::default()  // 2D physics
SymtropyPhysicsPlugin::<3>::default()  // 3D physics
SymtropyPhysicsPlugin::<4>::default()  // 4D physics (projects to 3D)
```

## Examples

`examples/pendulum_swarm.rs` — 100-pendulum 2D scene where Phi (computed
from neighborhood velocity variance) modulates each bob's `linear_damping`.
Coherent neighborhoods get near-conservative damping; incoherent ones damp
fast. Click to inject horizontal velocity into the nearest bob.

```bash
cargo run --example pendulum_swarm --release
```

Design notes, empirical constants, and tuning targets live in
`examples/pendulum_swarm.md`. Regression guards on the load-bearing
constants live in `tests/pendulum_swarm_invariants.rs`:

```bash
cargo test -p symtropy-bevy --test pendulum_swarm_invariants --release
```
