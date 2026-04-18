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

The `pendulum_swarm` series demonstrates Phi-coupled physics at 2D, 3D,
and 4D — same coupling (neighborhood velocity variance → coherence →
Phi → per-body damping), three rendering targets:

```bash
cargo run --example pendulum_swarm     --release  # 2D, 100 bobs, Sprite + Camera2d
cargo run --example pendulum_swarm_3d  --release  # 3D, 100 bobs, PBR Mesh3d + Camera3d
cargo run --example pendulum_swarm_4d  --release  # 4D, 75 bobs across 3 W-layers,
                                                  # hyperplane slicing + [/] keys to move slice
```

Each example has a side-by-side design doc (`examples/pendulum_swarm{,_3d,_4d}.md`)
covering layout, coupling formulas, known polish targets, and headless capture
mode for visual verification:

```bash
PENDULUM_CAPTURE_DIR=/tmp/caps cargo run --example pendulum_swarm_3d --release
# Saves three timed PNGs and auto-exits at t=8.5 s.
```

Regression guards on the 2D demo's load-bearing physics constants:

```bash
cargo test -p symtropy-bevy --test pendulum_swarm_invariants --release
```
