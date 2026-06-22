# symtropy-bevy-core

Permissively-licensed Bevy plugin for [`symtropy-physics`](https://crates.io/crates/symtropy-physics) N-dimensional physics.

**Zero AGPL dependencies.** Safe for proprietary projects.

## What you get

- `BevyPhysics<D>` — Bevy `Resource` wrapping a `PhysicsWorld<D>`.
- `PhysicsBody` — Bevy `Component` linking an entity to a physics body.
- `NoCouplingResource` — identity-modulation `PhysicsCallback` for physics-only usage.
- `BevyPhysicsPlugin<D>` — drop-in plugin with `NoCoupling` step (2D / 3D / 4D).
- `BevyPhysicsCorePlugin<D>` — minimal plugin without a step system (users add their own).
- `step_physics<D, C>` — generic step system for any `PhysicsCallback + Resource`.
- `sync_transforms<D>` — write physics positions to Bevy `Transform`.

## Quickstart

```rust
use bevy::prelude::*;
use symtropy_bevy_core::{BevyPhysicsPlugin, BevyPhysics, PhysicsBody};
use symtropy_physics::body::Point;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(BevyPhysicsPlugin::<3>::with_gravity([0.0, -9.81, 0.0]))
        .add_systems(Startup, spawn_ball)
        .run();
}

fn spawn_ball(
    mut commands: Commands,
    mut physics: ResMut<BevyPhysics<3>>,
) {
    let handle = physics.world.add_sphere(Point::new([0.0, 10.0, 0.0]), 1.0, 1.0);
    commands.spawn((
        PhysicsBody::new(handle, 1.0),
        Transform::default(),
    ));
}
```

## Bring your own coupling

If you want to couple a custom per-body metric (health, trust, skill, wealth, …) to physics, implement `PhysicsCallback<D>` on your own Resource type:

```rust
use bevy::prelude::*;
use nalgebra::SVector;
use symtropy_physics::{BodyHandle, CollisionEvent, PhysicsCallback};
use symtropy_bevy_core::{BevyPhysicsCorePlugin, step_physics, sync_transforms};
use std::collections::HashMap;

#[derive(Resource, Default)]
struct HealthField(HashMap<BodyHandle, f64>);

impl<const D: usize> PhysicsCallback<D> for HealthField {
    fn modulate_force(&self, body: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D> {
        let hp = self.0.get(&body).copied().unwrap_or(1.0);
        force * hp  // low HP → weaker motor authority
    }
    fn modulate_impulse(&self, impulse: f64, _: &SVector<f64, D>) -> f64 { impulse }
    fn friction_multiplier(&self, _: &SVector<f64, D>, _: BodyHandle) -> f64 { 1.0 }
    fn on_collision(&mut self, _: &CollisionEvent<D>) {}
    fn record_dissipation(&mut self, _: f64) {}
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(BevyPhysicsCorePlugin::<3>::with_gravity([0.0, -9.81, 0.0]))
        .insert_resource(HealthField::default())
        .add_systems(FixedUpdate, (
            step_physics::<3, HealthField>,
            sync_transforms::<3>,
        ).chain())
        .run();
}
```

## For Φ-coupling (IIT research)

Use the AGPL sibling crate `symtropy-bevy`, which wires `ConsciousnessField` from [`symtropy-consciousness-physics`](https://crates.io/crates/symtropy-consciousness-physics). Same plugin API, adds 5-channel Φ coupling, biometrics integration, and macro-world modifiers.

## Status

Phase 0.5 of the [Symtropy roadmap](https://github.com/luminous-dynamics/symtropy/blob/main/ROADMAP.md). First release extracts the physics-only Bevy plumbing into a permissively-licensed crate so game studios can adopt Symtropy without AGPL obligations.
