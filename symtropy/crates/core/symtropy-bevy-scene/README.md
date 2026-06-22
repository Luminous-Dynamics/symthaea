# symtropy-bevy-scene

Opinionated scene/camera/light scaffolding for Bevy 0.18 apps. Eliminates the
camera-light-clear-color boilerplate every Symtropy demo otherwise repeats.

```toml
[dependencies]
symtropy-bevy-scene = "0.1"
```

```rust
use bevy::prelude::*;
use symtropy_bevy_scene::{SymtropyScenePlugin, fixed_camera};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SymtropyScenePlugin::default())
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(fixed_camera(Vec3::new(0.0, 1.5, 6.0), Vec3::ZERO));
}
```

## What it does

- `ClearColor` — dark cool indigo background
- `GlobalAmbientLight` — cool tint, brightness 200 cd/m² so shadowed sides
  aren't fully black
- One `DirectionalLight` ("sun") at a stage-lighting angle (upper-front-right),
  illuminance 8000 lux

Everything is overridable via `SymtropyScenePlugin::with_config(...)`.

## Helpers

- `fixed_camera(position: Vec3, target: Vec3)` — `Camera3d + Transform`
  one-liner.
- `fixed_light(illuminance, color, euler_xyz)` — secondary directional
  light spawn.

## License

Apache-2.0 OR MIT (permissive). No AGPL deps.

## Relationship to symtropy-bevy

`symtropy-bevy-scene` is the *visual* sibling of `symtropy-bevy-core`
(physics-only). They compose:

```rust
App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(symtropy_bevy_scene::SymtropyScenePlugin::default())
    .add_plugins(symtropy_bevy::SymtropyPhysicsPlugin::<3>::with_gravity([0.0, -9.81, 0.0]))
    .run();
```
