# {{project_name}}

Symtropy 3D research scene — generated from `symtropy new --template 3d-research`.

## Run

```bash
cargo run --release
```

A 1280×720 window opens with one sphere swinging from a fixed pivot. Press
**F1** to toggle the dev console (left panel: Scene controls + Φ Inspector).

## What's loaded

- `bevy` — game engine framework
- `symtropy-bevy` — N-D physics + Phi-coupling field
- `symtropy-bevy-scene` — opinionated camera/light/clear-color defaults
- `symtropy-devconsole` (with `phi-panel`) — F1-toggleable dev console

## Where to go from here

- Add more bodies in `setup()` — see `symtropy-bevy/examples/pendulum_swarm_3d.rs`
  for a 100-pendulum scene.
- Add input handling, sprites, gameplay logic — bare Bevy patterns work.
- Switch to `SymtropyPhysicsPlugin::<2>` for 2D, or `::<4>` for 4D.
- Browse the [Symtropy book](https://github.com/luminous-dynamics/symtropy) for
  tutorials on Phi-coupling, ND physics, and consciousness-aware game design.
