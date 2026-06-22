# {{project_name}}

Symtropy 2D scene — generated from `symtropy new --template 2d-game`.

## Run

```bash
cargo run --release
```

A single sprite bounces around the screen. Click to kick it toward the cursor.
Press **F1** for the dev console.

## What's loaded

- `bevy` — game engine framework
- `symtropy-bevy` — 2D physics + Phi-coupling
- `symtropy-devconsole` (with `phi-panel`) — F1 dev console

This template is intentionally tiny — pick it up as a starting point and add
sprites, tilemaps, gameplay logic from there.

## Where to go from here

- See `symtropy-bevy/examples/pendulum_swarm.rs` for a 100-bob 2D scene with
  Phi-coupled damping.
- Add `bevy_ecs_tilemap` for tile-based 2D worlds.
- Add `bevy_kira_audio` for richer sound.
- Browse the [Symtropy book](https://github.com/luminous-dynamics/symtropy).
