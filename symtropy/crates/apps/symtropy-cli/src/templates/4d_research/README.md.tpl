# {{project_name}}

Symtropy 4D research scene — generated from `symtropy new --template 4d-research`.

## Run

```bash
cargo run --release
```

Three bobs sit at W = -1, 0, +1. Press `[` / `]` to move the W-slice plane.
Bobs near the current slice are visible; bobs further along W fade out. The
full 4D simulation always runs in the background; only the rendered
cross-section moves. Press **F1** for the dev console.

## What's loaded

- `bevy` — game engine framework
- `symtropy-bevy` — N-D physics + Phi-coupling
- `symtropy-bevy-scene` — opinionated 3D scene defaults
- `symtropy-devconsole` (with `phi-panel`) — F1 dev console

## Where to go from here

- See `symtropy-bevy/examples/pendulum_swarm_4d.rs` for a 75-bob swarm
  with 4D motion across W-layers.
- Browse the [Symtropy book](https://github.com/luminous-dynamics/symtropy)
  for the Miegakure-style tutorial on hyperplane slicing.
