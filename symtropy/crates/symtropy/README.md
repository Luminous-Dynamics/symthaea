# symtropy

The Symtropy distribution: an opinionated Bevy bundle for consciousness-
coupled simulation and game development.

```toml
[dependencies]
symtropy = "0.1"
```

```rust
use symtropy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SymtropyScenePlugin::default())
        .add_plugins(SymtropyPhysicsPlugin::<3>::with_gravity([0.0, -9.81, 0.0]))
        .add_plugins(SymtropyDevConsolePlugin)
        .run();
}
```

That's the whole setup. PBR rendering, Phi-coupled physics, F1 dev console
with Φ Inspector — all wired.

## What's bundled

| Crate | Purpose | License |
|---|---|---|
| `bevy` 0.18 | Game-engine framework | MIT/Apache-2.0 |
| `symtropy-bevy` | N-D physics + Phi-coupling | AGPL-3.0 |
| `symtropy-bevy-scene` | Scene scaffolding | Apache-2.0/MIT |
| `symtropy-devconsole` | F1 dev console + Φ Inspector | Apache-2.0/MIT |

The meta-crate itself is Apache-2.0/MIT but enabling the default
`devconsole-phi` feature pulls in `symtropy-bevy` (AGPL transitively). For
permissive-only distribution, disable defaults and add the components you
need manually:

```toml
symtropy = { version = "0.1", default-features = false }
```

## Features

| Feature | What | Default? |
|---|---|---|
| `devconsole-phi` | Φ Inspector panel in devconsole (pulls AGPL deps) | yes |
| `low-level` | Re-exports `symtropy-physics` + `symtropy-math` for low-level access | no |

## Get started

```bash
cargo install symtropy-cli
symtropy new my-game
cd my-game && cargo run --release
```

## Related

- [`symtropy-launcher`](https://github.com/luminous-dynamics/symtropy) — the
  flagship application (The Room That Remembers You + Sol Atlas).
- [`symtropy-cli`](https://crates.io/crates/symtropy-cli) — `symtropy new`
  project scaffolding.
- [The Symtropy Book](https://github.com/luminous-dynamics/symtropy) —
  tutorials, design docs, the consciousness-coupling story.
