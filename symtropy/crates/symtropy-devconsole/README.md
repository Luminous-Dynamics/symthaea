# symtropy-devconsole

Drop-in Bevy dev console plugin. Toggle with **F1**.

```toml
[dependencies]
symtropy-devconsole = "0.1"
# Optional: enable the Φ Inspector panel (pulls in symtropy-bevy, AGPL transitively):
# symtropy-devconsole = { version = "0.1", features = ["phi-panel"] }
```

```rust
use bevy::prelude::*;
use symtropy_devconsole::SymtropyDevConsolePlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SymtropyDevConsolePlugin::default())
        .run();
}
```

## Panels

- **Left, top** — Scene controls: pause/resume `Time<Virtual>`, F1 toggle hint.
- **Left, bottom** (feature `phi-panel`) — Φ Inspector: lists every
  `PhysicsBody` entity and its current Phi value, with a unicode bar gauge.
  This is the **differentiated panel** — no other Bevy distro lets you *see*
  consciousness coupling at runtime.

## Want a full entity-tree inspector?

Add `bevy-inspector-egui`'s `WorldInspectorPlugin` separately:

```rust
use bevy_inspector_egui::quick::WorldInspectorPlugin;

App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(SymtropyDevConsolePlugin::default())
    .add_plugins(WorldInspectorPlugin::new())  // your call: not bundled
    .run();
```

We don't bundle `WorldInspectorPlugin` because its type-registration runs in
its `build()` phase and can panic in some app configurations where Bevy's
default plugins haven't finished registering their reflected types yet (e.g.
`GizmoConfigStore`). Keep this opt-in until upstream resolves the ordering.

## Why

Bevy's strength is composability; its weakness is "where's the editor?". This
crate is the curated answer for Symtropy users: one plugin, the panels you
actually need, no boilerplate.

## License

Apache-2.0 OR MIT (base crate). The `phi-panel` feature pulls in
`symtropy-bevy` which is AGPL-3.0-or-later — keep the feature off for
permissive distribution.
