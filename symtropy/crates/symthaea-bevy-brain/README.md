# symthaea-bevy-brain

Bevy plugin that gives any entity a cognitive architecture. Not a behavior tree — a real neural system with HDC perception, CfC temporal dynamics, predictive processing, and integrated information (Phi) metrics.

## Usage

```rust
use bevy::prelude::*;
use symthaea_bevy_brain::{SymthaeaBrainPlugin, CognitiveBrain};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SymthaeaBrainPlugin::default())
        .add_systems(Startup, spawn_npc)
        .add_systems(Update, feed_perception)
        .add_systems(Update, read_outputs)
        .run();
}

fn spawn_npc(mut commands: Commands) {
    commands.spawn((
        Transform::default(),
        CognitiveBrain::new(64, "npc_guard_01"),
    ));
}

fn feed_perception(mut brains: Query<(&Transform, &mut CognitiveBrain)>) {
    for (transform, mut brain) in &mut brains {
        brain.perception_input = format!(
            "position:{:.1},{:.1}",
            transform.translation.x, transform.translation.y
        );
    }
}

fn read_outputs(brains: Query<&CognitiveBrain>) {
    for brain in &brains {
        println!(
            "Phi: {:.3}, prediction_error: {:.3}, learned: {}",
            brain.phi(), brain.prediction_error(), brain.learned()
        );
        if let Some(text) = brain.language() {
            println!("NPC says: {text}");
        }
    }
}
```

## What Each Brain Has

- **16,384-dimensional HDC state vector** — hyperdimensional perception encoding
- **CfC neural network** — closed-form continuous-time temporal dynamics
- **Predictive processing** — prediction error drives learning and attention
- **Phi (integrated information)** — consciousness metric from IIT
- **Episodic memory** — remembers past experiences
- **Language generation** — via BrocaLite (when consciousness is high enough)

## Performance

Each brain is ~500KB (128 CfC neurons + 16,384D HDC). Cognitive cycle runs at ~31Hz measured.

Default scheduling: cognitive cycle every 3 physics ticks (~21Hz at 64Hz physics). Configurable via `brain.cycle_interval`.

## Coupling to Physics

Wire brain Phi into symtropy-bevy's physics coupling:

```rust
fn sync_brain_to_physics(
    brains: Query<(&CognitiveBrain, &PhysicsBody)>,
    mut physics: ResMut<SymtropyPhysics<2>>,
) {
    for (brain, body) in &brains {
        physics.field.set_metric(body.handle, brain.phi());
    }
}
```

This closes the loop: perception → cognition → Phi → physics → environment → perception.
