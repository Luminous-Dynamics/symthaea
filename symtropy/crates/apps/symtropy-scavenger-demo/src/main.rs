use bevy::prelude::*;
use symtropy_scavenger_demo::plugin::ScavengerDemoPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Symtropy Scavenger Agent Demo".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(ScavengerDemoPlugin)
        .run();
}
