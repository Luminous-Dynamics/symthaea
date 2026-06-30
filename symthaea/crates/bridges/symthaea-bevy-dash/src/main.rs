use bevy::prelude::*;
use symthaea_bevy_dash::SymthaeaDashPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Symthaea Cognitive Dashboard".to_string(),
                resolution: (1280.0, 720.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(SymthaeaDashPlugin)
        .run();
}
