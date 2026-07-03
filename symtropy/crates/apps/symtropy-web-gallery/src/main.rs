// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use bevy::prelude::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn run() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins);

    app.add_systems(Startup, setup_gallery);

    app.run();
}

fn main() {
    run();
}

fn setup_gallery(mut commands: Commands) {
    // Setup camera
    commands.spawn(Camera2d::default());

    // UI: Web Gallery of 63 Consciousness Experiments
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                Text::new("Symtropy Consciousness Gallery"),
                TextFont {
                    font_size: FontSize::Px(40.0),
                    ..default()
                },
            ));

            parent.spawn((
                Text::new("63 Experiments in Integrated Information & Physics"),
                TextFont {
                    font_size: FontSize::Px(20.0),
                    ..default()
                },
            ));
        });
}
