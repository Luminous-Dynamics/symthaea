// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symtropy: consciousness-driven survival game.
//!
//! "The First Room" — sneak through a sleeping Leviathan, steal its fusion core,
//! and escape before your own panic wakes it up.
//!
//! Run:
//! ```sh
//! PKG_CONFIG_PATH="<alsa-lib-dev>/lib/pkgconfig:$PKG_CONFIG_PATH" cargo run -p symtropy
//! ```

mod components;
mod plugin;
mod resources;
mod systems;

use bevy::prelude::*;
use plugin::SymtropyPlugin;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Symtropy: The First Room".into(),
                        resolution: bevy::window::WindowResolution::new(960, 720),
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
        )
        .add_plugins(SymtropyPlugin)
        .run();
}
