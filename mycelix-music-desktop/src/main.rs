// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mycelix Music Desktop — consciousness instrument.
//!
//! Full-quality native audio, GPU visualization, egui sliders.
//! Run: cargo run -p mycelix-music-desktop

mod audio;
mod consciousness;
mod ui;

use bevy::prelude::*;
use bevy::render::settings::WgpuSettings;
use bevy::render::RenderPlugin;
use bevy::window::WindowResolution;
fn main() {
    eprintln!("[mycelix-music] Starting consciousness instrument...");

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Mycelix Music — What does your consciousness sound like?".into(),
                    resolution: WindowResolution::new(1280, 820),
                    present_mode: bevy::window::PresentMode::AutoVsync,
                    ..default()
                }),
                ..default()
            })
            .set(RenderPlugin {
                render_creation: bevy::render::settings::RenderCreation::Automatic(WgpuSettings::default()),
                ..default()
            }),
    )
    .add_plugins(consciousness::ConsciousnessPlugin)
    .add_plugins(audio::AudioPlugin)
    .add_plugins(ui::UiPlugin)
    .add_systems(Startup, setup_scene)
    .run();
}

fn setup_scene(mut commands: Commands) {
    commands.spawn(Camera2d);
}
