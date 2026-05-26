// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! # symtropy-devconsole
//!
//! Drop-in Bevy dev console plugin. Two panels (more coming):
//!
//! - **Scene controls** (left) — pause/resume `Time<Virtual>`, F1 toggle hint.
//! - **Φ Inspector** (left, feature `phi-panel`) — lists every `PhysicsBody`
//!   entity and its current Phi value.
//!
//! Toggle the whole console with **F1**.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};

/// Toggle state for the dev console.
#[derive(Resource, Debug, Clone, Copy)]
pub struct DevConsoleVisible(pub bool);

impl Default for DevConsoleVisible {
    fn default() -> Self {
        Self(true)
    }
}

/// Pause flag for downstream physics systems.
#[derive(Resource, Debug, Clone, Copy)]
pub struct DevConsolePaused(pub bool);

impl Default for DevConsolePaused {
    fn default() -> Self {
        Self(false)
    }
}

/// Drop-in dev console plugin. See crate docs.
#[derive(Default, Clone)]
pub struct SymtropyDevConsolePlugin;

impl Plugin for SymtropyDevConsolePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<EguiPlugin>() {
            app.add_plugins(EguiPlugin::default());
        }
        app.init_resource::<DevConsoleVisible>()
            .init_resource::<DevConsolePaused>()
            .add_systems(Update, toggle_console_keybind)
            .add_systems(
                EguiPrimaryContextPass,
                (
                    scene_controls_panel,
                    #[cfg(feature = "phi-panel")]
                    phi_inspector_panel,
                )
                    .chain(),
            );
    }
}

fn toggle_console_keybind(keys: Res<ButtonInput<KeyCode>>, mut vis: ResMut<DevConsoleVisible>) {
    if keys.just_pressed(KeyCode::F1) {
        vis.0 = !vis.0;
    }
}

fn scene_controls_panel(
    mut contexts: EguiContexts,
    vis: Res<DevConsoleVisible>,
    mut paused: ResMut<DevConsolePaused>,
    mut time: ResMut<Time<Virtual>>,
) {
    if !vis.0 {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::SidePanel::left("dev_console_scene_controls")
        .default_width(220.0)
        .show(ctx, |ui| {
            ui.heading("Scene");
            ui.separator();
            let label = if paused.0 { "▶ Resume" } else { "⏸ Pause" };
            if ui.button(label).clicked() {
                paused.0 = !paused.0;
                if paused.0 {
                    time.pause();
                } else {
                    time.unpause();
                }
            }
            ui.separator();
            ui.label("F1: Toggle Console");
        });
}

#[cfg(feature = "phi-panel")]
fn phi_inspector_panel(
    mut contexts: EguiContexts,
    vis: Res<DevConsoleVisible>,
    query: Query<(Entity, &symtropy_bevy_core::PhysicsBody)>,
) {
    if !vis.0 {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::SidePanel::left("dev_console_phi")
        .default_width(220.0)
        .show(ctx, |ui| {
            ui.heading("Φ Inspector");
            ui.label(format!(
                "{} body(ies)",
                query.iter().count()
            ));
            ui.separator();
            ui.label("Phi tracking decoupled to break cycles.");
        });
}
