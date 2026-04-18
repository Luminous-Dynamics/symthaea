// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! # symtropy-devconsole
//!
//! Drop-in Bevy dev console plugin. Two panels (more coming):
//!
//! - **Scene controls** (left) — pause/resume `Time<Virtual>`, F1 toggle hint.
//! - **Φ Inspector** (left, feature `phi-panel`) — lists every `PhysicsBody`
//!   entity and its current Phi value with a unicode bar gauge. The
//!   differentiated panel; behind a feature gate so the base crate stays
//!   Apache/MIT.
//!
//! Toggle the whole console with **F1**.
//!
//! For a full entity-tree inspector, add
//! [`bevy_inspector_egui::quick::WorldInspectorPlugin`] separately — it manages
//! its own egui window and toggle. We don't bundle it here because of plugin-
//! order issues (its type registration runs before some Bevy default plugins
//! finish, causing reflection-registry panics in some app configurations).
//!
//! ## Quick start
//!
//! ```ignore
//! use bevy::prelude::*;
//! use symtropy_devconsole::SymtropyDevConsolePlugin;
//!
//! App::new()
//!     .add_plugins(DefaultPlugins)
//!     .add_plugins(SymtropyDevConsolePlugin::default())
//!     .run();
//! ```
//!
//! With the Φ panel:
//!
//! ```toml
//! symtropy-devconsole = { version = "0.1", features = ["phi-panel"] }
//! ```

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};

/// Toggle state for the dev console. Flipping this resource hides/shows all
/// panels. Spawned by [`SymtropyDevConsolePlugin`] and toggled by F1 via the
/// built-in keybind system.
#[derive(Resource, Debug, Clone, Copy)]
pub struct DevConsoleVisible(pub bool);

impl Default for DevConsoleVisible {
    fn default() -> Self {
        Self(true)
    }
}

/// Pause flag for downstream physics systems. The dev console flips it; user
/// systems may inspect `Res<DevConsolePaused>` and skip their logic when paused.
///
/// Note: `bevy_time` exposes `Time::pause`/`Time::unpause` on `Time<Virtual>`
/// in Bevy 0.18; if you advance physics by `Time<Virtual>::delta_secs` (the
/// default in most plugins) you don't need to read this resource — pausing
/// `Time<Virtual>` is enough. This resource is for systems that step on
/// `Time<Real>` or want a separate pause concept.
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
        // bevy_egui's default plugin enables multi-pass mode (recommended).
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
                    crate::phi_panel::phi_inspector_panel,
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
            ui.label("Time controls operate on Time<Virtual>.");
            ui.label("Press F1 to hide this panel.");
        });
}

#[cfg(feature = "phi-panel")]
mod phi_panel {
    use super::*;
    use bevy_egui::EguiContexts;
    use symtropy_bevy::{PhysicsBody, SymtropyPhysics};

    /// Lists every entity with a `PhysicsBody` and shows its current Phi value
    /// from the 2D `ConsciousnessField`. (3D / 4D variants would need parallel
    /// systems gated on the dimension's resource.)
    pub fn phi_inspector_panel(
        mut contexts: EguiContexts,
        vis: Res<DevConsoleVisible>,
        physics2: Option<Res<SymtropyPhysics<2>>>,
        physics3: Option<Res<SymtropyPhysics<3>>>,
        physics4: Option<Res<SymtropyPhysics<4>>>,
        query: Query<(Entity, &PhysicsBody)>,
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
                    "{} entity(ies) with PhysicsBody",
                    query.iter().count()
                ));
                ui.separator();
                egui::ScrollArea::vertical()
                    .max_height(400.0)
                    .show(ui, |ui| {
                        for (entity, body) in &query {
                            let phi = if let Some(p) = physics2.as_deref() {
                                Some(p.field.phi(body.handle))
                            } else if let Some(p) = physics3.as_deref() {
                                Some(p.field.phi(body.handle))
                            } else if let Some(p) = physics4.as_deref() {
                                Some(p.field.phi(body.handle))
                            } else {
                                None
                            };
                            if let Some(phi) = phi {
                                let phi_norm = (phi / 0.314).clamp(0.0, 1.0);
                                let bar = "█".repeat((phi_norm * 12.0) as usize);
                                ui.label(format!("{:?}  {:.3}  {}", entity, phi, bar));
                            } else {
                                ui.label(format!("{:?}  (no field)", entity));
                            }
                        }
                    });
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_default_visible() {
        let v = DevConsoleVisible::default();
        assert!(v.0);
    }

    #[test]
    fn paused_default_false() {
        let p = DevConsolePaused::default();
        assert!(!p.0);
    }

    #[test]
    fn plugin_constructs() {
        let _plugin = SymtropyDevConsolePlugin;
    }
}
