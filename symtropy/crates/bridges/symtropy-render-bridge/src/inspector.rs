// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Symtropy Inspector — immediate-mode UI for 4D manifold and coupling control.

use crate::material::NdSlicingMaterial;
use bevy::pbr::MeshMaterial3d;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

pub struct SymtropyInspectorPlugin;

impl Plugin for SymtropyInspectorPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, inspector_ui);
    }
}

fn inspector_ui(
    mut contexts: EguiContexts,
    mut query_material: Query<&mut MeshMaterial3d<NdSlicingMaterial>>,
    mut materials: ResMut<Assets<NdSlicingMaterial>>,
) {
    let ctx = contexts.ctx_mut();
    // In Bevy 0.18 / bevy_egui 0.40, ctx_mut returns a Result
    let ctx = match ctx {
        Ok(c) => c,
        Err(_) => return,
    };

    egui::Window::new("💠 Symtropy Manifold Inspector")
        .default_width(320.0)
        .show(ctx, |ui| {
            ui.heading("Minkowski Scrubber");
            ui.separator();

            // In a real implementation, we'd iterate over materials
            // For now, assume one active heatmap material
            #[allow(clippy::collapsible_if)]
            if let Some(mesh_mat) = query_material.iter_mut().next() {
                if let Some(mut material) = materials.get_mut(&*mesh_mat) {
                    let settings = &mut material.extension.settings;

                    ui.add(
                        egui::Slider::new(&mut settings.w_slice, -10.0..=10.0).text("4D Slice (w)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut settings.slice_thickness, 0.01..=2.0)
                            .text("Thickness"),
                    );

                    ui.add_space(10.0);
                    ui.heading("Coupling Gain Matrix");
                    ui.separator();

                    ui.add(
                        egui::Slider::new(&mut settings.phi_global, 0.0..=1.0)
                            .text("Integration (Φ)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut settings.surprise_global, 0.0..=1.0)
                            .text("Surprise (Δ)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut settings.harmony_global, 0.0..=1.0)
                            .text("Harmony (Ω)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut settings.energy_level, 0.0..=1.0)
                            .text("Energy (Joules)"),
                    );
                }
            }

            ui.add_space(10.0);
            ui.heading("Thermodynamic Dashboard");
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Landauer Bound:");
                ui.colored_label(egui::Color32::GREEN, "2.3e-21 J/bit");
            });
            ui.horizontal(|ui| {
                ui.label("Status:");
                ui.colored_label(egui::Color32::from_rgb(0, 214, 200), "Homeostasis");
            });
        });
}
