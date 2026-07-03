// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bevy 0.18 plugin for Sol Atlas globe rendering.
//!
//! Add `TerraAtlasPlugin` to your Bevy app to get a 3D globe with
//! orbital camera, markers, and arc routes — all driven by
//! `terra-atlas-core` shared types.

pub mod camera;
pub mod cell_entry;
pub mod cell_walk;
pub mod clouds;
pub mod data;
pub mod desci_evidence;
pub mod frame_capture;
pub mod globe;
pub mod h3_grid;
pub mod holographic_material;
pub mod markers;
pub mod selection;
pub mod timeline;

use bevy::prelude::*;

/// Main plugin — registers all Sol Atlas systems and resources.
pub struct TerraAtlasPlugin;

impl Plugin for TerraAtlasPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<camera::OrbitalCameraConfig>()
            .init_resource::<h3_grid::HoveredCell>()
            .init_resource::<cell_entry::CellZoomTransition>()
            .add_plugins(holographic_material::HolographicMaterialPlugin)
            .add_plugins(clouds::CloudMaterialPlugin)
            .add_plugins(desci_evidence::DeSciEvidencePlugin)
            .add_systems(Startup, (globe::spawn_globe, camera::spawn_camera))
            .add_systems(
                Update,
                (
                    camera::orbital_camera_system,
                    h3_grid::hover_cell_system,
                    h3_grid::draw_hovered_cell_system,
                    cell_entry::trigger_cell_zoom_system,
                    cell_entry::cancel_zoom_on_manual_input,
                    cell_entry::cell_zoom_transition_system,
                    clouds::update_cloud_time,
                )
                    .chain(),
            );
    }
}

/// Convert terra-atlas-core `[f32; 3]` to Bevy `Vec3`.
#[inline]
pub fn to_bevy_vec3(v: [f32; 3]) -> Vec3 {
    Vec3::new(v[0], v[1], v[2])
}

/// Convert Bevy `Vec3` to terra-atlas-core `[f32; 3]`.
#[inline]
pub fn from_bevy_vec3(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}
