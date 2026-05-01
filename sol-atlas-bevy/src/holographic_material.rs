// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Custom holographic material for the Sol Atlas globe.
//! Uses Bevy's ExtendedMaterial pattern to add Fresnel edge glow,
//! scanlines, and noise on top of the standard PBR material.

use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

/// Holographic material extension — adds Fresnel, scanlines, and noise
/// on top of StandardMaterial's PBR pipeline.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct HolographicExtension {
    /// Fresnel glow color (default: Mycelix cyan).
    #[uniform(100)]
    pub fresnel_color: LinearRgba,
    /// Fresnel falloff power (higher = tighter edge glow). Default: 3.0.
    #[uniform(100)]
    pub fresnel_power: f32,
    /// Scanline scroll speed (units per second). Default: 0.5.
    #[uniform(100)]
    pub scanline_speed: f32,
    /// Scanline density (lines per world unit). Default: 15.0.
    #[uniform(100)]
    pub scanline_density: f32,
    /// Base hologram transparency. Default: 0.6.
    #[uniform(100)]
    pub hologram_alpha: f32,
    /// Current time (updated each frame by the time system).
    #[uniform(100)]
    pub time: f32,
    /// Enable holographic effects (1.0 = full, 0.0 = PBR only for Satellite mode).
    #[uniform(100)]
    pub enable_holographic: f32,
    // Padding for 16-byte alignment
    #[uniform(100)]
    pub _padding: f32,
}

impl Default for HolographicExtension {
    fn default() -> Self {
        Self {
            fresnel_color: LinearRgba::new(0.0, 0.87, 1.0, 1.0), // Mycelix cyan
            fresnel_power: 3.0,
            scanline_speed: 0.5,
            scanline_density: 15.0,
            hologram_alpha: 0.6,
            time: 0.0,
            enable_holographic: 1.0,
            _padding: 0.0,
        }
    }
}

impl MaterialExtension for HolographicExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/holographic.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        "shaders/holographic.wgsl".into()
    }
}

/// Type alias for the combined material.
pub type HolographicMaterial = ExtendedMaterial<StandardMaterial, HolographicExtension>;

/// System to update the time uniform on holographic materials each frame.
pub fn update_holographic_time(
    time: Res<Time>,
    mut materials: ResMut<Assets<HolographicMaterial>>,
) {
    let t = time.elapsed_secs();
    for (_, mat) in materials.iter_mut() {
        mat.extension.time = t;
    }
}
