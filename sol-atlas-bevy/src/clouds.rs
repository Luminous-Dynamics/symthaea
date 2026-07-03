// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Animated cloud shell material — not true volumetric raymarching (out of
//! scope for a sphere-shell layer), but dual-layer parallax scroll +
//! contrast-enhanced coverage + sun-facing silver-lining rim glow, the
//! combination that reads as "real" in satellite cloud photography without
//! the cost of a raymarched 3D noise field.

use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

pub type CloudMaterial = ExtendedMaterial<StandardMaterial, CloudExtension>;

#[derive(Asset, AsBindGroup, Debug, Clone, Reflect)]
pub struct CloudExtension {
    #[uniform(100)]
    pub settings: CloudSettings,
    #[texture(101)]
    #[sampler(102)]
    pub cloud_texture: Handle<Image>,
}

#[derive(Debug, Clone, Copy, Reflect, ShaderType)]
pub struct CloudSettings {
    pub time: f32,
    /// UV scroll speed for the primary (lower, faster-drifting) layer.
    pub scroll_speed_1: f32,
    /// UV scroll speed for the secondary (offset-scaled, slower) layer —
    /// combining two independently-drifting samples of the same texture
    /// gives a cheap sense of layered depth without true volumetrics.
    pub scroll_speed_2: f32,
    /// Contrast power applied to sampled luminance — higher = more
    /// distinct, puffier formations instead of a flat haze.
    pub contrast: f32,
    /// Overall coverage/opacity multiplier.
    pub coverage: f32,
    /// Silver-lining rim glow intensity (the bright edge real
    /// satellite-photographed clouds show against the sun).
    pub silver_lining: f32,
    pub _padding: f32,
    pub _padding2: f32,
}

impl Default for CloudSettings {
    fn default() -> Self {
        Self {
            time: 0.0,
            scroll_speed_1: 0.008,
            scroll_speed_2: -0.0045,
            contrast: 2.2,
            coverage: 0.8,
            silver_lining: 1.6,
            _padding: 0.0,
            _padding2: 0.0,
        }
    }
}

impl MaterialExtension for CloudExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/clouds.wgsl".into()
    }
}

/// Registers `CloudMaterial` with Bevy's asset/render pipeline. Without
/// this, `Assets<CloudMaterial>` never exists as a resource — same
/// missing-MaterialPlugin bug class found and fixed twice already this
/// session (NdSlicingPlugin, HolographicMaterialPlugin).
pub struct CloudMaterialPlugin;

impl Plugin for CloudMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<CloudMaterial>::default());
    }
}

/// Advances the cloud shader's time uniform each frame — mirrors
/// `holographic_material::update_holographic_time`.
pub fn update_cloud_time(time: Res<Time>, mut materials: ResMut<Assets<CloudMaterial>>) {
    for (_, mat) in materials.iter_mut() {
        mat.extension.settings.time = time.elapsed_secs();
    }
}
