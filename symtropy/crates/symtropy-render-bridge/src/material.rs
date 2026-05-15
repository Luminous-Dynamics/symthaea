// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bevy Material extension for 4D-to-3D cross-section slicing.

use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

pub struct NdSlicingPlugin;

impl Plugin for NdSlicingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<
            ExtendedMaterial<StandardMaterial, NdSlicingExtension>,
        >::default());
    }
}

/// Type alias for the 4D slicing material.
pub type NdSlicingMaterial = ExtendedMaterial<StandardMaterial, NdSlicingExtension>;

/// Material extension that slices 4D objects into 3D cross-sections.
#[derive(Asset, AsBindGroup, Debug, Clone, Reflect)]
pub struct NdSlicingExtension {
    #[uniform(100)]
    pub settings: NdSlicingSettings,
}

#[derive(Default, Debug, Clone, Copy, Reflect, ShaderType)]
pub struct NdSlicingSettings {
    pub w_pos: f32,
    pub w_slice: f32,
    pub slice_thickness: f32,
    pub edge_fade: f32,
    pub time: f32,
}

impl MaterialExtension for NdSlicingExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/nd_slicing.wgsl".into()
    }
}

impl Default for NdSlicingExtension {
    fn default() -> Self {
        Self {
            settings: NdSlicingSettings {
                w_pos: 0.0,
                w_slice: 0.0,
                slice_thickness: 1.0,
                edge_fade: 1.0,
                time: 0.0,
            },
        }
    }
}
