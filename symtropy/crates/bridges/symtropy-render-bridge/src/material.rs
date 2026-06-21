// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Bevy Material extension for 4D-to-3D cross-section slicing.

use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, StandardMaterial};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::render::storage::ShaderStorageBuffer;
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
    pub phi_global: f32,
    pub surprise_global: f32,
    pub harmony_global: f32,
    pub energy_level: f32,
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
                phi_global: 0.0,
                surprise_global: 0.0,
                harmony_global: 0.0,
                energy_level: 0.0,
            },
        }
    }
}

/// Type alias for the telemetry visualization material.
pub type TelemetryMaterial = ExtendedMaterial<StandardMaterial, TelemetryExtension>;

/// Material extension that deforms vertices and tints surfaces using active telemetry buffers.
#[derive(Asset, AsBindGroup, Debug, Clone, Reflect)]
pub struct TelemetryExtension {
    #[uniform(100)]
    pub settings: TelemetrySettings,

    #[storage(101, read_only)]
    pub nodes: Handle<ShaderStorageBuffer>,
}

#[derive(Default, Debug, Clone, Copy, Reflect, ShaderType)]
pub struct TelemetrySettings {
    pub node_index: u32,
    pub time: f32,
    pub deformation_scale: f32,
    pub noise_frequency: f32,
}

impl MaterialExtension for TelemetryExtension {
    fn vertex_shader() -> ShaderRef {
        "shaders/telemetry_deform.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/telemetry_deform.wgsl".into()
    }
}

pub struct TelemetryMaterialPlugin;

impl Plugin for TelemetryMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<TelemetryMaterial>::default());
    }
}
