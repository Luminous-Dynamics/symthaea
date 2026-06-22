// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! High-frequency telemetric observation system.

use crate::celestial::CrystalFieldResource;
use bevy::prelude::*;
use symthaea_core::hdc::unified_hv::ContinuousHV;

/// Tracks global system integrity (entropy, Phi, and energy flux).
#[derive(Resource, Default)]
pub struct TelemetricRegistry {
    pub global_entropy: f32,
    pub active_phi: f32,
}

/// System that computes system-wide telemetry from the crystal field.
pub fn telemetric_observation_system(
    crystal: Res<CrystalFieldResource>,
    mut registry: ResMut<TelemetricRegistry>,
) {
    // Entropy is computed as 1 - global_cosmic_vector's magnitude variance
    let vec = &crystal.field.global_cosmic_vector.values;
    let mean: f32 = vec.iter().sum::<f32>() / vec.len() as f32;
    let variance: f32 = vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / vec.len() as f32;

    registry.global_entropy = 1.0 - variance.min(1.0);
    registry.active_phi = vec.iter().map(|&x| x.abs()).sum::<f32>() / vec.len() as f32;
}

pub struct TelemetryPlugin;

impl Plugin for TelemetryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TelemetricRegistry>()
            .add_systems(Update, telemetric_observation_system);
    }
}
