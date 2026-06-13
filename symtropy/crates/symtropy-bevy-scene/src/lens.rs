// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! High-performance holographic lens system for gradient-based physics.

use bevy::prelude::*;
use nalgebra::Vector3;
use symtropy_core_stable::spacetime::SpacetimeCrystalField;

/// Marker component for bodies affected by crystallized gravity.
#[derive(Component, Default)]
pub struct GravitationalLens;

/// System that computes the hyperdimensional potential gradient
/// and injects forces into bodies in the simulation.
pub fn apply_crystal_gravity_system(
    crystal: Res<crate::celestial::CrystalFieldResource>,
    mut query: Query<&mut Transform, With<GravitationalLens>>,
    time: Res<Time>,
) {
    let epsilon = 0.05; // Gradient step
    let g_lens = 5e6;   // Force scaling factor
    let dt = time.delta_secs();

    for mut transform in query.iter_mut() {
        let pos = Vector3::new(
            transform.translation.x as f64,
            transform.translation.y as f64,
            transform.translation.z as f64,
        );
        
        let p_base = crystal.field.probe(pos);
        let px = crystal.field.probe(pos + Vector3::x() * epsilon);
        let py = crystal.field.probe(pos + Vector3::y() * epsilon);
        let pz = crystal.field.probe(pos + Vector3::z() * epsilon);
        
        let gx = (px - p_base) / epsilon as f32;
        let gy = (py - p_base) / epsilon as f32;
        let gz = (pz - p_base) / epsilon as f32;
        
        let force = Vec3::new(-gx, -gy, -gz) * g_lens;
        
        // Direct integration into position/transform
        transform.translation += force * dt * dt * 0.5;
    }
}
