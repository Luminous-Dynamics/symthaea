// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use nalgebra::SVector;

use crate::body::BodyHandle;

/// Collision event emitted when two bodies collide.
#[derive(Clone, Debug)]
pub struct CollisionEvent<const D: usize> {
    pub body_a: BodyHandle,
    pub body_b: BodyHandle,
    /// Impulse magnitude applied during resolution.
    pub impulse: f64,
    /// Contact normal.
    pub normal: SVector<f64, D>,
    /// Penetration depth.
    pub depth: f64,
}

/// Sensor overlap event (no collision resolution, just notification).
#[derive(Clone, Debug)]
pub struct SensorEvent {
    /// The sensor body.
    pub sensor: BodyHandle,
    /// The other body overlapping the sensor.
    pub other: BodyHandle,
}

/// Contact information from a collision between two bodies.
#[derive(Clone, Debug)]
pub struct ContactManifold<const D: usize> {
    /// Handle of body A.
    pub body_a: BodyHandle,
    /// Handle of body B.
    pub body_b: BodyHandle,
    /// Contact normal pointing from A to B.
    pub normal: SVector<f64, D>,
    /// Penetration depth (positive = overlapping).
    pub depth: f64,
    /// Contact point in world space (midpoint of contact).
    pub point: SVector<f64, D>,
}

impl<const D: usize> ContactManifold<D> {
    /// Impulse magnitude for elastic collision with given restitution.
    ///
    /// j = -(1 + e) * v_rel · n / (1/m_a + 1/m_b)
    pub fn impulse_magnitude(
        &self,
        relative_velocity: &SVector<f64, D>,
        inv_mass_a: f64,
        inv_mass_b: f64,
        restitution: f64,
    ) -> f64 {
        let v_rel_n = relative_velocity.dot(&self.normal);

        // Separating — no impulse needed
        if v_rel_n > 0.0 {
            return 0.0;
        }

        let denom = inv_mass_a + inv_mass_b;
        if denom < 1e-15 {
            return 0.0;
        }

        -(1.0 + restitution) * v_rel_n / denom
    }
}
