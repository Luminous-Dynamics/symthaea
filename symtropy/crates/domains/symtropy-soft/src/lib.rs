// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Soft-body and cloth dynamics for Symtropy using XPBD.

use nalgebra::SVector;
use serde::{Deserialize, Serialize};

/// A single particle in a soft body.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoftBodyParticle<const D: usize> {
    pub position: SVector<f64, D>,
    pub previous_position: SVector<f64, D>,
    pub velocity: SVector<f64, D>,
    pub inv_mass: f64,
}

/// XPBD constraint (e.g., distance, volume).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XpbdConstraint {
    pub particle_a: usize,
    pub particle_b: usize,
    pub rest_length: f64,
    pub compliance: f64, // Inverse stiffness
}

/// A soft body instance (cloth or volume mesh).
pub struct SoftBody<const D: usize> {
    pub particles: Vec<SoftBodyParticle<D>>,
    pub constraints: Vec<XpbdConstraint>,
}

impl<const D: usize> SoftBody<D> {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Step the soft body simulation using XPBD.
    pub fn step(&mut self, dt: f64, gravity: &SVector<f64, D>, iterations: usize) {
        if dt <= 0.0 {
            return;
        }

        // 1. Prediction (explicit integration)
        for p in &mut self.particles {
            if p.inv_mass > 0.0 {
                p.previous_position = p.position;
                p.velocity += gravity * dt;
                p.position += p.velocity * dt;
            }
        }

        // 2. Constraint Projection (XPBD loop)
        let sub_dt = dt / iterations as f64;
        for _ in 0..iterations {
            self.solve_constraints(sub_dt);
        }

        // 3. Update Velocities (implicit)
        for p in &mut self.particles {
            if p.inv_mass > 0.0 {
                p.velocity = (p.position - p.previous_position) / dt;
            }
        }
    }

    fn solve_constraints(&mut self, dt: f64) {
        for c in &self.constraints {
            let p_a_pos = self.particles[c.particle_a].position;
            let p_b_pos = self.particles[c.particle_b].position;
            let inv_m_a = self.particles[c.particle_a].inv_mass;
            let inv_m_b = self.particles[c.particle_b].inv_mass;

            let dir = p_a_pos - p_b_pos;
            let len = dir.norm();
            if len < 1e-9 {
                continue;
            }

            let constraint_val = len - c.rest_length;
            let alpha = c.compliance / (dt * dt);

            // Lagrangrian multiplier update
            let w_sum = inv_m_a + inv_m_b;
            if w_sum + alpha <= 0.0 {
                continue;
            }

            let d_lambda = -constraint_val / (w_sum + alpha);
            let p = dir * (d_lambda / len);

            if inv_m_a > 0.0 {
                self.particles[c.particle_a].position += p * inv_m_a;
            }
            if inv_m_b > 0.0 {
                self.particles[c.particle_b].position -= p * inv_m_b;
            }
        }
    }
}
