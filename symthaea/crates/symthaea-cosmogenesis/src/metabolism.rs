// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::types::SemanticParticle;

/// Manages the metabolic energy expenditure of semantic particles.
pub struct SemanticMetabolizer {
    pub maintenance_cost_rate: f32,
}

impl SemanticMetabolizer {
    pub fn new(maintenance_cost_rate: f32) -> Self {
        Self {
            maintenance_cost_rate,
        }
    }

    /// Particles drain energy proportional to their mass and complexity.
    /// If energy reaches zero, the particle becomes 'unstable' (evaporates).
    pub fn metabolize(&self, particles: &mut Vec<SemanticParticle>) {
        for p in particles.iter_mut() {
            let cost = p.latent_mass * self.maintenance_cost_rate;
            p.energy -= cost;
        }
    }
}
