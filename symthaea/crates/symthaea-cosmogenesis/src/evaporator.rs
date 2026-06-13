// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::types::SemanticParticle;

/// Manages the 'pruning' of transient, low-utility concepts from the semantic manifold.
pub struct SemanticEvaporator {
    pub survival_threshold: f32,
}

impl SemanticEvaporator {
    pub fn new(survival_threshold: f32) -> Self {
        Self { survival_threshold }
    }

    /// Removes particles that do not meet the survival threshold.
    /// Particles are assessed based on their 'latent mass' (historical priority).
    pub fn evaporate(&self, particles: &mut Vec<SemanticParticle>) {
        particles.retain(|p| p.latent_mass > self.survival_threshold);
    }
}
