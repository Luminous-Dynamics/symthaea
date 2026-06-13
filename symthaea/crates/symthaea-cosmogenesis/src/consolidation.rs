// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::types::SemanticParticle;

/// Handles the 'Freezing' of learned semantic clusters into persistent long-term memory.
pub struct MemoryConsolidator {
    pub stability_threshold: f32,
}

impl MemoryConsolidator {
    pub fn new(stability_threshold: f32) -> Self {
        Self {
            stability_threshold,
        }
    }

    /// Freezes stable particles into 'Anchors'.
    /// Anchors have their velocity set to zero and mass significantly increased.
    pub fn consolidate(&self, particles: &mut Vec<SemanticParticle>, cluster_variances: &[f32]) {
        for (i, p) in particles.iter_mut().enumerate() {
            // If the particle is part of a stable cluster (low variance), anchor it
            if cluster_variances[i] < self.stability_threshold {
                p.velocity = vec![0.0; p.velocity.len()];
                p.mass *= 10.0; // Anchor mass significantly increased
                p.latent_mass *= 5.0;
            }
        }
    }
}
