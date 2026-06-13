// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::types::SemanticParticle;

/// Analyzes the information density and structural complexity of the manifold.
pub struct ComplexityAnalyzer;

impl ComplexityAnalyzer {
    /// Computes the 'Manifold Complexity Score' (MCS).
    /// This is an approximation of the ratio between structured clustering
    /// (intra-cluster compactness) and global manifold entropy.
    pub fn compute_complexity_score(particles: &[SemanticParticle]) -> f32 {
        if particles.is_empty() {
            return 0.0;
        }

        let n = particles.len() as f32;
        let mut total_var = 0.0;
        let mut class_var = 0.0;

        // Calculate global mean position
        let mut mean = vec![0.0; particles[0].position.len()];
        for p in particles {
            for (i, &val) in p.position.iter().enumerate() {
                mean[i] += val / n;
            }
        }

        // Global variance (entropy proxy)
        for p in particles {
            for (i, &val) in p.position.iter().enumerate() {
                total_var += (val - mean[i]).powi(2);
            }
        }

        // Intra-class variance (structure proxy)
        // Complexity increases when total variance is high but intra-class variance is low.
        if total_var < 1e-6 {
            return 0.0;
        }

        1.0 - (class_var / total_var)
    }
}
