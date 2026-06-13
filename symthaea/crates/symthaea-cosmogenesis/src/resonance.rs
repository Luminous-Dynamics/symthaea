// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::types::SemanticParticle;

/// Analyzes semantic resonance (harmonic synchronization) between particles.
pub struct ResonanceAnalyzer {
    pub resonance_coupling: f32,
}

impl ResonanceAnalyzer {
    pub fn new(resonance_coupling: f32) -> Self {
        Self { resonance_coupling }
    }

    /// Computes the resonant force between two particles based on frequency alignment.
    pub fn compute_resonance_force(
        &self,
        p1: &SemanticParticle,
        p2: &SemanticParticle,
        dist_vec: &[f32],
    ) -> Vec<f32> {
        let freq_diff = (p1.frequency - p2.frequency).abs();
        // Coupling is strong when frequency difference is close to zero
        let coupling = self.resonance_coupling * (1.0 / (1.0 + freq_diff));

        dist_vec.iter().map(|&d| -coupling * d).collect()
    }
}
