// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};

/// Learnable cross-modulation matrix (4×4 Hebbian weights).
///
/// Entry `weights[i][j]` = how transmitter `i` modulates transmitter `j`'s production.
/// Positive = excitatory, negative = inhibitory. Initialized with biological priors
/// and updated via Hebbian co-activation of phasic bursts.
///
/// Science: Hebb (1949) — neurons that fire together wire together.
/// Hasselmo (2006) — ACh/DA/NE/5-HT interact through learned modulatory pathways.
///
/// Indices: DA=0, NE=1, 5-HT=2, ACh=3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModulationMatrix {
    /// 4×4 weight matrix: [source][target]
    pub weights: [[f32; 4]; 4],
    /// Hebbian learning rate (very slow: 0.001)
    learning_rate: f32,
}

impl Default for CrossModulationMatrix {
    fn default() -> Self {
        // Biological priors from known neurotransmitter interactions
        let mut weights = [[0.0_f32; 4]; 4];
        weights[0][1] = -0.03; // DA→NE: exploitation suppresses exploration
        weights[2][1] = -0.02; // 5-HT→NE: contentment dampens arousal
        weights[1][3] = 0.02; // NE→ACh: arousal sharpens attention
        Self {
            weights,
            learning_rate: 0.001,
        }
    }
}

impl CrossModulationMatrix {
    /// Compute modulation deltas for each channel based on current levels.
    #[inline]
    pub fn apply(&self, levels: &[f32; 4]) -> [f32; 4] {
        let mut deltas = [0.0_f32; 4];
        for (src, &level) in levels.iter().enumerate() {
            for (tgt, delta) in deltas.iter_mut().enumerate() {
                if src != tgt {
                    *delta += self.weights[src][tgt] * level;
                }
            }
        }
        deltas
    }

    /// Hebbian update from phasic co-activation.
    /// Δw[i][j] = lr × phasic[i] × phasic[j] with weight decay to prevent runaway.
    pub fn hebbian_update(&mut self, phasics: &[f32; 4]) {
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    // Hebbian: co-activation strengthens connection
                    self.weights[i][j] += self.learning_rate * phasics[i] * phasics[j];
                    // Weight decay prevents runaway (×0.999/cycle)
                    self.weights[i][j] *= 0.999;
                    // Clamp to prevent extreme modulation
                    self.weights[i][j] = self.weights[i][j].clamp(-0.1, 0.1);
                }
            }
        }
    }
}
