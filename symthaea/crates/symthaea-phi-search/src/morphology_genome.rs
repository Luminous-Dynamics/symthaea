// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Morphological Genome — Physical Architecture Search.
//!
//! Encodes the physical form (limb lengths, joint limits, torque bias)
//! into the Phi-guided search space, allowing robots to evolve their
//! own hardware to minimize systemic surprise.

use serde::{Deserialize, Serialize};

/// Encodes a physical robotic morphology for evolutionary search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphologicalGenome {
    pub base_mass: f64,
    pub limb_length_scales: Vec<f32>, // Scaling per segment
    pub joint_stiffness_bias: Vec<f32>,
    pub max_torque_multipliers: Vec<f32>,
    pub degree_of_freedom_map: Vec<u8>, // Bits encoding active axes
    pub seed: u64,
}

impl MorphologicalGenome {
    pub fn flagship_64dof() -> Self {
        Self {
            base_mass: 95.0,
            limb_length_scales: vec![1.0; 8], // 8 major limb segments
            joint_stiffness_bias: vec![1.0; 64],
            max_torque_multipliers: vec![1.0; 64],
            degree_of_freedom_map: vec![1; 64], // All 64 active
            seed: 42,
        }
    }

    /// Mutate the physical form based on Phi gradient.
    pub fn evolve_physical_form(&mut self, surprise_gradient: f32, rate: f32) {
        let delta = surprise_gradient * rate;
        for scale in &mut self.limb_length_scales {
            // If surprise is high, attempt morphological variance
            *scale += delta * 0.1;
            *scale = scale.clamp(0.5, 2.0);
        }
    }
}
