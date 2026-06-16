// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Autonomous Curiosity — Phi-Gradient Driven Exploration
//!
//! Uses the Streaming Phi Gradient to drive the robot toward high-complexity
//! environments where new structural information can be integrated.

use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::tiered_phi::streaming::{GradientConfig, StreamingPhiGradient};

/// Drives autonomous exploration based on the gradient of integrated information.
pub struct CuriosityEngine {
    /// Tracks Φ-gradient over recent observations
    pub gradient_tracker: StreamingPhiGradient,
    /// Current curiosity level (0.0 - 1.0)
    pub curiosity_level: f64,
}

impl CuriosityEngine {
    pub fn new() -> Self {
        Self {
            gradient_tracker: StreamingPhiGradient::new(GradientConfig::default()),
            curiosity_level: 0.5,
        }
    }

    /// Update curiosity based on a set of cognitive components.
    /// Returns an exploration vector (HDC) that points toward novelty.
    pub fn update(&mut self, components: &[BinaryHV]) -> Option<BinaryHV> {
        let grad = self.gradient_tracker.compute_gradient(components);

        // If the mean gradient is negative or low, curiosity increases
        // (the system is bored and wants to find more 'integratable' content).
        let mean_grad: f64 = grad.component_gradients.iter().sum::<f64>() / components.len() as f64;

        if mean_grad < 0.1 {
            self.curiosity_level = (self.curiosity_level + 0.1).min(1.0);
        } else {
            self.curiosity_level = (self.curiosity_level - 0.05).max(0.0);
        }

        if self.curiosity_level > 0.7 {
            // High curiosity: bundle the 'negative' components (the boring ones)
            // and return a vector that is ORTHOGONAL to them.
            let boring_indices: Vec<usize> = grad
                .component_gradients
                .iter()
                .enumerate()
                .filter(|(_, &g)| g < 0.2)
                .map(|(i, _)| i)
                .collect();

            if !boring_indices.is_empty() {
                let boring_hvs: Vec<BinaryHV> = boring_indices
                    .iter()
                    .map(|&i| components[i].clone())
                    .collect();
                let boring_consensus = BinaryHV::bundle(&boring_hvs);
                // Return a random vector as a "search direction" away from boring content
                return Some(BinaryHV::random(0).bind(&boring_consensus));
            }
        }

        None
    }
}

impl Default for CuriosityEngine {
    fn default() -> Self {
        Self::new()
    }
}
