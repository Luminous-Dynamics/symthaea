// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fractal cognitive architecture: self-similar processing at every scale.
//!
//! Each cognitive layer (perception, cognition, meta-cognition) implements
//! the same FractalCognitiveLayer trait, creating a recursive structure
//! where each level mirrors the whole. This is architecturally inspired
//! by:
//!
//! - Hofstadter's "strange loops" (consciousness from self-reference)
//! - Tononi's IIT (integration at every scale)
//! - Scale-free networks in neuroscience (self-similar connectivity)
//!
//! The fractal structure enables:
//! 1. Meta-cognition that uses the same mechanisms as base cognition
//! 2. Scale-invariant consciousness metrics (fractal dimension)
//! 3. Recursive introspection (thinking about thinking about thinking)
//!
//! # Architecture
//!
//! ```text
//! Level 0 (Perception):  sense → encode → predict → surprise
//! Level 1 (Cognition):   sense(L0) → encode → predict → surprise
//! Level 2 (Meta):        sense(L1) → encode → predict → surprise
//! Level 3 (Meta-meta):   sense(L2) → encode → predict → surprise
//! ```
//!
//! Each level has the same interface but operates on the output of the
//! level below. The fractal dimension of the combined system exceeds
//! that of any individual level.

use serde::{Deserialize, Serialize};

/// A single cognitive processing layer in the fractal hierarchy.
///
/// Every layer — from raw perception to meta-meta-cognition — implements
/// this same interface. The fractal property emerges from nesting:
/// each layer's `input` comes from the layer below's `output`.
pub trait FractalCognitiveLayer {
    /// Process input and produce output for the next level.
    fn process(&mut self, input: &[f32]) -> Vec<f32>;

    /// Prediction error at this level (surprise).
    fn prediction_error(&self) -> f32;

    /// Integration measure at this level (local Phi).
    fn integration(&self) -> f32;

    /// Fractal depth (0 = perception, 1 = cognition, 2 = meta, ...).
    fn depth(&self) -> usize;
}

/// A recursive stack of cognitive layers forming a fractal hierarchy.
///
/// Each cycle, input propagates bottom-up through all layers, then
/// prediction errors propagate top-down. The aggregate fractal dimension
/// measures the complexity of the combined state trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalStack {
    /// Number of active levels (2 = basic, 4 = deep meta-cognition).
    pub n_levels: usize,
    /// Per-level prediction errors from last cycle.
    pub level_errors: Vec<f32>,
    /// Per-level integration (Phi) from last cycle.
    pub level_integration: Vec<f32>,
    /// Aggregate fractal dimension across all levels.
    pub aggregate_fractal_dim: f32,
    /// Whether the stack has achieved self-referential closure
    /// (top level's output influences bottom level).
    pub strange_loop_active: bool,
}

impl FractalStack {
    /// Create a new fractal stack with the given depth.
    pub fn new(n_levels: usize) -> Self {
        Self {
            n_levels: n_levels.max(2),
            level_errors: vec![0.0; n_levels],
            level_integration: vec![0.0; n_levels],
            aggregate_fractal_dim: 1.0,
            strange_loop_active: false,
        }
    }

    /// Update the stack with current cycle data.
    ///
    /// `level_data` is a slice of (prediction_error, integration) per level.
    pub fn update(&mut self, level_data: &[(f32, f32)]) {
        for (i, &(pe, phi)) in level_data.iter().enumerate().take(self.n_levels) {
            self.level_errors[i] = pe;
            self.level_integration[i] = phi;
        }

        // Aggregate fractal dimension: scale-invariant if errors are similar across levels
        // D ≈ 1 + entropy(normalized_errors)
        let total_error: f32 = self.level_errors.iter().sum::<f32>().max(1e-6);
        let mut entropy = 0.0f32;
        for &e in &self.level_errors {
            let p = e / total_error;
            if p > 1e-8 {
                entropy -= p * p.ln();
            }
        }
        let max_entropy = (self.n_levels as f32).ln().max(1e-6);
        self.aggregate_fractal_dim = 1.0 + entropy / max_entropy;

        // Strange loop: detected when top-level error correlates with bottom-level output
        // (simplified: top error < 0.3 AND bottom integration > 0.5 means the system
        // is successfully modeling itself)
        self.strange_loop_active = self.n_levels >= 3
            && self.level_errors.last().copied().unwrap_or(1.0) < 0.3
            && self.level_integration.first().copied().unwrap_or(0.0) > 0.5;
    }

    /// Get a summary suitable for telemetry.
    pub fn summary(&self) -> FractalSummary {
        FractalSummary {
            n_levels: self.n_levels,
            aggregate_fractal_dim: self.aggregate_fractal_dim,
            strange_loop_active: self.strange_loop_active,
            mean_integration: if self.level_integration.is_empty() {
                0.0
            } else {
                self.level_integration.iter().sum::<f32>() / self.level_integration.len() as f32
            },
        }
    }
}

/// Compact fractal state for telemetry and display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalSummary {
    pub n_levels: usize,
    pub aggregate_fractal_dim: f32,
    pub strange_loop_active: bool,
    pub mean_integration: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractal_stack_basic() {
        let mut stack = FractalStack::new(3);
        stack.update(&[(0.5, 0.3), (0.4, 0.4), (0.3, 0.5)]);
        assert!(stack.aggregate_fractal_dim > 1.0);
        assert!(stack.aggregate_fractal_dim < 2.5);
    }

    #[test]
    fn test_strange_loop_detection() {
        let mut stack = FractalStack::new(3);
        // Top error low, bottom integration high → strange loop
        stack.update(&[(0.2, 0.6), (0.3, 0.5), (0.1, 0.4)]);
        assert!(stack.strange_loop_active);
    }

    #[test]
    fn test_no_strange_loop_when_too_shallow() {
        let mut stack = FractalStack::new(2);
        stack.update(&[(0.2, 0.6), (0.1, 0.5)]);
        assert!(!stack.strange_loop_active); // Needs 3+ levels
    }

    #[test]
    fn test_uniform_errors_high_dimension() {
        let mut stack = FractalStack::new(4);
        // All equal errors → maximum entropy → highest fractal dimension
        stack.update(&[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]);
        assert!(stack.aggregate_fractal_dim > 1.8);
    }

    #[test]
    fn test_concentrated_error_low_dimension() {
        let mut stack = FractalStack::new(4);
        // All error at one level → low entropy → low fractal dimension
        stack.update(&[(1.0, 0.5), (0.01, 0.5), (0.01, 0.5), (0.01, 0.5)]);
        assert!(stack.aggregate_fractal_dim < 1.3);
    }
}
