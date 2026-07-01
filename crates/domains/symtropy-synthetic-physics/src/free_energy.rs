// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Structural Free Energy — the control law for graph update acceptance.
//!
//! This is NOT called "gravity." It is called `StructuralFreeEnergy`.
//!
//! ## Objective
//!
//! ```text
//! F_structural = α · complexity_cost
//!              + β · prediction_error
//!              + γ · entropy_growth
//!              + δ · |dimension - target_dimension|
//!              - ε · coherence_reward
//! ```
//!
//! A graph update is accepted if `ΔF < 0` OR if safety guards approve it.
//!
//! ## Rationale
//!
//! The graph should not update arbitrarily. This objective keeps it bounded:
//! - **complexity_cost**: penalizes edge count growth (prevents hairball)
//! - **prediction_error**: penalizes departure from a generative model (FEP-like)
//! - **entropy_growth**: penalizes increasing degree entropy (prevents disorder)
//! - **dimension_deviation**: pulls toward the target intrinsic dimension
//! - **coherence_reward**: rewards clustering coefficient (local structure)

use serde::{Deserialize, Serialize};

use crate::{graph::SyntheticGraph, update_rules::UpdateRule};

/// Weights for the structural free energy objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralFreeEnergy {
    /// Weight on edge count growth (complexity).
    pub alpha: f64,
    /// Weight on departure from generative model (prediction error).
    pub beta: f64,
    /// Weight on degree entropy growth.
    pub gamma: f64,
    /// Weight on intrinsic dimension deviation from target.
    pub delta: f64,
    /// Weight on clustering coefficient (coherence reward — negative contribution).
    pub epsilon: f64,
    /// Target intrinsic dimension.
    pub target_dimension: f64,
    /// Reference edge density (edges / nodes). Updates that exceed this are penalized.
    pub reference_density: f64,
}

impl Default for StructuralFreeEnergy {
    fn default() -> Self {
        Self {
            alpha: 0.1,    // mild complexity cost
            beta: 0.2,     // moderate prediction error
            gamma: 0.15,   // moderate entropy cost
            delta: 0.3,    // strong dimension pull (2D target)
            epsilon: 0.25, // reward for local coherence
            target_dimension: 2.0,
            reference_density: 3.0, // avg degree ≈ 3
        }
    }
}

impl StructuralFreeEnergy {
    /// Compute the change in structural free energy if the rule were applied.
    ///
    /// Returns `ΔF`. Negative = update improves objective. Positive = update worsens it.
    ///
    /// In v0.1 this uses the current graph state as a proxy; a true ΔF would require
    /// computing the post-update graph metrics (Phase 2 refinement).
    pub fn compute_delta(&self, graph: &SyntheticGraph, rule: &UpdateRule) -> f64 {
        let n = graph.node_count() as f64;
        let e = graph.edge_count() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let current_density = e / n;
        let current_dim = graph.estimate_dimension();
        let current_entropy = graph.degree_entropy();
        let current_clustering = graph.clustering_coefficient();

        // Complexity cost: how much does this rule tend to add edges?
        let complexity_cost = self.alpha * (current_density - self.reference_density).max(0.0);

        // Prediction error: departure from target dimension
        let prediction_error = self.beta * (current_dim - self.target_dimension).abs();

        // Entropy growth proxy: current entropy relative to log(n) maximum
        let max_entropy = (n as f64).ln();
        let entropy_excess = (current_entropy / max_entropy.max(1.0)).min(1.0);
        let entropy_cost = self.gamma * entropy_excess;

        // Dimension deviation (signed pull toward target)
        let dim_deviation = self.delta * (current_dim - self.target_dimension).abs();

        // Coherence reward (clustering coefficient)
        let coherence_reward = self.epsilon * current_clustering;

        // Rule-specific modifier: aggressive rules cost more
        let rule_modifier = rule.complexity_modifier();

        (complexity_cost + prediction_error + entropy_cost + dim_deviation - coherence_reward)
            * rule_modifier
    }

    /// The current absolute structural free energy (not a delta — for logging).
    pub fn compute_absolute(&self, graph: &SyntheticGraph) -> f64 {
        let n = graph.node_count() as f64;
        let e = graph.edge_count() as f64;
        if n == 0.0 {
            return f64::INFINITY;
        }

        let density = e / n;
        let dim = graph.estimate_dimension();
        let entropy = graph.degree_entropy();
        let clustering = graph.clustering_coefficient();
        let max_entropy = n.ln().max(1.0);

        self.alpha * (density - self.reference_density).max(0.0)
            + self.beta * (dim - self.target_dimension).abs()
            + self.gamma * (entropy / max_entropy).min(1.0)
            + self.delta * (dim - self.target_dimension).powi(2)
            - self.epsilon * clustering
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::SyntheticGraph;

    #[test]
    fn free_energy_finite_on_initial_graph() {
        let graph = SyntheticGraph::new_with_seed(42, 16);
        let fe = StructuralFreeEnergy::default();
        let f = fe.compute_absolute(&graph);
        assert!(f.is_finite(), "free energy must be finite: {f}");
        println!("Initial structural free energy: {f:.4}");
    }

    #[test]
    fn delta_finite_on_initial_graph() {
        use crate::update_rules::UpdateRule;
        let graph = SyntheticGraph::new_with_seed(42, 16);
        let fe = StructuralFreeEnergy::default();
        let rule = UpdateRule::TriangulationPressure { probability: 0.1 };
        let delta = fe.compute_delta(&graph, &rule);
        assert!(delta.is_finite(), "ΔF must be finite: {delta}");
    }
}
