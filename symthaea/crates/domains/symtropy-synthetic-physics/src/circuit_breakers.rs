// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hard circuit breakers for the Synthetic Physics Lab.
//!
//! These guards are checked on every tick. A violation triggers either
//! a rejected update (recoverable) or full quarantine (halt run).
//!
//! **Immutable doctrine**: No graph update rule graduates to production until
//! it survives N ticks with these guards active.

use serde::{Deserialize, Serialize};

/// Hard safety limits for graph evolution.
///
/// All limits are conservative by design. Loosen only with evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSafetyGuards {
    /// Maximum degree of any single node. Prevents hub explosion.
    pub max_node_degree: u32,

    /// Maximum number of edge additions/removals per tick. Prevents churn explosion.
    pub max_edge_churn_per_tick: usize,

    /// Minimum estimated intrinsic dimension. Below this = string collapse.
    pub min_estimated_dimension: f64,

    /// Maximum estimated intrinsic dimension. Above this = hairball.
    pub max_estimated_dimension: f64,

    /// Maximum spectral radius of the adjacency matrix. Above this = instability.
    pub max_spectral_radius: f64,

    /// Maximum multiplier for free energy increase per tick. 2.0 = at most double.
    pub max_energy_increase_factor: f64,

    /// Maximum number of connected components (Betti-0). Above this = fragmentation.
    pub max_betti0_fragmentation: usize,

    /// Maximum normalized holonomy drift per tick. Above this = curvature explosion.
    pub max_holonomy_drift: f64,

    /// Maximum entropy growth rate (bits per tick). Above this = information explosion.
    pub max_entropy_growth_rate: f64,

    /// If true, roll back the graph state on guard violation (instead of just rejecting).
    pub rollback_on_violation: bool,

    /// If true, quarantine (halt entire run) when a strange-attractor signature is detected.
    pub quarantine_strange_attractor: bool,

    /// Maximum consecutive rejected updates before quarantine is triggered.
    pub max_consecutive_rejections: usize,
}

impl Default for GraphSafetyGuards {
    fn default() -> Self {
        Self {
            max_node_degree: 16,
            max_edge_churn_per_tick: 50,
            min_estimated_dimension: 0.5,
            max_estimated_dimension: 8.0,
            max_spectral_radius: 100.0,
            max_energy_increase_factor: 2.0,
            max_betti0_fragmentation: 10,
            max_holonomy_drift: 0.5,
            max_entropy_growth_rate: 0.1,
            rollback_on_violation: true,
            quarantine_strange_attractor: true,
            max_consecutive_rejections: 20,
        }
    }
}

/// A strict variant for early experimental runs (tighter bounds).
impl GraphSafetyGuards {
    pub fn strict() -> Self {
        Self {
            max_node_degree: 8,
            max_edge_churn_per_tick: 20,
            min_estimated_dimension: 1.0,
            max_estimated_dimension: 4.0,
            max_spectral_radius: 20.0,
            max_energy_increase_factor: 1.5,
            max_betti0_fragmentation: 3,
            max_holonomy_drift: 0.2,
            max_entropy_growth_rate: 0.05,
            rollback_on_violation: true,
            quarantine_strange_attractor: true,
            max_consecutive_rejections: 10,
        }
    }

    /// Relaxed variant for well-behaved rules that have passed `strict()`.
    pub fn relaxed() -> Self {
        Self {
            max_node_degree: 32,
            max_edge_churn_per_tick: 100,
            min_estimated_dimension: 0.3,
            max_estimated_dimension: 12.0,
            max_spectral_radius: 500.0,
            max_energy_increase_factor: 3.0,
            max_betti0_fragmentation: 20,
            max_holonomy_drift: 0.8,
            max_entropy_growth_rate: 0.3,
            rollback_on_violation: false,
            quarantine_strange_attractor: true,
            max_consecutive_rejections: 50,
        }
    }

    /// Check a set of candidate metrics against these guards.
    ///
    /// Returns `Ok(())` if all guards pass, or `Err(reason)` if violated.
    pub fn check(
        &self,
        max_degree: u32,
        edge_churn: usize,
        estimated_dim: f64,
        spectral_radius: f64,
        energy_increase_factor: f64,
        betti0: usize,
        holonomy_drift: f64,
        entropy_growth_rate: f64,
    ) -> Result<(), String> {
        if max_degree > self.max_node_degree {
            return Err(format!(
                "node degree {max_degree} exceeds max {}",
                self.max_node_degree
            ));
        }
        if edge_churn > self.max_edge_churn_per_tick {
            return Err(format!(
                "edge churn {edge_churn} exceeds max {}",
                self.max_edge_churn_per_tick
            ));
        }
        if estimated_dim < self.min_estimated_dimension {
            return Err(format!(
                "estimated dimension {estimated_dim:.3} below min {} (string collapse)",
                self.min_estimated_dimension
            ));
        }
        if estimated_dim > self.max_estimated_dimension {
            return Err(format!(
                "estimated dimension {estimated_dim:.3} above max {} (hairball)",
                self.max_estimated_dimension
            ));
        }
        if spectral_radius > self.max_spectral_radius {
            return Err(format!(
                "spectral radius {spectral_radius:.3} exceeds max {}",
                self.max_spectral_radius
            ));
        }
        if energy_increase_factor > self.max_energy_increase_factor {
            return Err(format!(
                "energy increase factor {energy_increase_factor:.3} exceeds max {}",
                self.max_energy_increase_factor
            ));
        }
        if betti0 > self.max_betti0_fragmentation {
            return Err(format!(
                "Betti-0 = {betti0} connected components exceeds max {} (fragmentation)",
                self.max_betti0_fragmentation
            ));
        }
        if holonomy_drift > self.max_holonomy_drift {
            return Err(format!(
                "holonomy drift {holonomy_drift:.3} exceeds max {} (curvature explosion)",
                self.max_holonomy_drift
            ));
        }
        if entropy_growth_rate > self.max_entropy_growth_rate {
            return Err(format!(
                "entropy growth rate {entropy_growth_rate:.4} bits/tick exceeds max {}",
                self.max_entropy_growth_rate
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_guards_pass_nominal() {
        let g = GraphSafetyGuards::default();
        assert!(g.check(8, 10, 2.0, 5.0, 1.1, 2, 0.1, 0.05).is_ok());
    }

    #[test]
    fn degree_violation_detected() {
        let g = GraphSafetyGuards::default();
        let result = g.check(100, 10, 2.0, 5.0, 1.1, 2, 0.1, 0.05);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("degree"));
    }

    #[test]
    fn string_collapse_detected() {
        let g = GraphSafetyGuards::strict();
        let result = g.check(4, 5, 0.2, 5.0, 1.0, 1, 0.05, 0.01);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("string collapse"));
    }

    #[test]
    fn hairball_detected() {
        let g = GraphSafetyGuards::default();
        let result = g.check(8, 10, 15.0, 5.0, 1.0, 1, 0.05, 0.01);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("hairball"));
    }

    #[test]
    fn fragmentation_detected() {
        let g = GraphSafetyGuards::default();
        let result = g.check(4, 5, 2.0, 5.0, 1.0, 15, 0.05, 0.01);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("fragmentation"));
    }
}
