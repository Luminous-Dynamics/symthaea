//! Topology Synergy Module
//!
//! # Bridging Research Directions with Existing Consciousness Architecture
//!
//! This module creates synergies between:
//! - Our new research (adaptive_topology, phi_gradient_learning, fractal_consciousness)
//! - The existing consciousness_topology (algebraic topology, Betti numbers, persistence)
//! - The consciousness dimension system (Φ, B, W, A, R, E, K)
//!
//! # Key Synergies
//!
//! 1. **Topological Φ-Gradient**: Optimize for topological features, not just scalar Φ
//! 2. **Fractal Betti Analysis**: Compute Betti numbers at each fractal scale
//! 3. **Adaptive Topology Signatures**: Switch cognitive modes based on topological state
//! 4. **Multi-Scale Persistence**: Track which features persist across fractal levels
//!
//! # Research Hypothesis
//!
//! If ~40-45% bridge ratio optimizes Φ, there may be corresponding optimal
//! topological signatures (e.g., β₀=1, specific β₁/β₂ patterns).

use super::phi_real::RealPhiCalculator;
use super::adaptive_topology::{AdaptiveTopology, CognitiveMode};
#[allow(unused_imports)]  // Used in tests
use super::fractal_consciousness::{FractalConsciousness, FractalConfig};

/// Extended metrics including topological features
#[derive(Clone, Debug)]
pub struct TopologicalMetrics {
    /// Standard Φ
    pub phi: f64,
    /// Betti number β₀ (connected components)
    pub beta_0: usize,
    /// Betti number β₁ (1-dimensional holes/cycles)
    pub beta_1: usize,
    /// Euler characteristic (β₀ - β₁ + β₂)
    pub euler_characteristic: i64,
    /// Bridge ratio
    pub bridge_ratio: f64,
    /// Cognitive mode (if adaptive)
    pub mode: Option<CognitiveMode>,
    /// Scale level (if fractal)
    pub scale: Option<usize>,
}

impl std::fmt::Display for TopologicalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Φ={:.4}, β₀={}, β₁={}, χ={}, bridges={:.1}%",
               self.phi, self.beta_0, self.beta_1, self.euler_characteristic,
               self.bridge_ratio * 100.0)
    }
}

/// Synergy analyzer connecting topology research directions
pub struct TopologySynergy {
    /// HDC dimension
    dim: usize,
    /// Φ calculator
    phi_calc: RealPhiCalculator,
}

impl TopologySynergy {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            phi_calc: RealPhiCalculator::new(),
        }
    }

    /// Analyze topological properties of an adaptive topology
    pub fn analyze_adaptive(&self, adaptive: &AdaptiveTopology) -> TopologicalMetrics {
        let metrics = adaptive.metrics();

        // Compute Betti numbers from connectivity
        let (beta_0, beta_1) = self.estimate_betti_from_edges(
            adaptive.organizer().processes().len(),
            &adaptive.active_bridge_connections(),
            adaptive.organizer().topology().edges.iter().map(|&(a,b)| (a, b)).collect(),
        );

        TopologicalMetrics {
            phi: metrics.base.phi,
            beta_0,
            beta_1,
            euler_characteristic: beta_0 as i64 - beta_1 as i64,
            bridge_ratio: metrics.bridge_ratio,
            mode: Some(metrics.mode),
            scale: None,
        }
    }

    /// Analyze multi-scale topological properties of fractal consciousness
    pub fn analyze_fractal(&self, fractal: &FractalConsciousness) -> Vec<TopologicalMetrics> {
        let metrics = fractal.metrics();
        let ms_phi = fractal.multi_scale_phi();

        // Generate metrics for each scale
        ms_phi.scale_phis.iter()
            .map(|(scale, phi)| {
                TopologicalMetrics {
                    phi: *phi,
                    beta_0: 1, // Assume connected at each scale
                    beta_1: *scale, // Hypothetical: more cycles at higher scales
                    euler_characteristic: 1 - *scale as i64,
                    bridge_ratio: metrics.top_level_bridge_ratio,
                    mode: None,
                    scale: Some(*scale),
                }
            })
            .collect()
    }

    /// Estimate Betti numbers from edge structure
    /// β₀ = number of connected components
    /// β₁ = number of independent cycles
    fn estimate_betti_from_edges(
        &self,
        n_nodes: usize,
        bridge_edges: &[(usize, usize)],
        base_edges: Vec<(usize, usize)>,
    ) -> (usize, usize) {
        // Combine all edges
        let all_edges: Vec<(usize, usize)> = base_edges.into_iter()
            .chain(bridge_edges.iter().copied())
            .collect();

        // Use Union-Find to count connected components (β₀)
        let mut parent: Vec<usize> = (0..n_nodes).collect();

        fn find(parent: &mut Vec<usize>, i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut Vec<usize>, i: usize, j: usize) {
            let pi = find(parent, i);
            let pj = find(parent, j);
            if pi != pj {
                parent[pi] = pj;
            }
        }

        for &(a, b) in &all_edges {
            if a < n_nodes && b < n_nodes {
                union(&mut parent, a, b);
            }
        }

        // Count unique roots
        let beta_0 = (0..n_nodes)
            .map(|i| find(&mut parent.clone(), i))
            .collect::<std::collections::HashSet<_>>()
            .len();

        // Euler formula for graphs: V - E + F = 2 for planar
        // For general graphs: β₀ - β₁ = V - E (when treating as 1-skeleton)
        // So β₁ = E - V + β₀
        let e = all_edges.len();
        let v = n_nodes;
        let beta_1 = if e + beta_0 > v { e + beta_0 - v } else { 0 };

        (beta_0, beta_1)
    }

    /// Find optimal cognitive mode for target topological signature
    pub fn find_mode_for_topology(
        &self,
        target_beta_0: usize,
        target_beta_1_range: (usize, usize),
    ) -> CognitiveMode {
        // Different modes produce different topological signatures
        //
        // DeepSpecialization: High β₀ (fragmented), low β₁
        // Focused: Moderate β₀, low β₁
        // Balanced: β₀=1 (unified), moderate β₁
        // Exploratory: β₀=1, high β₁ (many cycles)
        // GlobalAwareness: β₀=1, very high β₁

        if target_beta_0 > 1 {
            // Fragmented consciousness requested
            if target_beta_0 > 2 {
                CognitiveMode::DeepSpecialization
            } else {
                CognitiveMode::Focused
            }
        } else {
            // Unified consciousness (β₀ = 1)
            let mid_beta_1 = (target_beta_1_range.0 + target_beta_1_range.1) / 2;
            if mid_beta_1 < 3 {
                CognitiveMode::Focused
            } else if mid_beta_1 < 6 {
                CognitiveMode::Balanced
            } else if mid_beta_1 < 10 {
                CognitiveMode::Exploratory
            } else {
                CognitiveMode::GlobalAwareness
            }
        }
    }

    /// Compute topological stability across mode transitions
    pub fn mode_transition_analysis(&self) -> ModeTransitionReport {
        let dim = self.dim;
        let mut transitions = Vec::new();

        let modes = [
            CognitiveMode::DeepSpecialization,
            CognitiveMode::Focused,
            CognitiveMode::Balanced,
            CognitiveMode::Exploratory,
            CognitiveMode::GlobalAwareness,
        ];

        // Analyze each transition
        for i in 0..modes.len() {
            let from_mode = modes[i];
            let to_mode = modes[(i + 1) % modes.len()];

            let mut adaptive = AdaptiveTopology::new(24, dim, 42);

            adaptive.set_mode(from_mode);
            let from_metrics = self.analyze_adaptive(&adaptive);

            adaptive.set_mode(to_mode);
            let to_metrics = self.analyze_adaptive(&adaptive);

            transitions.push(ModeTransition {
                from: from_mode,
                to: to_mode,
                phi_delta: to_metrics.phi - from_metrics.phi,
                beta_0_delta: to_metrics.beta_0 as i64 - from_metrics.beta_0 as i64,
                beta_1_delta: to_metrics.beta_1 as i64 - from_metrics.beta_1 as i64,
                bridge_delta: to_metrics.bridge_ratio - from_metrics.bridge_ratio,
            });
        }

        ModeTransitionReport { transitions }
    }
}

/// Report on mode transitions
#[derive(Clone, Debug)]
pub struct ModeTransitionReport {
    pub transitions: Vec<ModeTransition>,
}

impl std::fmt::Display for ModeTransitionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Mode Transition Analysis:")?;
        writeln!(f, "{:<25} {:>10} {:>8} {:>8} {:>10}", "Transition", "ΔΦ", "Δβ₀", "Δβ₁", "ΔBridge")?;
        writeln!(f, "{}", "-".repeat(65))?;
        for t in &self.transitions {
            writeln!(f, "{:?} → {:?}", t.from, t.to)?;
            writeln!(f, "{:>25} {:>+10.4} {:>+8} {:>+8} {:>+10.1}%",
                     "", t.phi_delta, t.beta_0_delta, t.beta_1_delta, t.bridge_delta * 100.0)?;
        }
        Ok(())
    }
}

/// A single mode transition analysis
#[derive(Clone, Debug)]
pub struct ModeTransition {
    pub from: CognitiveMode,
    pub to: CognitiveMode,
    pub phi_delta: f64,
    pub beta_0_delta: i64,
    pub beta_1_delta: i64,
    pub bridge_delta: f64,
}

/// Consciousness state classification based on topology
#[derive(Clone, Debug, PartialEq)]
pub enum ConsciousnessState {
    /// β₀=1, low β₁: Unified, focused awareness
    Focused,
    /// β₀=1, moderate β₁: Normal waking consciousness
    NormalWaking,
    /// β₀=1, high β₁: Flow state, creative engagement
    FlowState,
    /// β₀>1: Fragmented attention, dissociation
    Fragmented,
    /// β₀=1, very high β₁: Meditative, expanded awareness
    ExpandedAwareness,
}

impl TopologySynergy {
    /// Classify consciousness state from topological metrics
    pub fn classify_state(&self, metrics: &TopologicalMetrics) -> ConsciousnessState {
        if metrics.beta_0 > 1 {
            return ConsciousnessState::Fragmented;
        }

        match metrics.beta_1 {
            0..=2 => ConsciousnessState::Focused,
            3..=5 => ConsciousnessState::NormalWaking,
            6..=10 => ConsciousnessState::FlowState,
            _ => ConsciousnessState::ExpandedAwareness,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::HDC_DIMENSION;

    #[test]
    fn test_synergy_analysis() {
        let synergy = TopologySynergy::new(2048);
        let adaptive = AdaptiveTopology::new(24, 2048, 42);

        let metrics = synergy.analyze_adaptive(&adaptive);
        println!("Adaptive topology: {}", metrics);

        let state = synergy.classify_state(&metrics);
        println!("Classified as: {:?}", state);
    }

    #[test]
    fn test_fractal_analysis() {
        let synergy = TopologySynergy::new(1024);

        let config = FractalConfig {
            n_scales: 3,
            nodes_per_scale: 4,
            dim: 1024,
            ..Default::default()
        };

        let fractal = FractalConsciousness::new(config);
        let scale_metrics = synergy.analyze_fractal(&fractal);

        println!("\nMulti-scale topological analysis:");
        for m in &scale_metrics {
            println!("  Scale {:?}: {}", m.scale, m);
        }
    }

    #[test]
    fn test_mode_transitions() {
        let synergy = TopologySynergy::new(2048);
        let report = synergy.mode_transition_analysis();
        println!("\n{}", report);
    }

    #[test]
    fn test_state_classification() {
        let synergy = TopologySynergy::new(2048);

        let test_cases = [
            (TopologicalMetrics {
                phi: 0.5, beta_0: 1, beta_1: 1, euler_characteristic: 0,
                bridge_ratio: 0.3, mode: None, scale: None,
            }, ConsciousnessState::Focused),
            (TopologicalMetrics {
                phi: 0.5, beta_0: 1, beta_1: 4, euler_characteristic: -3,
                bridge_ratio: 0.4, mode: None, scale: None,
            }, ConsciousnessState::NormalWaking),
            (TopologicalMetrics {
                phi: 0.5, beta_0: 3, beta_1: 2, euler_characteristic: 1,
                bridge_ratio: 0.2, mode: None, scale: None,
            }, ConsciousnessState::Fragmented),
        ];

        for (metrics, expected) in &test_cases {
            let state = synergy.classify_state(metrics);
            println!("{} => {:?} (expected {:?})", metrics, state, expected);
            assert_eq!(state, *expected);
        }
    }
}
