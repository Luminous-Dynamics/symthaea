// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gradient coherence gating for federated learning.
//!
//! **This is NOT IIT Phi.** This module computes a lightweight gradient
//! coherence score from gradient statistics (L2 norm + entropy proxy) to
//! gate gradient submissions. Only participants whose local model state
//! achieves sufficient coherence are permitted to submit.
//!
//! For true IIT Phi computation, see the Symthaea PhiEngine.
//!
//! # Naming Disambiguation
//!
//! | Concept | Module | Description |
//! |---------|--------|-------------|
//! | **IIT Phi** | `symthaea_core::phi_engine` | True integrated information |
//! | **Gradient Coherence** | this module | Proxy from gradient statistics |
//! | **Consciousness Level** | governance bridge | Agent's attested consciousness |
//!
//! # Canonical Thresholds
//!
//! See `mycelix_bridge_common::consciousness_thresholds` for the single source of truth.
//! Default values: veto=0.1, dampen=0.3, boost=0.6.

use mycelix_bridge_common::consciousness_thresholds::consciousness_thresholds;
use serde::{Deserialize, Serialize};

/// Configuration for gradient coherence gating.
///
/// Default values are imported from `mycelix_bridge_common::consciousness_thresholds` —
/// the single source of truth for all Mycelix consciousness thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientCoherenceConfig {
    /// Minimum coherence score required to submit a gradient (default: 0.1)
    pub min_coherence: f32,
    /// Coherence boost threshold — participants above this get a trust bonus (default: 0.6)
    pub boost_threshold: f32,
    /// Coherence veto threshold — participants below this are vetoed (default: 0.1)
    pub veto_threshold: f32,
}

impl Default for GradientCoherenceConfig {
    fn default() -> Self {
        let t = consciousness_thresholds();
        Self {
            min_coherence: t.fl_veto,
            boost_threshold: t.fl_boost,
            veto_threshold: t.fl_veto,
        }
    }
}

/// Gradient coherence score for a submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceScore {
    /// Estimated coherence value in [0.0, 1.0]
    pub coherence: f32,
    /// Whether this submission passes the gate
    pub passes_gate: bool,
    /// Trust multiplier derived from coherence (0.5 for veto, 1.0 for normal, 1.2 for boost)
    pub trust_multiplier: f32,
}

/// Gate decision for a gradient submission
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateDecision {
    /// Submission passes — include in aggregation
    Pass,
    /// Submission is boosted — higher trust weight
    Boost,
    /// Submission is dampened — lower trust weight
    Dampen,
    /// Submission is vetoed — exclude from aggregation
    Veto,
}

/// Gradient coherence gate for gradient submissions.
///
/// Computes a coherence score from gradient statistics and applies
/// a gating decision based on the configured thresholds.
///
/// **This is NOT IIT Phi.** It is a lightweight proxy based on
/// gradient L2 norm and entropy.
pub struct GradientCoherenceGate {
    config: GradientCoherenceConfig,
}

impl GradientCoherenceGate {
    pub fn new(config: GradientCoherenceConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(GradientCoherenceConfig::default())
    }

    /// Compute a gradient coherence score from gradient data.
    ///
    /// Uses gradient entropy and L2 norm coherence as a lightweight proxy
    /// for model integration quality. This is NOT IIT Phi — full Phi
    /// computation requires the Symthaea PhiEngine.
    pub fn compute_gradient_coherence(&self, gradient: &[f32]) -> f32 {
        if gradient.is_empty() {
            return 0.0;
        }

        let n = gradient.len() as f32;

        // Component 1: Normalized L2 norm (contribution diversity)
        let l2_sq: f32 = gradient.iter().map(|&x| x * x).sum();
        let l2 = l2_sq.sqrt();
        let l2_norm = (l2 / n.sqrt()).clamp(0.0, 1.0);

        // Component 2: Entropy proxy via std deviation
        let mean: f32 = gradient.iter().sum::<f32>() / n;
        let variance: f32 = gradient.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();
        // Normalize std_dev to [0, 1] via sigmoid-like mapping
        let entropy_proxy = 1.0 - (-std_dev).exp();

        // Coherence: geometric mean of L2 coherence and entropy
        (l2_norm * entropy_proxy).sqrt().clamp(0.0, 1.0)
    }

    /// Evaluate a coherence score and determine the gate decision.
    pub fn evaluate(&self, coherence: f32) -> CoherenceScore {
        let decision = self.gate_decision(coherence);
        let (passes_gate, trust_multiplier) = match decision {
            GateDecision::Veto => (false, 0.5),
            GateDecision::Dampen => (true, 0.7),
            GateDecision::Pass => (true, 1.0),
            GateDecision::Boost => (true, 1.2),
        };
        CoherenceScore {
            coherence,
            passes_gate,
            trust_multiplier,
        }
    }

    /// Compute gate decision from coherence score.
    pub fn gate_decision(&self, coherence: f32) -> GateDecision {
        if coherence < self.config.veto_threshold {
            GateDecision::Veto
        } else if coherence < self.config.min_coherence {
            GateDecision::Dampen
        } else if coherence >= self.config.boost_threshold {
            GateDecision::Boost
        } else {
            GateDecision::Pass
        }
    }

    /// Compute gradient coherence and gate a gradient in one step.
    pub fn gate_gradient(&self, gradient: &[f32]) -> CoherenceScore {
        let coherence = self.compute_gradient_coherence(gradient);
        self.evaluate(coherence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_gradient_zero() {
        let gate = GradientCoherenceGate::with_defaults();
        let coherence = gate.compute_gradient_coherence(&[]);
        assert_eq!(coherence, 0.0);
    }

    #[test]
    fn test_high_quality_gradient_passes() {
        let gate = GradientCoherenceGate::with_defaults();
        let gradient: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let score = gate.gate_gradient(&gradient);
        assert!(score.passes_gate);
    }

    #[test]
    fn test_zero_gradient_vetoed() {
        let gate = GradientCoherenceGate::with_defaults();
        let gradient = vec![0.0_f32; 100];
        let coherence = gate.compute_gradient_coherence(&gradient);
        assert!(coherence < gate.config.min_coherence);
    }

    #[test]
    fn test_gate_decision_veto() {
        let gate = GradientCoherenceGate::with_defaults();
        assert_eq!(gate.gate_decision(0.01), GateDecision::Veto);
    }

    #[test]
    fn test_gate_decision_boost() {
        let gate = GradientCoherenceGate::with_defaults();
        assert_eq!(gate.gate_decision(0.8), GateDecision::Boost);
    }

    #[test]
    fn test_coherence_score_range() {
        let gate = GradientCoherenceGate::with_defaults();
        for i in 0..=10 {
            let gradient: Vec<f32> = vec![i as f32 * 0.1; 50];
            let coherence = gate.compute_gradient_coherence(&gradient);
            assert!(coherence >= 0.0);
            assert!(coherence <= 1.0);
        }
    }

    #[test]
    fn test_defaults_match_canonical_thresholds() {
        let config = GradientCoherenceConfig::default();
        let canonical = consciousness_thresholds();
        assert_eq!(config.veto_threshold, canonical.fl_veto);
        assert_eq!(config.boost_threshold, canonical.fl_boost);
        assert_eq!(config.min_coherence, canonical.fl_veto);
    }
}
