//! Phi (Φ) coherence gating for federated learning.
//!
//! Uses Integrated Information Theory (IIT) Phi as a coherence measure to
//! gate gradient submissions. Only participants whose local model state
//! achieves sufficient integrated information are permitted to submit.
//!
//! In the Mycelix context, Phi serves as a proxy for model coherence:
//! a high Phi score indicates the model's representations are well-integrated
//! (not fragmented), which correlates with honest, useful updates.
//!
//! # Status
//! Stub implementation — Phi proxy via cosine coherence is functional.
//! Full IIT Phi computation requires the Symthaea engine (not WASM-compatible).

use serde::{Deserialize, Serialize};

/// Configuration for Phi coherence gating
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiGateConfig {
    /// Minimum Phi score required to submit a gradient (default: 0.1)
    pub min_phi: f32,
    /// Phi boost threshold — participants above this get a trust bonus (default: 0.6)
    pub boost_threshold: f32,
    /// Phi veto threshold — participants below this are vetoed (default: 0.05)
    pub veto_threshold: f32,
}

impl Default for PhiGateConfig {
    fn default() -> Self {
        Self {
            min_phi: 0.1,
            boost_threshold: 0.6,
            veto_threshold: 0.05,
        }
    }
}

/// Phi coherence score for a gradient submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiScore {
    /// Estimated Phi value in [0.0, 1.0]
    pub phi: f32,
    /// Whether this submission passes the gate
    pub passes_gate: bool,
    /// Trust multiplier derived from Phi (0.5 for veto, 1.0 for normal, 1.2 for boost)
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

/// Phi coherence gate for gradient submissions.
///
/// Computes a proxy Phi score from gradient statistics and applies
/// a gating decision based on the configured thresholds.
pub struct PhiGate {
    config: PhiGateConfig,
}

impl PhiGate {
    pub fn new(config: PhiGateConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(PhiGateConfig::default())
    }

    /// Compute a proxy Phi score from gradient data.
    ///
    /// Uses gradient entropy and L2 norm coherence as a lightweight proxy
    /// for Integrated Information. Full IIT Phi computation is deferred to
    /// the Symthaea engine which is not available in WASM context.
    pub fn compute_phi_proxy(&self, gradient: &[f32]) -> f32 {
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

        // Phi proxy: geometric mean of L2 coherence and entropy
        (l2_norm * entropy_proxy).sqrt().clamp(0.0, 1.0)
    }

    /// Evaluate a Phi score and determine the gate decision.
    pub fn evaluate(&self, phi: f32) -> PhiScore {
        let decision = self.gate_decision(phi);
        let (passes_gate, trust_multiplier) = match decision {
            GateDecision::Veto => (false, 0.5),
            GateDecision::Dampen => (true, 0.7),
            GateDecision::Pass => (true, 1.0),
            GateDecision::Boost => (true, 1.2),
        };
        PhiScore {
            phi,
            passes_gate,
            trust_multiplier,
        }
    }

    /// Compute gate decision from Phi score.
    pub fn gate_decision(&self, phi: f32) -> GateDecision {
        if phi < self.config.veto_threshold {
            GateDecision::Veto
        } else if phi < self.config.min_phi {
            GateDecision::Dampen
        } else if phi >= self.config.boost_threshold {
            GateDecision::Boost
        } else {
            GateDecision::Pass
        }
    }

    /// Compute Phi proxy and gate a gradient in one step.
    pub fn gate_gradient(&self, gradient: &[f32]) -> PhiScore {
        let phi = self.compute_phi_proxy(gradient);
        self.evaluate(phi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_gradient_phi_zero() {
        let gate = PhiGate::with_defaults();
        let phi = gate.compute_phi_proxy(&[]);
        assert_eq!(phi, 0.0);
    }

    #[test]
    fn test_high_quality_gradient_passes() {
        let gate = PhiGate::with_defaults();
        // A gradient with good structure should pass
        let gradient: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let score = gate.gate_gradient(&gradient);
        assert!(score.passes_gate);
    }

    #[test]
    fn test_zero_gradient_vetoed() {
        let gate = PhiGate::with_defaults();
        // All-zeros gradient has zero entropy → very low Phi
        let gradient = vec![0.0_f32; 100];
        let phi = gate.compute_phi_proxy(&gradient);
        assert!(phi < gate.config.min_phi);
    }

    #[test]
    fn test_gate_decision_veto() {
        let gate = PhiGate::with_defaults();
        assert_eq!(gate.gate_decision(0.01), GateDecision::Veto);
    }

    #[test]
    fn test_gate_decision_boost() {
        let gate = PhiGate::with_defaults();
        assert_eq!(gate.gate_decision(0.8), GateDecision::Boost);
    }

    #[test]
    fn test_phi_score_range() {
        let gate = PhiGate::with_defaults();
        for i in 0..=10 {
            let gradient: Vec<f32> = vec![i as f32 * 0.1; 50];
            let phi = gate.compute_phi_proxy(&gradient);
            assert!(phi >= 0.0);
            assert!(phi <= 1.0);
        }
    }
}
