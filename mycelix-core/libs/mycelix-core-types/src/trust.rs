//! Trust System Types
//!
//! Implements the MATL (Multi-Agent Trust Layer) trust formula and related types.
//! Trust Formula: T = 0.4×PoGQ + 0.3×TCDM + 0.3×Entropy

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::k_vector::KVector;

/// MATL trust weights
pub mod matl_weights {
    /// Weight for Proof of Gradient Quality
    pub const POGQ: f32 = 0.4;
    /// Weight for Trust-Confidence-Decay Model
    pub const TCDM: f32 = 0.3;
    /// Weight for Entropy contribution
    pub const ENTROPY: f32 = 0.3;
}

/// Trust score computed via MATL formula
///
/// T = 0.4×PoGQ + 0.3×TCDM + 0.3×Entropy
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrustScore {
    /// Proof of Gradient Quality score (0.0-1.0)
    pub pogq: f32,
    /// Trust-Confidence-Decay Model score (0.0-1.0)
    pub tcdm: f32,
    /// Entropy contribution (0.0-1.0)
    pub entropy: f32,
    /// Computed trust score (cached)
    pub total: f32,
}

impl TrustScore {
    /// Create a new trust score from components
    pub fn new(pogq: f32, tcdm: f32, entropy: f32) -> Self {
        let total = Self::compute_total(pogq, tcdm, entropy);
        Self {
            pogq,
            tcdm,
            entropy,
            total,
        }
    }

    /// Compute total trust score
    fn compute_total(pogq: f32, tcdm: f32, entropy: f32) -> f32 {
        matl_weights::POGQ * pogq + matl_weights::TCDM * tcdm + matl_weights::ENTROPY * entropy
    }

    /// Recalculate total from current components
    pub fn recalculate(&mut self) {
        self.total = Self::compute_total(self.pogq, self.tcdm, self.entropy);
    }

    /// Create a zero trust score
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Create a maximum trust score
    pub fn max() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    /// Check if trust is above threshold for participation
    pub fn is_trusted(&self, threshold: f32) -> bool {
        self.total >= threshold
    }

    /// Validate all components are in [0.0, 1.0]
    pub fn is_valid(&self) -> bool {
        let in_range = |v: f32| (0.0..=1.0).contains(&v) && v.is_finite();
        in_range(self.pogq) && in_range(self.tcdm) && in_range(self.entropy)
    }
}

impl Default for TrustScore {
    fn default() -> Self {
        Self::new(0.5, 0.5, 0.5)
    }
}

/// Proof of Gradient Quality (PoGQ) metrics
///
/// Used in federated learning to validate gradient contributions.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PoGQMetrics {
    /// Whether differential privacy requirements are met
    pub dp_compliant: bool,
    /// Epsilon value for differential privacy
    pub epsilon: f32,
    /// Sigma (noise) value for differential privacy
    pub sigma: f32,
    /// L2 norm of the gradient
    pub l2_norm: f32,
    /// Whether gradient bounds are satisfied
    pub bounds_valid: bool,
    /// Computed quality score (0.0-1.0)
    pub quality_score: f32,
}

impl PoGQMetrics {
    /// Create metrics indicating a valid contribution
    pub fn valid(epsilon: f32, sigma: f32, l2_norm: f32) -> Self {
        Self {
            dp_compliant: true,
            epsilon,
            sigma,
            l2_norm,
            bounds_valid: true,
            quality_score: 0.85, // Default good score
        }
    }

    /// Create metrics indicating an invalid contribution
    pub fn invalid(_reason: &str) -> Self {
        Self {
            dp_compliant: false,
            epsilon: 0.0,
            sigma: 0.0,
            l2_norm: 0.0,
            bounds_valid: false,
            quality_score: 0.0,
        }
    }

    /// Check if the contribution is acceptable
    pub fn is_acceptable(&self) -> bool {
        self.dp_compliant && self.bounds_valid && self.quality_score > 0.5
    }

    /// Compute PoGQ score based on metrics
    pub fn compute_pogq_score(&self) -> f32 {
        if !self.dp_compliant || !self.bounds_valid {
            return 0.0;
        }

        // Score components
        let dp_score = if self.epsilon <= 10.0 && self.sigma >= 5.0 {
            1.0
        } else {
            0.5
        };

        let bounds_score = if self.bounds_valid { 1.0 } else { 0.0 };

        // Combined score
        0.4 * dp_score + 0.3 * bounds_score + 0.3 * self.quality_score
    }
}

impl Default for PoGQMetrics {
    fn default() -> Self {
        Self {
            dp_compliant: false,
            epsilon: 10.0,
            sigma: 5.0,
            l2_norm: 0.0,
            bounds_valid: false,
            quality_score: 0.0,
        }
    }
}

/// Trust-Confidence-Decay Model (TCDM) state
///
/// Models how trust evolves over time with decay and confidence adjustments.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TCDMState {
    /// Current trust level (0.0-1.0)
    pub trust_level: f32,
    /// Confidence in the trust estimate (0.0-1.0)
    pub confidence: f32,
    /// Decay rate per time unit (0.0-1.0, where 1.0 = no decay)
    pub decay_rate: f32,
    /// Last update timestamp (Unix epoch seconds)
    pub last_update: i64,
    /// Number of interactions used to compute trust
    pub interaction_count: u32,
}

impl TCDMState {
    /// Create initial TCDM state for a new agent
    pub fn initial() -> Self {
        Self {
            trust_level: 0.5, // Neutral starting trust
            confidence: 0.1, // Low confidence initially
            decay_rate: 0.99, // Slow decay
            last_update: 0,
            interaction_count: 0,
        }
    }

    /// Apply time-based decay
    pub fn apply_decay(&mut self, current_time: i64) {
        if self.last_update == 0 {
            self.last_update = current_time;
            return;
        }

        let elapsed = (current_time - self.last_update) as f32;
        let time_units = elapsed / 86400.0; // Days

        // Exponential decay
        let decay_factor = self.decay_rate.powf(time_units);
        self.trust_level *= decay_factor;

        // Confidence also decays
        self.confidence *= decay_factor.sqrt();

        self.last_update = current_time;
    }

    /// Update trust based on a new interaction
    pub fn update_from_interaction(&mut self, interaction_success: bool, weight: f32) {
        self.interaction_count += 1;

        // Exponential moving average
        let alpha = weight.min(0.3); // Cap learning rate
        let observation = if interaction_success { 1.0 } else { 0.0 };

        self.trust_level = self.trust_level * (1.0 - alpha) + observation * alpha;

        // Confidence increases with more interactions
        self.confidence = (self.confidence + 0.05).min(1.0);
    }

    /// Compute TCDM score for MATL formula
    pub fn compute_tcdm_score(&self) -> f32 {
        // Trust weighted by confidence
        self.trust_level * self.confidence
    }
}

impl Default for TCDMState {
    fn default() -> Self {
        Self::initial()
    }
}

/// Complete trust state for an agent
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AgentTrust {
    /// Agent identifier
    pub agent_id: String,
    /// K-Vector trust representation
    pub k_vector: KVector,
    /// MATL trust score
    pub matl_score: TrustScore,
    /// TCDM state for temporal modeling
    pub tcdm: TCDMState,
    /// Latest PoGQ metrics (if applicable)
    pub pogq: Option<PoGQMetrics>,
    /// Whether the agent is currently trusted
    pub trusted: bool,
    /// Reason for current trust status
    pub trust_reason: String,
}

impl AgentTrust {
    /// Create initial trust state for a new agent
    pub fn new(agent_id: String) -> Self {
        Self {
            agent_id,
            k_vector: KVector::neutral(),
            matl_score: TrustScore::default(),
            tcdm: TCDMState::initial(),
            pogq: None,
            trusted: false,
            trust_reason: "New agent - insufficient history".to_string(),
        }
    }

    /// Update trust based on new information
    pub fn update(&mut self, current_time: i64) {
        // Apply time decay
        self.tcdm.apply_decay(current_time);

        // Recalculate MATL score
        let pogq_score = self.pogq.as_ref().map(|p| p.compute_pogq_score()).unwrap_or(0.5);
        let tcdm_score = self.tcdm.compute_tcdm_score();
        let entropy_score = self.calculate_entropy_score();

        self.matl_score = TrustScore::new(pogq_score, tcdm_score, entropy_score);

        // Update k_vector from MATL
        self.k_vector.k_r = self.matl_score.total;
        self.k_vector.k_h = self.tcdm.confidence;

        // Determine trust status
        let threshold = 0.45; // Match Byzantine tolerance
        self.trusted = self.matl_score.total >= threshold;
        self.trust_reason = if self.trusted {
            format!("Trust score {:.2} >= threshold {:.2}", self.matl_score.total, threshold)
        } else {
            format!("Trust score {:.2} < threshold {:.2}", self.matl_score.total, threshold)
        };
    }

    /// Calculate entropy contribution score
    fn calculate_entropy_score(&self) -> f32 {
        // Entropy measures diversity of trust sources
        // Higher when trust comes from multiple dimensions
        let values = self.k_vector.to_array();
        let sum: f32 = values.iter().sum();

        if sum == 0.0 {
            return 0.0;
        }

        // Shannon entropy normalized
        let entropy: f32 = values
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| {
                let p = v / sum;
                -p * p.ln()
            })
            .sum();

        // Normalize to [0, 1] (max entropy is ln(8))
        (entropy / 8.0_f32.ln()).min(1.0)
    }
}

/// Trust threshold levels for different operations
pub mod trust_thresholds {
    /// Minimum trust to participate in FL rounds
    pub const FL_PARTICIPATION: f32 = 0.30;
    /// Minimum trust to validate others' contributions
    pub const FL_VALIDATION: f32 = 0.50;
    /// Minimum trust to coordinate FL rounds
    pub const FL_COORDINATION: f32 = 0.70;
    /// Minimum trust for governance participation
    pub const GOVERNANCE: f32 = 0.45;
    /// Minimum trust for identity verification
    pub const IDENTITY: f32 = 0.60;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_score_calculation() {
        let ts = TrustScore::new(0.8, 0.6, 0.5);
        let expected = 0.4 * 0.8 + 0.3 * 0.6 + 0.3 * 0.5;
        assert!((ts.total - expected).abs() < 0.001);
    }

    #[test]
    fn test_tcdm_decay() {
        let mut tcdm = TCDMState::initial();
        tcdm.trust_level = 1.0;
        tcdm.confidence = 1.0;
        tcdm.last_update = 1000; // Start with a non-zero timestamp

        // Apply 30 days of decay
        tcdm.apply_decay(1000 + 30 * 86400);

        assert!(tcdm.trust_level < 1.0);
        assert!(tcdm.confidence < 1.0);
    }

    #[test]
    fn test_tcdm_interaction_update() {
        let mut tcdm = TCDMState::initial();

        // Positive interactions should increase trust
        for _ in 0..10 {
            tcdm.update_from_interaction(true, 0.2);
        }

        assert!(tcdm.trust_level > 0.5);
        assert!(tcdm.confidence > 0.1);
    }

    #[test]
    fn test_pogq_scoring() {
        let valid = PoGQMetrics::valid(10.0, 5.0, 0.5);
        assert!(valid.is_acceptable());
        assert!(valid.compute_pogq_score() > 0.5);

        let invalid = PoGQMetrics::invalid("test");
        assert!(!invalid.is_acceptable());
        assert!((invalid.compute_pogq_score() - 0.0).abs() < 0.001);
    }
}
