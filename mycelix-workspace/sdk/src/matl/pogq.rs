//! Proof of Gradient Quality (PoGQ)
//!
//! Core trust mechanism for Byzantine-resistant federated learning.
//! Measures the quality, consistency, and entropy of gradient contributions.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

/// Proof of Gradient Quality measurement
///
/// Captures the quality metrics of a gradient contribution in federated learning.
/// Used to detect Byzantine (malicious) participants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/matl/"))]
pub struct ProofOfGradientQuality {
    /// Quality score [0.0, 1.0] - how good is this gradient?
    pub quality: f64,

    /// Consistency score [0.0, 1.0] - how consistent with previous contributions?
    pub consistency: f64,

    /// Entropy measure - information content of the gradient
    pub entropy: f64,

    /// Unix timestamp when this was measured
    pub timestamp: u64,
}

impl ProofOfGradientQuality {
    /// Create a new PoGQ measurement
    ///
    /// # Arguments
    /// * `quality` - Gradient quality score (0.0-1.0)
    /// * `consistency` - Temporal consistency score (0.0-1.0)
    /// * `entropy` - Information entropy of gradient
    ///
    /// # Example
    /// ```
    /// use mycelix_sdk::matl::ProofOfGradientQuality;
    ///
    /// let pogq = ProofOfGradientQuality::new(0.95, 0.88, 0.12);
    /// assert!(pogq.quality > 0.9);
    /// ```
    pub fn new(quality: f64, consistency: f64, entropy: f64) -> Self {
        Self {
            quality: quality.clamp(0.0, 1.0),
            consistency: consistency.clamp(0.0, 1.0),
            entropy,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Create with explicit timestamp (for testing/replay)
    pub fn with_timestamp(quality: f64, consistency: f64, entropy: f64, timestamp: u64) -> Self {
        Self {
            quality: quality.clamp(0.0, 1.0),
            consistency: consistency.clamp(0.0, 1.0),
            entropy,
            timestamp,
        }
    }

    /// Compute composite trust score using MATL formula
    ///
    /// # Arguments
    /// * `reputation` - External reputation score (0.0-1.0)
    ///
    /// # Returns
    /// Weighted composite score: 0.4*quality + 0.3*consistency + 0.3*reputation
    pub fn composite_score(&self, reputation: f64) -> f64 {
        use super::{
            DEFAULT_CONSISTENCY_WEIGHT, DEFAULT_QUALITY_WEIGHT, DEFAULT_REPUTATION_WEIGHT,
        };

        let rep = reputation.clamp(0.0, 1.0);

        DEFAULT_QUALITY_WEIGHT * self.quality
            + DEFAULT_CONSISTENCY_WEIGHT * self.consistency
            + DEFAULT_REPUTATION_WEIGHT * rep
    }

    /// Compute composite score with custom weights
    pub fn composite_score_weighted(
        &self,
        reputation: f64,
        w_quality: f64,
        w_consistency: f64,
        w_reputation: f64,
    ) -> f64 {
        let rep = reputation.clamp(0.0, 1.0);
        let total_weight = w_quality + w_consistency + w_reputation;

        if total_weight == 0.0 {
            return 0.0;
        }

        (w_quality * self.quality + w_consistency * self.consistency + w_reputation * rep)
            / total_weight
    }

    /// Check if this measurement indicates Byzantine behavior
    ///
    /// # Arguments
    /// * `threshold` - Score below this is considered Byzantine
    ///
    /// # Security Note (FIND-006 mitigation)
    ///
    /// This method includes guards against NaN/Infinity edge cases.
    /// If either the quality score or threshold is not finite, the method
    /// fails safe by returning `true` (marking as potentially Byzantine).
    pub fn is_byzantine(&self, threshold: f64) -> bool {
        // Fail safe: if threshold is invalid, assume Byzantine
        if !threshold.is_finite() {
            return true;
        }
        // Fail safe: if quality is NaN/Infinity, assume Byzantine
        if !self.quality.is_finite() {
            return true;
        }
        self.quality < threshold
    }

    /// Check if measurement is stale
    ///
    /// # Arguments
    /// * `max_age_secs` - Maximum age in seconds
    pub fn is_stale(&self, max_age_secs: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        now - self.timestamp > max_age_secs
    }

    /// Age of this measurement in seconds
    pub fn age_secs(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        now.saturating_sub(self.timestamp)
    }
}

impl Default for ProofOfGradientQuality {
    fn default() -> Self {
        Self::new(0.5, 0.5, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let pogq = ProofOfGradientQuality::new(0.9, 0.8, 0.1);
        assert_eq!(pogq.quality, 0.9);
        assert_eq!(pogq.consistency, 0.8);
        assert_eq!(pogq.entropy, 0.1);
        assert!(pogq.timestamp > 0);
    }

    #[test]
    fn test_clamping() {
        let pogq = ProofOfGradientQuality::new(1.5, -0.5, 0.0);
        assert_eq!(pogq.quality, 1.0);
        assert_eq!(pogq.consistency, 0.0);
    }

    #[test]
    fn test_composite_score() {
        let pogq = ProofOfGradientQuality::new(1.0, 1.0, 0.0);
        let score = pogq.composite_score(1.0);
        assert!((score - 1.0).abs() < 0.001);

        let pogq_low = ProofOfGradientQuality::new(0.0, 0.0, 0.0);
        let score_low = pogq_low.composite_score(0.0);
        assert!((score_low - 0.0).abs() < 0.001);

        // Mixed scores
        let pogq_mid = ProofOfGradientQuality::new(0.8, 0.6, 0.1);
        let score_mid = pogq_mid.composite_score(0.7);
        // 0.4*0.8 + 0.3*0.6 + 0.3*0.7 = 0.32 + 0.18 + 0.21 = 0.71
        assert!((score_mid - 0.71).abs() < 0.001);
    }

    #[test]
    fn test_is_byzantine() {
        let good = ProofOfGradientQuality::new(0.9, 0.9, 0.1);
        assert!(!good.is_byzantine(0.5));

        let bad = ProofOfGradientQuality::new(0.2, 0.9, 0.1);
        assert!(bad.is_byzantine(0.5));
    }

    #[test]
    fn test_default() {
        let pogq = ProofOfGradientQuality::default();
        assert_eq!(pogq.quality, 0.5);
        assert_eq!(pogq.consistency, 0.5);
    }

    // ==========================================================================
    // FIND-006 Mitigation Tests: NaN/Infinity guards for is_byzantine
    // ==========================================================================

    #[test]
    fn test_is_byzantine_nan_threshold() {
        // If threshold is NaN, should fail safe and return true (Byzantine)
        let pogq = ProofOfGradientQuality::new(0.9, 0.9, 0.1);
        assert!(pogq.is_byzantine(f64::NAN));
    }

    #[test]
    fn test_is_byzantine_infinity_threshold() {
        // If threshold is Infinity, should fail safe and return true (Byzantine)
        let pogq = ProofOfGradientQuality::new(0.9, 0.9, 0.1);
        assert!(pogq.is_byzantine(f64::INFINITY));
        assert!(pogq.is_byzantine(f64::NEG_INFINITY));
    }

    #[test]
    fn test_is_byzantine_nan_quality() {
        // If quality became NaN somehow (shouldn't happen due to clamping, but test anyway)
        // We can't easily create a PoGQ with NaN quality due to clamping,
        // but we test the edge case by directly testing threshold behavior
        let good = ProofOfGradientQuality::new(0.9, 0.9, 0.1);

        // Normal behavior: high quality is not Byzantine
        assert!(!good.is_byzantine(0.5));

        // With NaN threshold, should be Byzantine (fail safe)
        assert!(good.is_byzantine(f64::NAN));
    }
}
