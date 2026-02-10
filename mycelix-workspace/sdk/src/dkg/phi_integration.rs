//! Phi-DKG Integration
//!
//! Connects consciousness metrics (Phi) from the Phi-lab to DKG confidence scoring.
//! Coherent agents' claims receive higher confidence; incoherent agents are penalized.
//!
//! # Integration Points
//!
//! 1. **Phi → Confidence Boost**: High Phi (coherent) claims get +15% confidence
//! 2. **Coherence State → Degradation**: Critical/Degraded states flag confidence as degraded
//! 3. **Harmonic Level → Bonus**: Network-aligned claims (H2-H4) get additional boost
//!
//! # Coherence Thresholds
//!
//! - Phi >= 0.7: Coherent (full trust)
//! - Phi 0.5-0.7: Stable (normal operation)
//! - Phi 0.3-0.5: Unstable (requires review)
//! - Phi < 0.3: Degraded/Critical (restricted operations)

use serde::{Deserialize, Serialize};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

/// Coherence state derived from Phi measurement
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub enum CoherenceState {
    /// Phi >= 0.7: Highly coherent, full operational capacity
    Coherent = 4,
    /// Phi 0.5-0.7: Stable, normal operations
    Stable = 3,
    /// Phi 0.3-0.5: Unstable, requires monitoring
    Unstable = 2,
    /// Phi 0.15-0.3: Degraded, restricted operations
    Degraded = 1,
    /// Phi < 0.15: Critical, suspended operations
    Critical = 0,
}

impl CoherenceState {
    /// Derive coherence state from Phi value
    pub fn from_phi(phi: f64) -> Self {
        if phi >= 0.7 {
            Self::Coherent
        } else if phi >= 0.5 {
            Self::Stable
        } else if phi >= 0.3 {
            Self::Unstable
        } else if phi >= 0.15 {
            Self::Degraded
        } else {
            Self::Critical
        }
    }

    /// Get multiplier for confidence calculations
    pub fn confidence_multiplier(&self) -> f64 {
        match self {
            Self::Coherent => 1.15, // +15% boost
            Self::Stable => 1.0,    // Normal
            Self::Unstable => 0.85, // -15% penalty
            Self::Degraded => 0.6,  // -40% penalty
            Self::Critical => 0.3,  // -70% penalty
        }
    }

    /// Check if state allows high-stakes operations
    pub fn allows_high_stakes(&self) -> bool {
        matches!(self, Self::Coherent | Self::Stable)
    }

    /// Check if state requires human approval
    pub fn requires_approval(&self) -> bool {
        matches!(self, Self::Unstable | Self::Degraded | Self::Critical)
    }
}

/// Consciousness metrics from Phi-lab
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct ConsciousnessMetrics {
    /// Integrated Information (Phi) - measure of coherence (0.0-1.0)
    pub phi: f64,
    /// Derived coherence state
    pub coherence_state: CoherenceState,
    /// Harmonic level from epistemic cube (H0-H4)
    pub harmonic_level: u8,
    /// Unix timestamp of measurement
    pub measured_at: u64,
    /// Optional: window size used for measurement
    pub window_size: Option<usize>,
}

impl ConsciousnessMetrics {
    /// Create new metrics from Phi measurement
    pub fn new(phi: f64, harmonic_level: u8, measured_at: u64) -> Self {
        Self {
            phi: phi.clamp(0.0, 1.0),
            coherence_state: CoherenceState::from_phi(phi),
            harmonic_level: harmonic_level.min(4),
            measured_at,
            window_size: None,
        }
    }

    /// Create with window size
    pub fn with_window(mut self, window_size: usize) -> Self {
        self.window_size = Some(window_size);
        self
    }

    /// Check if metrics are stale (older than 1 hour)
    pub fn is_stale(&self, current_time: u64) -> bool {
        current_time.saturating_sub(self.measured_at) > 3600
    }
}

/// Factors derived from consciousness metrics for confidence adjustment
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct PhiConfidenceFactors {
    /// Coherence factor: Higher Phi = more internally consistent claims (0.3-1.15)
    pub coherence_factor: f64,
    /// Harmonic factor: Claims aligned with network harmony (0.9-1.1)
    pub harmonic_factor: f64,
    /// State factor: Penalty for degraded/critical states (0.3-1.0)
    pub state_factor: f64,
    /// Combined multiplier for confidence score
    pub combined_multiplier: f64,
}

impl PhiConfidenceFactors {
    /// Create factors from consciousness metrics
    pub fn from_metrics(metrics: &ConsciousnessMetrics) -> Self {
        let coherence_factor = metrics.coherence_state.confidence_multiplier();

        // Harmonic factor: H0=0.9, H1=0.95, H2=1.0, H3=1.05, H4=1.1
        let harmonic_factor = 0.9 + (metrics.harmonic_level as f64 * 0.05);

        // State factor: additional penalty for bad states
        let state_factor = match metrics.coherence_state {
            CoherenceState::Coherent => 1.0,
            CoherenceState::Stable => 1.0,
            CoherenceState::Unstable => 0.9,
            CoherenceState::Degraded => 0.7,
            CoherenceState::Critical => 0.4,
        };

        let combined = coherence_factor * harmonic_factor * state_factor;

        Self {
            coherence_factor,
            harmonic_factor,
            state_factor,
            combined_multiplier: combined.clamp(0.2, 1.5),
        }
    }
}

/// Convert Phi value to confidence boost factor
///
/// Maps Phi (0.0-1.0) to a multiplicative factor for confidence scores.
/// - Phi >= 0.7: +15% boost (factor 1.15)
/// - Phi = 0.5: neutral (factor 1.0)
/// - Phi <= 0.3: -25% penalty (factor 0.75)
///
/// # Arguments
/// * `phi` - Integrated Information value (0.0-1.0)
///
/// # Returns
/// Multiplicative factor for confidence (0.75-1.15)
pub fn phi_to_confidence_boost(phi: f64) -> f64 {
    // Linear interpolation from (0.0, 0.75) to (1.0, 1.15)
    // boost = 0.75 + 0.4 * phi
    // At phi=0.5: boost = 0.75 + 0.2 = 0.95 (slight penalty)
    // We want phi=0.5 to be neutral, so:
    // boost = 0.8 + 0.4 * phi at midpoint of 0.5 = 1.0

    // Better formula: center at 0.5, range ±0.2
    // boost = 1.0 + (phi - 0.5) * 0.4
    // At phi=0.0: 1.0 - 0.2 = 0.8
    // At phi=0.5: 1.0
    // At phi=1.0: 1.0 + 0.2 = 1.2

    let centered = phi - 0.5;
    let boost = 1.0 + (centered * 0.4);
    boost.clamp(0.75, 1.2)
}

/// Apply Phi boost to base confidence score
///
/// # Arguments
/// * `base_confidence` - Original confidence (0.0-1.0)
/// * `phi` - Integrated Information value (0.0-1.0)
/// * `harmonic_level` - Harmonic level from epistemic classification (0-4)
///
/// # Returns
/// Adjusted confidence score (0.0-0.99)
pub fn apply_phi_to_confidence(base_confidence: f64, phi: f64, harmonic_level: u8) -> f64 {
    let phi_boost = phi_to_confidence_boost(phi);

    // Harmonic bonus: H0=0%, H1=1%, H2=2%, H3=3%, H4=4%
    let harmonic_bonus = 1.0 + (harmonic_level as f64 * 0.01);

    let adjusted = base_confidence * phi_boost * harmonic_bonus;
    adjusted.clamp(0.0, 0.99)
}

/// Apply consciousness metrics to confidence calculation
///
/// This is the main integration point for DKG confidence scoring.
///
/// # Arguments
/// * `base_confidence` - Original confidence from 5-factor calculation
/// * `metrics` - Consciousness metrics from Phi-lab
///
/// # Returns
/// Adjusted confidence with Phi factors applied
pub fn apply_consciousness_to_confidence(
    base_confidence: f64,
    metrics: &ConsciousnessMetrics,
) -> f64 {
    let factors = PhiConfidenceFactors::from_metrics(metrics);
    let adjusted = base_confidence * factors.combined_multiplier;
    adjusted.clamp(0.0, 0.99)
}

/// Enhanced confidence input with consciousness metrics
pub struct PhiEnhancedConfidenceInput<'a> {
    /// Base confidence from standard calculation
    pub base_confidence: f64,
    /// Consciousness metrics (optional - if None, no adjustment)
    pub consciousness: Option<&'a ConsciousnessMetrics>,
    /// Whether to apply harmonic bonus
    pub apply_harmonic: bool,
}

/// Calculate Phi-enhanced confidence
pub fn calculate_phi_enhanced_confidence(input: &PhiEnhancedConfidenceInput) -> f64 {
    match input.consciousness {
        Some(metrics) => {
            if input.apply_harmonic {
                apply_consciousness_to_confidence(input.base_confidence, metrics)
            } else {
                // Apply only Phi boost, not harmonic
                let phi_boost = phi_to_confidence_boost(metrics.phi);
                (input.base_confidence * phi_boost).clamp(0.0, 0.99)
            }
        }
        None => input.base_confidence,
    }
}

/// Check if agent coherence allows a specific operation
pub fn coherence_allows_operation(
    metrics: &ConsciousnessMetrics,
    operation_type: &str,
) -> Result<(), String> {
    match operation_type {
        "high_stakes" | "financial" | "governance" => {
            if metrics.coherence_state.allows_high_stakes() {
                Ok(())
            } else {
                Err(format!(
                    "Operation '{}' requires Coherent or Stable state, got {:?}",
                    operation_type, metrics.coherence_state
                ))
            }
        }
        "query_complex" | "batch_update" => {
            if !matches!(metrics.coherence_state, CoherenceState::Critical) {
                Ok(())
            } else {
                Err("Agent is in Critical state, operations suspended".into())
            }
        }
        _ => Ok(()), // Simple operations always allowed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherence_state_from_phi() {
        assert_eq!(CoherenceState::from_phi(0.85), CoherenceState::Coherent);
        assert_eq!(CoherenceState::from_phi(0.7), CoherenceState::Coherent);
        assert_eq!(CoherenceState::from_phi(0.6), CoherenceState::Stable);
        assert_eq!(CoherenceState::from_phi(0.4), CoherenceState::Unstable);
        assert_eq!(CoherenceState::from_phi(0.2), CoherenceState::Degraded);
        assert_eq!(CoherenceState::from_phi(0.1), CoherenceState::Critical);
    }

    #[test]
    fn test_phi_to_confidence_boost() {
        // High Phi should boost
        let boost_high = phi_to_confidence_boost(0.9);
        assert!(boost_high > 1.1);

        // Neutral Phi should be ~1.0
        let boost_mid = phi_to_confidence_boost(0.5);
        assert!((boost_mid - 1.0).abs() < 0.01);

        // Low Phi should penalize
        let boost_low = phi_to_confidence_boost(0.2);
        assert!(boost_low < 0.9);
    }

    #[test]
    fn test_apply_phi_to_confidence() {
        // High coherence should boost
        let boosted = apply_phi_to_confidence(0.6, 0.8, 2);
        assert!(boosted > 0.6);

        // Low coherence should penalize
        let penalized = apply_phi_to_confidence(0.6, 0.2, 0);
        assert!(penalized < 0.6);

        // Never exceed 0.99
        let capped = apply_phi_to_confidence(0.95, 1.0, 4);
        assert!(capped <= 0.99);
    }

    #[test]
    fn test_consciousness_metrics() {
        let metrics = ConsciousnessMetrics::new(0.75, 3, 1700000000);
        assert_eq!(metrics.coherence_state, CoherenceState::Coherent);
        assert_eq!(metrics.harmonic_level, 3);

        // Check staleness
        assert!(!metrics.is_stale(1700001000)); // Within 1 hour
        assert!(metrics.is_stale(1700010000)); // Over 1 hour
    }

    #[test]
    fn test_phi_confidence_factors() {
        let metrics = ConsciousnessMetrics::new(0.8, 4, 1700000000);
        let factors = PhiConfidenceFactors::from_metrics(&metrics);

        // Coherent state: 1.15x
        assert!((factors.coherence_factor - 1.15).abs() < 0.01);

        // H4: 1.1x harmonic
        assert!((factors.harmonic_factor - 1.1).abs() < 0.01);

        // Combined should be > 1.2
        assert!(factors.combined_multiplier > 1.2);
    }

    #[test]
    fn test_operation_permissions() {
        let coherent = ConsciousnessMetrics::new(0.8, 2, 1700000000);
        let critical = ConsciousnessMetrics::new(0.1, 0, 1700000000);

        assert!(coherence_allows_operation(&coherent, "high_stakes").is_ok());
        assert!(coherence_allows_operation(&critical, "high_stakes").is_err());
        assert!(coherence_allows_operation(&critical, "query_complex").is_err());
    }
}
