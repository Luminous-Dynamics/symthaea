// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness-to-MATL Adapter
//!
//! Bridges Symthaea KosmicSong consciousness metrics to MATL trust components.
//!
//! ## Mapping Strategy
//!
//! | KosmicSong Metric | MATL Component | Rationale |
//! |-------------------|----------------|-----------|
//! | Φ (phi) | PoGQ quality_score | Higher consciousness integration → better contribution quality |
//! | Coherence | TCDM trust_level | Synthesis score → reliable trust baseline |
//! | Harmonic Alignment | DP compliance | Balanced values → ethical data handling |
//! | Moral Uncertainty | Anomaly score | High uncertainty → potential anomalous behavior |
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐          ┌─────────────────────┐          ┌─────────────────┐
//! │   KosmicSong    │──────────│  ConsciousnessAdapter│─────────▶│   MatlBridge    │
//! │                 │          │                     │          │                 │
//! │  • Φ           │          │  • to_gradient()    │          │  • process()    │
//! │  • Coherence   │──────────│  • to_interaction() │─────────▶│  • compute()    │
//! │  • Harmonics   │          │  • to_k_vector()    │          │  • TCDM track   │
//! │  • Uncertainty │          │                     │          │                 │
//! └─────────────────┘          └─────────────────────┘          └─────────────────┘
//! ```

use serde::{Deserialize, Serialize};

use crate::error::MatlResult;
use crate::pogq::{GradientContribution, DEFAULT_EPSILON, DEFAULT_SIGMA};
use crate::tcdm::{Interaction, InteractionType};
use crate::bridge::MatlBridge;
use mycelix_core_types::{KVector, TrustScore};

/// Consciousness metrics extracted from KosmicSong
///
/// This struct captures the essential consciousness state without requiring
/// a direct dependency on the Symthaea crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetrics {
    /// Agent identifier
    pub agent_id: String,

    /// Φ (phi) - Integrated Information (0.0 to ~0.5)
    /// Higher values indicate greater consciousness integration
    pub phi: f32,

    /// Coherence score from synthesis (0.0 to 1.0)
    /// Product of Φ contribution × harmonic alignment × moral clarity
    pub coherence: f32,

    /// Harmonic alignment score (0.0 to 1.0)
    /// Measures how balanced the eight harmonies are
    pub harmonic_alignment: f32,

    /// Individual harmony activations (8 values, each 0.0 to 1.0)
    pub harmony_activations: [f32; 8],

    /// Moral uncertainty components
    pub moral_uncertainty: MoralUncertaintyMetrics,

    /// GIS (Graceful Ignorance System) uncertainty
    pub epistemic_uncertainty: f32,

    /// Current FL round (if participating)
    pub current_round: Option<u64>,
}

/// Moral uncertainty breakdown
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MoralUncertaintyMetrics {
    /// Epistemic uncertainty (knowledge gaps)
    pub epistemic: f32,
    /// Axiological uncertainty (value conflicts)
    pub axiological: f32,
    /// Deontic uncertainty (duty conflicts)
    pub deontic: f32,
}

impl MoralUncertaintyMetrics {
    /// Create new moral uncertainty metrics
    pub fn new(epistemic: f32, axiological: f32, deontic: f32) -> Self {
        Self {
            epistemic: epistemic.clamp(0.0, 1.0),
            axiological: axiological.clamp(0.0, 1.0),
            deontic: deontic.clamp(0.0, 1.0),
        }
    }

    /// Total moral uncertainty (average of components)
    pub fn total(&self) -> f32 {
        (self.epistemic + self.axiological + self.deontic) / 3.0
    }

    /// Confidence (inverse of uncertainty)
    pub fn confidence(&self) -> f32 {
        1.0 - self.total()
    }
}

impl ConsciousnessMetrics {
    /// Create from raw values
    pub fn new(agent_id: String, phi: f32, coherence: f32) -> Self {
        Self {
            agent_id,
            phi: phi.clamp(0.0, 0.5),
            coherence: coherence.clamp(0.0, 1.0),
            harmonic_alignment: 1.0 / 7.0, // Default balanced
            harmony_activations: [1.0 / 7.0; 7],
            moral_uncertainty: MoralUncertaintyMetrics::default(),
            epistemic_uncertainty: 0.5,
            current_round: None,
        }
    }

    /// Builder: set harmonic alignment
    pub fn with_harmonic_alignment(mut self, alignment: f32) -> Self {
        self.harmonic_alignment = alignment.clamp(0.0, 1.0);
        self
    }

    /// Builder: set harmony activations
    pub fn with_harmony_activations(mut self, activations: [f32; 8]) -> Self {
        // Normalize
        let sum: f32 = activations.iter().sum();
        if sum > 0.01 {
            for (i, a) in activations.iter().enumerate() {
                self.harmony_activations[i] = a / sum;
            }
        }
        self
    }

    /// Builder: set moral uncertainty
    pub fn with_moral_uncertainty(mut self, epistemic: f32, axiological: f32, deontic: f32) -> Self {
        self.moral_uncertainty = MoralUncertaintyMetrics::new(epistemic, axiological, deontic);
        self
    }

    /// Builder: set epistemic uncertainty
    pub fn with_epistemic_uncertainty(mut self, uncertainty: f32) -> Self {
        self.epistemic_uncertainty = uncertainty.clamp(0.0, 1.0);
        self
    }

    /// Builder: set current round
    pub fn with_round(mut self, round: u64) -> Self {
        self.current_round = Some(round);
        self
    }

    /// Calculate quality score from consciousness metrics
    ///
    /// Maps Φ and coherence to a quality score suitable for PoGQ
    fn quality_score(&self) -> f32 {
        // Φ contributes 40%, coherence 40%, moral clarity 20%
        let phi_normalized = self.phi / 0.5; // Normalize Φ to 0-1
        let moral_clarity = self.moral_uncertainty.confidence();

        0.4 * phi_normalized + 0.4 * self.coherence + 0.2 * moral_clarity
    }

    /// Calculate anomaly score from uncertainty metrics
    ///
    /// Higher uncertainty = higher anomaly potential
    fn anomaly_score(&self) -> f32 {
        // Combine moral and epistemic uncertainty
        let moral = self.moral_uncertainty.total();
        let epistemic = self.epistemic_uncertainty;

        // Weight epistemic higher as it represents knowledge gaps
        0.6 * epistemic + 0.4 * moral
    }

    /// Check if agent should be considered Byzantine-suspicious
    pub fn is_byzantine_suspicious(&self) -> bool {
        // High uncertainty + low coherence = suspicious
        self.anomaly_score() > 0.7 && self.coherence < 0.3
    }
}

/// Adapter for converting consciousness metrics to MATL inputs
pub struct ConsciousnessAdapter {
    /// Φ threshold for high-quality classification
    phi_high_threshold: f32,
    /// Φ threshold for low-quality classification
    phi_low_threshold: f32,
    /// Coherence threshold for trust
    coherence_threshold: f32,
    /// Moral uncertainty threshold for flagging
    moral_uncertainty_threshold: f32,
}

impl ConsciousnessAdapter {
    /// Create with default thresholds
    pub fn new() -> Self {
        Self {
            phi_high_threshold: 0.35,    // 70% of max Φ
            phi_low_threshold: 0.15,     // 30% of max Φ
            coherence_threshold: 0.5,
            moral_uncertainty_threshold: 0.7,
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(
        phi_high: f32,
        phi_low: f32,
        coherence: f32,
        moral_uncertainty: f32,
    ) -> Self {
        Self {
            phi_high_threshold: phi_high,
            phi_low_threshold: phi_low,
            coherence_threshold: coherence,
            moral_uncertainty_threshold: moral_uncertainty,
        }
    }

    /// Convert consciousness metrics to a gradient contribution
    ///
    /// This creates a pseudo-gradient that represents the agent's consciousness-based
    /// contribution quality, suitable for processing by MatlBridge.
    pub fn to_gradient_contribution(&self, metrics: &ConsciousnessMetrics) -> GradientContribution {
        let round = metrics.current_round.unwrap_or(0);
        let quality = metrics.quality_score();
        let _anomaly = metrics.anomaly_score(); // Used for future Byzantine detection enhancements

        // Generate gradient hash from consciousness state
        let gradient_hash = format!(
            "consciousness_{}_{:.4}_{:.4}",
            metrics.agent_id,
            metrics.phi,
            metrics.coherence
        );

        // Map consciousness metrics to gradient statistics
        // Higher Φ → lower variance (more stable)
        let variance = 1.0 + (1.0 - metrics.phi / 0.5) * 2.0; // 1.0 to 3.0

        // Harmonic imbalance → skewness
        let max_activation = metrics.harmony_activations.iter().cloned().fold(0.0f32, f32::max);
        let skewness = (max_activation - 1.0 / 7.0) * 7.0; // 0 to ~1

        // Moral uncertainty → kurtosis deviation
        let kurtosis = 3.0 + metrics.moral_uncertainty.total() * 2.0; // 3.0 to 5.0

        // L2 norm based on coherence
        let l2_norm = 5.0 + (1.0 - metrics.coherence) * 3.0; // 5.0 to 8.0

        // DP parameters based on quality
        // Higher quality → stronger privacy (lower epsilon, higher sigma)
        let epsilon = DEFAULT_EPSILON - quality * 3.0; // 7.0 to 10.0
        let sigma = DEFAULT_SIGMA + quality * 2.0; // 5.0 to 7.0

        GradientContribution::new(
            metrics.agent_id.clone(),
            round,
            gradient_hash,
        )
        .with_stats(
            l2_norm,
            1000, // Nominal element count
            0.0,  // Mean centered at 0
            variance,
            skewness,
            kurtosis,
        )
        .with_dp(sigma, epsilon)
    }

    /// Convert consciousness metrics to a TCDM interaction
    ///
    /// Classifies the consciousness state into an appropriate interaction type.
    pub fn to_interaction(&self, metrics: &ConsciousnessMetrics) -> Interaction {
        let interaction_type = self.classify_interaction(metrics);

        let mut interaction = Interaction::new(
            metrics.agent_id.clone(),
            interaction_type,
        );

        if let Some(round) = metrics.current_round {
            interaction = interaction.with_round(round);
        }

        let context = format!(
            "phi={:.3},coherence={:.3},harmony={:.3},uncertainty={:.3}",
            metrics.phi,
            metrics.coherence,
            metrics.harmonic_alignment,
            metrics.moral_uncertainty.total()
        );

        interaction.with_context(context)
    }

    /// Classify consciousness state into interaction type
    fn classify_interaction(&self, metrics: &ConsciousnessMetrics) -> InteractionType {
        // Check for Byzantine indicators first
        if metrics.is_byzantine_suspicious() {
            return InteractionType::ByzantineBehavior;
        }

        // Check moral uncertainty threshold
        if metrics.moral_uncertainty.total() > self.moral_uncertainty_threshold {
            return InteractionType::LowQualityContribution;
        }

        // Classify by Φ and coherence
        if metrics.phi >= self.phi_high_threshold && metrics.coherence >= self.coherence_threshold {
            InteractionType::HighQualityContribution
        } else if metrics.phi < self.phi_low_threshold || metrics.coherence < 0.3 {
            InteractionType::LowQualityContribution
        } else {
            InteractionType::FLParticipation
        }
    }

    /// Convert consciousness metrics directly to a K-Vector
    ///
    /// Maps consciousness state to the 8-dimensional K-Vector:
    /// - k_r: Reputation from coherence
    /// - k_a: Activity from epistemic certainty
    /// - k_i: Integrity from moral clarity
    /// - k_p: Performance from quality score
    /// - k_m: Membership (default, needs external tracking)
    /// - k_s: Stake from harmonic alignment
    /// - k_h: Historical from Φ normalized
    /// - k_topo: Topology contribution from coherence * quality
    pub fn to_k_vector(&self, metrics: &ConsciousnessMetrics) -> KVector {
        let quality = metrics.quality_score();

        KVector {
            // k_r (reputation) from coherence
            k_r: metrics.coherence,
            // k_a (activity) from epistemic certainty
            k_a: 1.0 - metrics.epistemic_uncertainty,
            // k_i (integrity) from moral clarity
            k_i: metrics.moral_uncertainty.confidence(),
            // k_p (performance) from quality score
            k_p: quality,
            // k_m (membership) - default, needs external tracking
            k_m: 0.5,
            // k_s (stake) from harmonic alignment
            k_s: metrics.harmonic_alignment,
            // k_h (historical) from Φ normalized
            k_h: metrics.phi / 0.5,
            // k_topo (topology contribution) from coherence * quality
            k_topo: metrics.coherence * quality,
        }
    }

    /// Process consciousness metrics through MATL bridge
    ///
    /// This is the primary integration point - takes consciousness metrics
    /// and updates the MATL trust system accordingly.
    pub fn process(&self, bridge: &mut MatlBridge, metrics: &ConsciousnessMetrics) -> MatlResult<TrustScore> {
        // Check for Byzantine behavior first
        if metrics.is_byzantine_suspicious() {
            bridge.record_byzantine(
                &metrics.agent_id,
                &format!(
                    "Consciousness-based detection: low coherence ({:.3}) with high uncertainty ({:.3})",
                    metrics.coherence,
                    metrics.anomaly_score()
                ),
            )?;
        }

        // Convert to gradient contribution and process
        let contribution = self.to_gradient_contribution(metrics);
        bridge.process_gradient(contribution)
    }

    /// Batch process multiple consciousness states
    pub fn process_batch(
        &self,
        bridge: &mut MatlBridge,
        metrics_batch: &[ConsciousnessMetrics],
    ) -> Vec<MatlResult<TrustScore>> {
        metrics_batch
            .iter()
            .map(|m| self.process(bridge, m))
            .collect()
    }

    /// Evaluate consciousness state for governance eligibility
    ///
    /// Returns thresholds based on consciousness level.
    pub fn governance_eligibility(&self, metrics: &ConsciousnessMetrics) -> GovernanceEligibility {
        let phi_normalized = metrics.phi / 0.5;

        GovernanceEligibility {
            can_propose: phi_normalized >= 0.3 && metrics.coherence >= 0.4,
            can_vote: phi_normalized >= 0.4 && metrics.coherence >= 0.5,
            can_constitutional: phi_normalized >= 0.6 && metrics.coherence >= 0.7,
            voting_weight: metrics.coherence * phi_normalized,
            trust_signal: phi_normalized * 0.8 + 0.2, // score = Φ × 0.8 + 0.2
        }
    }
}

impl Default for ConsciousnessAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Governance eligibility based on consciousness state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceEligibility {
    /// Can create governance proposals
    pub can_propose: bool,
    /// Can vote on proposals
    pub can_vote: bool,
    /// Can participate in constitutional amendments
    pub can_constitutional: bool,
    /// Voting weight multiplier
    pub voting_weight: f32,
    /// Trust signal for external systems
    pub trust_signal: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_metrics() -> ConsciousnessMetrics {
        ConsciousnessMetrics::new("agent-1".to_string(), 0.35, 0.7)
            .with_harmonic_alignment(0.8)
            .with_moral_uncertainty(0.2, 0.3, 0.2)
            .with_round(1)
    }

    #[test]
    fn test_quality_score_calculation() {
        let metrics = create_test_metrics();
        let quality = metrics.quality_score();

        // phi=0.35 → 0.7 normalized, coherence=0.7, moral_clarity≈0.77
        // 0.4*0.7 + 0.4*0.7 + 0.2*0.77 ≈ 0.714
        assert!(quality > 0.6 && quality < 0.8);
    }

    #[test]
    fn test_anomaly_score_calculation() {
        let metrics = create_test_metrics()
            .with_epistemic_uncertainty(0.8)
            .with_moral_uncertainty(0.6, 0.7, 0.6);

        let anomaly = metrics.anomaly_score();
        assert!(anomaly > 0.6);
    }

    #[test]
    fn test_byzantine_detection() {
        // High uncertainty, low coherence should be suspicious
        let metrics = ConsciousnessMetrics::new("agent-1".to_string(), 0.1, 0.2)
            .with_epistemic_uncertainty(0.9)
            .with_moral_uncertainty(0.8, 0.8, 0.8);

        assert!(metrics.is_byzantine_suspicious());
    }

    #[test]
    fn test_to_gradient_contribution() {
        let adapter = ConsciousnessAdapter::new();
        let metrics = create_test_metrics();

        let contribution = adapter.to_gradient_contribution(&metrics);

        assert_eq!(contribution.agent_id, "agent-1");
        assert_eq!(contribution.round, 1);
        assert!(contribution.l2_norm > 0.0);
        assert!(contribution.noise_sigma >= DEFAULT_SIGMA);
    }

    #[test]
    fn test_interaction_classification() {
        let adapter = ConsciousnessAdapter::new();

        // High quality
        let high_quality = ConsciousnessMetrics::new("agent-1".to_string(), 0.4, 0.8);
        let interaction = adapter.to_interaction(&high_quality);
        assert_eq!(interaction.interaction_type, InteractionType::HighQualityContribution);

        // Low quality
        let low_quality = ConsciousnessMetrics::new("agent-2".to_string(), 0.1, 0.2);
        let interaction = adapter.to_interaction(&low_quality);
        assert_eq!(interaction.interaction_type, InteractionType::LowQualityContribution);
    }

    #[test]
    fn test_k_vector_mapping() {
        let adapter = ConsciousnessAdapter::new();
        let metrics = create_test_metrics();

        let k_vector = adapter.to_k_vector(&metrics);

        // Coherence should map to k_r
        assert!((k_vector.k_r - 0.7).abs() < 0.01);
        // Phi normalized (0.35/0.5 = 0.7) should map to k_h
        assert!((k_vector.k_h - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_governance_eligibility() {
        let adapter = ConsciousnessAdapter::new();

        // High consciousness agent
        let high = ConsciousnessMetrics::new("agent-1".to_string(), 0.4, 0.8);
        let eligibility = adapter.governance_eligibility(&high);
        assert!(eligibility.can_propose);
        assert!(eligibility.can_vote);

        // Low consciousness agent
        let low = ConsciousnessMetrics::new("agent-2".to_string(), 0.1, 0.3);
        let eligibility = adapter.governance_eligibility(&low);
        assert!(!eligibility.can_propose);
        assert!(!eligibility.can_vote);
    }

    #[test]
    fn test_process_through_bridge() {
        let adapter = ConsciousnessAdapter::new();
        let mut bridge = MatlBridge::new();
        let metrics = create_test_metrics();

        let result = adapter.process(&mut bridge, &metrics);
        assert!(result.is_ok());

        let trust = result.unwrap();
        assert!(trust.total > 0.0);
    }
}
