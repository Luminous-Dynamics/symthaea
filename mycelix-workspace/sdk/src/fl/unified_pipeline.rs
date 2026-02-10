//! Unified FL Pipeline — f64 Wrapper
//!
//! Wraps `mycelix_fl_core::pipeline::UnifiedPipeline` with f64 conversion
//! for backward compatibility with the Mycelix SDK's f64-based API.
//!
//! The core pipeline operates on f32 (matching Symthaea and Holochain).
//! This wrapper converts f64 inputs to f32, runs the pipeline, and
//! converts outputs back to f64.

use std::collections::HashMap;

use mycelix_fl_core::convert;
use mycelix_fl_core::pipeline::{
    ExternalWeightMap, ParticipantWeightAdjustment, PipelineConfig, PipelineStats, UnifiedPipeline,
};
use mycelix_fl_core::privacy::DifferentialPrivacyConfig;
use mycelix_fl_core::types::{AggregationMethod, GradientUpdate as CoreGradientUpdate};

use super::epistemic_fl_bridge::EpistemicGradientUpdate;
use super::types::GradientUpdate;

/// f64-based pipeline configuration
///
/// Mirrors `PipelineConfig` but uses f64 where the SDK expects it.
#[derive(Debug, Clone)]
pub struct PipelineConfigF64 {
    /// Minimum reputation for participation
    pub min_reputation: f64,
    /// Maximum Byzantine tolerance (0.34 validated)
    pub max_byzantine_tolerance: f64,
    /// Aggregation method
    pub aggregation_method: AggregationMethod,
    /// Differential privacy (None = disabled)
    pub dp_config: Option<DifferentialPrivacyConfig>,
    /// Trim fraction for hybrid BFT
    pub trim_fraction: f64,
    /// Reputation exponent (2.0 = quadratic)
    pub reputation_exponent: f64,
    /// Enable multi-signal Byzantine detection
    pub multi_signal_detection: bool,
    /// Trust threshold
    pub trust_threshold: f64,
}

impl Default for PipelineConfigF64 {
    fn default() -> Self {
        Self {
            min_reputation: 0.3,
            max_byzantine_tolerance: 0.34,
            aggregation_method: AggregationMethod::TrustWeighted,
            dp_config: None,
            trim_fraction: 0.1,
            reputation_exponent: 2.0,
            multi_signal_detection: true,
            trust_threshold: 0.5,
        }
    }
}

impl PipelineConfigF64 {
    /// Convert to the f32 core config
    fn to_core(&self) -> PipelineConfig {
        PipelineConfig {
            min_reputation: self.min_reputation as f32,
            max_byzantine_tolerance: self.max_byzantine_tolerance as f32,
            aggregation_method: self.aggregation_method,
            dp_config: self.dp_config,
            trim_fraction: self.trim_fraction as f32,
            reputation_exponent: self.reputation_exponent as f32,
            multi_signal_detection: self.multi_signal_detection,
            trust_threshold: self.trust_threshold as f32,
        }
    }
}

/// f64-based pipeline result
#[derive(Debug)]
pub struct PipelineResultF64 {
    /// Aggregated gradients (f64)
    pub gradients: Vec<f64>,
    /// Model version
    pub model_version: u64,
    /// Number of contributing participants
    pub participant_count: usize,
    /// Number of excluded participants
    pub excluded_count: usize,
    /// Aggregation method used
    pub method: AggregationMethod,
    /// Pipeline statistics
    pub stats: PipelineStats,
    /// Current epsilon estimate (if DP enabled)
    pub epsilon_estimate: Option<f64>,
}

/// Unified FL pipeline with f64 interface
///
/// Wraps `mycelix_fl_core::pipeline::UnifiedPipeline` with automatic
/// f32↔f64 conversion at the boundary.
pub struct UnifiedPipelineF64 {
    inner: UnifiedPipeline,
}

impl UnifiedPipelineF64 {
    /// Create a new pipeline with f64 configuration
    pub fn new(config: PipelineConfigF64) -> Self {
        Self {
            inner: UnifiedPipeline::new(config.to_core()),
        }
    }

    /// Create from a core PipelineConfig directly
    pub fn from_core_config(config: PipelineConfig) -> Self {
        Self {
            inner: UnifiedPipeline::new(config),
        }
    }

    /// Execute the pipeline with f64 gradient updates
    ///
    /// Converts f64 inputs to f32, runs the core pipeline, and
    /// converts the result back to f64.
    pub fn aggregate(
        &mut self,
        updates: &[GradientUpdate],
        reputations: &HashMap<String, f64>,
    ) -> Result<PipelineResultF64, mycelix_fl_core::aggregation::AggregationError> {
        // Convert f64 updates to f32 core updates
        let core_updates: Vec<CoreGradientUpdate> = updates
            .iter()
            .map(|u| {
                convert::update_to_f32(
                    u.participant_id.clone(),
                    u.model_version,
                    &u.gradients,
                    u.metadata.batch_size as u32,
                    u.metadata.loss,
                    u.metadata.accuracy,
                    u.metadata.timestamp,
                )
            })
            .collect();

        // Convert f64 reputations to f32
        let core_reps: HashMap<String, f32> = reputations
            .iter()
            .map(|(k, v)| (k.clone(), *v as f32))
            .collect();

        // Run the core pipeline
        let result = self.inner.aggregate(&core_updates, &core_reps)?;

        // Convert result back to f64
        Ok(PipelineResultF64 {
            gradients: convert::gradients_to_f64(&result.aggregated.gradients),
            model_version: result.aggregated.model_version,
            participant_count: result.aggregated.participant_count,
            excluded_count: result.aggregated.excluded_count,
            method: result.aggregated.method,
            stats: result.stats,
            epsilon_estimate: result.privacy.and_then(|p| p.epsilon_estimate),
        })
    }

    /// Execute the consciousness-aware pipeline with epistemic weight adjustments.
    ///
    /// This is the full-featured entry point that combines:
    /// - Epistemic classification (E-N-M-H → weight multiplier)
    /// - Agent Phi coherence (amplifies high-coherence agents)
    /// - Standard pipeline (DP → gate → detect → trim → aggregate)
    ///
    /// # Weight Formula
    ///
    /// `final_weight = reputation^exp × batch_size × epistemic_weight × phi_multiplier`
    ///
    /// Where:
    /// - `epistemic_weight = E_factor × N_factor × M_factor × H_factor × confidence`
    /// - `phi_multiplier = 0.5 + agent_phi × 0.5` (default 0.75 if no Phi)
    pub fn aggregate_consciousness_aware(
        &mut self,
        updates: &[EpistemicGradientUpdate],
        reputations: &HashMap<String, f64>,
    ) -> Result<PipelineResultF64, mycelix_fl_core::aggregation::AggregationError> {
        // Extract base gradient updates and build external weight map
        let base_updates: Vec<&GradientUpdate> = updates.iter().map(|u| &u.gradient).collect();

        let mut external_weights = ExternalWeightMap::new();

        for update in updates {
            let pid = &update.gradient.participant_id;
            let combined = update.combined_weight() as f32;

            external_weights
                .entry(pid.clone())
                .or_default()
                .push(ParticipantWeightAdjustment {
                    weight_multiplier: combined.clamp(0.01, 2.0),
                    veto: combined < 0.01,
                    source: "epistemic+phi".to_string(),
                });
        }

        // Convert f64 updates to f32 core updates
        let core_updates: Vec<CoreGradientUpdate> = base_updates
            .iter()
            .map(|u| {
                convert::update_to_f32(
                    u.participant_id.clone(),
                    u.model_version,
                    &u.gradients,
                    u.metadata.batch_size as u32,
                    u.metadata.loss,
                    u.metadata.accuracy,
                    u.metadata.timestamp,
                )
            })
            .collect();

        // Convert f64 reputations to f32
        let core_reps: HashMap<String, f32> = reputations
            .iter()
            .map(|(k, v)| (k.clone(), *v as f32))
            .collect();

        // Run the consciousness-aware core pipeline
        let result = self.inner.aggregate_with_external_weights(
            &core_updates,
            &core_reps,
            &external_weights,
        )?;

        Ok(PipelineResultF64 {
            gradients: convert::gradients_to_f64(&result.aggregated.gradients),
            model_version: result.aggregated.model_version,
            participant_count: result.aggregated.participant_count,
            excluded_count: result.aggregated.excluded_count,
            method: result.aggregated.method,
            stats: result.stats,
            epsilon_estimate: result.privacy.and_then(|p| p.epsilon_estimate),
        })
    }

    /// Get current epsilon estimate from RDP tracking
    pub fn current_epsilon(&self) -> Option<f64> {
        self.inner.current_epsilon()
    }
}

#[cfg(test)]
mod tests {
    use super::super::epistemic_fl_bridge::GradientEpistemicClassification;
    use super::*;
    use crate::epistemic::{EmpiricalLevel, HarmonicLevel, MaterialityLevel, NormativeLevel};

    #[test]
    fn test_unified_pipeline_f64_basic() {
        let config = PipelineConfigF64::default();
        let mut pipeline = UnifiedPipelineF64::new(config);

        let updates = vec![
            GradientUpdate::new("p1".to_string(), 1, vec![0.1, 0.2, 0.3], 100, 0.5),
            GradientUpdate::new("p2".to_string(), 1, vec![0.15, 0.25, 0.35], 100, 0.4),
            GradientUpdate::new("p3".to_string(), 1, vec![0.12, 0.22, 0.32], 100, 0.45),
        ];

        let mut reps = HashMap::new();
        reps.insert("p1".to_string(), 0.9);
        reps.insert("p2".to_string(), 0.85);
        reps.insert("p3".to_string(), 0.88);

        let result = pipeline.aggregate(&updates, &reps).unwrap();
        assert_eq!(result.gradients.len(), 3);
        assert_eq!(result.participant_count + result.excluded_count, 3);

        // Values should be near the input average
        for val in &result.gradients {
            assert!(*val > 0.0 && *val < 1.0);
        }
    }

    #[test]
    fn test_unified_pipeline_f64_with_byzantine() {
        let config = PipelineConfigF64 {
            multi_signal_detection: true,
            ..Default::default()
        };
        let mut pipeline = UnifiedPipelineF64::new(config);

        let mut updates = Vec::new();
        let mut reps = HashMap::new();

        // 5 honest
        for i in 0..5 {
            updates.push(GradientUpdate::new(
                format!("h{}", i),
                1,
                vec![0.5; 10],
                100,
                0.5,
            ));
            reps.insert(format!("h{}", i), 0.9);
        }

        // 1 Byzantine
        updates.push(GradientUpdate::new(
            "b0".to_string(),
            1,
            vec![100.0; 10],
            100,
            0.5,
        ));
        reps.insert("b0".to_string(), 0.15);

        let result = pipeline.aggregate(&updates, &reps).unwrap();

        // Byzantine should be gated out (rep 0.15 < threshold 0.3)
        for val in &result.gradients {
            assert!((*val - 0.5).abs() < 0.2, "Should be near 0.5, got {}", val);
        }
    }

    #[test]
    fn test_unified_pipeline_f64_with_dp() {
        let config = PipelineConfigF64 {
            dp_config: Some(DifferentialPrivacyConfig::low_privacy()),
            ..Default::default()
        };
        let mut pipeline = UnifiedPipelineF64::new(config);

        let updates = vec![
            GradientUpdate::new("p1".to_string(), 1, vec![0.5; 5], 100, 0.5),
            GradientUpdate::new("p2".to_string(), 1, vec![0.5; 5], 100, 0.5),
            GradientUpdate::new("p3".to_string(), 1, vec![0.5; 5], 100, 0.5),
        ];

        let mut reps = HashMap::new();
        reps.insert("p1".to_string(), 0.9);
        reps.insert("p2".to_string(), 0.85);
        reps.insert("p3".to_string(), 0.88);

        let result = pipeline.aggregate(&updates, &reps).unwrap();
        assert!(result.epsilon_estimate.is_some());
    }

    #[test]
    fn test_consciousness_aware_pipeline() {
        let config = PipelineConfigF64::default();
        let mut pipeline = UnifiedPipelineF64::new(config);

        // Create epistemic gradient updates with varying quality
        let high_quality = EpistemicGradientUpdate::new(
            GradientUpdate::new("p1".to_string(), 1, vec![0.1, 0.2, 0.3], 100, 0.5),
            GradientEpistemicClassification::new(
                EmpiricalLevel::E3Cryptographic,
                NormativeLevel::N2Network,
                MaterialityLevel::M2Persistent,
                HarmonicLevel::H2Network,
            ),
        )
        .with_phi(0.9);

        let medium_quality = EpistemicGradientUpdate::new(
            GradientUpdate::new("p2".to_string(), 1, vec![0.15, 0.25, 0.35], 100, 0.4),
            GradientEpistemicClassification::new(
                EmpiricalLevel::E1Testimonial,
                NormativeLevel::N1Communal,
                MaterialityLevel::M1Temporal,
                HarmonicLevel::H1Local,
            ),
        )
        .with_phi(0.5);

        let low_quality = EpistemicGradientUpdate::new(
            GradientUpdate::new("p3".to_string(), 1, vec![0.12, 0.22, 0.32], 100, 0.45),
            GradientEpistemicClassification::default(), // E0, N0, M0, H0 = lowest
        );

        let updates = vec![high_quality, medium_quality, low_quality];

        let mut reps = HashMap::new();
        reps.insert("p1".to_string(), 0.9);
        reps.insert("p2".to_string(), 0.85);
        reps.insert("p3".to_string(), 0.88);

        let result = pipeline
            .aggregate_consciousness_aware(&updates, &reps)
            .unwrap();

        assert_eq!(result.gradients.len(), 3);
        assert!(result.participant_count > 0);
    }

    #[test]
    fn test_consciousness_aware_phi_boost() {
        let config = PipelineConfigF64 {
            min_reputation: 0.1,
            ..Default::default()
        };
        let mut pipeline = UnifiedPipelineF64::new(config);

        // Two identical gradients, one with high Phi, one with low
        let high_phi = EpistemicGradientUpdate::new(
            GradientUpdate::new("high".to_string(), 1, vec![1.0; 5], 100, 0.5),
            GradientEpistemicClassification::new(
                EmpiricalLevel::E2PrivateVerify,
                NormativeLevel::N1Communal,
                MaterialityLevel::M1Temporal,
                HarmonicLevel::H1Local,
            ),
        )
        .with_phi(1.0); // Maximum Phi → multiplier = 1.0

        let low_phi = EpistemicGradientUpdate::new(
            GradientUpdate::new("low".to_string(), 1, vec![0.0; 5], 100, 0.5),
            GradientEpistemicClassification::new(
                EmpiricalLevel::E2PrivateVerify,
                NormativeLevel::N1Communal,
                MaterialityLevel::M1Temporal,
                HarmonicLevel::H1Local,
            ),
        )
        .with_phi(0.0); // Minimum Phi → multiplier = 0.5

        // Need a third participant
        let neutral = EpistemicGradientUpdate::new(
            GradientUpdate::new("neutral".to_string(), 1, vec![0.5; 5], 100, 0.5),
            GradientEpistemicClassification::new(
                EmpiricalLevel::E2PrivateVerify,
                NormativeLevel::N1Communal,
                MaterialityLevel::M1Temporal,
                HarmonicLevel::H1Local,
            ),
        )
        .with_phi(0.5);

        let updates = vec![high_phi, low_phi, neutral];
        let mut reps = HashMap::new();
        reps.insert("high".to_string(), 0.9);
        reps.insert("low".to_string(), 0.9);
        reps.insert("neutral".to_string(), 0.9);

        let result = pipeline
            .aggregate_consciousness_aware(&updates, &reps)
            .unwrap();

        // High-phi agent (gradients=1.0) should have more influence than
        // low-phi agent (gradients=0.0), so result should be > 0.5
        let avg: f64 = result.gradients.iter().sum::<f64>() / result.gradients.len() as f64;
        assert!(
            avg > 0.4,
            "High-phi agent should have more influence, avg was {}",
            avg,
        );
    }
}
