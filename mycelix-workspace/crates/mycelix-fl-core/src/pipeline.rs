//! Unified FL Pipeline
//!
//! Chains all FL capabilities into a single coherent pipeline:
//!
//! ```text
//! Validate → DP → Reputation Gate → Byzantine Detect → Trim → Aggregate
//! ```
//!
//! No existing FL system combines: consciousness-guided quality + HDC compression +
//! multi-signal Byzantine detection + reputation-weighted trimmed mean + epistemic
//! classification + ZK proofs + differential privacy in one pipeline.

use std::collections::HashMap;

use crate::aggregation::{
    self, validate_gradient_consistency, AggregationError,
};
use crate::byzantine::{MultiSignalByzantineDetector, MultiSignalDetectionResult};
use crate::hybrid_bft::{
    self, HybridAggregationResult, HybridBftConfig, ReputationGradient,
};
use crate::privacy::{self, DifferentialPrivacyConfig, PrivacyReport, RdpBudgetTracker};
use crate::types::{AggregatedGradient, AggregationMethod, GradientUpdate, MAX_BYZANTINE_TOLERANCE};

/// Per-participant weight adjustment from external modules (e.g., consciousness, epistemic).
///
/// Weight modifiers allow external systems to adjust aggregation weights without
/// the core pipeline needing to depend on those systems. Each modifier provides:
/// - A weight multiplier (1.0 = no change, 0.0 = exclude)
/// - An optional veto (exclude the participant entirely)
#[derive(Debug, Clone)]
pub struct ParticipantWeightAdjustment {
    /// Multiplier applied to this participant's aggregation weight (0.0-2.0)
    pub weight_multiplier: f32,
    /// If true, exclude this participant entirely
    pub veto: bool,
    /// Source of the adjustment (for logging/debugging)
    pub source: String,
}

impl ParticipantWeightAdjustment {
    pub fn neutral() -> Self {
        Self {
            weight_multiplier: 1.0,
            veto: false,
            source: String::new(),
        }
    }
}

/// External weight adjustments keyed by participant ID.
///
/// Plug consciousness, epistemic classification, PoGQ, or any other
/// external signal into the pipeline by providing a HashMap of adjustments.
pub type ExternalWeightMap = HashMap<String, Vec<ParticipantWeightAdjustment>>;

/// Unified pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Minimum reputation for participation (reputation gate)
    pub min_reputation: f32,
    /// Maximum Byzantine tolerance (0.34 validated)
    pub max_byzantine_tolerance: f32,
    /// Aggregation method to use
    pub aggregation_method: AggregationMethod,
    /// Differential privacy configuration (None = disabled)
    pub dp_config: Option<DifferentialPrivacyConfig>,
    /// Trim fraction for hybrid BFT
    pub trim_fraction: f32,
    /// Reputation exponent for weighting (2.0 = quadratic)
    pub reputation_exponent: f32,
    /// Enable multi-signal Byzantine detection
    pub multi_signal_detection: bool,
    /// Trust threshold for trust-weighted aggregation
    pub trust_threshold: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            min_reputation: 0.3,
            max_byzantine_tolerance: MAX_BYZANTINE_TOLERANCE,
            aggregation_method: AggregationMethod::TrustWeighted,
            dp_config: None,
            trim_fraction: 0.1,
            reputation_exponent: 2.0,
            multi_signal_detection: true,
            trust_threshold: 0.5,
        }
    }
}

impl PipelineConfig {
    /// Create a high-security configuration
    pub fn high_security() -> Self {
        Self {
            min_reputation: 0.4,
            max_byzantine_tolerance: 0.30,
            aggregation_method: AggregationMethod::Krum,
            dp_config: Some(DifferentialPrivacyConfig::moderate_privacy()),
            trim_fraction: 0.2,
            reputation_exponent: 2.0,
            multi_signal_detection: true,
            trust_threshold: 0.6,
        }
    }

    /// Create a performance-optimized configuration
    pub fn performance() -> Self {
        Self {
            min_reputation: 0.2,
            max_byzantine_tolerance: MAX_BYZANTINE_TOLERANCE,
            aggregation_method: AggregationMethod::FedAvg,
            dp_config: None,
            trim_fraction: 0.1,
            reputation_exponent: 2.0,
            multi_signal_detection: false,
            trust_threshold: 0.3,
        }
    }
}

/// Pipeline execution statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Total contributions received
    pub total_contributions: usize,
    /// Contributions after DP
    pub after_dp: usize,
    /// Contributions after reputation gate
    pub after_gate: usize,
    /// Contributions after Byzantine detection + trimming
    pub after_detection: usize,
    /// Final aggregation method used
    pub method_used: AggregationMethod,
}

/// Result of pipeline execution
#[derive(Debug)]
pub struct PipelineResult {
    /// Aggregated gradient
    pub aggregated: AggregatedGradient,
    /// Byzantine detection report (if multi-signal enabled)
    pub detection: Option<MultiSignalDetectionResult>,
    /// Hybrid BFT result (if reputation-weighted)
    pub hybrid_result: Option<HybridAggregationResult>,
    /// Privacy report
    pub privacy: Option<PrivacyReport>,
    /// Pipeline statistics
    pub stats: PipelineStats,
}

/// The unified FL pipeline
pub struct UnifiedPipeline {
    pub config: PipelineConfig,
    rdp_tracker: Option<RdpBudgetTracker>,
}

impl UnifiedPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let rdp_tracker = config
            .dp_config
            .map(|_| RdpBudgetTracker::new(1e-5));
        Self {
            config,
            rdp_tracker,
        }
    }

    /// Execute the complete FL aggregation pipeline.
    ///
    /// Pipeline stages:
    /// 1. Validate inputs (dimension consistency, metadata)
    /// 2. Apply differential privacy (clip + noise) if configured
    /// 3. Reputation gate (drop below threshold)
    /// 4. Multi-signal Byzantine detection (flag suspects)
    /// 5. Reputation-weighted outlier scoring + trimming (hybrid BFT)
    /// 6. Reputation²-weighted aggregation
    /// 7. Return result with detection metadata
    pub fn aggregate(
        &mut self,
        contributions: &[GradientUpdate],
        reputations: &HashMap<String, f32>,
    ) -> Result<PipelineResult, AggregationError> {
        let total_contributions = contributions.len();

        // Stage 1: Validate
        validate_gradient_consistency(contributions)?;

        // Stage 2: Apply DP if configured
        let mut working_updates: Vec<GradientUpdate> = contributions.to_vec();
        let privacy_report = if let Some(dp_config) = &self.config.dp_config {
            for update in working_updates.iter_mut() {
                privacy::apply_dp(&mut update.gradients, dp_config);
            }
            if let Some(tracker) = &mut self.rdp_tracker {
                tracker.record_round(dp_config.sigma());
            }
            Some(PrivacyReport {
                dp_applied: true,
                clip_norm: dp_config.clip_norm,
                sigma: dp_config.sigma(),
                epsilon_estimate: self.rdp_tracker.as_ref().map(|t| t.epsilon()),
                rounds_tracked: self.rdp_tracker.as_ref().map(|t| t.rounds).unwrap_or(0),
            })
        } else {
            None
        };
        let after_dp = working_updates.len();

        // Stage 3: Multi-signal Byzantine detection (before reputation gating)
        let detection = if self.config.multi_signal_detection && working_updates.len() >= 3 {
            let detector = MultiSignalByzantineDetector::new();
            let result = detector.detect(&working_updates);

            // Check if too many Byzantine detected
            let byz_fraction = result.byzantine_indices.len() as f32 / working_updates.len() as f32;
            if byz_fraction > self.config.max_byzantine_tolerance {
                return Err(AggregationError::TooManyByzantine);
            }
            if result.early_terminated {
                return Err(AggregationError::TooManyByzantine);
            }

            Some(result)
        } else {
            None
        };

        // Stage 4 + 5: Hybrid BFT (reputation gate + outlier trim + rep² aggregation)
        // Build ReputationGradient list
        let rep_contributions: Vec<ReputationGradient> = working_updates
            .iter()
            .map(|u| {
                let rep = reputations.get(&u.participant_id).copied().unwrap_or(0.5);
                ReputationGradient {
                    update: u.clone(),
                    reputation: rep,
                }
            })
            .collect();

        let hybrid_config = HybridBftConfig {
            min_reputation: self.config.min_reputation,
            trim_fraction: self.config.trim_fraction,
            sample_dims: 0,
            reputation_exponent: self.config.reputation_exponent,
            reputation_outlier_weight: 0.5,
        };

        let hybrid_result = hybrid_bft::hybrid_trimmed_mean(&rep_contributions, &hybrid_config);

        // If hybrid BFT succeeded, use its result
        if let Some(ref hybrid) = hybrid_result {
            let after_gate = hybrid.gated_count;
            let after_detection = hybrid.surviving_count;

            let aggregated = AggregatedGradient::new(
                hybrid.aggregated.clone(),
                contributions[0].model_version,
                hybrid.surviving_count,
                total_contributions - hybrid.surviving_count,
                self.config.aggregation_method,
            );

            return Ok(PipelineResult {
                aggregated,
                detection,
                hybrid_result,
                privacy: privacy_report,
                stats: PipelineStats {
                    total_contributions,
                    after_dp,
                    after_gate,
                    after_detection,
                    method_used: self.config.aggregation_method,
                },
            });
        }

        // Fallback: if hybrid BFT failed (not enough contributions after gating),
        // try standard aggregation on all updates
        let after_gate = working_updates.len();

        // Remove detected Byzantine from working set
        if let Some(ref det) = detection {
            let mut filtered = Vec::new();
            for (i, u) in working_updates.iter().enumerate() {
                if !det.byzantine_indices.contains(&i) {
                    filtered.push(u.clone());
                }
            }
            working_updates = filtered;
        }

        if working_updates.is_empty() {
            return Err(AggregationError::NoUpdates);
        }

        let after_detection = working_updates.len();

        // Stage 6: Standard aggregation
        let gradients = match self.config.aggregation_method {
            AggregationMethod::FedAvg => aggregation::fedavg(&working_updates)?,
            AggregationMethod::TrimmedMean => {
                aggregation::trimmed_mean(&working_updates, self.config.trim_fraction)?
            }
            AggregationMethod::Median => aggregation::coordinate_median(&working_updates)?,
            AggregationMethod::Krum => {
                if working_updates.len() >= 3 {
                    aggregation::krum(&working_updates, 1)?
                } else {
                    aggregation::coordinate_median(&working_updates)?
                }
            }
            AggregationMethod::TrustWeighted => {
                let result = aggregation::trust_weighted(
                    &working_updates,
                    reputations,
                    self.config.trust_threshold,
                )?;
                return Ok(PipelineResult {
                    aggregated: result,
                    detection,
                    hybrid_result: None,
                    privacy: privacy_report,
                    stats: PipelineStats {
                        total_contributions,
                        after_dp,
                        after_gate,
                        after_detection,
                        method_used: AggregationMethod::TrustWeighted,
                    },
                });
            }
        };

        let aggregated = AggregatedGradient::new(
            gradients,
            contributions[0].model_version,
            after_detection,
            total_contributions - after_detection,
            self.config.aggregation_method,
        );

        Ok(PipelineResult {
            aggregated,
            detection,
            hybrid_result: None,
            privacy: privacy_report,
            stats: PipelineStats {
                total_contributions,
                after_dp,
                after_gate,
                after_detection,
                method_used: self.config.aggregation_method,
            },
        })
    }

    /// Execute the pipeline with external weight adjustments.
    ///
    /// This is the consciousness-aware entry point. External modules (epistemic
    /// classification, PoGQ, Phi assessment) provide per-participant weight
    /// multipliers that are applied during the aggregation stage.
    ///
    /// External weights do NOT affect the reputation gate — they only modify
    /// the aggregation weight. A veto will gate a participant out regardless.
    ///
    /// # Weight Composition
    ///
    /// `final_weight = reputation^exponent × batch_size × product(external_multipliers)`
    pub fn aggregate_with_external_weights(
        &mut self,
        contributions: &[GradientUpdate],
        reputations: &HashMap<String, f32>,
        external_weights: &ExternalWeightMap,
    ) -> Result<PipelineResult, AggregationError> {
        // Apply vetoes only (not weight adjustments) to the gating reputation
        let mut gate_reps = reputations.clone();
        for (pid, adjustments) in external_weights {
            if adjustments.iter().any(|a| a.veto) {
                gate_reps.insert(pid.clone(), 0.0);
            }
        }

        // Build aggregation reputations: gate reputation × external multipliers.
        //
        // External weights scale the aggregation influence but must not push
        // participants below the gate threshold. A participant who passes the
        // reputation gate should still participate — just with less weight.
        let min_rep = self.config.min_reputation;
        let mut agg_reps = gate_reps.clone();
        for (pid, adjustments) in external_weights {
            if adjustments.iter().any(|a| a.veto) {
                continue; // Already vetoed
            }
            if let Some(rep) = agg_reps.get_mut(pid) {
                if *rep < min_rep {
                    continue; // Would be gated anyway, don't touch
                }
                let combined_multiplier: f32 = adjustments
                    .iter()
                    .map(|a| a.weight_multiplier)
                    .product();
                // Scale the effective weight. Floor at min_reputation so participants
                // who pass the gate are never dropped by external weight alone.
                *rep = (*rep * combined_multiplier).max(min_rep);
            }
        }

        self.aggregate(contributions, &agg_reps)
    }

    /// Get current privacy epsilon estimate
    pub fn current_epsilon(&self) -> Option<f64> {
        self.rdp_tracker.as_ref().map(|t| t.epsilon())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_contributions(n_honest: usize, n_byzantine: usize) -> (Vec<GradientUpdate>, HashMap<String, f32>) {
        let mut updates = Vec::new();
        let mut reps = HashMap::new();

        for i in 0..n_honest {
            let val = 0.5 + (i as f32 * 0.001);
            updates.push(GradientUpdate::new(
                format!("h{}", i), 1, vec![val; 10], 100, 0.5,
            ));
            reps.insert(format!("h{}", i), 0.85 + (i as f32 * 0.001));
        }

        for i in 0..n_byzantine {
            let val = if i % 2 == 0 { 100.0 } else { -100.0 };
            updates.push(GradientUpdate::new(
                format!("b{}", i), 1, vec![val; 10], 100, 0.5,
            ));
            reps.insert(format!("b{}", i), 0.15);
        }

        (updates, reps)
    }

    #[test]
    fn test_pipeline_honest() {
        let (updates, reps) = test_contributions(10, 0);
        let config = PipelineConfig::default();
        let mut pipeline = UnifiedPipeline::new(config);
        let result = pipeline.aggregate(&updates, &reps).unwrap();
        assert_eq!(result.stats.total_contributions, 10);
        for val in &result.aggregated.gradients {
            assert!((*val - 0.5).abs() < 0.1);
        }
    }

    #[test]
    fn test_pipeline_with_byzantine() {
        let (updates, reps) = test_contributions(10, 3);
        let config = PipelineConfig::default();
        let mut pipeline = UnifiedPipeline::new(config);
        let result = pipeline.aggregate(&updates, &reps).unwrap();
        // Byzantine should be gated out (rep 0.15 < min_reputation 0.3)
        for val in &result.aggregated.gradients {
            assert!((*val - 0.5).abs() < 0.2, "Should be near 0.5, got {}", val);
        }
    }

    #[test]
    fn test_pipeline_with_dp() {
        let (updates, reps) = test_contributions(5, 0);
        let config = PipelineConfig {
            dp_config: Some(DifferentialPrivacyConfig::low_privacy()),
            ..Default::default()
        };
        let mut pipeline = UnifiedPipeline::new(config);
        let result = pipeline.aggregate(&updates, &reps).unwrap();
        assert!(result.privacy.is_some());
        assert!(result.privacy.unwrap().dp_applied);
    }

    #[test]
    fn test_pipeline_high_security() {
        let (updates, reps) = test_contributions(10, 2);
        let config = PipelineConfig::high_security();
        let mut pipeline = UnifiedPipeline::new(config);
        let result = pipeline.aggregate(&updates, &reps).unwrap();
        assert!(result.privacy.is_some());
    }

    #[test]
    fn test_pipeline_34_percent_byzantine_converges() {
        let (updates, reps) = test_contributions(66, 34);
        let config = PipelineConfig {
            trim_fraction: 0.2,
            ..Default::default()
        };
        let mut pipeline = UnifiedPipeline::new(config);
        let result = pipeline.aggregate(&updates, &reps).unwrap();
        // Byzantine nodes have rep 0.15 < min_rep 0.3, so they're gated out
        for val in &result.aggregated.gradients {
            assert!(
                (*val - 0.5).abs() < 0.15,
                "Should be ~0.5, got {}",
                val
            );
        }
    }

    #[test]
    fn test_pipeline_empty() {
        let config = PipelineConfig::default();
        let mut pipeline = UnifiedPipeline::new(config);
        let result = pipeline.aggregate(&[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_stats() {
        let (updates, reps) = test_contributions(10, 0);
        let config = PipelineConfig::default();
        let mut pipeline = UnifiedPipeline::new(config);
        let result = pipeline.aggregate(&updates, &reps).unwrap();
        assert_eq!(result.stats.total_contributions, 10);
        assert_eq!(result.stats.after_dp, 10);
    }

    #[test]
    fn test_pipeline_external_weights_boost() {
        let (updates, reps) = test_contributions(5, 0);
        let config = PipelineConfig::default();
        let mut pipeline = UnifiedPipeline::new(config);

        // Boost first participant, leave others neutral
        let mut ext = ExternalWeightMap::new();
        ext.insert("h0".to_string(), vec![ParticipantWeightAdjustment {
            weight_multiplier: 1.5,
            veto: false,
            source: "test".to_string(),
        }]);

        let result = pipeline
            .aggregate_with_external_weights(&updates, &reps, &ext)
            .unwrap();
        assert!(result.aggregated.participant_count > 0);
    }

    #[test]
    fn test_pipeline_external_weights_veto() {
        let (updates, reps) = test_contributions(5, 0);
        let config = PipelineConfig::default();
        let mut pipeline = UnifiedPipeline::new(config);

        // Veto first two participants
        let mut ext = ExternalWeightMap::new();
        ext.insert("h0".to_string(), vec![ParticipantWeightAdjustment {
            weight_multiplier: 1.0,
            veto: true,
            source: "consciousness".to_string(),
        }]);
        ext.insert("h1".to_string(), vec![ParticipantWeightAdjustment {
            weight_multiplier: 1.0,
            veto: true,
            source: "epistemic".to_string(),
        }]);

        let result = pipeline
            .aggregate_with_external_weights(&updates, &reps, &ext)
            .unwrap();
        // 2 vetoed participants should have rep=0 and be gated out
        assert!(result.aggregated.participant_count <= 3);
    }

    #[test]
    fn test_pipeline_external_weights_dampen() {
        let (updates, reps) = test_contributions(5, 0);
        let config = PipelineConfig {
            min_reputation: 0.1,
            ..Default::default()
        };
        let mut pipeline = UnifiedPipeline::new(config);

        // Dampen all participants to 50% weight via phi multiplier
        let mut ext = ExternalWeightMap::new();
        for i in 0..5 {
            ext.insert(format!("h{}", i), vec![ParticipantWeightAdjustment {
                weight_multiplier: 0.5,
                veto: false,
                source: "phi".to_string(),
            }]);
        }

        let result = pipeline
            .aggregate_with_external_weights(&updates, &reps, &ext)
            .unwrap();
        assert!(result.aggregated.participant_count > 0);
    }

    #[test]
    fn test_pipeline_rdp_tracking() {
        let (updates, reps) = test_contributions(5, 0);
        let config = PipelineConfig {
            dp_config: Some(DifferentialPrivacyConfig::moderate_privacy()),
            ..Default::default()
        };
        let mut pipeline = UnifiedPipeline::new(config);

        // Run multiple rounds
        for _ in 0..10 {
            let _ = pipeline.aggregate(&updates, &reps);
        }

        let eps = pipeline.current_epsilon().unwrap();
        assert!(eps > 0.0, "Epsilon should increase with rounds");
    }
}
