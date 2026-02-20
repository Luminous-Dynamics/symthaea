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

use crate::aggregation::{self, validate_gradient_consistency, AggregationError};
use crate::byzantine::{MultiSignalByzantineDetector, MultiSignalDetectionResult};
use crate::hybrid_bft::{self, HybridAggregationResult, HybridBftConfig, ReputationGradient};
use crate::privacy::{self, DifferentialPrivacyConfig, PrivacyReport, RdpBudgetTracker};
use crate::types::{
    AggregatedGradient, AggregationMethod, GradientUpdate, MAX_BYZANTINE_TOLERANCE,
};

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

    /// Create an adaptive configuration that learns from Byzantine detection history.
    ///
    /// Uses MetaLearningByzantinePlugin to adapt signal weights and track
    /// per-participant exclusion rates across rounds. Best for long-running
    /// federated sessions where attack patterns evolve.
    pub fn adaptive() -> Self {
        Self {
            min_reputation: 0.3,
            max_byzantine_tolerance: MAX_BYZANTINE_TOLERANCE,
            aggregation_method: AggregationMethod::TrustWeighted,
            dp_config: None,
            trim_fraction: 0.15,
            reputation_exponent: 2.0,
            multi_signal_detection: true,
            trust_threshold: 0.5,
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
///
/// # Presets
///
/// - `PipelineConfig::default()` — balanced defaults
/// - `PipelineConfig::high_security()` — Krum + DP, 30% BFT limit
/// - `PipelineConfig::adaptive()` — for use with `MetaLearningByzantinePlugin`
/// - `PipelineConfig::performance()` — FedAvg, no detection, fastest
pub struct UnifiedPipeline {
    pub config: PipelineConfig,
    rdp_tracker: Option<RdpBudgetTracker>,
}

impl UnifiedPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let rdp_tracker = config.dp_config.map(|_| RdpBudgetTracker::new(1e-5));
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
            AggregationMethod::MultiKrum => {
                if working_updates.len() >= 3 {
                    aggregation::multi_krum(&working_updates, 1, 3)?
                } else {
                    aggregation::coordinate_median(&working_updates)?
                }
            }
            AggregationMethod::GeometricMedian => {
                aggregation::geometric_median(&working_updates, 100, 1e-6)?
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
                let combined_multiplier: f32 =
                    adjustments.iter().map(|a| a.weight_multiplier).product();
                // Scale the effective weight. Floor at min_reputation so participants
                // who pass the gate are never dropped by external weight alone.
                *rep = (*rep * combined_multiplier).max(min_rep);
            }
        }

        self.aggregate(contributions, &agg_reps)
    }

    /// Execute the pipeline with all plugin stages.
    ///
    /// This is the full-capability entry point that runs:
    /// 1. Standard pipeline (validate → DP → gate → detect → trim → aggregate)
    /// 2. Byzantine plugins contribute to `ExternalWeightMap`
    /// 3. Post-aggregation verification (if plugin provided)
    ///
    /// Compression plugins are not applied inside the pipeline — they should
    /// be used by callers to compress/decompress before/after pipeline execution.
    /// This keeps the pipeline operating on raw gradients for maximum accuracy.
    pub fn aggregate_with_plugins(
        &mut self,
        contributions: &[GradientUpdate],
        reputations: &HashMap<String, f32>,
        plugins: &mut crate::plugins::PipelinePlugins<'_>,
    ) -> Result<PluginPipelineResult, AggregationError> {
        // Collect external weights from all Byzantine plugins
        let mut merged_weights = ExternalWeightMap::new();
        for plugin in &mut plugins.byzantine {
            let weights = plugin.analyze(contributions);
            for (pid, adjustments) in weights {
                merged_weights.entry(pid).or_default().extend(adjustments);
            }
        }

        // Run the pipeline with merged external weights
        let result = if merged_weights.is_empty() {
            self.aggregate(contributions, reputations)?
        } else {
            self.aggregate_with_external_weights(contributions, reputations, &merged_weights)?
        };

        // Post-aggregation verification
        let verification = if let Some(verifier) = &mut plugins.verification {
            Some(verifier.verify(contributions, &result.aggregated.gradients, reputations))
        } else {
            None
        };

        // Record outcome for meta-learning plugins
        let excluded: Vec<String> = if let Some(ref det) = result.detection {
            det.byzantine_indices
                .iter()
                .filter_map(|&i| contributions.get(i).map(|u| u.participant_id.clone()))
                .collect()
        } else {
            Vec::new()
        };

        for plugin in &mut plugins.byzantine {
            plugin.record_outcome(result.aggregated.model_version, &excluded);
        }

        Ok(PluginPipelineResult {
            result,
            verification,
            plugin_weights_applied: !merged_weights.is_empty(),
        })
    }

    /// Get current privacy epsilon estimate
    pub fn current_epsilon(&self) -> Option<f64> {
        self.rdp_tracker.as_ref().map(|t| t.epsilon())
    }
}

/// Extended pipeline result including plugin outputs.
#[derive(Debug)]
pub struct PluginPipelineResult {
    /// Standard pipeline result
    pub result: PipelineResult,
    /// Post-aggregation verification (if plugin was provided)
    pub verification: Option<crate::plugins::VerificationResult>,
    /// Whether any Byzantine plugin weights were applied
    pub plugin_weights_applied: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_contributions(
        n_honest: usize,
        n_byzantine: usize,
    ) -> (Vec<GradientUpdate>, HashMap<String, f32>) {
        let mut updates = Vec::new();
        let mut reps = HashMap::new();

        for i in 0..n_honest {
            let val = 0.5 + (i as f32 * 0.001);
            updates.push(GradientUpdate::new(
                format!("h{}", i),
                1,
                vec![val; 10],
                100,
                0.5,
            ));
            reps.insert(format!("h{}", i), 0.85 + (i as f32 * 0.001));
        }

        for i in 0..n_byzantine {
            let val = if i % 2 == 0 { 100.0 } else { -100.0 };
            updates.push(GradientUpdate::new(
                format!("b{}", i),
                1,
                vec![val; 10],
                100,
                0.5,
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
        // Use 20 honest nodes: DP noise (sigma=1.0 for low_privacy) applied
        // before detection means we need enough honest nodes that noised
        // gradients aren't falsely flagged as Byzantine (>34% threshold).
        let (updates, reps) = test_contributions(20, 0);
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
        // Use 50 honest nodes: moderate DP noise (sigma=1.1) applied before
        // multi-signal detection means we need enough honest nodes that
        // noised gradients aren't falsely flagged as Byzantine.
        let (updates, reps) = test_contributions(50, 2);
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
            assert!((*val - 0.5).abs() < 0.15, "Should be ~0.5, got {}", val);
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
        ext.insert(
            "h0".to_string(),
            vec![ParticipantWeightAdjustment {
                weight_multiplier: 1.5,
                veto: false,
                source: "test".to_string(),
            }],
        );

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
        ext.insert(
            "h0".to_string(),
            vec![ParticipantWeightAdjustment {
                weight_multiplier: 1.0,
                veto: true,
                source: "consciousness".to_string(),
            }],
        );
        ext.insert(
            "h1".to_string(),
            vec![ParticipantWeightAdjustment {
                weight_multiplier: 1.0,
                veto: true,
                source: "epistemic".to_string(),
            }],
        );

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
            ext.insert(
                format!("h{}", i),
                vec![ParticipantWeightAdjustment {
                    weight_multiplier: 0.5,
                    veto: false,
                    source: "consciousness_gate".to_string(),
                }],
            );
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

    /// Byzantine phase diagram: sweep Byzantine% × reputation disparity.
    ///
    /// Validates that the pipeline converges to honest mean when:
    /// - Byzantine fraction ≤ 34% AND Byzantine reputation < honest reputation
    /// - Byzantine nodes are gated out by min_reputation
    ///
    /// This is the core safety property of the unified pipeline.
    #[test]
    fn test_byzantine_phase_diagram() {
        let target = 0.5_f32;
        let dim = 20;

        // Sweep: (byzantine_pct, byzantine_rep, honest_rep)
        let scenarios: Vec<(usize, f32, f32, bool)> = vec![
            // (byz_count out of 100, byz_rep, honest_rep, should_converge)
            (10, 0.15, 0.85, true), // 10% low-rep: trivially safe
            (20, 0.15, 0.85, true), // 20% low-rep: safe
            (30, 0.15, 0.85, true), // 30% low-rep: safe (gated)
            (34, 0.15, 0.85, true), // 34% low-rep: safe (gated)
            (34, 0.50, 0.85, true), // 34% medium-rep: safe (trimming helps)
            (10, 0.85, 0.85, true), // 10% same-rep: classical BFT handles
            (20, 0.85, 0.85, true), // 20% same-rep: trimmed mean handles
            (30, 0.85, 0.85, true), // 30% same-rep: at limit but works
        ];

        for (byz_count, byz_rep, honest_rep, should_converge) in &scenarios {
            let honest_count = 100 - byz_count;

            let mut updates = Vec::new();
            let mut reps = HashMap::new();

            for i in 0..honest_count {
                let val = target + (i as f32 * 0.001);
                updates.push(GradientUpdate::new(
                    format!("h{}", i),
                    1,
                    vec![val; dim],
                    100,
                    0.5,
                ));
                reps.insert(format!("h{}", i), *honest_rep);
            }

            for i in 0..*byz_count {
                let val = if i % 2 == 0 { 100.0 } else { -100.0 };
                updates.push(GradientUpdate::new(
                    format!("b{}", i),
                    1,
                    vec![val; dim],
                    100,
                    0.5,
                ));
                reps.insert(format!("b{}", i), *byz_rep);
            }

            let config = PipelineConfig {
                min_reputation: 0.3,
                trim_fraction: 0.2,
                multi_signal_detection: true,
                ..Default::default()
            };
            let mut pipeline = UnifiedPipeline::new(config);
            let result = pipeline.aggregate(&updates, &reps);

            if *should_converge {
                let result = result.unwrap_or_else(|e| {
                    panic!(
                        "Scenario byz={}% rep={}/{} should converge, got: {:?}",
                        byz_count, byz_rep, honest_rep, e
                    )
                });
                let max_error = result
                    .aggregated
                    .gradients
                    .iter()
                    .map(|v| (v - target).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    max_error < 0.5,
                    "Scenario byz={}% rep={}/{}: max_error={:.4} (should be <0.5)",
                    byz_count,
                    byz_rep,
                    honest_rep,
                    max_error
                );
            }
        }
    }

    #[test]
    fn test_pipeline_with_byzantine_plugin() {
        use crate::plugins::{ByzantinePlugin, PipelinePlugins};

        struct FlagBadPlugin;
        impl ByzantinePlugin for FlagBadPlugin {
            fn analyze(&mut self, updates: &[GradientUpdate]) -> ExternalWeightMap {
                let mut weights = ExternalWeightMap::new();
                for u in updates {
                    if u.participant_id.starts_with("b") {
                        weights.insert(
                            u.participant_id.clone(),
                            vec![ParticipantWeightAdjustment {
                                weight_multiplier: 0.0,
                                veto: true,
                                source: "test_plugin".into(),
                            }],
                        );
                    }
                }
                weights
            }
            fn name(&self) -> &str {
                "test_flag_bad"
            }
        }

        let (updates, reps) = test_contributions(8, 2);
        let config = PipelineConfig {
            multi_signal_detection: false, // Disable to isolate plugin effect
            ..Default::default()
        };
        let mut pipeline = UnifiedPipeline::new(config);

        let mut plugin = FlagBadPlugin;
        let mut plugins = PipelinePlugins {
            compression: None,
            byzantine: vec![&mut plugin],
            verification: None,
        };

        let result = pipeline
            .aggregate_with_plugins(&updates, &reps, &mut plugins)
            .unwrap();
        assert!(result.plugin_weights_applied);
        // Byzantine nodes should be vetoed
        for val in &result.result.aggregated.gradients {
            assert!((*val - 0.5).abs() < 0.2, "Should be near 0.5, got {}", val);
        }
    }

    /// Verify the pipeline correctly reports the number of gated/trimmed nodes.
    #[test]
    fn test_pipeline_gating_counts() {
        let mut updates = Vec::new();
        let mut reps = HashMap::new();

        // 8 honest high-rep
        for i in 0..8 {
            updates.push(GradientUpdate::new(
                format!("h{}", i),
                1,
                vec![0.5; 10],
                100,
                0.5,
            ));
            reps.insert(format!("h{}", i), 0.9);
        }

        // 2 low-rep outliers (25% < 34% so multi-signal won't reject)
        for i in 0..2 {
            updates.push(GradientUpdate::new(
                format!("l{}", i),
                1,
                vec![50.0; 10],
                100,
                0.5,
            ));
            reps.insert(format!("l{}", i), 0.1);
        }

        let config = PipelineConfig {
            min_reputation: 0.3,
            ..Default::default()
        };
        let mut pipeline = UnifiedPipeline::new(config);
        let result = pipeline.aggregate(&updates, &reps).unwrap();

        assert_eq!(result.stats.total_contributions, 10);
        // 2 low-rep should be gated out by hybrid BFT
        assert!(
            result.aggregated.participant_count <= 8,
            "At most 8 should survive, got {}",
            result.aggregated.participant_count
        );
    }

    #[test]
    fn test_adaptive_preset() {
        let config = PipelineConfig::adaptive();
        assert_eq!(config.trim_fraction, 0.15);
        assert!(config.multi_signal_detection);
        assert_eq!(config.aggregation_method, AggregationMethod::TrustWeighted);

        // Should work with the pipeline
        let (updates, reps) = test_contributions(10, 2);
        let mut pipeline = UnifiedPipeline::new(config);
        let result = pipeline.aggregate(&updates, &reps).unwrap();
        assert!(result.aggregated.participant_count > 0);
    }

    #[test]
    fn test_adaptive_preset_with_meta_learning_plugin() {
        use crate::meta_learning::MetaLearningByzantinePlugin;
        use crate::plugins::PipelinePlugins;

        let config = PipelineConfig::adaptive();
        let mut pipeline = UnifiedPipeline::new(config);
        let mut meta_plugin = MetaLearningByzantinePlugin::new();

        // Run 3 rounds — meta-learning should track participant behavior
        for round in 0..3 {
            let (updates, reps) = test_contributions(8, 2);
            let mut plugins = PipelinePlugins {
                compression: None,
                byzantine: vec![&mut meta_plugin],
                verification: None,
            };
            let result = pipeline
                .aggregate_with_plugins(&updates, &reps, &mut plugins)
                .unwrap();
            assert!(
                result.result.aggregated.participant_count > 0,
                "Round {} should produce results",
                round
            );
        }

        // After 3 rounds, the plugin should have learned something
        // test_contributions uses "b0" for Byzantine IDs
        let profile = meta_plugin.get_participant_profile("b0");
        assert!(
            profile.is_some(),
            "Should have tracked Byzantine participant"
        );
        assert!(profile.unwrap().rounds_seen > 0, "Should have seen rounds");
    }
}
