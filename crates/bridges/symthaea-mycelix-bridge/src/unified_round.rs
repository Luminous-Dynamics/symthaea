// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unified Conscious FL Round
//!
//! A single-call function that runs one complete round of consciousness-guided
//! federated learning: assess → weight → aggregate.
//!
//! # Flow
//!
//! 1. For each `HyperGradient`, call `SymthaeaBackend::assess_update()` to get a
//!    `QualityScore` (Phi, epistemic confidence, anomaly severity).
//! 2. Feed the quality scores into `SymthaeaQualityPlugin`.
//! 3. Convert each `HyperGradient` into a `GradientUpdate` for the pipeline.
//! 4. Run `UnifiedPipeline::aggregate_with_plugins()` with the quality plugin.
//! 5. Return the aggregated result and per-node consciousness metadata.
//!
//! # Example
//!
//! ```ignore
//! use symthaea_mycelix_bridge::unified_round::*;
//!
//! let mut round = ConsciousFlRound::new(backend, pipeline_config);
//! let result = round.run(&snapshot, &gradients, &reputations)?;
//! println!("Aggregated {} dims, {} vetoed", result.aggregated_dim, result.vetoed_count);
//! ```

use std::collections::HashMap;

use mycelix_fl_core::pipeline::{PipelineConfig, UnifiedPipeline};
use mycelix_fl_core::plugins::PipelinePlugins;
use mycelix_fl_core::types::GradientUpdate;
use mycelix_sdk::hyperfeel::HyperGradient;

use crate::fl_plugin::{QualityPluginConfig, SymthaeaQualityPlugin};
use crate::{
    BridgeError, ConsciousAnomalySeverity, ConsciousnessBackend, ConsciousnessVector,
    ModelSnapshot, Result, SpectralConnectivityAssessment, SymthaeaBackend,
    pogq_from_quality_score,
};

/// Configuration for a conscious FL round.
#[derive(Debug, Clone)]
pub struct ConsciousRoundConfig {
    /// Pipeline config (defense strategy, thresholds, etc.)
    pub pipeline: PipelineConfig,
    /// Quality plugin config (confidence thresholds, veto/boost behavior)
    pub quality_plugin: QualityPluginConfig,
    /// Number of gradient dimensions to extract from each HyperGradient.
    /// Uses the first N values of the hypervector, projected to f32.
    /// Default: 128 (matches common FL gradient sizes).
    pub gradient_dim: usize,
}

impl Default for ConsciousRoundConfig {
    fn default() -> Self {
        Self {
            pipeline: PipelineConfig::default(),
            quality_plugin: QualityPluginConfig::default(),
            gradient_dim: 128,
        }
    }
}

impl ConsciousRoundConfig {
    /// High-security configuration: Krum defense + strict quality gating.
    pub fn high_security() -> Self {
        Self {
            pipeline: PipelineConfig::high_security(),
            quality_plugin: QualityPluginConfig {
                confidence_threshold: 0.4,
                veto_confidence: 0.15,
                veto_severe: true,
                ..QualityPluginConfig::default()
            },
            gradient_dim: 128,
        }
    }

    /// Performance configuration: relaxed thresholds for faster convergence.
    pub fn performance() -> Self {
        Self {
            pipeline: PipelineConfig::performance(),
            quality_plugin: QualityPluginConfig {
                confidence_threshold: 0.2,
                veto_confidence: 0.05,
                veto_severe: false,
                ..QualityPluginConfig::default()
            },
            gradient_dim: 128,
        }
    }
}

/// Result of a conscious FL round.
#[derive(Debug)]
pub struct ConsciousRoundResult {
    /// Aggregated gradient values.
    pub aggregated: Vec<f32>,
    /// Number of participants whose updates were included.
    pub included_count: usize,
    /// Number of participants vetoed by consciousness gating.
    pub vetoed_count: usize,
    /// Number of participants dampened (reduced weight).
    pub dampened_count: usize,
    /// Number of participants boosted (increased weight).
    pub boosted_count: usize,
    /// Per-node quality assessment from Symthaea.
    pub node_scores: HashMap<String, NodeRoundScore>,
    /// Whether the pipeline applied any Byzantine plugin weights.
    pub plugin_weights_applied: bool,
}

/// Per-node consciousness assessment for a round.
#[derive(Debug, Clone)]
pub struct NodeRoundScore {
    /// Spectral connectivity assessment (before/after/gain).
    pub spectral: SpectralConnectivityAssessment,
    /// Multi-dimensional consciousness vector.
    pub consciousness_vector: ConsciousnessVector,
    /// Epistemic confidence [0,1].
    pub epistemic_confidence: f32,
    /// Whether the node was flagged as anomalous.
    pub is_anomalous: bool,
    /// Anomaly severity classification.
    pub severity: ConsciousAnomalySeverity,
    /// PoGQ quality score derived from the QualityScore.
    pub pogq_quality: f64,
    /// PoGQ consistency score.
    pub pogq_consistency: f64,
    /// Machine-readable anomaly causes.
    pub causes: Vec<String>,
}

/// Stateful runner for conscious FL rounds.
///
/// Holds the `SymthaeaBackend` and `UnifiedPipeline`, accumulating node state
/// across rounds for trend-aware anomaly detection.
pub struct ConsciousFlRound {
    backend: SymthaeaBackend,
    pipeline: UnifiedPipeline,
    config: ConsciousRoundConfig,
    /// Cumulative round count for statistics.
    rounds_completed: u64,
}

impl ConsciousFlRound {
    /// Create a new round runner with default backend.
    pub fn new(config: ConsciousRoundConfig) -> Self {
        Self {
            backend: SymthaeaBackend::new(),
            pipeline: UnifiedPipeline::new(config.pipeline.clone()),
            config,
            rounds_completed: 0,
        }
    }

    /// Create with a custom backend.
    pub fn with_backend(backend: SymthaeaBackend, config: ConsciousRoundConfig) -> Self {
        Self {
            backend,
            pipeline: UnifiedPipeline::new(config.pipeline.clone()),
            config,
            rounds_completed: 0,
        }
    }

    /// Number of rounds completed so far.
    pub fn rounds_completed(&self) -> u64 {
        self.rounds_completed
    }

    /// Access the backend for node state inspection.
    pub fn backend(&self) -> &SymthaeaBackend {
        &self.backend
    }

    /// Run one complete round of consciousness-guided FL.
    ///
    /// # Arguments
    ///
    /// * `snapshot` - Current model state for validation evaluation
    /// * `gradients` - HyperGradients from all participants this round
    /// * `reputations` - MATL reputation scores per participant (0.0-1.0)
    ///
    /// # Returns
    ///
    /// A `ConsciousRoundResult` with the aggregated gradient, per-node
    /// consciousness scores, and veto/boost/dampen counts.
    pub fn run(
        &mut self,
        snapshot: &dyn ModelSnapshot,
        gradients: &[HyperGradient],
        reputations: &HashMap<String, f32>,
    ) -> Result<ConsciousRoundResult> {
        if gradients.is_empty() {
            return Err(BridgeError::Mycelix("No gradients provided".into()));
        }

        // Phase 1: Assess each gradient with the consciousness backend
        let mut quality_scores = HashMap::new();
        let mut node_scores = HashMap::new();

        for gradient in gradients {
            let node_id = gradient.node_id.clone();

            let quality = self.backend.assess_update(snapshot, gradient)?;

            // Derive PoGQ for metadata
            let pogq = pogq_from_quality_score(&quality);

            node_scores.insert(
                node_id.clone(),
                NodeRoundScore {
                    spectral: quality.spectral.clone(),
                    consciousness_vector: quality.consciousness_vector.clone(),
                    epistemic_confidence: quality.epistemic_confidence,
                    is_anomalous: quality.is_anomalous,
                    severity: quality.severity,
                    pogq_quality: pogq.quality,
                    pogq_consistency: pogq.consistency,
                    causes: quality.causes.clone(),
                },
            );

            quality_scores.insert(node_id, quality);
        }

        // Phase 2: Set up the quality plugin with scores
        let mut quality_plugin =
            SymthaeaQualityPlugin::with_config(self.config.quality_plugin.clone());
        quality_plugin.set_quality_scores(quality_scores);

        // Phase 3: Convert HyperGradients to GradientUpdates for the pipeline
        let gradient_updates: Vec<GradientUpdate> = gradients
            .iter()
            .map(|hg| {
                // Extract gradient values from the hypervector bytes
                let dim = self.config.gradient_dim.min(hg.hypervector.len());
                let values: Vec<f32> = hg.hypervector[..dim]
                    .iter()
                    .map(|&b| (b as f32 / 255.0) * 2.0 - 1.0)
                    .collect();

                GradientUpdate::new(
                    hg.node_id.clone(),
                    hg.round_num as u64,
                    values,
                    1,   // batch_size placeholder
                    0.0, // loss placeholder (quality assessed via Phi)
                )
            })
            .collect();

        // Phase 4: Run the pipeline with the quality plugin
        let mut plugins = PipelinePlugins {
            compression: None,
            byzantine: vec![&mut quality_plugin],
            verification: None,
        };

        let pipeline_result = self
            .pipeline
            .aggregate_with_plugins(&gradient_updates, reputations, &mut plugins)
            .map_err(|e| BridgeError::Mycelix(format!("Pipeline error: {}", e)))?;

        // Phase 5: Gather statistics from the plugin
        let (vetoed, boosted, dampened, _rounds) = quality_plugin.stats();

        self.rounds_completed += 1;

        Ok(ConsciousRoundResult {
            aggregated: pipeline_result.result.aggregated.gradients,
            included_count: gradients.len() - vetoed as usize,
            vetoed_count: vetoed as usize,
            dampened_count: dampened as usize,
            boosted_count: boosted as usize,
            node_scores,
            plugin_weights_applied: pipeline_result.plugin_weights_applied,
        })
    }
}

/// Convenience function for one-shot rounds without maintaining state.
///
/// Creates a fresh `SymthaeaBackend` and `UnifiedPipeline` each call.
/// For multi-round FL, prefer [`ConsciousFlRound`] which accumulates
/// node state across rounds.
pub fn run_conscious_fl_round(
    snapshot: &dyn ModelSnapshot,
    gradients: &[HyperGradient],
    reputations: &HashMap<String, f32>,
    config: ConsciousRoundConfig,
) -> Result<ConsciousRoundResult> {
    let mut round = ConsciousFlRound::new(config);

    // Warm-up pass: calibrate integration_before for each node so the first
    // assessment doesn't produce false Severe anomalies from the 0.5 default.
    for gradient in gradients {
        let _ = round.backend.assess_update(snapshot, gradient);
    }
    round.backend.reset_all();

    // Real assessment
    // Re-initialize pipeline and backend state except node phi history
    let mut round = ConsciousFlRound::new(round.config.clone());
    for gradient in gradients {
        let _ = round.backend.assess_update(snapshot, gradient);
    }

    // Now run with calibrated integration_before values
    round.run(snapshot, gradients, reputations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mycelix_sdk::hyperfeel::{HV16_BYTES, HyperGradient};

    struct DummySnapshot;

    impl ModelSnapshot for DummySnapshot {
        fn apply_update_clone(&self, _update: &HyperGradient) -> Result<Box<dyn ModelSnapshot>> {
            Ok(Box::new(DummySnapshot))
        }

        fn evaluate(&self) -> Result<(f32, f32)> {
            Ok((0.85, 0.15))
        }

        fn current_accuracy(&self) -> Option<f32> {
            Some(0.8)
        }
    }

    fn make_gradient(node_id: &str, round: u32, fill: u8) -> HyperGradient {
        HyperGradient::new(
            node_id.to_string(),
            round,
            vec![fill; HV16_BYTES],
            0.5,
            4_000_000,
            1.0,
            [fill; 32],
        )
    }

    #[test]
    fn test_conscious_fl_round_basic() {
        let config = ConsciousRoundConfig {
            pipeline: PipelineConfig::performance(),
            ..ConsciousRoundConfig::default()
        };
        let mut round = ConsciousFlRound::new(config);
        let snapshot = DummySnapshot;

        let gradients = vec![
            make_gradient("node-1", 1, 100),
            make_gradient("node-2", 1, 110),
            make_gradient("node-3", 1, 120),
        ];

        // Warm-up pass
        for g in &gradients {
            let _ = round.backend.assess_update(&snapshot, g);
        }

        let mut reputations = HashMap::new();
        reputations.insert("node-1".to_string(), 0.8);
        reputations.insert("node-2".to_string(), 0.7);
        reputations.insert("node-3".to_string(), 0.9);

        let result = round.run(&snapshot, &gradients, &reputations).unwrap();

        assert!(!result.aggregated.is_empty());
        assert_eq!(result.node_scores.len(), 3);
        assert!(result.included_count > 0);

        // All nodes should have scores
        for node in ["node-1", "node-2", "node-3"] {
            let score = result.node_scores.get(node).unwrap();
            assert!(score.epistemic_confidence >= 0.0);
            assert!(score.pogq_quality >= 0.0);
        }
    }

    #[test]
    fn test_multi_round_accumulates_state() {
        let config = ConsciousRoundConfig {
            pipeline: PipelineConfig::performance(),
            ..ConsciousRoundConfig::default()
        };
        let mut round = ConsciousFlRound::new(config);
        let snapshot = DummySnapshot;

        let gradients = vec![
            make_gradient("node-1", 1, 100),
            make_gradient("node-2", 1, 110),
        ];

        // Warm-up
        for g in &gradients {
            let _ = round.backend.assess_update(&snapshot, g);
        }

        let mut reputations = HashMap::new();
        reputations.insert("node-1".to_string(), 0.8);
        reputations.insert("node-2".to_string(), 0.7);

        // Round 1
        let _r1 = round.run(&snapshot, &gradients, &reputations).unwrap();
        assert_eq!(round.rounds_completed(), 1);

        // Round 2 (same gradients but backend has connectivity history)
        let r2 = round.run(&snapshot, &gradients, &reputations).unwrap();
        assert_eq!(round.rounds_completed(), 2);
        assert!(!r2.aggregated.is_empty());
    }

    #[test]
    fn test_convenience_function() {
        let snapshot = DummySnapshot;
        let gradients = vec![
            make_gradient("node-1", 1, 100),
            make_gradient("node-2", 1, 110),
            make_gradient("node-3", 1, 120),
        ];

        let mut reputations = HashMap::new();
        reputations.insert("node-1".to_string(), 0.8);
        reputations.insert("node-2".to_string(), 0.7);
        reputations.insert("node-3".to_string(), 0.9);

        let config = ConsciousRoundConfig {
            pipeline: PipelineConfig::performance(),
            ..ConsciousRoundConfig::default()
        };

        let result = run_conscious_fl_round(&snapshot, &gradients, &reputations, config).unwrap();
        assert!(!result.aggregated.is_empty());
        assert_eq!(result.node_scores.len(), 3);
    }

    #[test]
    fn test_empty_gradients_returns_error() {
        let config = ConsciousRoundConfig::default();
        let mut round = ConsciousFlRound::new(config);
        let snapshot = DummySnapshot;

        let result = round.run(&snapshot, &[], &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_high_security_config() {
        let config = ConsciousRoundConfig::high_security();
        assert!(config.quality_plugin.confidence_threshold > 0.3);
        assert!(config.quality_plugin.veto_severe);
    }

    #[test]
    fn test_performance_config() {
        let config = ConsciousRoundConfig::performance();
        assert!(config.quality_plugin.confidence_threshold < 0.3);
        assert!(!config.quality_plugin.veto_severe);
    }
}
