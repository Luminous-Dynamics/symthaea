//! Plugin traits for extending the unified FL pipeline.
//!
//! These traits define extension points that allow external modules
//! (consciousness, HDC compression, ZK verification, epistemic classification)
//! to plug into the pipeline without adding dependencies to this core crate.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                    UnifiedPipeline                           │
//! │                                                              │
//! │  Validate → DP → Gate → Detect → Trim → Aggregate           │
//! │      ↑                    ↑                    ↓             │
//! │  CompressionPlugin   ByzantinePlugin   VerificationPlugin   │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! Plugins are passed to `UnifiedPipeline::aggregate_with_plugins()`.
//! Each plugin is optional — the pipeline works without any plugins.

use std::collections::HashMap;

use crate::pipeline::ExternalWeightMap;
use crate::types::GradientUpdate;

/// Plugin for gradient compression (e.g., HyperFeel HDC encoding).
///
/// Compresses gradients before transmission and decompresses after aggregation.
/// The canonical implementation is `HyperFeelFLBridge` in the Mycelix SDK.
///
/// # Example Integration
///
/// ```rust,ignore
/// use mycelix_fl_core::plugins::CompressionPlugin;
///
/// struct HyperFeelPlugin { bridge: HyperFeelFLBridge }
///
/// impl CompressionPlugin for HyperFeelPlugin {
///     fn compress(&mut self, update: &GradientUpdate) -> CompressedGradient {
///         let hg = self.bridge.compress_gradient(
///             &update.gradients, 0, &update.participant_id
///         );
///         CompressedGradient {
///             participant_id: update.participant_id.clone(),
///             data: hg.hypervector,
///             compression_ratio: hg.compression_ratio,
///         }
///     }
///
///     fn decompress(&mut self, compressed: &CompressedGradient, dim: usize) -> Vec<f32> {
///         self.bridge.decode_hypervector(&compressed.data, dim)
///     }
/// }
/// ```
pub trait CompressionPlugin {
    /// Compress a gradient update for efficient transmission.
    fn compress(&mut self, update: &GradientUpdate) -> CompressedGradient;

    /// Decompress a gradient back to f32 vector.
    fn decompress(&mut self, compressed: &CompressedGradient, output_dim: usize) -> Vec<f32>;

    /// Name of this compression scheme (for logging/reports).
    fn name(&self) -> &str;

    /// Expected compression ratio (e.g., 2000.0 for HyperFeel).
    fn compression_ratio(&self) -> f32;
}

/// A compressed gradient representation.
#[derive(Debug, Clone)]
pub struct CompressedGradient {
    /// Participant who produced this gradient.
    pub participant_id: String,
    /// Compressed data (format depends on the plugin).
    pub data: Vec<u8>,
    /// Achieved compression ratio.
    pub compression_ratio: f32,
}

/// Plugin for Byzantine detection with causal/meta-learning capabilities.
///
/// External Byzantine analyzers (e.g., Symthaea's CausalByzantineDefense)
/// implement this trait to provide per-participant weight adjustments
/// that feed into the pipeline's `ExternalWeightMap`.
///
/// # Example Integration
///
/// ```rust,ignore
/// use mycelix_fl_core::plugins::ByzantinePlugin;
///
/// struct CausalByzantinePlugin { defense: CausalByzantineDefense }
///
/// impl ByzantinePlugin for CausalByzantinePlugin {
///     fn analyze(&mut self, updates: &[GradientUpdate]) -> ExternalWeightMap {
///         let mut weights = ExternalWeightMap::new();
///         for update in updates {
///             let features = self.gradient_to_features(update);
///             let (is_malicious, confidence) = self.defense.predict_malicious(&features);
///             if is_malicious {
///                 weights.insert(update.participant_id.clone(), vec![
///                     ParticipantWeightAdjustment {
///                         weight_multiplier: 1.0 - confidence as f32,
///                         veto: confidence > 0.9,
///                         source: "causal_byzantine".into(),
///                     }
///                 ]);
///             }
///         }
///         weights
///     }
/// }
/// ```
pub trait ByzantinePlugin {
    /// Analyze gradient updates and return per-participant weight adjustments.
    ///
    /// The returned `ExternalWeightMap` is merged with any other external weights
    /// before being passed to the pipeline's aggregation stage.
    fn analyze(&mut self, updates: &[GradientUpdate]) -> ExternalWeightMap;

    /// Name of this Byzantine analysis plugin (for logging/reports).
    fn name(&self) -> &str;

    /// Optional: record the outcome of aggregation for meta-learning.
    ///
    /// Called after each successful aggregation round. Plugins that implement
    /// meta-learning can use this feedback to improve future predictions.
    fn record_outcome(&mut self, _round: u64, _excluded_ids: &[String]) {
        // Default: no-op. Override for meta-learning.
    }
}

/// Plugin for post-aggregation verification (e.g., ZK proofs).
///
/// Verifies the correctness of aggregation results. The canonical
/// implementation is `ZkProofBridge` in the Mycelix SDK.
///
/// # Example Integration
///
/// ```rust,ignore
/// use mycelix_fl_core::plugins::VerificationPlugin;
///
/// struct ZkVerifier { prover: SimulationProver }
///
/// impl VerificationPlugin for ZkVerifier {
///     fn verify(&mut self, inputs: &[GradientUpdate], result: &[f32]) -> VerificationResult {
///         let proof = self.prover.prove(inputs, result);
///         VerificationResult {
///             verified: proof.is_valid(),
///             proof_data: Some(proof.to_bytes()),
///             verifier: "zk-snark".into(),
///         }
///     }
/// }
/// ```
pub trait VerificationPlugin {
    /// Verify the aggregation result given the inputs.
    fn verify(
        &mut self,
        inputs: &[GradientUpdate],
        aggregated: &[f32],
        reputations: &HashMap<String, f32>,
    ) -> VerificationResult;

    /// Name of this verification scheme.
    fn name(&self) -> &str;
}

/// Result of post-aggregation verification.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the aggregation was verified as correct.
    pub verified: bool,
    /// Optional proof data (ZK proof bytes, signature, etc.).
    pub proof_data: Option<Vec<u8>>,
    /// Name of the verifier that produced this result.
    pub verifier: String,
}

/// Collected plugins for pipeline execution.
///
/// Pass this to `UnifiedPipeline::aggregate_with_plugins()` to enable
/// all plugin stages. All fields are optional.
pub struct PipelinePlugins<'a> {
    /// Compression plugin (e.g., HyperFeel HDC).
    pub compression: Option<&'a mut dyn CompressionPlugin>,
    /// Byzantine analysis plugins (can have multiple).
    pub byzantine: Vec<&'a mut dyn ByzantinePlugin>,
    /// Verification plugin (e.g., ZK proofs).
    pub verification: Option<&'a mut dyn VerificationPlugin>,
}

impl<'a> PipelinePlugins<'a> {
    /// Create an empty plugin set (no plugins enabled).
    pub fn none() -> Self {
        Self {
            compression: None,
            byzantine: Vec::new(),
            verification: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::ParticipantWeightAdjustment;

    // A simple test plugin that flags participants with "bad" in their ID
    struct TestByzantinePlugin;

    impl ByzantinePlugin for TestByzantinePlugin {
        fn analyze(&mut self, updates: &[GradientUpdate]) -> ExternalWeightMap {
            let mut weights = ExternalWeightMap::new();
            for update in updates {
                if update.participant_id.contains("bad") {
                    weights.insert(
                        update.participant_id.clone(),
                        vec![ParticipantWeightAdjustment {
                            weight_multiplier: 0.0,
                            veto: true,
                            source: "test".into(),
                        }],
                    );
                }
            }
            weights
        }

        fn name(&self) -> &str {
            "test_byzantine"
        }
    }

    #[test]
    fn test_byzantine_plugin_produces_weights() {
        let updates = vec![
            GradientUpdate::new("good1".into(), 1, vec![0.5; 10], 100, 0.5),
            GradientUpdate::new("bad1".into(), 1, vec![100.0; 10], 100, 0.5),
            GradientUpdate::new("good2".into(), 1, vec![0.5; 10], 100, 0.5),
        ];

        let mut plugin = TestByzantinePlugin;
        let weights = plugin.analyze(&updates);

        assert_eq!(weights.len(), 1);
        assert!(weights.contains_key("bad1"));
        assert!(weights["bad1"][0].veto);
    }

    #[test]
    fn test_verification_result() {
        let result = VerificationResult {
            verified: true,
            proof_data: Some(vec![1, 2, 3, 4]),
            verifier: "test".into(),
        };
        assert!(result.verified);
        assert_eq!(result.proof_data.unwrap().len(), 4);
    }

    #[test]
    fn test_compressed_gradient() {
        let cg = CompressedGradient {
            participant_id: "node1".into(),
            data: vec![0xFF; 2048],
            compression_ratio: 2000.0,
        };
        assert_eq!(cg.data.len(), 2048);
        assert!(cg.compression_ratio > 1000.0);
    }

    #[test]
    fn test_pipeline_plugins_none() {
        let plugins = PipelinePlugins::none();
        assert!(plugins.compression.is_none());
        assert!(plugins.byzantine.is_empty());
        assert!(plugins.verification.is_none());
    }
}
