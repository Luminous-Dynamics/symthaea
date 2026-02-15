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

/// Built-in random projection compression plugin (Johnson-Lindenstrauss).
///
/// Compresses gradients to a fixed-size representation using sparse random
/// projection, then quantizes to bytes. Output size is always
/// `target_bytes + 8` bytes regardless of input dimension.
///
/// This is the honest baseline compression that works without Symthaea.
/// For HDC-based compression with better reconstruction, use
/// `ContinuousHvPlugin` from the symthaea crate.
///
/// # Compression Ratios
///
/// | Input Params | Output Bytes | Ratio |
/// |-------------|-------------|-------|
/// | 1,000       | 2,056       | ~2x   |
/// | 10,000      | 2,056       | ~20x  |
/// | 100,000     | 2,056       | ~200x |
/// | 1,000,000   | 2,056       | ~2,000x |
pub struct RandomProjectionPlugin {
    /// Target compressed size in bytes (default: 2048)
    pub target_bytes: usize,
    /// Seed for deterministic projection matrix
    pub seed: u64,
}

impl RandomProjectionPlugin {
    /// Create with default settings (2048 bytes output, seed=42).
    pub fn new() -> Self {
        Self {
            target_bytes: 2048,
            seed: 42,
        }
    }

    /// Create with custom target size and seed.
    pub fn with_config(target_bytes: usize, seed: u64) -> Self {
        Self { target_bytes, seed }
    }

    fn generate_projection_row(&self, row: usize, dim: usize) -> Vec<f32> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(row as u64));
        let mut proj = vec![0.0f32; dim];
        // Sparse random projection: ~10% non-zero entries (JL-optimal sparsity)
        let scale = (10.0 / dim as f32).sqrt();
        for val in proj.iter_mut() {
            let r: f32 = rng.gen();
            if r < 0.05 {
                *val = scale;
            } else if r < 0.10 {
                *val = -scale;
            }
        }
        proj
    }
}

impl Default for RandomProjectionPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressionPlugin for RandomProjectionPlugin {
    fn compress(&mut self, update: &GradientUpdate) -> CompressedGradient {
        let dim = update.gradients.len();
        let n_proj = self.target_bytes;

        // Project: dot product with random vector per output dimension
        let mut projected = Vec::with_capacity(n_proj);
        for row in 0..n_proj {
            let proj_vec = self.generate_projection_row(row, dim);
            let dot: f32 = update
                .gradients
                .iter()
                .zip(proj_vec.iter())
                .map(|(g, p)| g * p)
                .sum();
            projected.push(dot);
        }

        // Quantize to bytes: map range to [0, 255]
        let min_val = projected.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = projected.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(1e-10);

        let mut data = Vec::with_capacity(n_proj + 8);
        data.extend_from_slice(&min_val.to_le_bytes());
        data.extend_from_slice(&range.to_le_bytes());
        for &v in &projected {
            let normalized = ((v - min_val) / range * 255.0).round() as u8;
            data.push(normalized);
        }

        let original_bytes = dim * 4;
        let ratio = original_bytes as f32 / data.len() as f32;

        CompressedGradient {
            participant_id: update.participant_id.clone(),
            data,
            compression_ratio: ratio,
        }
    }

    fn decompress(&mut self, compressed: &CompressedGradient, output_dim: usize) -> Vec<f32> {
        if compressed.data.len() < 8 {
            return vec![0.0; output_dim];
        }

        let min_bytes: [u8; 4] = compressed.data[0..4].try_into().unwrap();
        let range_bytes: [u8; 4] = compressed.data[4..8].try_into().unwrap();
        let min_val = f32::from_le_bytes(min_bytes);
        let range = f32::from_le_bytes(range_bytes);

        let quantized = &compressed.data[8..];
        let n_proj = quantized.len();
        let mut projected = Vec::with_capacity(n_proj);
        for &byte in quantized {
            projected.push(min_val + (byte as f32 / 255.0) * range);
        }

        // Pseudo-inverse reconstruction via transpose projection
        let mut result = vec![0.0f32; output_dim];
        for (row, &proj_val) in projected.iter().enumerate() {
            let proj_vec = self.generate_projection_row(row, output_dim);
            for (i, &p) in proj_vec.iter().enumerate() {
                result[i] += proj_val * p;
            }
        }

        let scale = 1.0 / (n_proj as f32).sqrt();
        for val in result.iter_mut() {
            *val *= scale;
        }

        result
    }

    fn name(&self) -> &str {
        "random_projection"
    }

    fn compression_ratio(&self) -> f32 {
        // Nominal — actual ratio depends on input size
        1000.0
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

    #[test]
    fn test_random_projection_compress_decompress() {
        let mut plugin = RandomProjectionPlugin::new();
        let update = GradientUpdate::new("n1".into(), 1, vec![0.5; 1000], 100, 0.5);

        let compressed = plugin.compress(&update);
        assert_eq!(compressed.data.len(), 2048 + 8); // target_bytes + header
        assert!(compressed.compression_ratio > 1.0);

        let decompressed = plugin.decompress(&compressed, 1000);
        assert_eq!(decompressed.len(), 1000);
        // Lossy but finite
        for val in &decompressed {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_random_projection_compression_ratio() {
        let mut plugin = RandomProjectionPlugin::new();

        // 10K params = 40KB → 2056 bytes ≈ 20x compression
        let update = GradientUpdate::new("n1".into(), 1, vec![0.1; 10_000], 100, 0.5);
        let compressed = plugin.compress(&update);
        let ratio = (10_000 * 4) as f32 / compressed.data.len() as f32;
        assert!(
            ratio > 15.0 && ratio < 25.0,
            "Expected ~20x, got {:.1}x",
            ratio
        );
    }

    #[test]
    fn test_random_projection_deterministic() {
        let update = GradientUpdate::new("n1".into(), 1, vec![0.3; 500], 100, 0.5);

        let mut p1 = RandomProjectionPlugin::new();
        let mut p2 = RandomProjectionPlugin::new();

        let c1 = p1.compress(&update);
        let c2 = p2.compress(&update);

        assert_eq!(
            c1.data, c2.data,
            "Same seed should produce identical compression"
        );
    }

    #[test]
    fn test_random_projection_cosine_similarity() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(99);
        let gradients: Vec<f32> = (0..5000).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let update = GradientUpdate::new("n1".into(), 1, gradients.clone(), 100, 0.5);

        let mut plugin = RandomProjectionPlugin::new();
        let compressed = plugin.compress(&update);
        let decompressed = plugin.decompress(&compressed, 5000);

        // Cosine similarity should be positive (preserves direction)
        let dot: f32 = gradients
            .iter()
            .zip(decompressed.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = gradients.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = decompressed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_sim = dot / (norm_a * norm_b);

        assert!(
            cos_sim > 0.0,
            "Cosine similarity should be positive, got {}",
            cos_sim
        );
    }

    #[test]
    fn test_random_projection_custom_config() {
        let mut plugin = RandomProjectionPlugin::with_config(512, 123);
        let update = GradientUpdate::new("n1".into(), 1, vec![0.5; 1000], 100, 0.5);

        let compressed = plugin.compress(&update);
        assert_eq!(compressed.data.len(), 512 + 8);
    }
}
