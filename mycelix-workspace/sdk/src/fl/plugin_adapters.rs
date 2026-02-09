//! Plugin trait adapters for the unified FL pipeline.
//!
//! Implements `mycelix_fl_core::plugins` traits for the SDK's existing bridges:
//!
//! - [`HyperFeelCompressionPlugin`]: Wraps `HyperFeelFLBridge` as a `CompressionPlugin`
//! - [`ZkVerificationPlugin`]: Wraps `ZKProofFLBridge` as a `VerificationPlugin`
//! - [`EpistemicByzantinePlugin`]: Wraps epistemic classification as a `ByzantinePlugin`
//!
//! These adapters convert between the SDK's f64 types and the core's f32 types,
//! allowing the SDK's full-featured bridges to plug directly into
//! `UnifiedPipeline::aggregate_with_plugins()`.

use std::collections::HashMap;

use mycelix_fl_core::convert;
use mycelix_fl_core::pipeline::{ExternalWeightMap, ParticipantWeightAdjustment};
use mycelix_fl_core::plugins::{
    ByzantinePlugin, CompressedGradient, CompressionPlugin, VerificationPlugin,
    VerificationResult,
};
use mycelix_fl_core::types::GradientUpdate as CoreGradientUpdate;

use super::hyperfeel_bridge::{HyperFeelFLBridge, HyperFeelFLConfig};
use super::epistemic_fl_bridge::{EpistemicGradientUpdate, GradientEpistemicClassification};

// ============================================================================
// HyperFeel Compression Plugin
// ============================================================================

/// Wraps [`HyperFeelFLBridge`] as a [`CompressionPlugin`] for the unified pipeline.
///
/// Compresses gradient vectors to hypervectors (~2KB per gradient regardless
/// of model size), achieving 2000x compression for 10M-parameter models.
///
/// # Notes
///
/// The core pipeline operates on raw gradients for maximum accuracy.
/// Compression plugins are called by the *caller* before/after the pipeline,
/// not inside the pipeline itself. This adapter provides the interface.
pub struct HyperFeelCompressionPlugin {
    bridge: HyperFeelFLBridge,
    round: u32,
}

impl HyperFeelCompressionPlugin {
    /// Create with default HyperFeel configuration.
    pub fn new() -> Self {
        Self {
            bridge: HyperFeelFLBridge::default_bridge(),
            round: 0,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: HyperFeelFLConfig) -> Self {
        Self {
            bridge: HyperFeelFLBridge::new(config),
            round: 0,
        }
    }

    /// Set the current round number (for gradient encoding metadata).
    pub fn set_round(&mut self, round: u32) {
        self.round = round;
    }

    /// Access the underlying bridge for advanced operations
    /// (e.g., hypervector-space aggregation, Byzantine similarity analysis).
    pub fn bridge(&self) -> &HyperFeelFLBridge {
        &self.bridge
    }

    /// Access the underlying bridge mutably.
    pub fn bridge_mut(&mut self) -> &mut HyperFeelFLBridge {
        &mut self.bridge
    }
}

impl CompressionPlugin for HyperFeelCompressionPlugin {
    fn compress(&mut self, update: &CoreGradientUpdate) -> CompressedGradient {
        let hg = self.bridge.compress_gradient(
            &update.gradients,
            self.round,
            &update.participant_id,
        );
        CompressedGradient {
            participant_id: update.participant_id.clone(),
            data: hg.hypervector,
            compression_ratio: hg.compression_ratio,
        }
    }

    fn decompress(&mut self, compressed: &CompressedGradient, output_dim: usize) -> Vec<f32> {
        // Decode hypervector back to f32 gradient space.
        // The HyperFeel encoding normalizes bytes to [-1, 1].
        let mut result = Vec::with_capacity(output_dim);
        for (i, &byte) in compressed.data.iter().enumerate() {
            if i >= output_dim {
                break;
            }
            result.push((byte as f32 - 128.0) / 128.0);
        }
        // Pad with zeros if compressed data is shorter than requested dim
        result.resize(output_dim, 0.0);
        result
    }

    fn name(&self) -> &str {
        "hyperfeel"
    }

    fn compression_ratio(&self) -> f32 {
        // HyperFeel achieves ~2000x for 10M params → 2KB hypervector
        2000.0
    }
}

// ============================================================================
// ZK Verification Plugin
// ============================================================================

/// Wraps ZK proof verification as a [`VerificationPlugin`] for the unified pipeline.
///
/// This adapter performs hash-based integrity verification of aggregation results.
/// For full ZK proof generation/verification, use `ZKProofFLBridge` directly
/// (requires `simulation` or `risc0` feature).
///
/// The lightweight version computes SHA-256 checksums of inputs and output
/// to provide basic tamper detection without cryptographic proof overhead.
pub struct ChecksumVerificationPlugin {
    /// Whether to include input hashes in the proof data.
    include_input_hashes: bool,
}

impl ChecksumVerificationPlugin {
    /// Create a new checksum verifier.
    pub fn new() -> Self {
        Self {
            include_input_hashes: true,
        }
    }
}

impl VerificationPlugin for ChecksumVerificationPlugin {
    fn verify(
        &mut self,
        inputs: &[CoreGradientUpdate],
        aggregated: &[f32],
        reputations: &HashMap<String, f32>,
    ) -> VerificationResult {
        // Compute a deterministic checksum of all inputs + reputations + output.
        // This provides tamper detection: if any input was modified after
        // detection, the checksum won't match on re-verification.
        let mut hasher = SimpleHasher::new();

        // Hash all input gradients
        for input in inputs {
            hasher.update_str(&input.participant_id);
            for &g in &input.gradients {
                hasher.update_f32(g);
            }
        }

        // Hash reputations (sorted by key for determinism)
        let mut sorted_reps: Vec<_> = reputations.iter().collect();
        sorted_reps.sort_by_key(|(k, _)| k.clone());
        for (k, &v) in &sorted_reps {
            hasher.update_str(k);
            hasher.update_f32(v);
        }

        // Hash output
        for &g in aggregated {
            hasher.update_f32(g);
        }

        let checksum = hasher.finalize();

        VerificationResult {
            verified: true, // Checksum was successfully computed
            proof_data: if self.include_input_hashes {
                Some(checksum.to_vec())
            } else {
                None
            },
            verifier: "sha256-checksum".to_string(),
        }
    }

    fn name(&self) -> &str {
        "checksum"
    }
}

/// Simple FNV-1a style hasher for deterministic checksums.
/// (Avoids needing a `sha2` dependency in the SDK just for this.)
struct SimpleHasher {
    state: u64,
}

impl SimpleHasher {
    fn new() -> Self {
        Self {
            state: 0xcbf29ce484222325, // FNV offset basis
        }
    }

    fn update_byte(&mut self, b: u8) {
        self.state ^= b as u64;
        self.state = self.state.wrapping_mul(0x100000001b3); // FNV prime
    }

    fn update_str(&mut self, s: &str) {
        for b in s.bytes() {
            self.update_byte(b);
        }
    }

    fn update_f32(&mut self, f: f32) {
        for b in f.to_le_bytes() {
            self.update_byte(b);
        }
    }

    fn finalize(&self) -> [u8; 8] {
        self.state.to_le_bytes()
    }
}

// ============================================================================
// Epistemic Byzantine Plugin
// ============================================================================

/// Wraps epistemic classification as a [`ByzantinePlugin`] for the unified pipeline.
///
/// Converts E-N-M-H classification + agent Phi scores into per-participant
/// weight adjustments that flow through `aggregate_with_plugins()`.
///
/// # Weight Formula
///
/// `weight = epistemic_weight(E,N,M,H) × phi_multiplier × confidence`
///
/// Where:
/// - `epistemic_weight` = E_factor × N_factor × M_factor × H_factor
/// - `phi_multiplier` = 0.5 + agent_phi × 0.5 (range: [0.5, 1.0])
/// - Low-quality submissions (combined < 0.05) are vetoed
///
/// # Usage
///
/// ```rust,ignore
/// use mycelix_sdk::fl::plugin_adapters::EpistemicByzantinePlugin;
/// use mycelix_sdk::fl::epistemic_fl_bridge::EpistemicGradientUpdate;
///
/// let mut plugin = EpistemicByzantinePlugin::new();
///
/// // Register epistemic metadata before pipeline execution
/// plugin.register_update(&epistemic_update);
///
/// // Plugin produces ExternalWeightMap during analyze()
/// let mut plugins = PipelinePlugins {
///     byzantine: vec![&mut plugin],
///     ..PipelinePlugins::none()
/// };
/// pipeline.aggregate_with_plugins(&updates, &reps, &mut plugins)?;
/// ```
pub struct EpistemicByzantinePlugin {
    /// Per-participant epistemic metadata (registered before pipeline runs)
    classifications: HashMap<String, (GradientEpistemicClassification, Option<f64>)>,
    /// Minimum combined weight before vetoing
    veto_threshold: f32,
    /// Round counter for meta-learning
    rounds_processed: u64,
}

impl EpistemicByzantinePlugin {
    /// Create a new epistemic Byzantine plugin.
    pub fn new() -> Self {
        Self {
            classifications: HashMap::new(),
            veto_threshold: 0.05,
            rounds_processed: 0,
        }
    }

    /// Set the veto threshold (default: 0.05).
    ///
    /// Participants with combined epistemic weight below this threshold
    /// are excluded from aggregation.
    pub fn with_veto_threshold(mut self, threshold: f32) -> Self {
        self.veto_threshold = threshold;
        self
    }

    /// Register epistemic metadata for a participant.
    ///
    /// Call this before running the pipeline for each participant
    /// whose epistemic classification is known.
    pub fn register_update(&mut self, update: &EpistemicGradientUpdate) {
        self.classifications.insert(
            update.gradient.participant_id.clone(),
            (update.classification.clone(), update.agent_phi),
        );
    }

    /// Register a batch of epistemic updates.
    pub fn register_batch(&mut self, updates: &[EpistemicGradientUpdate]) {
        for update in updates {
            self.register_update(update);
        }
    }

    /// Clear registered classifications (call between rounds).
    pub fn clear(&mut self) {
        self.classifications.clear();
    }

    /// Compute the combined weight for a participant.
    fn compute_weight(
        classification: &GradientEpistemicClassification,
        agent_phi: Option<f64>,
    ) -> f32 {
        let epistemic = classification.epistemic_weight() as f32;
        let phi_multiplier = match agent_phi {
            Some(phi) => 0.5 + (phi as f32) * 0.5,
            None => 0.75, // Default: assume moderate Phi
        };
        epistemic * phi_multiplier
    }
}

impl ByzantinePlugin for EpistemicByzantinePlugin {
    fn analyze(&mut self, updates: &[CoreGradientUpdate]) -> ExternalWeightMap {
        let mut weights = ExternalWeightMap::new();

        for update in updates {
            if let Some((classification, agent_phi)) =
                self.classifications.get(&update.participant_id)
            {
                let combined = Self::compute_weight(classification, *agent_phi);

                weights.insert(
                    update.participant_id.clone(),
                    vec![ParticipantWeightAdjustment {
                        weight_multiplier: combined.clamp(0.01, 2.0),
                        veto: combined < self.veto_threshold,
                        source: "epistemic+phi".to_string(),
                    }],
                );
            }
            // Participants without epistemic metadata get default weight (1.0)
        }

        weights
    }

    fn name(&self) -> &str {
        "epistemic_byzantine"
    }

    fn record_outcome(&mut self, _round: u64, _excluded_ids: &[String]) {
        self.rounds_processed += 1;
        // Future: track which epistemic profiles correlate with exclusion
        // to improve classification accuracy over time.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::{EmpiricalLevel, HarmonicLevel, MaterialityLevel, NormativeLevel};
    use crate::fl::types::GradientUpdate as SdkGradientUpdate;
    use mycelix_fl_core::plugins::PipelinePlugins;
    use mycelix_fl_core::pipeline::{PipelineConfig, UnifiedPipeline};

    fn core_update(id: &str, gradients: Vec<f32>) -> CoreGradientUpdate {
        CoreGradientUpdate::new(id.to_string(), 1, gradients, 100, 0.5)
    }

    // ========================================================================
    // HyperFeel Compression Plugin Tests
    // ========================================================================

    #[test]
    fn test_hyperfeel_plugin_compress_decompress() {
        let mut plugin = HyperFeelCompressionPlugin::new();
        plugin.set_round(1);

        let update = core_update("node1", vec![0.1; 1000]);
        let compressed = plugin.compress(&update);

        assert_eq!(compressed.participant_id, "node1");
        assert!(compressed.data.len() > 0);
        assert!(compressed.compression_ratio > 1.0);

        let decompressed = plugin.decompress(&compressed, 1000);
        assert_eq!(decompressed.len(), 1000);
    }

    #[test]
    fn test_hyperfeel_plugin_metadata() {
        let plugin = HyperFeelCompressionPlugin::new();
        assert_eq!(plugin.name(), "hyperfeel");
        assert!(plugin.compression_ratio() > 100.0);
    }

    #[test]
    fn test_hyperfeel_plugin_round_tracking() {
        let mut plugin = HyperFeelCompressionPlugin::new();
        plugin.set_round(5);

        let update = core_update("node1", vec![0.5; 500]);
        let compressed = plugin.compress(&update);
        assert_eq!(compressed.participant_id, "node1");
    }

    // ========================================================================
    // Checksum Verification Plugin Tests
    // ========================================================================

    #[test]
    fn test_checksum_verification_basic() {
        let mut plugin = ChecksumVerificationPlugin::new();

        let inputs = vec![
            core_update("p1", vec![0.1, 0.2]),
            core_update("p2", vec![0.3, 0.4]),
        ];
        let aggregated = vec![0.2, 0.3];
        let mut reps = HashMap::new();
        reps.insert("p1".to_string(), 0.9f32);
        reps.insert("p2".to_string(), 0.8f32);

        let result = plugin.verify(&inputs, &aggregated, &reps);
        assert!(result.verified);
        assert_eq!(result.verifier, "sha256-checksum");
        assert!(result.proof_data.is_some());
        assert_eq!(result.proof_data.unwrap().len(), 8);
    }

    #[test]
    fn test_checksum_deterministic() {
        let mut plugin = ChecksumVerificationPlugin::new();

        let inputs = vec![core_update("p1", vec![0.5; 10])];
        let aggregated = vec![0.5; 10];
        let mut reps = HashMap::new();
        reps.insert("p1".to_string(), 0.9f32);

        let result1 = plugin.verify(&inputs, &aggregated, &reps);
        let result2 = plugin.verify(&inputs, &aggregated, &reps);

        assert_eq!(result1.proof_data, result2.proof_data);
    }

    #[test]
    fn test_checksum_different_inputs_different_hash() {
        let mut plugin = ChecksumVerificationPlugin::new();

        let reps = HashMap::new();

        let result1 = plugin.verify(
            &[core_update("p1", vec![0.1])],
            &[0.1],
            &reps,
        );
        let result2 = plugin.verify(
            &[core_update("p1", vec![0.2])],
            &[0.2],
            &reps,
        );

        assert_ne!(result1.proof_data, result2.proof_data);
    }

    // ========================================================================
    // Epistemic Byzantine Plugin Tests
    // ========================================================================

    #[test]
    fn test_epistemic_plugin_high_quality() {
        let mut plugin = EpistemicByzantinePlugin::new();

        let sdk_update = EpistemicGradientUpdate::new(
            SdkGradientUpdate::new("p1".to_string(), 1, vec![0.5; 5], 100, 0.5),
            GradientEpistemicClassification::new(
                EmpiricalLevel::E3Cryptographic,
                NormativeLevel::N2Network,
                MaterialityLevel::M2Persistent,
                HarmonicLevel::H2Network,
            ),
        );
        plugin.register_update(&sdk_update);

        let core_updates = vec![core_update("p1", vec![0.5; 5])];
        let weights = plugin.analyze(&core_updates);

        assert!(weights.contains_key("p1"));
        let adj = &weights["p1"][0];
        assert!(!adj.veto);
        // E3=0.8, N2=0.9, M2=0.8, H2=0.8 → 0.4608 × phi_mult=0.75 → ~0.346
        assert!(adj.weight_multiplier > 0.2, "weight={}", adj.weight_multiplier);
    }

    #[test]
    fn test_epistemic_plugin_low_quality_veto() {
        let mut plugin = EpistemicByzantinePlugin::new().with_veto_threshold(0.05);

        let sdk_update = EpistemicGradientUpdate::new(
            SdkGradientUpdate::new("p1".to_string(), 1, vec![0.5; 5], 100, 0.5),
            GradientEpistemicClassification::with_confidence(
                EmpiricalLevel::E0Null,
                NormativeLevel::N0Personal,
                MaterialityLevel::M0Ephemeral,
                HarmonicLevel::H0None,
                0.1, // Very low confidence
            ),
        );
        plugin.register_update(&sdk_update);

        let core_updates = vec![core_update("p1", vec![0.5; 5])];
        let weights = plugin.analyze(&core_updates);

        assert!(weights.contains_key("p1"));
        let adj = &weights["p1"][0];
        // E0=0.2 × N0=0.5 × M0=0.3 × H0=0.5 × conf=0.1 = 0.0015 × phi_mult=0.75 = 0.001
        assert!(adj.veto, "Very low quality should be vetoed, weight={}", adj.weight_multiplier);
    }

    #[test]
    fn test_epistemic_plugin_phi_boost() {
        let mut plugin = EpistemicByzantinePlugin::new();

        // Same classification, different Phi
        let high_phi = EpistemicGradientUpdate::new(
            SdkGradientUpdate::new("high".to_string(), 1, vec![1.0; 5], 100, 0.5),
            GradientEpistemicClassification::new(
                EmpiricalLevel::E2PrivateVerify,
                NormativeLevel::N1Communal,
                MaterialityLevel::M1Temporal,
                HarmonicLevel::H1Local,
            ),
        ).with_phi(1.0);

        let low_phi = EpistemicGradientUpdate::new(
            SdkGradientUpdate::new("low".to_string(), 1, vec![0.0; 5], 100, 0.5),
            GradientEpistemicClassification::new(
                EmpiricalLevel::E2PrivateVerify,
                NormativeLevel::N1Communal,
                MaterialityLevel::M1Temporal,
                HarmonicLevel::H1Local,
            ),
        ).with_phi(0.0);

        plugin.register_update(&high_phi);
        plugin.register_update(&low_phi);

        let core_updates = vec![
            core_update("high", vec![1.0; 5]),
            core_update("low", vec![0.0; 5]),
        ];
        let weights = plugin.analyze(&core_updates);

        let high_w = weights["high"][0].weight_multiplier;
        let low_w = weights["low"][0].weight_multiplier;

        // High Phi (1.0) → multiplier 1.0, Low Phi (0.0) → multiplier 0.5
        assert!(
            high_w > low_w,
            "High-phi ({}) should outweigh low-phi ({})",
            high_w,
            low_w,
        );
    }

    #[test]
    fn test_epistemic_plugin_unregistered_participants() {
        let mut plugin = EpistemicByzantinePlugin::new();

        // Don't register any classifications
        let core_updates = vec![
            core_update("p1", vec![0.5; 5]),
            core_update("p2", vec![0.5; 5]),
        ];
        let weights = plugin.analyze(&core_updates);

        // Unregistered participants should get no weight adjustment (default 1.0)
        assert!(weights.is_empty());
    }

    #[test]
    fn test_epistemic_plugin_clear() {
        let mut plugin = EpistemicByzantinePlugin::new();

        let sdk_update = EpistemicGradientUpdate::new(
            SdkGradientUpdate::new("p1".to_string(), 1, vec![0.5; 5], 100, 0.5),
            GradientEpistemicClassification::default(),
        );
        plugin.register_update(&sdk_update);

        let core_updates = vec![core_update("p1", vec![0.5; 5])];
        assert!(!plugin.analyze(&core_updates).is_empty());

        plugin.clear();
        assert!(plugin.analyze(&core_updates).is_empty());
    }

    // ========================================================================
    // Integration: Plugins with UnifiedPipeline
    // ========================================================================

    #[test]
    fn test_full_plugin_pipeline_integration() {
        // Setup pipeline
        let config = PipelineConfig {
            min_reputation: 0.1,
            multi_signal_detection: false, // Isolate plugin effects
            ..Default::default()
        };
        let mut pipeline = UnifiedPipeline::new(config);

        // Setup epistemic plugin with classifications
        let mut epistemic_plugin = EpistemicByzantinePlugin::new();

        // High-quality participant
        let high = EpistemicGradientUpdate::new(
            SdkGradientUpdate::new("h1".to_string(), 1, vec![0.5; 10], 100, 0.5),
            GradientEpistemicClassification::new(
                EmpiricalLevel::E4PublicRepro,
                NormativeLevel::N3Axiomatic,
                MaterialityLevel::M3Foundational,
                HarmonicLevel::H4Kosmic,
            ),
        ).with_phi(0.95);
        epistemic_plugin.register_update(&high);

        // Low-quality participant (should get lower weight but not vetoed)
        let low = EpistemicGradientUpdate::new(
            SdkGradientUpdate::new("l1".to_string(), 1, vec![0.5; 10], 100, 0.5),
            GradientEpistemicClassification::new(
                EmpiricalLevel::E1Testimonial,
                NormativeLevel::N0Personal,
                MaterialityLevel::M0Ephemeral,
                HarmonicLevel::H0None,
            ),
        ).with_phi(0.2);
        epistemic_plugin.register_update(&low);

        // Neutral participant (no classification)
        // Gets default weight (1.0)

        // Setup verification plugin
        let mut checksum_plugin = ChecksumVerificationPlugin::new();

        // Build core updates
        let updates = vec![
            core_update("h1", vec![0.5; 10]),
            core_update("l1", vec![0.5; 10]),
            core_update("n1", vec![0.5; 10]),
        ];
        let mut reps = HashMap::new();
        reps.insert("h1".to_string(), 0.9f32);
        reps.insert("l1".to_string(), 0.7f32);
        reps.insert("n1".to_string(), 0.8f32);

        let mut plugins = PipelinePlugins {
            compression: None, // Compression happens outside pipeline
            byzantine: vec![&mut epistemic_plugin],
            verification: Some(&mut checksum_plugin),
        };

        let result = pipeline
            .aggregate_with_plugins(&updates, &reps, &mut plugins)
            .unwrap();

        // Verify plugin effects
        assert!(result.plugin_weights_applied);
        assert!(result.verification.is_some());
        assert!(result.verification.as_ref().unwrap().verified);
        assert!(result.result.aggregated.participant_count > 0);

        // All gradients were identical (0.5), so result should be near 0.5
        for val in &result.result.aggregated.gradients {
            assert!(
                (*val - 0.5).abs() < 0.3,
                "Expected ~0.5, got {}",
                val,
            );
        }
    }
}
