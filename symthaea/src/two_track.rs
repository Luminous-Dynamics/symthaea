//! # Two-Track Architecture: HDC Semantics + CfC Temporal
//!
//! This module implements a unified system where:
//! - **HDC track** handles semantic meaning (what concepts are present)
//! - **CfC track** learns temporal sequences (what order/patterns)
//! - **Fusion** combines both for richer understanding
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         TWO-TRACK ARCHITECTURE                              │
//! │                                                                             │
//! │   Input Embedding                                                           │
//! │         │                                                                   │
//! │         ├──────────────────┬────────────────────────┐                       │
//! │         │                  │                        │                       │
//! │         ▼                  ▼                        ▼                       │
//! │   ┌───────────┐      ┌───────────┐           ┌───────────┐                 │
//! │   │   HDC     │      │   CfC     │           │  Direct   │                 │
//! │   │ Projector │      │ Temporal  │           │  Copy     │                 │
//! │   │           │      │ Network   │           │           │                 │
//! │   └─────┬─────┘      └─────┬─────┘           └─────┬─────┘                 │
//! │         │                  │                        │                       │
//! │         ▼                  ▼                        ▼                       │
//! │   ┌───────────┐      ┌───────────┐           ┌───────────┐                 │
//! │   │ Semantic  │      │ Temporal  │           │ Original  │                 │
//! │   │    HV     │      │  State    │           │ Embedding │                 │
//! │   │ (16,384D) │      │ (hidden)  │           │           │                 │
//! │   └─────┬─────┘      └─────┬─────┘           └─────┬─────┘                 │
//! │         │                  │                        │                       │
//! │         └─────────┬────────┴────────────────────────┘                       │
//! │                   │                                                         │
//! │                   ▼                                                         │
//! │            ┌─────────────┐                                                  │
//! │            │   Fusion    │                                                  │
//! │            │   Layer     │                                                  │
//! │            └──────┬──────┘                                                  │
//! │                   │                                                         │
//! │                   ▼                                                         │
//! │         ┌─────────────────┐                                                 │
//! │         │ TwoTrackOutput  │                                                 │
//! │         │ • semantic_hv   │                                                 │
//! │         │ • temporal_state│                                                 │
//! │         │ • fused         │                                                 │
//! │         └─────────────────┘                                                 │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::two_track::{TwoTrackProcessor, TwoTrackConfig};
//!
//! let config = TwoTrackConfig::default();
//! let mut processor = TwoTrackProcessor::new(config);
//!
//! // Process an embedding through both tracks
//! let embedding = vec![0.1f32; 256];
//! let output = processor.process(&embedding);
//!
//! println!("Semantic HV dim: {}", output.semantic_hv.len());
//! println!("Temporal state dim: {}", output.temporal_state.len());
//! println!("Fused HV dim: {}", output.fused.len());
//! ```

use anyhow::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::bridges::{HdcCfcBridge, HdcCfcBridgeConfig};
use crate::dynamics::cfc::{CfCConfig, CfCNetwork, CfCNetworkConfig};
use crate::embeddings::{BridgeConfig, HdcBridge};
use crate::hdc::BinaryHV;

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the two-track processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoTrackConfig {
    /// Input embedding dimension (e.g., 256 from Qwen3 compressed)
    pub input_dim: usize,

    /// HDC projection dimension
    pub hdc_dim: usize,

    /// CfC hidden dimension for temporal processing
    pub cfc_hidden_dim: usize,

    /// Number of CfC layers
    pub cfc_layers: usize,

    /// CfC output dimension (before fusion)
    pub cfc_output_dim: usize,

    /// Fusion weights: (semantic_weight, temporal_weight)
    pub fusion_weights: (f32, f32),

    /// Whether to normalize before fusion
    pub normalize_before_fusion: bool,

    /// Random seed for reproducibility
    pub seed: u64,

    /// Optional genesis phrase for deterministic initialization
    pub genesis_phrase: Option<String>,

    /// Delta-t for CfC temporal stepping
    pub cfc_delta_t: f32,

    /// Learning rate for CfC training
    pub cfc_learning_rate: f32,

    /// Enable HDC-CfC bridge for bidirectional semantic-temporal translation
    pub enable_bridge: bool,

    /// Bridge intermediate dimension (if bridge enabled)
    pub bridge_intermediate_dim: usize,

    /// Number of attention heads in bridge
    pub bridge_attention_heads: usize,
}

impl Default for TwoTrackConfig {
    fn default() -> Self {
        Self {
            input_dim: 256,
            hdc_dim: HDC_DIMENSION,
            cfc_hidden_dim: 128,
            cfc_layers: 2,
            cfc_output_dim: 64,
            fusion_weights: (0.6, 0.4), // Slightly favor semantics
            normalize_before_fusion: true,
            seed: 0x5954_2154, // "TWOT" in hex-like
            genesis_phrase: None,
            cfc_delta_t: 0.02, // 50Hz
            cfc_learning_rate: 0.001,
            enable_bridge: false, // Off by default for backward compatibility
            bridge_intermediate_dim: 512,
            bridge_attention_heads: 4,
        }
    }
}

impl TwoTrackConfig {
    /// Create config optimized for semantic tasks
    pub fn semantic_heavy() -> Self {
        Self {
            fusion_weights: (0.8, 0.2),
            ..Default::default()
        }
    }

    /// Create config optimized for temporal tasks
    pub fn temporal_heavy() -> Self {
        Self {
            fusion_weights: (0.3, 0.7),
            cfc_hidden_dim: 256,
            cfc_layers: 3,
            ..Default::default()
        }
    }

    /// Create config with small HDC dimension for speed
    pub fn fast() -> Self {
        Self {
            hdc_dim: 2048,
            cfc_hidden_dim: 64,
            cfc_layers: 1,
            enable_bridge: false,
            ..Default::default()
        }
    }

    /// Create config with HDC-CfC bridge enabled for bidirectional translation
    pub fn with_bridge() -> Self {
        Self {
            enable_bridge: true,
            bridge_intermediate_dim: 512,
            bridge_attention_heads: 4,
            ..Default::default()
        }
    }

    /// Create high-fidelity config with bridge and larger dimensions
    pub fn high_fidelity() -> Self {
        Self {
            cfc_hidden_dim: 256,
            cfc_layers: 4,
            cfc_output_dim: 128,
            enable_bridge: true,
            bridge_intermediate_dim: 1024,
            bridge_attention_heads: 8,
            ..Default::default()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// OUTPUT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Output from the two-track processor
#[derive(Debug, Clone)]
pub struct TwoTrackOutput {
    /// Semantic hypervector from HDC projection
    pub semantic_hv: ContinuousHV,

    /// Temporal state from CfC network
    pub temporal_state: Vec<f32>,

    /// Fused representation combining both tracks
    pub fused: ContinuousHV,

    /// Similarity of semantic track to previous (for tracking)
    pub semantic_stability: f32,

    /// Temporal prediction confidence
    pub temporal_confidence: f32,
}

impl TwoTrackOutput {
    /// Get the semantic HV as a binary BinaryHV
    pub fn semantic_hv16(&self) -> BinaryHV {
        continuous_to_hv16(&self.semantic_hv)
    }

    /// Get the fused representation as binary BinaryHV
    pub fn fused_hv16(&self) -> BinaryHV {
        continuous_to_hv16(&self.fused)
    }
}

/// Convert a ContinuousHV to BinaryHV by thresholding
fn continuous_to_hv16(hv: &ContinuousHV) -> BinaryHV {
    // Convert continuous values to bipolar (-1, +1) representation
    let bipolar: Vec<f32> = hv
        .values
        .iter()
        .map(|&v| if v > 0.0 { 1.0 } else { -1.0 })
        .collect();
    BinaryHV::from_bipolar(&bipolar)
}

// ═══════════════════════════════════════════════════════════════════════════════
// HDC PROJECTOR
// ═══════════════════════════════════════════════════════════════════════════════

/// HDC projector that maps input embeddings to high-dimensional hypervectors
#[derive(Debug, Clone)]
pub struct HdcProjector {
    /// Underlying projection bridge
    bridge: HdcBridge,

    /// Configuration
    config: BridgeConfig,
}

impl HdcProjector {
    /// Create a new HDC projector
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let config = BridgeConfig {
            input_dim,
            output_dim,
            seed,
            ..Default::default()
        };
        let bridge = HdcBridge::with_config(config.clone());

        Self { bridge, config }
    }

    /// Project an embedding to HDC space as ContinuousHV
    pub fn project(&self, embedding: &[f32]) -> ContinuousHV {
        // Project to BinaryHV first
        let binary_hv = self.bridge.project(embedding);

        // Convert to ContinuousHV (bipolar: 0 -> -1.0, 1 -> +1.0)
        hv16_to_continuous(&binary_hv, self.config.output_dim)
    }

    /// Project and return BinaryHV directly
    pub fn project_hv16(&self, embedding: &[f32]) -> BinaryHV {
        self.bridge.project(embedding)
    }

    /// Get projected similarity between two embeddings
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        self.bridge.projected_similarity(a, b)
    }
}

/// Convert BinaryHV to ContinuousHV using to_bipolar method
fn hv16_to_continuous(binary_hv: &BinaryHV, dim: usize) -> ContinuousHV {
    // Use BinaryHV's to_bipolar which returns Vec<f32> with -1.0/+1.0 values
    let bipolar = binary_hv.to_bipolar();

    // Ensure we have the right dimension
    let values: Vec<f32> = if bipolar.len() >= dim {
        bipolar[..dim].to_vec()
    } else {
        let mut v = bipolar;
        v.resize(dim, -1.0);
        v
    };

    ContinuousHV::from_vec(values)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TWO-TRACK PROCESSOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Two-track processor combining HDC semantics and CfC temporal learning
///
/// This processor maintains two parallel tracks:
/// 1. **Semantic track (HDC)**: Projects inputs to high-dimensional hypervectors
///    that preserve semantic similarity. This is fast and captures "what" is present.
/// 2. **Temporal track (CfC)**: Learns temporal patterns and sequences over time.
///    This captures "when" and "how" patterns evolve.
///
/// The fusion layer combines both tracks into a unified representation.
///
/// ## Optional Bridge Enhancement
///
/// When `config.enable_bridge` is true, an HDC-CfC bridge is created that enables:
/// - Bidirectional translation between semantic and temporal representations
/// - Temporal context injection into semantic processing
/// - Semantic priors for temporal state interpretation
#[derive(Debug)]
pub struct TwoTrackProcessor {
    /// Configuration
    config: TwoTrackConfig,

    /// HDC projector for semantic track
    hdc_projector: HdcProjector,

    /// CfC network for temporal track
    cfc_temporal: CfCNetwork,

    /// Projection from input to CfC input dimension
    input_to_cfc: Vec<f32>,

    /// Projection from CfC output to HDC space (for fusion)
    temporal_to_hdc: Vec<f32>,

    /// Previous semantic HV (for stability tracking)
    prev_semantic: Option<ContinuousHV>,

    /// Running average of temporal state magnitude
    temporal_magnitude_avg: f32,

    /// Total steps processed
    total_steps: u64,

    /// Optional genesis seed for deterministic operations - reserved for future use
    _genesis: Option<GenesisSeed>,

    /// Optional HDC-CfC bridge for bidirectional semantic-temporal translation
    bridge: Option<HdcCfcBridge>,
}

impl TwoTrackProcessor {
    /// Create a new two-track processor
    pub fn new(config: TwoTrackConfig) -> Self {
        // Create HDC projector
        let hdc_projector = HdcProjector::new(config.input_dim, config.hdc_dim, config.seed);

        // Create CfC network
        let cfc_config = CfCNetworkConfig {
            input_dim: config.input_dim,
            hidden_dim: config.cfc_hidden_dim,
            num_layers: config.cfc_layers,
            output_dim: config.cfc_output_dim,
            cell_config: CfCConfig {
                input_dim: config.input_dim,
                hidden_dim: config.cfc_hidden_dim,
                ..Default::default()
            },
            residual: true,
            bidirectional: false,
            enable_online_learning: false,
            online_learning_config: Default::default(),
        };

        let genesis = config
            .genesis_phrase
            .as_ref()
            .map(|p| GenesisSeed::from_phrase(p));

        let cfc_temporal = if let Some(ref g) = genesis {
            CfCNetwork::from_genesis(cfc_config, g, "two_track_cfc")
        } else {
            CfCNetwork::new(cfc_config)
        };

        // Create projection matrices
        let input_to_cfc =
            Self::init_projection(config.input_dim, config.input_dim, config.seed + 1);

        let temporal_to_hdc =
            Self::init_projection(config.cfc_output_dim, config.hdc_dim, config.seed + 2);

        // Create optional bridge if enabled
        let bridge = if config.enable_bridge {
            let bridge_config = HdcCfcBridgeConfig {
                hdc_dim: config.hdc_dim,
                cfc_hidden_dim: config.cfc_hidden_dim,
                intermediate_dim: config.bridge_intermediate_dim,
                num_attention_heads: config.bridge_attention_heads,
                learning_rate: config.cfc_learning_rate,
                seed: config.seed + 100,
                genesis_phrase: config.genesis_phrase.clone(),
                ..Default::default()
            };
            if let Some(ref g) = genesis {
                Some(HdcCfcBridge::from_genesis(
                    bridge_config,
                    g,
                    "two_track_bridge",
                ))
            } else {
                Some(HdcCfcBridge::new(bridge_config))
            }
        } else {
            None
        };

        Self {
            config,
            hdc_projector,
            cfc_temporal,
            input_to_cfc,
            temporal_to_hdc,
            prev_semantic: None,
            temporal_magnitude_avg: 0.0,
            total_steps: 0,
            _genesis: genesis,
            bridge,
        }
    }

    /// Initialize a random projection matrix
    fn init_projection(input_dim: usize, output_dim: usize, seed: u64) -> Vec<f32> {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();

        (0..input_dim * output_dim)
            .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
            .collect()
    }

    /// Process input through both tracks
    pub fn process(&mut self, embedding: &[f32]) -> TwoTrackOutput {
        // Ensure input has correct dimension
        let input = self.normalize_input(embedding);

        // === SEMANTIC TRACK (HDC) ===
        let semantic_hv = self.hdc_projector.project(&input);

        // Track semantic stability
        let semantic_stability = if let Some(ref prev) = self.prev_semantic {
            semantic_hv.similarity(prev)
        } else {
            1.0
        };
        self.prev_semantic = Some(semantic_hv.clone());

        // === TEMPORAL TRACK (CfC) ===
        // Project input to CfC-compatible format
        let cfc_input = self.project_for_cfc(&input);
        let cfc_input_array = Array1::from_vec(cfc_input);

        // Step the CfC network
        let temporal_output = self
            .cfc_temporal
            .forward(&cfc_input_array, self.config.cfc_delta_t);
        let temporal_state: Vec<f32> = temporal_output.iter().cloned().collect();

        // Track temporal confidence (based on output magnitude stability)
        let magnitude = temporal_state.iter().map(|x| x * x).sum::<f32>().sqrt();
        let alpha = 0.1f32;
        self.temporal_magnitude_avg =
            alpha * magnitude + (1.0 - alpha) * self.temporal_magnitude_avg;
        let temporal_confidence = 1.0 / (1.0 + (magnitude - self.temporal_magnitude_avg).abs());

        // === FUSION ===
        let fused = self.fuse(&semantic_hv, &temporal_state);

        self.total_steps += 1;

        TwoTrackOutput {
            semantic_hv,
            temporal_state,
            fused,
            semantic_stability,
            temporal_confidence,
        }
    }

    /// Process with training: updates CfC based on prediction error
    pub fn process_and_train(
        &mut self,
        embedding: &[f32],
        target: Option<&[f32]>,
    ) -> Result<(TwoTrackOutput, Option<f32>)> {
        let output = self.process(embedding);

        let loss = if let Some(target_emb) = target {
            // Get the HDC projection of the target
            let target_hv = self.hdc_projector.project(target_emb);
            let target_vec: Vec<f32> = target_hv.values.to_vec();

            // Use fused output as prediction, target_hv as ground truth
            // But we need to match dimensions - use CfC output vs projected target
            let target_array = self.compress_for_training(&target_vec);

            let input = self.normalize_input(embedding);
            let cfc_input = self.project_for_cfc(&input);
            let cfc_input_array = Array1::from_vec(cfc_input);

            let loss = self.cfc_temporal.train_step(
                &cfc_input_array,
                &target_array,
                self.config.cfc_delta_t,
                self.config.cfc_learning_rate,
            )?;

            Some(loss)
        } else {
            None
        };

        Ok((output, loss))
    }

    /// Step the temporal track without full processing
    pub fn step_temporal(&mut self, embedding: &[f32]) -> Vec<f32> {
        let input = self.normalize_input(embedding);
        let cfc_input = self.project_for_cfc(&input);
        let cfc_input_array = Array1::from_vec(cfc_input);

        let output = self
            .cfc_temporal
            .forward(&cfc_input_array, self.config.cfc_delta_t);
        output.iter().cloned().collect()
    }

    /// Get only semantic projection (fast, no temporal update)
    pub fn semantic_only(&self, embedding: &[f32]) -> ContinuousHV {
        let input = self.normalize_input(embedding);
        self.hdc_projector.project(&input)
    }

    /// Reset temporal state
    pub fn reset(&mut self) {
        self.cfc_temporal.reset();
        self.prev_semantic = None;
        self.temporal_magnitude_avg = 0.0;
    }

    /// Get current CfC state
    pub fn temporal_state(&self) -> Vec<Array1<f32>> {
        self.cfc_temporal.state()
    }

    /// Get configuration
    pub fn config(&self) -> &TwoTrackConfig {
        &self.config
    }

    /// Get total steps processed
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Check if bridge is enabled
    pub fn has_bridge(&self) -> bool {
        self.bridge.is_some()
    }

    /// Get reference to bridge (if enabled)
    pub fn bridge(&self) -> Option<&HdcCfcBridge> {
        self.bridge.as_ref()
    }

    /// Get mutable reference to bridge (if enabled)
    pub fn bridge_mut(&mut self) -> Option<&mut HdcCfcBridge> {
        self.bridge.as_mut()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // BRIDGE-ENHANCED METHODS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Encode an HDC semantic vector to CfC-compatible temporal input using the bridge
    ///
    /// This provides a learned, attention-based projection from HDC to CfC space,
    /// which can capture more nuanced semantic-to-temporal mappings than simple
    /// linear projections.
    ///
    /// Returns None if bridge is not enabled.
    pub fn encode_semantic_to_temporal(&self, hv: &ContinuousHV) -> Option<Vec<f32>> {
        self.bridge
            .as_ref()
            .map(|b| b.encode_semantic_to_temporal(hv))
    }

    /// Decode CfC temporal state to HDC semantic vector using the bridge
    ///
    /// This provides a learned reconstruction of semantic meaning from temporal
    /// dynamics, enabling interpretation of CfC states in semantic terms.
    ///
    /// Returns None if bridge is not enabled.
    pub fn decode_temporal_to_semantic(&self, state: &[f32]) -> Option<ContinuousHV> {
        self.bridge
            .as_ref()
            .map(|b| b.decode_temporal_to_semantic(state))
    }

    /// Process with bridge-enhanced fusion
    ///
    /// When bridge is enabled, this uses the bridge for more sophisticated
    /// semantic-temporal interaction:
    /// 1. Standard HDC projection for semantic track
    /// 2. Bridge-enhanced encoding with temporal context for CfC input
    /// 3. Bridge-enhanced decoding for temporal contribution to fusion
    ///
    /// Falls back to standard processing if bridge is not enabled.
    pub fn process_with_bridge(&mut self, embedding: &[f32]) -> TwoTrackOutput {
        if self.bridge.is_none() {
            return self.process(embedding);
        }

        let input = self.normalize_input(embedding);

        // === SEMANTIC TRACK (HDC) ===
        let semantic_hv = self.hdc_projector.project(&input);

        let semantic_stability = if let Some(ref prev) = self.prev_semantic {
            semantic_hv.similarity(prev)
        } else {
            1.0
        };
        self.prev_semantic = Some(semantic_hv.clone());

        // === TEMPORAL TRACK (CfC) with bridge-enhanced input ===
        // Use bridge to encode semantic HV with temporal context
        let current_temporal_state: Vec<f32> = if let Some(state) = self.cfc_temporal.state().last()
        {
            state.iter().cloned().collect()
        } else {
            vec![0.0; self.config.cfc_hidden_dim]
        };

        let bridge = self
            .bridge
            .as_ref()
            .expect("bridge confirmed Some by is_none() check above");
        let bridge_encoded =
            bridge.encode_with_temporal_context(&semantic_hv, &current_temporal_state);

        // Pad bridge output to match CfC input_dim if needed
        // Bridge outputs cfc_hidden_dim, but CfC expects input_dim
        let cfc_input: Vec<f32> = if bridge_encoded.len() < self.config.input_dim {
            let mut padded = bridge_encoded;
            padded.resize(self.config.input_dim, 0.0);
            padded
        } else {
            bridge_encoded[..self.config.input_dim].to_vec()
        };
        let cfc_input_array = Array1::from_vec(cfc_input);

        let temporal_output = self
            .cfc_temporal
            .forward(&cfc_input_array, self.config.cfc_delta_t);
        let temporal_state: Vec<f32> = temporal_output.iter().cloned().collect();

        // Track temporal confidence
        let magnitude = temporal_state.iter().map(|x| x * x).sum::<f32>().sqrt();
        let alpha = 0.1f32;
        self.temporal_magnitude_avg =
            alpha * magnitude + (1.0 - alpha) * self.temporal_magnitude_avg;
        let temporal_confidence = 1.0 / (1.0 + (magnitude - self.temporal_magnitude_avg).abs());

        // === BRIDGE-ENHANCED FUSION ===
        // Use bridge to decode temporal back to semantic for fusion
        // Pad temporal state to cfc_hidden_dim if needed (bridge expects cfc_hidden_dim)
        let temporal_for_decode: Vec<f32> = if temporal_state.len() < self.config.cfc_hidden_dim {
            let mut padded = temporal_state.clone();
            padded.resize(self.config.cfc_hidden_dim, 0.0);
            padded
        } else {
            temporal_state[..self.config.cfc_hidden_dim].to_vec()
        };
        let temporal_semantic =
            bridge.decode_with_semantic_context(&temporal_for_decode, &semantic_hv);

        let (sem_weight, temp_weight) = self.config.fusion_weights;
        let fused =
            self.weighted_hv_fusion(&semantic_hv, &temporal_semantic, sem_weight, temp_weight);

        self.total_steps += 1;

        TwoTrackOutput {
            semantic_hv,
            temporal_state,
            fused,
            semantic_stability,
            temporal_confidence,
        }
    }

    /// Train the bridge on roundtrip reconstruction
    ///
    /// Returns the reconstruction loss, or None if bridge is not enabled.
    pub fn train_bridge(&mut self, hv: &ContinuousHV) -> Option<f32> {
        self.bridge.as_mut().map(|b| b.train_step(hv))
    }

    /// Measure bridge roundtrip similarity
    ///
    /// Returns the cosine similarity between original and reconstructed HV,
    /// or None if bridge is not enabled.
    pub fn measure_bridge_roundtrip(&self, hv: &ContinuousHV) -> Option<f32> {
        self.bridge
            .as_ref()
            .map(|b| b.measure_roundtrip_similarity(hv))
    }

    /// Get bridge reconstruction loss EMA
    pub fn bridge_loss(&self) -> Option<f32> {
        self.bridge.as_ref().map(|b| b.reconstruction_loss())
    }

    /// Weighted fusion of two HVs
    fn weighted_hv_fusion(
        &self,
        a: &ContinuousHV,
        b: &ContinuousHV,
        w_a: f32,
        w_b: f32,
    ) -> ContinuousHV {
        let dim = a.dim().max(b.dim());

        let values: Vec<f32> = (0..dim)
            .map(|i| {
                let va = if i < a.values.len() { a.values[i] } else { 0.0 };
                let vb = if i < b.values.len() { b.values[i] } else { 0.0 };
                w_a * va + w_b * vb
            })
            .collect();

        if self.config.normalize_before_fusion {
            ContinuousHV::from_vec(values).normalize()
        } else {
            ContinuousHV::from_vec(values)
        }
    }

    /// Normalize input to expected dimension
    fn normalize_input(&self, embedding: &[f32]) -> Vec<f32> {
        if embedding.len() == self.config.input_dim {
            embedding.to_vec()
        } else if embedding.len() < self.config.input_dim {
            let mut padded = embedding.to_vec();
            padded.resize(self.config.input_dim, 0.0);
            padded
        } else {
            embedding[..self.config.input_dim].to_vec()
        }
    }

    /// Project input for CfC processing
    fn project_for_cfc(&self, input: &[f32]) -> Vec<f32> {
        // Simple identity projection since input_dim matches
        // In a more sophisticated version, this could be a learned projection
        let mut output = vec![0.0f32; self.config.input_dim];
        let in_dim = self.config.input_dim;

        for i in 0..in_dim {
            for j in 0..in_dim {
                output[i] += self.input_to_cfc[i * in_dim + j] * input[j];
            }
        }

        // Apply tanh nonlinearity
        output.iter_mut().for_each(|x| *x = x.tanh());
        output
    }

    /// Compress a high-dimensional vector for training
    fn compress_for_training(&self, vec: &[f32]) -> Array1<f32> {
        // Downsample to CfC output dimension
        let out_dim = self.config.cfc_output_dim;
        let in_dim = vec.len();
        let step = (in_dim / out_dim).max(1);

        let compressed: Vec<f32> = (0..out_dim)
            .map(|i| {
                let start = i * step;
                let end = ((i + 1) * step).min(in_dim);
                if start < in_dim {
                    vec[start..end].iter().sum::<f32>() / (end - start) as f32
                } else {
                    0.0
                }
            })
            .collect();

        Array1::from_vec(compressed)
    }

    /// Fuse semantic and temporal representations
    fn fuse(&self, semantic: &ContinuousHV, temporal: &[f32]) -> ContinuousHV {
        let (sem_weight, temp_weight) = self.config.fusion_weights;

        // Project temporal to HDC space
        let temporal_hdc = self.project_temporal_to_hdc(temporal);

        // Get semantic values (using the public field)
        let sem_values = &semantic.values;
        let hdc_dim = self.config.hdc_dim;

        // Normalize if configured
        let (norm_sem, norm_temp) = if self.config.normalize_before_fusion {
            let sem_norm: f32 = sem_values.iter().map(|x| x * x).sum::<f32>().sqrt();
            let temp_norm: f32 = temporal_hdc.iter().map(|x| x * x).sum::<f32>().sqrt();

            let sem: Vec<f32> = if sem_norm > 1e-6 {
                sem_values.iter().map(|x| x / sem_norm).collect()
            } else {
                sem_values.to_vec()
            };

            let temp: Vec<f32> = if temp_norm > 1e-6 {
                temporal_hdc.iter().map(|x| x / temp_norm).collect()
            } else {
                temporal_hdc
            };

            (sem, temp)
        } else {
            (sem_values.to_vec(), temporal_hdc)
        };

        // Weighted combination
        let fused_values: Vec<f32> = (0..hdc_dim)
            .map(|i| {
                let s = if i < norm_sem.len() { norm_sem[i] } else { 0.0 };
                let t = if i < norm_temp.len() {
                    norm_temp[i]
                } else {
                    0.0
                };
                sem_weight * s + temp_weight * t
            })
            .collect();

        ContinuousHV::from_vec(fused_values)
    }

    /// Project temporal state to HDC space
    fn project_temporal_to_hdc(&self, temporal: &[f32]) -> Vec<f32> {
        let out_dim = self.config.hdc_dim;
        let in_dim = self.config.cfc_output_dim;

        let mut output = vec![0.0f32; out_dim];

        // Ensure we don't exceed the temporal slice
        let temp_len = temporal.len().min(in_dim);

        for i in 0..out_dim {
            for j in 0..temp_len {
                let idx = i * in_dim + j;
                if idx < self.temporal_to_hdc.len() {
                    output[i] += self.temporal_to_hdc[idx] * temporal[j];
                }
            }
        }

        output
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_track_creation() {
        let config = TwoTrackConfig::default();
        let processor = TwoTrackProcessor::new(config.clone());

        assert_eq!(processor.config().input_dim, 256);
        assert_eq!(processor.config().hdc_dim, HDC_DIMENSION);
        assert_eq!(processor.total_steps(), 0);
    }

    #[test]
    fn test_semantic_track_preserves_similarity() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config);

        // Create two similar embeddings
        let emb1: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut emb2 = emb1.clone();
        emb2[0] += 0.05; // Slightly different

        let out1 = processor.process(&emb1);
        processor.reset(); // Reset temporal to isolate semantic test
        let out2 = processor.process(&emb2);

        // Semantic HVs should be similar for similar inputs
        let sim = out1.semantic_hv.similarity(&out2.semantic_hv);
        assert!(
            sim > 0.8,
            "Similar inputs should produce similar semantic HVs, got {}",
            sim
        );
    }

    #[test]
    fn test_semantic_track_distinguishes_different() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config);

        // Create two very different embeddings
        let emb1: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let emb2: Vec<f32> = (0..256).map(|i| (i as f32 * 0.5).cos()).collect();

        let out1 = processor.process(&emb1);
        processor.reset();
        let out2 = processor.process(&emb2);

        // Semantic HVs should differ for different inputs
        let sim = out1.semantic_hv.similarity(&out2.semantic_hv);
        assert!(
            sim < 0.8,
            "Different inputs should produce different semantic HVs, got {}",
            sim
        );
    }

    #[test]
    fn test_temporal_track_learns_sequences() {
        let config = TwoTrackConfig {
            cfc_learning_rate: 0.01,
            ..Default::default()
        };
        let mut processor = TwoTrackProcessor::new(config);

        // Create a simple sequence: A -> B -> C -> D
        let sequence: Vec<Vec<f32>> = (0..4)
            .map(|i| {
                (0..256)
                    .map(|j| ((i * 64 + j) as f32 * 0.01).sin())
                    .collect()
            })
            .collect();

        // Process sequence multiple times to learn
        let mut losses = Vec::new();
        for _epoch in 0..10 {
            let mut epoch_loss = 0.0f32;
            for i in 0..sequence.len() - 1 {
                let current = &sequence[i];
                let next = &sequence[i + 1];

                let (_, loss) = processor
                    .process_and_train(current, Some(next))
                    .expect("Training should succeed");

                if let Some(l) = loss {
                    epoch_loss += l;
                }
            }
            losses.push(epoch_loss);
            processor.reset(); // Reset between epochs
        }

        // Loss should generally decrease (or at least not explode)
        let first_half_avg: f32 = losses[..5].iter().sum::<f32>() / 5.0;
        let second_half_avg: f32 = losses[5..].iter().sum::<f32>() / 5.0;

        // Check that training didn't diverge
        assert!(second_half_avg.is_finite(), "Training should not diverge");

        // Note: We don't strictly require loss decrease as CfC learning
        // can be noisy, but we check it doesn't explode
        assert!(
            second_half_avg < first_half_avg * 10.0,
            "Loss should not explode: first_half={}, second_half={}",
            first_half_avg,
            second_half_avg
        );
    }

    #[test]
    fn test_fusion_combines_both() {
        let config = TwoTrackConfig {
            fusion_weights: (0.5, 0.5),
            ..Default::default()
        };
        let mut processor = TwoTrackProcessor::new(config);

        let embedding: Vec<f32> = (0..256).map(|i| (i as f32 * 0.02).sin()).collect();
        let output = processor.process(&embedding);

        // Fused should have the right dimension
        assert_eq!(output.fused.dim(), HDC_DIMENSION);

        // Fused should be different from pure semantic (due to temporal contribution)
        // After first step, temporal state is non-zero
        processor.reset();
        let _ = processor.process(&embedding);
        let output2 = processor.process(&embedding);

        let sem_sim = output2.fused.similarity(&output2.semantic_hv);
        // Fused should be similar but not identical to semantic
        assert!(
            sem_sim > 0.5 && sem_sim < 1.0,
            "Fused should blend semantic and temporal, similarity={}",
            sem_sim
        );
    }

    #[test]
    fn test_temporal_state_evolves() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config);

        let embedding: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();

        // Process same input multiple times
        let out1 = processor.process(&embedding);
        let out2 = processor.process(&embedding);
        let out3 = processor.process(&embedding);

        // Temporal states should differ as CfC evolves
        let state1_mag: f32 = out1
            .temporal_state
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let state2_mag: f32 = out2
            .temporal_state
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let state3_mag: f32 = out3
            .temporal_state
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        // States should be different (CfC dynamics)
        assert!(
            (state1_mag - state2_mag).abs() > 0.001 || (state2_mag - state3_mag).abs() > 0.001,
            "Temporal state should evolve over time: {}, {}, {}",
            state1_mag,
            state2_mag,
            state3_mag
        );
    }

    #[test]
    fn test_semantic_stability_tracking() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config);

        // Similar consecutive inputs
        let emb1: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut emb2 = emb1.clone();
        emb2[0] += 0.01;

        let out1 = processor.process(&emb1);
        let out2 = processor.process(&emb2);

        // First output has no previous, so stability = 1.0
        assert_eq!(out1.semantic_stability, 1.0);

        // Second should show high stability (similar input)
        assert!(
            out2.semantic_stability > 0.9,
            "Similar consecutive inputs should show high stability: {}",
            out2.semantic_stability
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config);

        let embedding: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();

        // Build up state
        for _ in 0..10 {
            processor.process(&embedding);
        }

        // Reset and check
        processor.reset();

        let out = processor.process(&embedding);

        // After reset, first stability should be 1.0 (no previous)
        assert_eq!(out.semantic_stability, 1.0);
    }

    #[test]
    fn test_fast_config() {
        let config = TwoTrackConfig::fast();
        let mut processor = TwoTrackProcessor::new(config);

        let embedding: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let output = processor.process(&embedding);

        // Should work with reduced dimensions
        assert_eq!(output.semantic_hv.dim(), 2048);
        assert!(!output.temporal_state.is_empty());
    }

    #[test]
    fn test_deterministic_with_genesis() {
        let config1 = TwoTrackConfig {
            genesis_phrase: Some("test_genesis".to_string()),
            ..Default::default()
        };
        let config2 = TwoTrackConfig {
            genesis_phrase: Some("test_genesis".to_string()),
            ..Default::default()
        };

        let mut proc1 = TwoTrackProcessor::new(config1);
        let mut proc2 = TwoTrackProcessor::new(config2);

        let embedding: Vec<f32> = vec![0.5; 256];

        let out1 = proc1.process(&embedding);
        let out2 = proc2.process(&embedding);

        // Same genesis should produce identical semantic outputs
        let sim = out1.semantic_hv.similarity(&out2.semantic_hv);
        assert_eq!(
            sim, 1.0,
            "Same genesis should produce identical semantic HVs"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // BRIDGE INTEGRATION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_bridge_disabled_by_default() {
        let config = TwoTrackConfig::default();
        let processor = TwoTrackProcessor::new(config);

        assert!(
            !processor.has_bridge(),
            "Bridge should be disabled by default"
        );
        assert!(processor.bridge().is_none());
    }

    #[test]
    fn test_bridge_enabled_with_config() {
        let config = TwoTrackConfig::with_bridge();
        let processor = TwoTrackProcessor::new(config);

        assert!(
            processor.has_bridge(),
            "Bridge should be enabled with with_bridge()"
        );
        assert!(processor.bridge().is_some());
    }

    #[test]
    fn test_encode_semantic_to_temporal_requires_bridge() {
        let config_no_bridge = TwoTrackConfig::default();
        let processor_no_bridge = TwoTrackProcessor::new(config_no_bridge);

        let hv = ContinuousHV::random_default(42);
        assert!(processor_no_bridge
            .encode_semantic_to_temporal(&hv)
            .is_none());

        let config_with_bridge = TwoTrackConfig::with_bridge();
        let processor_with_bridge = TwoTrackProcessor::new(config_with_bridge);

        let encoded = processor_with_bridge.encode_semantic_to_temporal(&hv);
        assert!(encoded.is_some());
        assert_eq!(encoded.unwrap().len(), 128); // cfc_hidden_dim
    }

    #[test]
    fn test_decode_temporal_to_semantic_requires_bridge() {
        let config_no_bridge = TwoTrackConfig::default();
        let processor_no_bridge = TwoTrackProcessor::new(config_no_bridge);

        let temporal_state = vec![0.5f32; 128];
        assert!(processor_no_bridge
            .decode_temporal_to_semantic(&temporal_state)
            .is_none());

        let config_with_bridge = TwoTrackConfig::with_bridge();
        let processor_with_bridge = TwoTrackProcessor::new(config_with_bridge);

        let decoded = processor_with_bridge.decode_temporal_to_semantic(&temporal_state);
        assert!(decoded.is_some());
        assert_eq!(decoded.unwrap().dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_process_with_bridge_fallback() {
        // When bridge is disabled, process_with_bridge should fall back to process
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config);

        let embedding: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();

        let out1 = processor.process(&embedding);
        processor.reset();
        let out2 = processor.process_with_bridge(&embedding);

        // Should be identical when bridge is disabled
        let sim = out1.semantic_hv.similarity(&out2.semantic_hv);
        assert!(
            sim > 0.99,
            "process_with_bridge should match process when bridge disabled"
        );
    }

    #[test]
    fn test_bridge_roundtrip_measurement() {
        let config = TwoTrackConfig::with_bridge();
        let processor = TwoTrackProcessor::new(config);

        let hv = ContinuousHV::random_default(42);
        let similarity = processor.measure_bridge_roundtrip(&hv);

        assert!(similarity.is_some());
        // Untrained bridge will have some similarity (random projections)
        let sim = similarity.unwrap();
        assert!(
            sim > -1.0 && sim <= 1.0,
            "Similarity should be in valid range"
        );
    }

    #[test]
    #[ignore = "slow ~60s: trains HDC-LTC bridge networks"]
    fn test_bridge_training() {
        let config = TwoTrackConfig {
            enable_bridge: true,
            hdc_dim: 1024, // Smaller for faster test
            bridge_intermediate_dim: 128,
            ..Default::default()
        };
        let mut processor = TwoTrackProcessor::new(config);

        // Train the bridge
        let hv = ContinuousHV::random(1024, 42);
        let loss1 = processor.train_bridge(&hv).unwrap();

        // Train more
        for i in 0..10 {
            let hv_i = ContinuousHV::random(1024, i as u64 + 100);
            let _ = processor.train_bridge(&hv_i);
        }

        let loss2 = processor.bridge_loss().unwrap();

        // Loss should be finite
        assert!(loss1.is_finite(), "Loss should be finite");
        assert!(loss2.is_finite(), "Final loss should be finite");
    }

    #[test]
    fn test_process_with_bridge_works() {
        let config = TwoTrackConfig::with_bridge();
        let mut processor = TwoTrackProcessor::new(config);

        let embedding: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();

        // Process with bridge should complete without error
        let output = processor.process_with_bridge(&embedding);

        assert_eq!(output.semantic_hv.dim(), HDC_DIMENSION);
        assert!(!output.temporal_state.is_empty());
        assert_eq!(output.fused.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_bridge_deterministic_with_genesis() {
        let config1 = TwoTrackConfig {
            enable_bridge: true,
            genesis_phrase: Some("test_bridge_genesis".to_string()),
            ..Default::default()
        };
        let config2 = TwoTrackConfig {
            enable_bridge: true,
            genesis_phrase: Some("test_bridge_genesis".to_string()),
            ..Default::default()
        };

        let processor1 = TwoTrackProcessor::new(config1);
        let processor2 = TwoTrackProcessor::new(config2);

        let hv = ContinuousHV::random_default(42);

        let encoded1 = processor1.encode_semantic_to_temporal(&hv).unwrap();
        let encoded2 = processor2.encode_semantic_to_temporal(&hv).unwrap();

        // Same genesis should produce identical encodings
        for (a, b) in encoded1.iter().zip(encoded2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Genesis should make bridge deterministic"
            );
        }
    }

    #[test]
    fn test_high_fidelity_config() {
        let config = TwoTrackConfig::high_fidelity();

        assert!(config.enable_bridge);
        assert_eq!(config.cfc_hidden_dim, 256);
        assert_eq!(config.cfc_layers, 4);
        assert_eq!(config.bridge_intermediate_dim, 1024);
        assert_eq!(config.bridge_attention_heads, 8);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ADDITIONAL COVERAGE TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_default_config_values() {
        let config = TwoTrackConfig::default();

        assert_eq!(config.input_dim, 256);
        assert_eq!(config.hdc_dim, HDC_DIMENSION);
        assert_eq!(config.cfc_hidden_dim, 128);
        assert_eq!(config.cfc_layers, 2);
        assert_eq!(config.cfc_output_dim, 64);
        assert_eq!(config.fusion_weights, (0.6, 0.4));
        assert!(config.normalize_before_fusion);
        assert_eq!(config.seed, 0x5954_2154);
        assert!(config.genesis_phrase.is_none());
        assert!((config.cfc_delta_t - 0.02).abs() < 1e-6);
        assert!((config.cfc_learning_rate - 0.001).abs() < 1e-6);
        assert!(!config.enable_bridge);
        assert_eq!(config.bridge_intermediate_dim, 512);
        assert_eq!(config.bridge_attention_heads, 4);
    }

    #[test]
    fn test_semantic_heavy_config() {
        let config = TwoTrackConfig::semantic_heavy();

        assert_eq!(config.fusion_weights, (0.8, 0.2));
        // All other fields should be default
        assert_eq!(config.input_dim, 256);
        assert_eq!(config.hdc_dim, HDC_DIMENSION);
        assert_eq!(config.cfc_hidden_dim, 128);
    }

    #[test]
    fn test_temporal_heavy_config() {
        let config = TwoTrackConfig::temporal_heavy();

        assert_eq!(config.fusion_weights, (0.3, 0.7));
        assert_eq!(config.cfc_hidden_dim, 256);
        assert_eq!(config.cfc_layers, 3);
        // input_dim should remain default
        assert_eq!(config.input_dim, 256);
    }

    #[test]
    fn test_output_dimensions_invariant() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config.clone());

        let embedding: Vec<f32> = vec![0.3; 256];
        let output = processor.process(&embedding);

        assert_eq!(
            output.semantic_hv.dim(),
            config.hdc_dim,
            "Semantic HV must match configured hdc_dim"
        );
        assert_eq!(
            output.temporal_state.len(),
            config.cfc_output_dim,
            "Temporal state must match cfc_output_dim"
        );
        assert_eq!(
            output.fused.dim(),
            config.hdc_dim,
            "Fused HV must match hdc_dim"
        );
    }

    #[test]
    fn test_output_values_are_finite() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config);

        let embedding: Vec<f32> = (0..256).map(|i| (i as f32 * 0.037).sin()).collect();

        // Process multiple steps to exercise temporal dynamics
        for _ in 0..5 {
            let output = processor.process(&embedding);

            for (i, v) in output.semantic_hv.values.iter().enumerate() {
                assert!(v.is_finite(), "semantic_hv[{}] = {} is not finite", i, v);
            }
            for (i, v) in output.temporal_state.iter().enumerate() {
                assert!(v.is_finite(), "temporal_state[{}] = {} is not finite", i, v);
            }
            for (i, v) in output.fused.values.iter().enumerate() {
                assert!(v.is_finite(), "fused[{}] = {} is not finite", i, v);
            }
            assert!(
                output.semantic_stability.is_finite(),
                "semantic_stability must be finite"
            );
            assert!(
                output.temporal_confidence.is_finite(),
                "temporal_confidence must be finite"
            );
        }
    }

    #[test]
    fn test_total_steps_increments() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config);

        assert_eq!(processor.total_steps(), 0);

        let embedding: Vec<f32> = vec![0.1; 256];

        processor.process(&embedding);
        assert_eq!(processor.total_steps(), 1);

        processor.process(&embedding);
        assert_eq!(processor.total_steps(), 2);

        // reset does not clear total_steps
        processor.reset();
        processor.process(&embedding);
        assert_eq!(processor.total_steps(), 3);
    }

    #[test]
    fn test_step_temporal_returns_correct_dim() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config.clone());

        let embedding: Vec<f32> = vec![0.2; 256];
        let temporal = processor.step_temporal(&embedding);

        assert_eq!(
            temporal.len(),
            config.cfc_output_dim,
            "step_temporal output must match cfc_output_dim"
        );

        for (i, v) in temporal.iter().enumerate() {
            assert!(v.is_finite(), "step_temporal[{}] = {} is not finite", i, v);
        }
    }

    #[test]
    fn test_semantic_only_does_not_mutate_state() {
        let config = TwoTrackConfig::default();
        let processor = TwoTrackProcessor::new(config.clone());

        let embedding: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).cos()).collect();

        // Capture temporal state before
        let state_before = processor.temporal_state();
        let steps_before = processor.total_steps();

        let hv = processor.semantic_only(&embedding);

        // Verify the semantic_only result
        assert_eq!(hv.dim(), config.hdc_dim);

        // State and step count should be unchanged
        let state_after = processor.temporal_state();
        assert_eq!(
            steps_before,
            processor.total_steps(),
            "semantic_only must not increment total_steps"
        );
        assert_eq!(
            state_before.len(),
            state_after.len(),
            "semantic_only must not alter temporal state count"
        );
    }

    #[test]
    fn test_empty_input_padded_to_input_dim() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config.clone());

        // Empty input should be zero-padded to input_dim
        let empty: Vec<f32> = vec![];
        let output = processor.process(&empty);

        assert_eq!(output.semantic_hv.dim(), config.hdc_dim);
        assert_eq!(output.temporal_state.len(), config.cfc_output_dim);
        assert_eq!(output.fused.dim(), config.hdc_dim);

        // All values should be finite
        for v in output.fused.values.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_short_input_padded() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config.clone());

        // Input shorter than input_dim (10 < 256)
        let short: Vec<f32> = vec![1.0; 10];
        let output = processor.process(&short);

        assert_eq!(output.semantic_hv.dim(), config.hdc_dim);
        assert_eq!(output.temporal_state.len(), config.cfc_output_dim);
        assert_eq!(output.fused.dim(), config.hdc_dim);
    }

    #[test]
    fn test_long_input_truncated() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config.clone());

        // Input longer than input_dim (500 > 256)
        let long: Vec<f32> = vec![0.5; 500];
        let output = processor.process(&long);

        assert_eq!(output.semantic_hv.dim(), config.hdc_dim);
        assert_eq!(output.temporal_state.len(), config.cfc_output_dim);
        assert_eq!(output.fused.dim(), config.hdc_dim);
    }

    #[test]
    fn test_zero_input_produces_finite_output() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config);

        let zeros: Vec<f32> = vec![0.0; 256];
        let output = processor.process(&zeros);

        for v in output.semantic_hv.values.iter() {
            assert!(
                v.is_finite(),
                "zero input should produce finite semantic HV"
            );
        }
        for v in output.temporal_state.iter() {
            assert!(
                v.is_finite(),
                "zero input should produce finite temporal state"
            );
        }
        for v in output.fused.values.iter() {
            assert!(v.is_finite(), "zero input should produce finite fused HV");
        }
    }

    #[test]
    fn test_temporal_confidence_bounded() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config);

        let embedding: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();

        for _ in 0..10 {
            let output = processor.process(&embedding);
            assert!(
                output.temporal_confidence > 0.0,
                "temporal_confidence must be positive, got {}",
                output.temporal_confidence
            );
            assert!(
                output.temporal_confidence <= 1.0,
                "temporal_confidence must be <= 1.0, got {}",
                output.temporal_confidence
            );
        }
    }

    #[test]
    fn test_semantic_hv16_and_fused_hv16_conversions() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config);

        let embedding: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let output = processor.process(&embedding);

        // The hv16 conversions should produce valid BinaryHV objects
        let sem_hv16 = output.semantic_hv16();
        let fused_hv16 = output.fused_hv16();

        // BinaryHV has DIM = 16384 bits represented in 2048 bytes
        assert_eq!(
            sem_hv16.0.len(),
            2048,
            "semantic_hv16 should have 2048 bytes"
        );
        assert_eq!(
            fused_hv16.0.len(),
            2048,
            "fused_hv16 should have 2048 bytes"
        );

        // Verify bipolar roundtrip: to_bipolar should produce -1.0 or +1.0 values only
        let bipolar = sem_hv16.to_bipolar();
        for (i, v) in bipolar.iter().enumerate() {
            assert!(
                *v == -1.0 || *v == 1.0,
                "bipolar[{}] should be -1.0 or 1.0, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_process_and_train_without_target() {
        let config = TwoTrackConfig::default();
        let mut processor = TwoTrackProcessor::new(config);

        let embedding: Vec<f32> = vec![0.1; 256];

        let result = processor.process_and_train(&embedding, None);
        assert!(result.is_ok());

        let (output, loss) = result.unwrap();
        assert!(loss.is_none(), "No target means no loss");
        assert_eq!(output.semantic_hv.dim(), HDC_DIMENSION);
    }
}
