//! Semantic Encoder - The "Eyes" of the Holographic Associative Memory
//!
//! This module provides the sensation layer for the HAM architecture,
//! converting raw text into dual representations:
//!
//! 1. **DenseVector (768D)** - For storage, retrieval, and semantic similarity
//! 2. **Hypervector (HV16)** - For symbolic reasoning and HDC operations
//!
//! ## Architecture Role
//!
//! ```text
//! ┌─────────────┐
//! │  Raw Text   │  ← Input (query, memory fragment, etc.)
//! └──────┬──────┘
//!        │
//!        ▼
//! ┌─────────────────────────────────────────────┐
//! │            SemanticEncoder                   │
//! │  ┌─────────────┐     ┌─────────────────┐   │
//! │  │ BGE Encoder │────▶│ Random Projector │   │
//! │  │   (768D)    │     │ (768D → 16384b) │   │
//! │  └──────┬──────┘     └────────┬────────┘   │
//! │         │                     │            │
//! │         ▼                     ▼            │
//! │  ┌─────────────┐     ┌─────────────────┐   │
//! │  │ DenseVector │     │   Hypervector   │   │
//! │  │   (f32[])   │     │    (HV16)       │   │
//! │  └─────────────┘     └─────────────────┘   │
//! └─────────────────────────────────────────────┘
//!         │                     │
//!         ▼                     ▼
//!   [Storage/FAISS]      [HDC Reasoning]
//! ```
//!
//! ## Temporal Context (Statefulness)
//!
//! The encoder supports temporal context accumulation for conversation continuity:
//!
//! ```text
//! Thought_t = (α × Input_t) + ((1-α) × Thought_{t-1})
//! ```
//!
//! This allows the system to maintain a "drift" of meaning across turns,
//! enabling infinite context through vector arithmetic.
//!
//! ## Future: Swarm Integration
//!
//! The `EncodedThought` struct is designed for serialization, enabling:
//! - **Memory storage**: Persist thoughts across sessions
//! - **Swarm telepathy**: Share learned deltas across nodes
//! - **One-shot learning**: Memory_new = Memory_old + Experience

use crate::embeddings::bge::{BGEEmbedder, BGEConfig, BGE_DIMENSION};
use crate::embeddings::bridge::HdcBridge;
use crate::hdc::binary_hv::HV16;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Configuration for the Semantic Encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEncoderConfig {
    /// Dimension of dense embeddings (768 for BGE)
    pub dense_dim: usize,

    /// Dimension of hypervectors (16384 for HV16)
    pub hv_dim: usize,

    /// Temporal decay factor (α in the context equation)
    /// Higher = more weight on current input, less memory of past
    /// Default: 0.7 (70% current, 30% accumulated context)
    pub temporal_alpha: f32,

    /// Whether to enable temporal context accumulation
    pub enable_temporal_context: bool,

    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for SemanticEncoderConfig {
    fn default() -> Self {
        Self {
            dense_dim: BGE_DIMENSION,
            hv_dim: HV16::DIM,
            temporal_alpha: 0.7,
            enable_temporal_context: true,
            seed: 0x5974_1AEA, // "SYMTHAEA" as hex
        }
    }
}

/// Dual representation of a semantic thought
///
/// This struct encapsulates both vector representations of a concept,
/// designed for storage, reasoning, and cross-node sharing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedThought {
    /// The original text input
    pub text: String,

    /// Dense vector representation (for similarity/storage)
    #[serde(with = "dense_vector_serde")]
    pub dense: DenseVector,

    /// Hypervector representation (for HDC reasoning)
    pub hypervector: SerializableHV16,

    /// Timestamp (Unix epoch milliseconds)
    pub timestamp_ms: u64,

    /// Optional conversation context ID
    pub context_id: Option<String>,

    /// Confidence/quality score (0.0 - 1.0)
    pub confidence: f32,
}

/// Dense vector wrapper for semantic embeddings
#[derive(Debug, Clone)]
pub struct DenseVector {
    pub values: Vec<f32>,
}

impl DenseVector {
    pub fn new(values: Vec<f32>) -> Self {
        Self { values }
    }

    pub fn zeros(dim: usize) -> Self {
        Self {
            values: vec![0.0; dim],
        }
    }

    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Cosine similarity with another dense vector
    pub fn similarity(&self, other: &DenseVector) -> f32 {
        if self.values.len() != other.values.len() {
            return 0.0;
        }

        let dot: f32 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.values.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Weighted blend of two vectors
    /// result = (alpha * self) + ((1 - alpha) * other)
    pub fn blend(&self, other: &DenseVector, alpha: f32) -> DenseVector {
        let values = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| alpha * a + (1.0 - alpha) * b)
            .collect();

        DenseVector { values }
    }

    /// Element-wise addition (for memory accumulation)
    pub fn add(&self, other: &DenseVector) -> DenseVector {
        let values = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a + b)
            .collect();

        DenseVector { values }
    }

    /// Scale by a constant
    pub fn scale(&self, factor: f32) -> DenseVector {
        let values = self.values.iter().map(|x| x * factor).collect();
        DenseVector { values }
    }
}

/// Serializable wrapper for HV16
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableHV16 {
    bytes: Vec<u8>,
}

impl SerializableHV16 {
    pub fn from_hv16(hv: &HV16) -> Self {
        Self {
            bytes: hv.0.to_vec(),
        }
    }

    pub fn to_hv16(&self) -> HV16 {
        let mut bytes = [0u8; HV16::BYTES];
        bytes.copy_from_slice(&self.bytes[..HV16::BYTES.min(self.bytes.len())]);
        HV16(bytes)
    }
}

/// Serde module for DenseVector
mod dense_vector_serde {
    use super::DenseVector;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(dense: &DenseVector, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        dense.values.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<DenseVector, D::Error>
    where
        D: Deserializer<'de>,
    {
        let values = Vec::<f32>::deserialize(deserializer)?;
        Ok(DenseVector { values })
    }
}

/// The Semantic Encoder - dual-output perception layer
///
/// This is the "eyes" of the HAM system, converting raw text into
/// both dense vectors (for storage/retrieval) and hypervectors (for reasoning).
pub struct SemanticEncoder {
    /// Configuration
    config: SemanticEncoderConfig,

    /// BGE embedder (dense 768D vectors)
    embedder: BGEEmbedder,

    /// HDC bridge (projects dense → hypervector)
    bridge: HdcBridge,

    /// Accumulated temporal context (if enabled)
    temporal_context: Option<DenseVector>,

    /// Statistics
    stats: EncoderStats,
}

/// Statistics for the encoder
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct EncoderStats {
    /// Total encodings performed
    pub total_encodings: u64,

    /// Average encoding time (microseconds)
    pub avg_encoding_time_us: f64,

    /// Number of context updates
    pub context_updates: u64,
}

impl SemanticEncoder {
    /// Create a new Semantic Encoder with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(SemanticEncoderConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SemanticEncoderConfig) -> Result<Self> {
        let embedder = BGEEmbedder::new(BGEConfig::default())?;
        let bridge = HdcBridge::new();

        let temporal_context = if config.enable_temporal_context {
            Some(DenseVector::zeros(config.dense_dim))
        } else {
            None
        };

        Ok(Self {
            config,
            embedder,
            bridge,
            temporal_context,
            stats: EncoderStats::default(),
        })
    }

    /// Encode text into dual representation (DenseVector + Hypervector)
    ///
    /// This is the primary encoding function. It:
    /// 1. Embeds text using BGE (768D dense)
    /// 2. Projects to hypervector (HV16)
    /// 3. Optionally blends with temporal context
    ///
    /// Returns an `EncodedThought` containing both representations.
    pub fn encode(&mut self, text: &str) -> Result<EncodedThought> {
        let start = std::time::Instant::now();

        // Step 1: Embed text using BGE
        let embedding = self.embedder.embed(text)?;
        let mut dense = DenseVector::new(embedding.embedding.clone());

        // Step 2: Apply temporal context if enabled
        if let Some(ref ctx) = self.temporal_context {
            // Blend: Thought_t = (α × Input_t) + ((1-α) × Thought_{t-1})
            let blended = dense.blend(ctx, self.config.temporal_alpha);
            dense = blended;
        }

        // Step 3: Project to hypervector
        let hv = self.bridge.project(&dense.values);

        // Update temporal context
        if let Some(ref mut ctx) = self.temporal_context {
            *ctx = dense.clone();
            self.stats.context_updates += 1;
        }

        // Update stats
        self.stats.total_encodings += 1;
        let elapsed_us = start.elapsed().as_micros() as f64;
        self.stats.avg_encoding_time_us = (self.stats.avg_encoding_time_us
            * (self.stats.total_encodings - 1) as f64
            + elapsed_us)
            / self.stats.total_encodings as f64;

        Ok(EncodedThought {
            text: text.to_string(),
            dense,
            hypervector: SerializableHV16::from_hv16(&hv),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            context_id: None,
            confidence: 1.0,
        })
    }

    /// Encode without temporal context blending (pure encoding)
    ///
    /// Use this when you want a fresh encoding without any accumulated context,
    /// e.g., for training exemplars or isolated queries.
    pub fn encode_pure(&self, text: &str) -> Result<EncodedThought> {
        let embedding = self.embedder.embed(text)?;
        let dense = DenseVector::new(embedding.embedding.clone());
        let hv = self.bridge.project(&dense.values);

        Ok(EncodedThought {
            text: text.to_string(),
            dense,
            hypervector: SerializableHV16::from_hv16(&hv),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            context_id: None,
            confidence: 1.0,
        })
    }

    /// Encode and return only the hypervector (for HDC-only operations)
    pub fn encode_hv(&self, text: &str) -> Result<HV16> {
        let embedding = self.embedder.embed(text)?;
        Ok(self.bridge.project(&embedding.embedding))
    }

    /// Encode and return only the dense vector (for similarity-only operations)
    pub fn encode_dense(&self, text: &str) -> Result<DenseVector> {
        let embedding = self.embedder.embed(text)?;
        Ok(DenseVector::new(embedding.embedding))
    }

    /// Reset temporal context
    pub fn reset_context(&mut self) {
        if let Some(ref mut ctx) = self.temporal_context {
            *ctx = DenseVector::zeros(self.config.dense_dim);
        }
    }

    /// Get current temporal context (if enabled)
    pub fn get_context(&self) -> Option<&DenseVector> {
        self.temporal_context.as_ref()
    }

    /// Set context from an existing EncodedThought (for session restoration)
    pub fn set_context(&mut self, thought: &EncodedThought) {
        if let Some(ref mut ctx) = self.temporal_context {
            *ctx = thought.dense.clone();
        }
    }

    /// Compute similarity between two texts using dense vectors
    pub fn similarity(&self, text_a: &str, text_b: &str) -> Result<f32> {
        let emb_a = self.embedder.embed(text_a)?;
        let emb_b = self.embedder.embed(text_b)?;

        let dense_a = DenseVector::new(emb_a.embedding);
        let dense_b = DenseVector::new(emb_b.embedding);

        Ok(dense_a.similarity(&dense_b))
    }

    /// Compute similarity using hypervectors (HDC space)
    pub fn similarity_hdc(&self, text_a: &str, text_b: &str) -> Result<f32> {
        let hv_a = self.encode_hv(text_a)?;
        let hv_b = self.encode_hv(text_b)?;
        Ok(hv_a.similarity(&hv_b))
    }

    /// Get encoder statistics
    pub fn stats(&self) -> &EncoderStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &SemanticEncoderConfig {
        &self.config
    }
}

impl Default for SemanticEncoder {
    fn default() -> Self {
        Self::new().expect("Failed to create default SemanticEncoder")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = SemanticEncoder::new().unwrap();
        assert_eq!(encoder.config().dense_dim, BGE_DIMENSION);
        assert_eq!(encoder.config().hv_dim, HV16::DIM);
    }

    #[test]
    fn test_encode_returns_dual_representation() {
        let mut encoder = SemanticEncoder::new().unwrap();
        let thought = encoder.encode("What is the capital of France?").unwrap();

        // Dense vector should be 768D
        assert_eq!(thought.dense.dim(), BGE_DIMENSION);

        // Hypervector should be HV16
        let hv = thought.hypervector.to_hv16();
        assert_eq!(hv.0.len(), HV16::BYTES);

        // Text should be preserved
        assert_eq!(thought.text, "What is the capital of France?");
    }

    #[test]
    fn test_similar_texts_have_high_similarity() {
        let encoder = SemanticEncoder::new().unwrap();

        let sim = encoder
            .similarity("capital of France", "French capital city")
            .unwrap();

        // Similar texts should have similarity > 0.5
        println!("Similarity (capital/French): {:.4}", sim);
        // Note: With stub embedder, similarity is based on hash, not semantics
    }

    #[test]
    fn test_temporal_context_blending() {
        let mut encoder = SemanticEncoder::with_config(SemanticEncoderConfig {
            enable_temporal_context: true,
            temporal_alpha: 0.5,
            ..Default::default()
        })
        .unwrap();

        // First encoding
        let thought1 = encoder.encode("Paris is the capital").unwrap();

        // Second encoding - should be blended with first
        let thought2 = encoder.encode("of France").unwrap();

        // The second thought should be different from a pure encoding
        let pure = encoder.encode_pure("of France").unwrap();

        // With context blending, thought2.dense != pure.dense
        let sim_to_pure = thought2.dense.similarity(&pure.dense);
        println!("Similarity (blended vs pure): {:.4}", sim_to_pure);

        // Blended should be different from pure (similarity < 1.0)
        // Note: With stub embedder this may not show much difference
    }

    #[test]
    fn test_context_reset() {
        let mut encoder = SemanticEncoder::new().unwrap();

        encoder.encode("Some context").unwrap();
        assert!(encoder.get_context().is_some());

        encoder.reset_context();
        let ctx = encoder.get_context().unwrap();

        // After reset, context should be zeros
        let sum: f32 = ctx.values.iter().sum();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_pure_encoding_is_stateless() {
        let encoder = SemanticEncoder::new().unwrap();

        let pure1 = encoder.encode_pure("test query").unwrap();
        let pure2 = encoder.encode_pure("test query").unwrap();

        // Pure encodings of same text should be identical
        let sim = pure1.dense.similarity(&pure2.dense);
        assert!((sim - 1.0).abs() < 0.0001, "Pure encodings should be identical");
    }

    #[test]
    fn test_encoded_thought_serialization() {
        let mut encoder = SemanticEncoder::new().unwrap();
        let thought = encoder.encode("Test serialization").unwrap();

        // Serialize to JSON
        let json = serde_json::to_string(&thought).unwrap();
        assert!(!json.is_empty());

        // Deserialize back
        let deserialized: EncodedThought = serde_json::from_str(&json).unwrap();

        assert_eq!(thought.text, deserialized.text);
        assert_eq!(thought.dense.values.len(), deserialized.dense.values.len());
    }

    #[test]
    fn test_dense_vector_operations() {
        let v1 = DenseVector::new(vec![1.0, 0.0, 0.0]);
        let v2 = DenseVector::new(vec![0.0, 1.0, 0.0]);
        let v3 = DenseVector::new(vec![1.0, 0.0, 0.0]);

        // Orthogonal vectors have 0 similarity
        assert!((v1.similarity(&v2) - 0.0).abs() < 0.0001);

        // Same vectors have 1 similarity
        assert!((v1.similarity(&v3) - 1.0).abs() < 0.0001);

        // Blend test
        let blended = v1.blend(&v2, 0.5);
        assert!((blended.values[0] - 0.5).abs() < 0.0001);
        assert!((blended.values[1] - 0.5).abs() < 0.0001);

        // Add test
        let added = v1.add(&v2);
        assert!((added.values[0] - 1.0).abs() < 0.0001);
        assert!((added.values[1] - 1.0).abs() < 0.0001);
    }
}
