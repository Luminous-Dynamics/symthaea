// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Modal Integration - Unifying All Senses into the Holographic Brain
//!
//! This module provides the revolutionary integration layer that projects all
//! sensory modalities (vision, voice, code, OCR) into a unified 10,000D hyperdimensional
//! concept space within Sophia's Holographic Liquid Brain (HLB).
//!
//! ## Revolutionary Concept: One Mind, Many Senses
//!
//! Traditional AI systems process different modalities separately:
//! - Vision models output embeddings in one space
//! - Language models in another
//! - Code analysis in yet another
//!
//! Sophia is different. ALL sensory input is projected into the SAME holographic
//! concept space, allowing true multi-modal reasoning:
//! - "This image shows code that implements what I just heard about"
//! - "The OCR text explains the visual scene"
//! - "The code structure mirrors the architectural diagram"
//!
//! This is consciousness-first computing: unified awareness across all modalities.

use anyhow::Result;
use std::time::Instant;

// Use local HDC dimension constant
use crate::HDC_DIMENSION;
const HDC_DIM: usize = HDC_DIMENSION;

/// Johnson-Lindenstrauss Random Projection Matrix
///
/// Implements distance-preserving projection from input space to HDC space.
/// Uses sparse Rademacher distribution (±1 with probability 1/2, 0 with probability 1-2s)
/// for computational efficiency.
///
/// Key property: For any two points u, v:
/// |‖Pu - Pv‖ - ‖u - v‖| < ε‖u - v‖ with high probability
struct JLProjector {
    /// Projection matrix stored as sparse entries: (row, col, sign)
    /// Using sparse representation since most entries are 0
    sparse_entries: Vec<(usize, usize, i8)>,

    /// Input dimension
    input_dim: usize,

    /// Output dimension
    output_dim: usize,

    /// Scaling factor (sqrt(3/s) for sparse JL)
    scale: f32,
}

impl JLProjector {
    /// Create a new JL projector with sparse Rademacher distribution
    ///
    /// Uses sparsity s = 1/3 (standard choice) which means:
    /// - P(entry = +1/sqrt(s)) = s/2
    /// - P(entry = -1/sqrt(s)) = s/2
    /// - P(entry = 0) = 1 - s
    fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut entries = Vec::new();
        let sparsity = 3; // 1/3 probability of non-zero

        // Generate sparse random projection matrix
        for row in 0..output_dim {
            for col in 0..input_dim {
                // Deterministic pseudo-random based on position and seed
                let mut hasher = DefaultHasher::new();
                (row, col, seed).hash(&mut hasher);
                let hash = hasher.finish();

                // Sparse: only ~1/3 of entries are non-zero
                if hash % sparsity as u64 == 0 {
                    // Sign: ±1 with equal probability
                    let sign = if (hash / sparsity as u64) % 2 == 0 {
                        1i8
                    } else {
                        -1i8
                    };
                    entries.push((row, col, sign));
                }
            }
        }

        // Scale factor for sparse JL: sqrt(3) since sparsity = 1/3
        let scale = (sparsity as f32).sqrt();

        Self {
            sparse_entries: entries,
            input_dim,
            output_dim,
            scale,
        }
    }

    /// Project a vector from input space to output space
    ///
    /// Uses JL scaling: each output is scaled by sqrt(sparsity/output_dim) to
    /// ensure distance preservation. For sparse Rademacher with sparsity=1/3,
    /// this means scaling by sqrt(3/k) where k is the output dimension.
    fn project(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim);

        let mut output = vec![0.0f32; self.output_dim];

        // Sparse matrix-vector multiplication
        for &(row, col, sign) in &self.sparse_entries {
            output[row] += (sign as f32) * input[col];
        }

        // Apply JL scaling: sqrt(sparsity / output_dim) = scale / sqrt(output_dim)
        // This ensures expected squared distance is preserved
        let jl_scale = self.scale / (self.output_dim as f32).sqrt();
        for val in &mut output {
            *val *= jl_scale;
        }

        output
    }

    /// Project to binary HDC (threshold at 0)
    fn project_to_binary(&self, input: &[f32]) -> Vec<bool> {
        self.project(input).into_iter().map(|v| v > 0.0).collect()
    }

    /// Project with L2 normalization applied first
    ///
    /// This ensures that input vectors of different magnitudes produce
    /// comparable output vectors, which is critical for cosine similarity
    /// and HDC bundling operations.
    fn project_normalized(&self, input: &[f32]) -> Vec<f32> {
        let normalized = l2_normalize(input);
        self.project(&normalized)
    }

    /// Project to binary HDC with L2 normalization applied first
    fn project_to_binary_normalized(&self, input: &[f32]) -> Vec<bool> {
        self.project_normalized(input)
            .into_iter()
            .map(|v| v > 0.0)
            .collect()
    }
}

/// L2 normalize a vector to unit length
///
/// Returns a zero vector if the input has zero magnitude (e.g., empty file, black image).
/// This prevents NaN propagation while maintaining a valid HDC representation.
fn l2_normalize(input: &[f32]) -> Vec<f32> {
    let norm_sq: f32 = input.iter().map(|x| x * x).sum();

    if norm_sq < 1e-12 {
        // Zero or near-zero vector - return zeros to avoid NaN
        // This represents "no information" in the HDC space
        return vec![0.0; input.len()];
    }

    let norm = norm_sq.sqrt();
    input.iter().map(|x| x / norm).collect()
}

/// Multi-modal HDC vector type - Uses HDC_DIMENSION (16,384D) holographic vector
///
/// Note: This is a simplified boolean-based HDC implementation optimized for
/// multi-modal perception. For consciousness measurement, use `hdc::RealHV`.
/// For binary operations, use `hdc::HV16`. This type focuses on fast bundling
/// and projection for sensory fusion tasks.
#[derive(Debug, Clone)]
pub struct HdcVector {
    /// Binary representation (public for consciousness bridge)
    pub bits: Vec<bool>,
}

impl HdcVector {
    /// Create a zero vector (all bits false)
    pub fn zero() -> Self {
        Self {
            bits: vec![false; HDC_DIM],
        }
    }

    /// Create a random hypervector with ~50% sparsity
    ///
    /// Random vectors are quasi-orthogonal in high dimensions - the expected
    /// Hamming similarity between two random vectors approaches 0.5.
    /// This is used to create atomic symbol representations.
    pub fn random(seed: u64) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut bits = vec![false; HDC_DIM];
        for i in 0..HDC_DIM {
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            bits[i] = hasher.finish() % 2 == 0;
        }
        Self { bits }
    }

    pub fn set_bit(&mut self, index: usize) {
        if index < self.bits.len() {
            self.bits[index] = true;
        }
    }

    /// Bundle two vectors using OR (disjunction)
    ///
    /// Bundling creates a vector that is similar to both inputs.
    /// Note: OR-based bundling increases sparsity. For multiple vectors,
    /// use `majority_bundle()` for better fidelity.
    pub fn bundle(&self, other: &HdcVector) -> HdcVector {
        let mut result = HdcVector::zero();
        for i in 0..HDC_DIM {
            result.bits[i] = self.bits[i] || other.bits[i];
        }
        result
    }

    /// Bundle multiple vectors using majority voting
    ///
    /// Each output bit is 1 if more than half of the input bits at that
    /// position are 1. This preserves ~50% sparsity and provides better
    /// similarity to all inputs than iterative OR bundling.
    ///
    /// For ties (even number of vectors), uses the first vector as tiebreaker.
    pub fn majority_bundle(vectors: &[&HdcVector]) -> HdcVector {
        if vectors.is_empty() {
            return HdcVector::zero();
        }
        if vectors.len() == 1 {
            return vectors[0].clone();
        }

        let n = vectors.len();
        let threshold = n / 2;
        let has_ties = n % 2 == 0; // Only even counts can have ties
        let mut result = HdcVector::zero();

        for i in 0..HDC_DIM {
            let count: usize = vectors.iter().filter(|v| v.bits[i]).count();
            // Majority vote: count > n/2
            // For even n, use tiebreaker when count == n/2
            result.bits[i] =
                count > threshold || (has_ties && count == threshold && vectors[0].bits[i]);
        }
        result
    }

    /// Bind two vectors using XOR (exclusive or)
    ///
    /// Binding creates a compositional representation: `A ⊗ B` represents
    /// "A in relation to B" or "A bound with B". Key properties:
    /// - Binding is commutative: A ⊗ B = B ⊗ A
    /// - Binding is self-inverse: A ⊗ B ⊗ B = A
    /// - Bound vector is dissimilar to both inputs (quasi-orthogonal)
    ///
    /// Example: `code ⊗ context` creates a representation of "code in this context"
    pub fn bind(&self, other: &HdcVector) -> HdcVector {
        let mut result = HdcVector::zero();
        for i in 0..HDC_DIM {
            result.bits[i] = self.bits[i] ^ other.bits[i];
        }
        result
    }

    /// Unbind a vector - same as bind since XOR is self-inverse
    ///
    /// If C = A ⊗ B, then A = C ⊗ B (unbinding B recovers A)
    #[inline]
    pub fn unbind(&self, other: &HdcVector) -> HdcVector {
        self.bind(other) // XOR is its own inverse
    }

    /// Permute vector by circular right shift
    ///
    /// Permutation encodes position or sequence information. Each unique
    /// shift creates a quasi-orthogonal vector from the original.
    ///
    /// Example: Encoding sequence "A then B then C":
    /// ```ignore
    /// let seq = a.permute(0).bundle(&b.permute(1)).bundle(&c.permute(2));
    /// ```
    pub fn permute(&self, positions: usize) -> HdcVector {
        if positions == 0 {
            return self.clone();
        }

        let shift = positions % HDC_DIM;
        let mut result = HdcVector::zero();

        for i in 0..HDC_DIM {
            let new_pos = (i + shift) % HDC_DIM;
            result.bits[new_pos] = self.bits[i];
        }
        result
    }

    /// Inverse permute (circular left shift)
    ///
    /// Undoes a permutation: `v.permute(n).inverse_permute(n) == v`
    pub fn inverse_permute(&self, positions: usize) -> HdcVector {
        if positions == 0 {
            return self.clone();
        }

        let shift = positions % HDC_DIM;
        let mut result = HdcVector::zero();

        for i in 0..HDC_DIM {
            let new_pos = (i + HDC_DIM - shift) % HDC_DIM;
            result.bits[new_pos] = self.bits[i];
        }
        result
    }

    /// Calculate Hamming similarity between two vectors
    ///
    /// Returns a value in [0.0, 1.0] where:
    /// - 1.0 = identical vectors
    /// - 0.5 = orthogonal (random/unrelated) vectors
    /// - 0.0 = opposite vectors (all bits flipped)
    ///
    /// This is the fundamental similarity measure in binary HDC.
    pub fn similarity(&self, other: &HdcVector) -> f32 {
        let matching: usize = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .filter(|&(&a, &b)| a == b)
            .count();
        matching as f32 / HDC_DIM as f32
    }

    /// Calculate Hamming distance (number of differing bits)
    pub fn hamming_distance(&self, other: &HdcVector) -> usize {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .filter(|&(&a, &b)| a != b)
            .count()
    }

    /// Check if this vector is similar to another (above threshold)
    ///
    /// Default threshold of 0.6 is well above the 0.5 expected for random vectors.
    pub fn is_similar(&self, other: &HdcVector, threshold: f32) -> bool {
        self.similarity(other) > threshold
    }

    /// Get dimensionality of the HDC vector
    pub fn dim(&self) -> usize {
        self.bits.len()
    }

    /// Calculate sparsity (fraction of bits that are set)
    pub fn sparsity(&self) -> f32 {
        let set_bits = self.bits.iter().filter(|&&b| b).count();
        set_bits as f32 / self.bits.len() as f32
    }

    /// Flip all bits (logical NOT)
    pub fn invert(&self) -> HdcVector {
        HdcVector {
            bits: self.bits.iter().map(|&b| !b).collect(),
        }
    }
}
use super::{ImageEmbedding, OcrResult, RustCodeSemantics, VisualFeatures};

/// Multi-modal perception result combining all sensory modalities
#[derive(Debug, Clone)]
pub struct MultiModalPerception {
    /// Unified HDC vector representing the combined perception
    pub unified_concept: HdcVector,

    /// Individual modality contributions (for inspection/debugging)
    pub modalities: Vec<ModalityContribution>,

    /// Overall confidence in the perception (0.0 to 1.0)
    pub confidence: f32,

    /// Timestamp when this perception was created
    pub timestamp: Instant,
}

/// Contribution from a single sensory modality
#[derive(Debug, Clone)]
pub struct ModalityContribution {
    /// Which sense provided this input
    pub modality: ModalityType,

    /// HDC projection of this modality's input
    pub hdc_projection: HdcVector,

    /// Confidence in this modality's input (0.0 to 1.0)
    pub confidence: f32,

    /// Weight given to this modality in the fusion (0.0 to 1.0)
    pub fusion_weight: f32,
}

/// Types of sensory modalities Sophia can perceive
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModalityType {
    /// Visual perception (images, screenshots)
    Vision,

    /// Voice/audio perception (speech, sounds)
    Voice,

    /// Code perception (source code analysis)
    Code,

    /// Text extraction via OCR
    Ocr,

    /// Semantic text input (direct text, not from OCR)
    Semantic,
}

impl ModalityType {
    /// Get a human-readable name for this modality
    pub fn name(&self) -> &'static str {
        match self {
            ModalityType::Vision => "Vision",
            ModalityType::Voice => "Voice",
            ModalityType::Code => "Code",
            ModalityType::Ocr => "OCR",
            ModalityType::Semantic => "Semantic",
        }
    }
}

/// Embedding dimension for Qwen3 (1024D)
pub const QWEN3_DIM: usize = 1024;

/// Embedding dimension for SigLIP (768D)
pub const SIGLIP_DIM: usize = 768;

/// Multi-modal integration system - the sensory fusion layer
pub struct MultiModalIntegrator {
    /// Default weights for each modality (can be adjusted dynamically)
    modality_weights: ModalityWeights,

    /// Whether to use adaptive weighting based on confidence
    adaptive_weighting: bool,

    /// JL projector for image embeddings (768D SigLIP → HDC_DIM)
    image_projector: JLProjector,

    /// JL projector for text embeddings (1024D Qwen3 → HDC_DIM)
    text_embedding_projector: JLProjector,

    /// JL projector for text/OCR n-grams (256D character n-grams → HDC_DIM)
    text_ngram_projector: JLProjector,

    /// JL projector for visual features (32D handcrafted → HDC_DIM)
    vision_feature_projector: JLProjector,

    /// JL projector for code semantics (150D structural+ngram features → HDC_DIM)
    code_projector: JLProjector,
}

/// Weights for combining different modalities
#[derive(Debug, Clone)]
pub struct ModalityWeights {
    pub vision: f32,
    pub voice: f32,
    pub code: f32,
    pub ocr: f32,
    pub semantic: f32,
}

impl Default for ModalityWeights {
    fn default() -> Self {
        Self {
            vision: 1.0,
            voice: 1.0,
            code: 1.0,
            ocr: 1.0,
            semantic: 1.0,
        }
    }
}

impl Default for MultiModalIntegrator {
    fn default() -> Self {
        // Use fixed seeds for deterministic projections across runs
        const IMAGE_PROJECTOR_SEED: u64 = 0xCAFE_BABE_1234_5678;
        const TEXT_EMBEDDING_PROJECTOR_SEED: u64 = 0xDEAD_BEEF_8765_4321;
        const TEXT_NGRAM_PROJECTOR_SEED: u64 = 0x5974_1AEA_CAFE_BABE;
        const VISION_FEATURE_PROJECTOR_SEED: u64 = 0xC0DE_CAFE_1337_BEEF;
        const CODE_PROJECTOR_SEED: u64 = 0xA57_5EED_C0DE_CAFE;

        Self {
            modality_weights: ModalityWeights::default(),
            adaptive_weighting: true,
            // 768D SigLIP embedding → 16,384D HDC
            image_projector: JLProjector::new(SIGLIP_DIM, HDC_DIM, IMAGE_PROJECTOR_SEED),
            // 1024D Qwen3 embedding → 16,384D HDC
            text_embedding_projector: JLProjector::new(
                QWEN3_DIM,
                HDC_DIM,
                TEXT_EMBEDDING_PROJECTOR_SEED,
            ),
            // 256D n-gram features → 16,384D HDC
            text_ngram_projector: JLProjector::new(256, HDC_DIM, TEXT_NGRAM_PROJECTOR_SEED),
            // 32D visual features → 16,384D HDC
            vision_feature_projector: JLProjector::new(32, HDC_DIM, VISION_FEATURE_PROJECTOR_SEED),
            // 150D code features (10 structural + 64 fn_ngrams + 64 struct_ngrams + 12 reserved) → 16,384D HDC
            code_projector: JLProjector::new(150, HDC_DIM, CODE_PROJECTOR_SEED),
        }
    }
}

impl MultiModalIntegrator {
    /// Create a new multi-modal integrator
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom modality weights
    pub fn with_weights(weights: ModalityWeights) -> Self {
        let default = Self::default();
        Self {
            modality_weights: weights,
            adaptive_weighting: true,
            image_projector: default.image_projector,
            text_embedding_projector: default.text_embedding_projector,
            text_ngram_projector: default.text_ngram_projector,
            vision_feature_projector: default.vision_feature_projector,
            code_projector: default.code_projector,
        }
    }

    /// Project a text embedding (from Qwen3 or similar) into HDC space
    ///
    /// Uses JL projection to map 1024D Qwen3 embedding to HDC_DIM (16,384D).
    pub fn project_text_embedding(&self, embedding: &[f32]) -> Result<HdcVector> {
        if embedding.len() != QWEN3_DIM {
            anyhow::bail!(
                "Text embedding dimension mismatch: expected {}, got {}",
                QWEN3_DIM,
                embedding.len()
            );
        }

        let projected_bits = self.text_embedding_projector.project_to_binary(embedding);

        Ok(HdcVector {
            bits: projected_bits,
        })
    }

    /// Enable or disable adaptive weighting based on confidence
    pub fn set_adaptive_weighting(&mut self, enabled: bool) {
        self.adaptive_weighting = enabled;
    }

    /// Project visual features into HDC space using JL projection
    ///
    /// Creates a dense feature vector from visual features, then applies
    /// Johnson-Lindenstrauss random projection to map to HDC space.
    pub fn project_vision(&self, features: &VisualFeatures) -> Result<HdcVector> {
        // Build a dense feature vector from visual features
        // Dimension: 3 (brightness, variance, edges) + up to 24 (8 colors × 3 RGB) = 27D max
        let mut feature_vec = Vec::with_capacity(32);

        // Core features (normalized to [0, 1])
        feature_vec.push(features.brightness);
        feature_vec.push(features.color_variance);
        feature_vec.push(features.edge_density);

        // Normalized dimensions (aspect ratio encoding)
        let (w, h) = features.dimensions;
        let aspect = w as f32 / h.max(1) as f32;
        feature_vec.push((aspect / 3.0).clamp(0.0, 1.0)); // Normalized aspect ratio
        feature_vec.push((w as f32 / 2048.0).clamp(0.0, 1.0)); // Normalized width
        feature_vec.push((h as f32 / 2048.0).clamp(0.0, 1.0)); // Normalized height

        // Encode dominant colors (up to 8 colors, 3 channels each = 24 features)
        for color in features.dominant_colors.iter().take(8) {
            feature_vec.push(color[0] as f32 / 255.0); // R
            feature_vec.push(color[1] as f32 / 255.0); // G
            feature_vec.push(color[2] as f32 / 255.0); // B
        }

        // Pad to fixed size (32D) for consistent projection
        while feature_vec.len() < 32 {
            feature_vec.push(0.0);
        }

        // Use the vision feature projector (32D -> HDC_DIM) with L2 normalization
        // L2 norm ensures images of different complexity produce comparable magnitude vectors
        let projected_bits = self
            .vision_feature_projector
            .project_to_binary_normalized(&feature_vec);

        Ok(HdcVector {
            bits: projected_bits,
        })
    }

    /// Project image embedding into HDC space using Johnson-Lindenstrauss projection
    ///
    /// Uses sparse random projection to map 768D SigLIP embedding to HDC_DIM (16,384D).
    /// The JL lemma guarantees that pairwise distances are approximately preserved:
    /// |‖P(u) - P(v)‖ - ‖u - v‖| < ε‖u - v‖ with high probability.
    pub fn project_image_embedding(&self, embedding: &ImageEmbedding) -> Result<HdcVector> {
        // Ensure input dimension matches (768D SigLIP)
        if embedding.vector.len() != 768 {
            anyhow::bail!(
                "Image embedding dimension mismatch: expected 768, got {}",
                embedding.vector.len()
            );
        }

        // Apply JL projection to get binary HDC vector
        let projected_bits = self.image_projector.project_to_binary(&embedding.vector);

        Ok(HdcVector {
            bits: projected_bits,
        })
    }

    /// Project raw text into HDC space using n-gram feature encoding
    ///
    /// Encodes text using character tri-grams projected through JL projection.
    /// This creates a distributed representation that captures character patterns.
    pub fn project_text(&self, text: &str) -> Result<HdcVector> {
        // Build n-gram feature vector (256 dimensions for character tri-grams)
        let mut ngram_features = vec![0.0f32; 256];

        // Extract character tri-grams
        let text_lower = text.to_lowercase();
        let chars: Vec<char> = text_lower.chars().collect();

        for window in chars.windows(3) {
            // Hash tri-gram to feature index
            let hash = window
                .iter()
                .fold(0u64, |acc, &c| acc.wrapping_mul(31).wrapping_add(c as u64));
            let idx = (hash % 256) as usize;
            ngram_features[idx] += 1.0;
        }

        // Normalize features
        let sum: f32 = ngram_features.iter().sum();
        if sum > 0.0 {
            for f in &mut ngram_features {
                *f /= sum;
            }
        }

        // Apply JL projection
        let projected_bits = self.text_ngram_projector.project_to_binary(&ngram_features);

        Ok(HdcVector {
            bits: projected_bits,
        })
    }

    /// Project OCR text into HDC space using n-gram feature encoding
    ///
    /// Encodes text using character tri-grams projected through JL projection.
    /// This creates a distributed representation that captures character patterns.
    pub fn project_ocr(&self, ocr: &OcrResult) -> Result<HdcVector> {
        // Build n-gram feature vector (256 dimensions for character tri-grams)
        let mut ngram_features = vec![0.0f32; 256];

        // Extract character tri-grams
        let text = ocr.text.to_lowercase();
        let chars: Vec<char> = text.chars().collect();

        for window in chars.windows(3) {
            // Hash tri-gram to feature index
            let hash = window
                .iter()
                .fold(0u64, |acc, &c| acc.wrapping_mul(31).wrapping_add(c as u64));
            let idx = (hash % 256) as usize;
            ngram_features[idx] += 1.0;
        }

        // Normalize features
        let sum: f32 = ngram_features.iter().sum();
        if sum > 0.0 {
            for f in &mut ngram_features {
                *f /= sum;
            }
        }

        // Apply JL projection
        let projected_bits = self.text_ngram_projector.project_to_binary(&ngram_features);

        Ok(HdcVector {
            bits: projected_bits,
        })
    }

    /// Project code semantics into HDC space using Johnson-Lindenstrauss projection
    ///
    /// Encodes structural code features into a 150D vector, then projects to HDC space.
    /// Features include counts (functions, structs, enums, traits, impls, modules),
    /// ratios (public_ratio), and n-gram encodings of identifiers with 64 buckets each.
    ///
    /// ## Feature Layout (150D total)
    /// - Indices 0-9: Structural features (counts, ratios)
    /// - Indices 10-73: Function name tri-gram buckets (64 buckets, ~40% collision rate)
    /// - Indices 74-137: Struct name tri-gram buckets (64 buckets)
    /// - Indices 138-149: Reserved for future features (12 dimensions)
    pub fn project_code(&self, semantics: &RustCodeSemantics) -> Result<HdcVector> {
        // N-gram bucket count - 64 gives ~40% collision rate vs 80% with 22
        const NGRAM_BUCKETS: usize = 64;

        // Build 150D feature vector from code semantics
        // Layout: 10 structural + 64 fn_ngrams + 64 struct_ngrams + 12 reserved = 150
        let mut features = vec![0.0f32; 150];

        // Structural counts (normalized by typical max values)
        // Indices 0-5: Count features (log-scaled for better distribution)
        features[0] = ((semantics.function_count as f32 + 1.0).ln() / 7.0).clamp(0.0, 1.0); // ~1000 max
        features[1] = ((semantics.struct_count as f32 + 1.0).ln() / 6.0).clamp(0.0, 1.0); // ~400 max
        features[2] = ((semantics.enum_count as f32 + 1.0).ln() / 5.0).clamp(0.0, 1.0); // ~150 max
        features[3] = ((semantics.trait_count as f32 + 1.0).ln() / 5.0).clamp(0.0, 1.0); // ~150 max
        features[4] = ((semantics.impl_count as f32 + 1.0).ln() / 6.0).clamp(0.0, 1.0); // ~400 max
        features[5] = ((semantics.module_count as f32 + 1.0).ln() / 5.0).clamp(0.0, 1.0); // ~150 max

        // Index 6: Public ratio (already normalized 0-1)
        features[6] = semantics.public_ratio;

        // Index 7: Total items (complexity indicator)
        let total_items = semantics.function_count
            + semantics.struct_count
            + semantics.enum_count
            + semantics.trait_count
            + semantics.impl_count
            + semantics.module_count;
        features[7] = ((total_items as f32 + 1.0).ln() / 8.0).clamp(0.0, 1.0);

        // Index 8: Struct-to-function ratio (architectural pattern)
        let fn_count = semantics.function_count.max(1) as f32;
        features[8] = (semantics.struct_count as f32 / fn_count).clamp(0.0, 1.0);

        // Index 9: Trait-to-struct ratio (abstraction level)
        let struct_count = semantics.struct_count.max(1) as f32;
        features[9] = (semantics.trait_count as f32 / struct_count).clamp(0.0, 1.0);

        // Indices 10-73: N-gram encoding of function names (64 buckets)
        let mut fn_name_ngrams = vec![0.0f32; NGRAM_BUCKETS];
        for name in &semantics.function_names {
            let name_lower = name.to_lowercase();
            let chars: Vec<char> = name_lower.chars().collect();
            for window in chars.windows(3) {
                let hash = window
                    .iter()
                    .fold(0u64, |acc, &c| acc.wrapping_mul(31).wrapping_add(c as u64));
                let idx = (hash % NGRAM_BUCKETS as u64) as usize;
                fn_name_ngrams[idx] += 1.0;
            }
        }
        // Normalize function name n-grams (distribution, not magnitude - L2 norm handles magnitude)
        let fn_sum: f32 = fn_name_ngrams.iter().sum();
        if fn_sum > 0.0 {
            for f in &mut fn_name_ngrams {
                *f /= fn_sum;
            }
        }
        features[10..74].copy_from_slice(&fn_name_ngrams);

        // Indices 74-137: N-gram encoding of struct names (64 buckets)
        let mut struct_name_ngrams = vec![0.0f32; NGRAM_BUCKETS];
        for name in &semantics.struct_names {
            let name_lower = name.to_lowercase();
            let chars: Vec<char> = name_lower.chars().collect();
            for window in chars.windows(3) {
                let hash = window
                    .iter()
                    .fold(0u64, |acc, &c| acc.wrapping_mul(31).wrapping_add(c as u64));
                let idx = (hash % NGRAM_BUCKETS as u64) as usize;
                struct_name_ngrams[idx] += 1.0;
            }
        }
        // Normalize struct name n-grams
        let struct_sum: f32 = struct_name_ngrams.iter().sum();
        if struct_sum > 0.0 {
            for f in &mut struct_name_ngrams {
                *f /= struct_sum;
            }
        }
        features[74..138].copy_from_slice(&struct_name_ngrams);

        // Indices 138-149: Reserved for future features (initialized to 0)
        // Could add: cyclomatic complexity, nesting depth, async/unsafe counts, etc.

        // Apply JL projection with L2 normalization (150D → HDC_DIM)
        // L2 norm ensures codebases of different sizes produce comparable magnitude vectors
        let projected_bits = self.code_projector.project_to_binary_normalized(&features);

        Ok(HdcVector {
            bits: projected_bits,
        })
    }

    /// Fuse multiple modality contributions into a unified perception
    pub fn fuse_modalities(
        &self,
        contributions: Vec<ModalityContribution>,
    ) -> Result<MultiModalPerception> {
        if contributions.is_empty() {
            anyhow::bail!("Cannot fuse zero modalities");
        }

        // Start with zero vector
        let mut unified = HdcVector::zero();
        let mut total_weight = 0.0f32;
        let mut weighted_confidence = 0.0f32;

        for contrib in &contributions {
            // Calculate effective weight (base weight * confidence if adaptive)
            let effective_weight = if self.adaptive_weighting {
                self.get_base_weight(contrib.modality) * contrib.confidence
            } else {
                self.get_base_weight(contrib.modality)
            };

            // Accumulate weighted contribution
            // In HDC, we can use bundling (element-wise OR) or weighted majority
            // For now, use simple bundling with the first modality as base
            if total_weight == 0.0 {
                unified = contrib.hdc_projection.clone();
            } else {
                // Bundle: combine vectors (similar to vector addition in concept space)
                unified = unified.bundle(&contrib.hdc_projection);
            }

            total_weight += effective_weight;
            weighted_confidence += contrib.confidence * effective_weight;
        }

        // Normalize confidence
        let overall_confidence = if total_weight > 0.0 {
            weighted_confidence / total_weight
        } else {
            0.0
        };

        Ok(MultiModalPerception {
            unified_concept: unified,
            modalities: contributions,
            confidence: overall_confidence,
            timestamp: Instant::now(),
        })
    }

    /// Get base weight for a modality type
    fn get_base_weight(&self, modality: ModalityType) -> f32 {
        match modality {
            ModalityType::Vision => self.modality_weights.vision,
            ModalityType::Voice => self.modality_weights.voice,
            ModalityType::Code => self.modality_weights.code,
            ModalityType::Ocr => self.modality_weights.ocr,
            ModalityType::Semantic => self.modality_weights.semantic,
        }
    }

    /// Create a multi-modal perception from a single image
    ///
    /// This is a convenience method that processes an image through
    /// multiple pathways (visual features, embeddings, OCR) and fuses
    /// the results into a unified perception.
    pub fn perceive_image(
        &self,
        visual: &VisualFeatures,
        embedding: Option<&ImageEmbedding>,
        ocr: Option<&OcrResult>,
    ) -> Result<MultiModalPerception> {
        let mut contributions = Vec::new();

        // Add visual features contribution
        let visual_hdc = self.project_vision(visual)?;
        contributions.push(ModalityContribution {
            modality: ModalityType::Vision,
            hdc_projection: visual_hdc,
            confidence: 0.8, // Visual features are generally reliable
            fusion_weight: self.modality_weights.vision,
        });

        // Add embedding contribution if available
        if let Some(emb) = embedding {
            let emb_hdc = self.project_image_embedding(emb)?;
            contributions.push(ModalityContribution {
                modality: ModalityType::Vision,
                hdc_projection: emb_hdc,
                confidence: 0.9, // Embeddings are high quality
                fusion_weight: self.modality_weights.vision * 1.2, // Higher weight for semantic
            });
        }

        // Add OCR contribution if available
        if let Some(ocr_result) = ocr {
            let ocr_hdc = self.project_ocr(ocr_result)?;
            contributions.push(ModalityContribution {
                modality: ModalityType::Ocr,
                hdc_projection: ocr_hdc,
                confidence: ocr_result.confidence,
                fusion_weight: self.modality_weights.ocr,
            });
        }

        self.fuse_modalities(contributions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_visual_features() -> VisualFeatures {
        VisualFeatures {
            dimensions: (800, 600),
            dominant_colors: vec![[128, 128, 128]],
            brightness: 0.5,
            color_variance: 0.5,
            edge_density: 0.5,
        }
    }

    fn create_test_embedding() -> ImageEmbedding {
        ImageEmbedding {
            vector: vec![0.5; 768],
            timestamp: Instant::now(),
            image_hash: 12345,
        }
    }

    fn create_test_ocr() -> OcrResult {
        OcrResult {
            text: "Test OCR text".to_string(),
            confidence: 0.85,
            method: crate::OcrMethod::RustOcr,
            duration_ms: 50,
            words: vec![],
        }
    }

    #[test]
    fn test_integrator_creation() {
        let integrator = MultiModalIntegrator::new();
        assert!(integrator.adaptive_weighting);
        assert_eq!(integrator.modality_weights.vision, 1.0);
    }

    #[test]
    fn test_custom_weights() {
        let weights = ModalityWeights {
            vision: 2.0,
            voice: 1.0,
            code: 1.5,
            ocr: 0.8,
            semantic: 1.0,
        };

        let integrator = MultiModalIntegrator::with_weights(weights.clone());
        assert_eq!(integrator.modality_weights.vision, 2.0);
        assert_eq!(integrator.modality_weights.ocr, 0.8);
    }

    #[test]
    fn test_vision_projection() {
        let integrator = MultiModalIntegrator::new();
        let features = create_test_visual_features();

        let hdc = integrator.project_vision(&features).unwrap();
        assert_eq!(hdc.dim(), HDC_DIM);
        assert!(hdc.sparsity() > 0.0); // Should have some bits set
    }

    #[test]
    fn test_embedding_projection() {
        let integrator = MultiModalIntegrator::new();
        let embedding = create_test_embedding();

        let hdc = integrator.project_image_embedding(&embedding).unwrap();
        assert_eq!(hdc.dim(), HDC_DIM);
        assert!(hdc.sparsity() > 0.0);
    }

    #[test]
    fn test_ocr_projection() {
        let integrator = MultiModalIntegrator::new();
        let ocr = create_test_ocr();

        let hdc = integrator.project_ocr(&ocr).unwrap();
        assert_eq!(hdc.dim(), HDC_DIM);
        assert!(hdc.sparsity() > 0.0);
    }

    #[test]
    fn test_single_modality_fusion() {
        let integrator = MultiModalIntegrator::new();
        let features = create_test_visual_features();
        let visual_hdc = integrator.project_vision(&features).unwrap();

        let contribution = ModalityContribution {
            modality: ModalityType::Vision,
            hdc_projection: visual_hdc,
            confidence: 0.8,
            fusion_weight: 1.0,
        };

        let perception = integrator.fuse_modalities(vec![contribution]).unwrap();
        assert_eq!(perception.modalities.len(), 1);
        assert!(perception.confidence > 0.0);
    }

    #[test]
    fn test_multi_modality_fusion() {
        let integrator = MultiModalIntegrator::new();
        let features = create_test_visual_features();
        let ocr = create_test_ocr();

        let visual_hdc = integrator.project_vision(&features).unwrap();
        let ocr_hdc = integrator.project_ocr(&ocr).unwrap();

        let contributions = vec![
            ModalityContribution {
                modality: ModalityType::Vision,
                hdc_projection: visual_hdc,
                confidence: 0.8,
                fusion_weight: 1.0,
            },
            ModalityContribution {
                modality: ModalityType::Ocr,
                hdc_projection: ocr_hdc,
                confidence: 0.85,
                fusion_weight: 1.0,
            },
        ];

        let perception = integrator.fuse_modalities(contributions).unwrap();
        assert_eq!(perception.modalities.len(), 2);
        assert!(perception.confidence > 0.0);
        assert_eq!(perception.unified_concept.dim(), HDC_DIM);
    }

    #[test]
    fn test_perceive_image_all_modalities() {
        let integrator = MultiModalIntegrator::new();
        let features = create_test_visual_features();
        let embedding = create_test_embedding();
        let ocr = create_test_ocr();

        let perception = integrator
            .perceive_image(&features, Some(&embedding), Some(&ocr))
            .unwrap();

        // Should have 3 contributions: visual features, embedding, OCR
        assert_eq!(perception.modalities.len(), 3);
        assert!(perception.confidence > 0.0);
    }

    #[test]
    fn test_adaptive_weighting() {
        let mut integrator = MultiModalIntegrator::new();
        let features = create_test_visual_features();
        let visual_hdc = integrator.project_vision(&features).unwrap();

        // High confidence contribution
        let high_conf = ModalityContribution {
            modality: ModalityType::Vision,
            hdc_projection: visual_hdc.clone(),
            confidence: 0.9,
            fusion_weight: 1.0,
        };

        // Low confidence contribution
        let low_conf = ModalityContribution {
            modality: ModalityType::Ocr,
            hdc_projection: visual_hdc.clone(),
            confidence: 0.3,
            fusion_weight: 1.0,
        };

        // With adaptive weighting
        integrator.set_adaptive_weighting(true);
        let adaptive_perception = integrator
            .fuse_modalities(vec![high_conf.clone(), low_conf.clone()])
            .unwrap();

        // Without adaptive weighting
        integrator.set_adaptive_weighting(false);
        let non_adaptive_perception = integrator
            .fuse_modalities(vec![high_conf, low_conf])
            .unwrap();

        // Adaptive should give higher overall confidence
        assert!(adaptive_perception.confidence > non_adaptive_perception.confidence);
    }

    #[test]
    fn test_modality_type_names() {
        assert_eq!(ModalityType::Vision.name(), "Vision");
        assert_eq!(ModalityType::Voice.name(), "Voice");
        assert_eq!(ModalityType::Code.name(), "Code");
        assert_eq!(ModalityType::Ocr.name(), "OCR");
    }

    #[test]
    fn test_l2_normalize_basic() {
        let input = vec![3.0, 4.0];
        let normalized = l2_normalize(&input);
        // Should be unit vector: [0.6, 0.8] (3/5, 4/5)
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let input = vec![0.0, 0.0, 0.0];
        let normalized = l2_normalize(&input);
        // Should return zeros without NaN
        assert!(normalized.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_l2_normalize_near_zero() {
        let input = vec![1e-15, 1e-15];
        let normalized = l2_normalize(&input);
        // Should handle near-zero gracefully
        assert!(normalized.iter().all(|x| x.is_finite()));
    }

    // =========================================================================
    // HDC Compositional Operation Tests
    // =========================================================================

    #[test]
    fn test_hdc_random_quasi_orthogonal() {
        // Two random vectors should have ~0.5 similarity (quasi-orthogonal)
        let a = HdcVector::random(12345);
        let b = HdcVector::random(67890);
        let sim = a.similarity(&b);

        // Should be close to 0.5 (within statistical variance)
        assert!(
            sim > 0.45 && sim < 0.55,
            "Random vectors should be quasi-orthogonal, got similarity {}",
            sim
        );
    }

    #[test]
    fn test_hdc_random_sparsity() {
        // Random vector should have ~50% sparsity
        let v = HdcVector::random(42);
        let sparsity = v.sparsity();

        assert!(
            sparsity > 0.45 && sparsity < 0.55,
            "Random vector should have ~50% sparsity, got {}",
            sparsity
        );
    }

    #[test]
    fn test_hdc_bind_self_inverse() {
        // A ⊗ B ⊗ B = A (binding is self-inverse)
        let a = HdcVector::random(111);
        let b = HdcVector::random(222);

        let bound = a.bind(&b);
        let unbound = bound.unbind(&b);

        // Should recover original vector exactly
        assert_eq!(
            a.similarity(&unbound),
            1.0,
            "Unbinding should recover original vector"
        );
    }

    #[test]
    fn test_hdc_bind_dissimilar_to_inputs() {
        // Bound vector should be dissimilar to both inputs
        let a = HdcVector::random(333);
        let b = HdcVector::random(444);
        let bound = a.bind(&b);

        let sim_a = bound.similarity(&a);
        let sim_b = bound.similarity(&b);

        // Should be quasi-orthogonal to both inputs
        assert!(
            sim_a > 0.45 && sim_a < 0.55,
            "Bound vector should be orthogonal to first input, got {}",
            sim_a
        );
        assert!(
            sim_b > 0.45 && sim_b < 0.55,
            "Bound vector should be orthogonal to second input, got {}",
            sim_b
        );
    }

    #[test]
    fn test_hdc_bind_commutative() {
        // A ⊗ B = B ⊗ A
        let a = HdcVector::random(555);
        let b = HdcVector::random(666);

        let ab = a.bind(&b);
        let ba = b.bind(&a);

        assert_eq!(ab.similarity(&ba), 1.0, "Binding should be commutative");
    }

    #[test]
    fn test_hdc_permute_inverse() {
        // permute(n).inverse_permute(n) = identity
        let v = HdcVector::random(777);

        for shift in [1, 10, 100, 1000, HDC_DIM / 2] {
            let permuted = v.permute(shift);
            let restored = permuted.inverse_permute(shift);

            assert_eq!(
                v.similarity(&restored),
                1.0,
                "Inverse permute should restore original for shift {}",
                shift
            );
        }
    }

    #[test]
    fn test_hdc_permute_creates_orthogonal() {
        // Permuted vector should be quasi-orthogonal to original
        let v = HdcVector::random(888);

        // Even small shifts create quasi-orthogonal vectors in high dimensions
        let permuted = v.permute(100);
        let sim = v.similarity(&permuted);

        assert!(
            sim > 0.45 && sim < 0.55,
            "Permuted vector should be quasi-orthogonal, got {}",
            sim
        );
    }

    #[test]
    fn test_hdc_permute_zero_is_identity() {
        let v = HdcVector::random(999);
        let permuted = v.permute(0);
        assert_eq!(v.similarity(&permuted), 1.0);
    }

    #[test]
    fn test_hdc_bundle_similar_to_both() {
        // Bundled vector should be similar to both inputs
        let a = HdcVector::random(1010);
        let b = HdcVector::random(2020);
        let bundled = a.bundle(&b);

        let sim_a = bundled.similarity(&a);
        let sim_b = bundled.similarity(&b);

        // OR-bundle should be more similar to inputs than random
        assert!(
            sim_a > 0.6,
            "Bundle should be similar to first input, got {}",
            sim_a
        );
        assert!(
            sim_b > 0.6,
            "Bundle should be similar to second input, got {}",
            sim_b
        );
    }

    #[test]
    fn test_hdc_majority_bundle() {
        // Majority bundle should be similar to all inputs
        let a = HdcVector::random(3030);
        let b = HdcVector::random(4040);
        let c = HdcVector::random(5050);

        let bundled = HdcVector::majority_bundle(&[&a, &b, &c]);

        // Should be similar to all three (but not as high as with OR-bundle)
        assert!(
            bundled.similarity(&a) > 0.55,
            "Majority bundle should be similar to inputs"
        );
        assert!(bundled.similarity(&b) > 0.55);
        assert!(bundled.similarity(&c) > 0.55);

        // Sparsity should be close to 50% (majority voting preserves this)
        assert!(
            bundled.sparsity() > 0.4 && bundled.sparsity() < 0.6,
            "Majority bundle should preserve ~50% sparsity, got {}",
            bundled.sparsity()
        );
    }

    #[test]
    fn test_hdc_similarity_bounds() {
        let a = HdcVector::random(6060);
        let b = HdcVector::random(7070);

        // Self-similarity should be 1.0
        assert_eq!(a.similarity(&a), 1.0);

        // Inverted should have 0.0 similarity
        let inverted = a.invert();
        assert_eq!(a.similarity(&inverted), 0.0);

        // Random should be ~0.5
        let sim = a.similarity(&b);
        assert!(sim >= 0.0 && sim <= 1.0);
    }

    #[test]
    fn test_hdc_hamming_distance() {
        let a = HdcVector::random(8080);
        let b = a.clone();

        // Same vector has distance 0
        assert_eq!(a.hamming_distance(&b), 0);

        // Inverted has maximum distance
        let inverted = a.invert();
        assert_eq!(a.hamming_distance(&inverted), HDC_DIM);

        // Relationship: similarity = 1 - distance/dim
        let random = HdcVector::random(9090);
        let dist = a.hamming_distance(&random);
        let sim = a.similarity(&random);
        let expected_sim = 1.0 - (dist as f32 / HDC_DIM as f32);
        assert!((sim - expected_sim).abs() < 1e-6);
    }

    #[test]
    fn test_hdc_compositional_query() {
        // Demonstrate compositional reasoning:
        // If we bind "code" with "context_A", we can later query
        // "what was bound with context_A?" and recover "code"

        let code = HdcVector::random(1111);
        let context_a = HdcVector::random(2222);
        let context_b = HdcVector::random(3333);

        // Create bound representation: "code in context_A"
        let code_in_a = code.bind(&context_a);

        // Query: unbind context_A to get back code
        let recovered = code_in_a.unbind(&context_a);
        assert_eq!(
            code.similarity(&recovered),
            1.0,
            "Should recover code by unbinding context_A"
        );

        // Wrong query: unbind context_B gives garbage
        let wrong_query = code_in_a.unbind(&context_b);
        let sim = code.similarity(&wrong_query);
        assert!(
            sim > 0.45 && sim < 0.55,
            "Wrong context should give quasi-orthogonal result, got {}",
            sim
        );
    }

    #[test]
    fn test_hdc_sequence_encoding() {
        // Encode sequence: A, B, C using permutation
        let a = HdcVector::random(4444);
        let b = HdcVector::random(5555);
        let c = HdcVector::random(6666);

        // Position-encoded sequence
        let sequence = HdcVector::majority_bundle(&[&a.permute(0), &b.permute(1), &c.permute(2)]);

        // Query: "what's at position 0?" - inverse permute and check similarity
        let query_pos0 = sequence.inverse_permute(0);
        let query_pos1 = sequence.inverse_permute(1);
        let query_pos2 = sequence.inverse_permute(2);

        // Each query should be most similar to the correct element
        assert!(query_pos0.similarity(&a) > query_pos0.similarity(&b));
        assert!(query_pos0.similarity(&a) > query_pos0.similarity(&c));

        assert!(query_pos1.similarity(&b) > query_pos1.similarity(&a));
        assert!(query_pos1.similarity(&b) > query_pos1.similarity(&c));

        assert!(query_pos2.similarity(&c) > query_pos2.similarity(&a));
        assert!(query_pos2.similarity(&c) > query_pos2.similarity(&b));
    }

    #[test]
    fn test_hdc_associative_memory() {
        // Create a simple associative memory: store (key, value) pairs
        // Key and value are bound together
        let key1 = HdcVector::random(7777);
        let val1 = HdcVector::random(8888);
        let key2 = HdcVector::random(9999);
        let val2 = HdcVector::random(1234);

        // Memory is bundle of bound pairs
        let memory = HdcVector::majority_bundle(&[&key1.bind(&val1), &key2.bind(&val2)]);

        // Query memory with key1 - should retrieve val1
        let retrieved1 = memory.unbind(&key1);
        assert!(
            retrieved1.similarity(&val1) > retrieved1.similarity(&val2),
            "Query with key1 should be more similar to val1"
        );

        // Query memory with key2 - should retrieve val2
        let retrieved2 = memory.unbind(&key2);
        assert!(
            retrieved2.similarity(&val2) > retrieved2.similarity(&val1),
            "Query with key2 should be more similar to val2"
        );
    }
}

/// Property-based tests for JL distance preservation
///
/// These tests verify that the Johnson-Lindenstrauss random projection
/// approximately preserves Euclidean distances between input vectors.
#[cfg(test)]
mod proptest_jl {
    use super::*;
    use proptest::prelude::*;

    /// Calculate Euclidean distance between two f32 slices
    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    proptest! {
        // Reduce test cases for faster iteration (256 default is slow)
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(64))]

        /// Test JL distance preservation for 32D vision feature projector
        ///
        /// JL Lemma guarantees: |‖P(u) - P(v)‖ - ‖u - v‖| < ε‖u - v‖
        /// With sparse Rademacher and small input dim, variance is high.
        /// We use relaxed bounds to catch bugs (wrong distribution, broken scaling)
        /// rather than proving tight theoretical bounds.
        #[test]
        fn test_jl_vision_distance_preservation(
            vec_a in proptest::collection::vec(0.1f32..0.9f32, 32),
            vec_b in proptest::collection::vec(0.1f32..0.9f32, 32)
        ) {
            let projector = JLProjector::new(32, HDC_DIM, 0xC0DE_CAFE_1337_BEEF);

            // Calculate original distance (on normalized vectors, as we do in practice)
            let a_norm = l2_normalize(&vec_a);
            let b_norm = l2_normalize(&vec_b);
            let dist_original = euclidean_distance(&a_norm, &b_norm);

            // Skip if vectors are nearly identical (avoid division issues)
            if dist_original < 0.01 {
                return Ok(());
            }

            // Project
            let a_proj = projector.project(&a_norm);
            let b_proj = projector.project(&b_norm);
            let dist_projected = euclidean_distance(&a_proj, &b_proj);

            // Verify ratio is bounded (relaxed for sparse Rademacher variance)
            // These bounds catch broken projections while allowing statistical variance
            let ratio = dist_projected / dist_original;

            prop_assert!(
                ratio > 0.1 && ratio < 10.0,
                "Vision projector severely distorted: ratio={:.3}, original={:.3}, projected={:.3}",
                ratio, dist_original, dist_projected
            );

            // Also check projected distance is non-zero (projection working)
            prop_assert!(
                dist_projected > 1e-6,
                "Projected distance collapsed to zero"
            );
        }

        /// Test JL distance preservation for 150D code feature projector
        ///
        /// Larger input dimension should give better distance preservation
        #[test]
        fn test_jl_code_distance_preservation(
            vec_a in proptest::collection::vec(0.1f32..0.9f32, 150),
            vec_b in proptest::collection::vec(0.1f32..0.9f32, 150)
        ) {
            let projector = JLProjector::new(150, HDC_DIM, 0xA57_5EED_C0DE_CAFE);

            // Calculate original distance (on normalized vectors)
            let a_norm = l2_normalize(&vec_a);
            let b_norm = l2_normalize(&vec_b);
            let dist_original = euclidean_distance(&a_norm, &b_norm);

            // Skip if vectors are nearly identical
            if dist_original < 0.01 {
                return Ok(());
            }

            // Project
            let a_proj = projector.project(&a_norm);
            let b_proj = projector.project(&b_norm);
            let dist_projected = euclidean_distance(&a_proj, &b_proj);

            // Verify ratio is bounded (150D should be more stable than 32D)
            let ratio = dist_projected / dist_original;

            prop_assert!(
                ratio > 0.1 && ratio < 10.0,
                "Code projector severely distorted: ratio={:.3}, original={:.3}, projected={:.3}",
                ratio, dist_original, dist_projected
            );
        }

        /// Test that similar vectors stay similar after projection
        ///
        /// This verifies the key property: if ‖u - v‖ is small, then ‖P(u) - P(v)‖ is also small
        #[test]
        fn test_jl_similar_vectors_stay_similar(
            base in proptest::collection::vec(0.2f32..0.8f32, 64),
            perturbation in proptest::collection::vec(-0.05f32..0.05f32, 64)
        ) {
            let projector = JLProjector::new(64, HDC_DIM, 0xDEAD_BEEF_8765_4321);

            // Create similar vector by small perturbation
            let similar: Vec<f32> = base.iter()
                .zip(perturbation.iter())
                .map(|(b, p)| (b + p).clamp(0.1, 0.9))
                .collect();

            let base_norm = l2_normalize(&base);
            let similar_norm = l2_normalize(&similar);

            let dist_original = euclidean_distance(&base_norm, &similar_norm);

            // Skip if perturbation made them identical
            if dist_original < 0.001 {
                return Ok(());
            }

            let base_proj = projector.project(&base_norm);
            let similar_proj = projector.project(&similar_norm);
            let dist_projected = euclidean_distance(&base_proj, &similar_proj);

            // Similar vectors should not diverge extremely
            let ratio = dist_projected / dist_original;

            prop_assert!(
                ratio > 0.05 && ratio < 20.0,
                "Similar vectors diverged extremely: ratio={:.3}, original={:.3}, projected={:.3}",
                ratio, dist_original, dist_projected
            );
        }

        /// Test that orthogonal vectors remain distinguishable
        ///
        /// Orthogonal unit vectors have distance √2. After projection, they should
        /// remain sufficiently separated to be distinguishable.
        #[test]
        fn test_jl_orthogonal_vectors_stay_separated(
            dim in 32usize..=150usize,
            idx1 in 0usize..30usize,
            idx2 in 0usize..30usize
        ) {
            // Skip if same index (would create identical vectors)
            if idx1 == idx2 {
                return Ok(());
            }

            let projector = JLProjector::new(dim, HDC_DIM, 0xCAFE_BABE_1234_5678);

            // Create two orthogonal unit vectors (one-hot)
            let mut vec_a = vec![0.0f32; dim];
            let mut vec_b = vec![0.0f32; dim];
            vec_a[idx1 % dim] = 1.0;
            vec_b[idx2 % dim] = 1.0;

            // Original distance is √2 for orthogonal unit vectors
            let dist_original = euclidean_distance(&vec_a, &vec_b);

            let a_proj = projector.project(&vec_a);
            let b_proj = projector.project(&vec_b);
            let dist_projected = euclidean_distance(&a_proj, &b_proj);

            // Key check: projected distance should be non-zero (vectors distinguishable)
            prop_assert!(
                dist_projected > 0.001,
                "Orthogonal vectors collapsed to same point: dist_projected={:.6}",
                dist_projected
            );

            // Also check ratio isn't extreme
            let ratio = dist_projected / dist_original;
            prop_assert!(
                ratio > 0.01 && ratio < 100.0,
                "Orthogonal vectors had extreme distortion: ratio={:.3}",
                ratio
            );
        }
    }

    /// Deterministic test to verify scaling factor correctness
    #[test]
    fn test_jl_scaling_factor() {
        // For sparse JL with sparsity=3 (1/3 non-zero), scale should be sqrt(3)
        let projector = JLProjector::new(64, 1000, 12345);
        assert!((projector.scale - 3.0f32.sqrt()).abs() < 1e-6);
    }

    /// Test that zero input produces zero output (with normalization handling)
    #[test]
    fn test_jl_zero_input_handling() {
        let projector = JLProjector::new(32, HDC_DIM, 12345);
        let zero_input = vec![0.0f32; 32];

        // Without normalization - should produce zeros
        let output = projector.project(&zero_input);
        assert!(output.iter().all(|&x| x == 0.0));

        // With normalization - l2_normalize returns zeros for zero input
        let normalized = l2_normalize(&zero_input);
        let output_normalized = projector.project(&normalized);
        assert!(output_normalized.iter().all(|&x| x == 0.0));
    }
}
