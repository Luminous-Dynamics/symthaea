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

// Use central HDC dimension constant
use crate::hdc::HDC_DIMENSION;
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
                    let sign = if (hash / sparsity as u64) % 2 == 0 { 1i8 } else { -1i8 };
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
    fn project(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim);

        let mut output = vec![0.0f32; self.output_dim];

        // Sparse matrix-vector multiplication
        for &(row, col, sign) in &self.sparse_entries {
            output[row] += (sign as f32) * input[col];
        }

        // Apply scaling
        for val in &mut output {
            *val *= self.scale / (self.input_dim as f32).sqrt();
        }

        output
    }

    /// Project to binary HDC (threshold at 0)
    fn project_to_binary(&self, input: &[f32]) -> Vec<bool> {
        self.project(input).into_iter().map(|v| v > 0.0).collect()
    }
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
    pub fn zero() -> Self {
        Self {
            bits: vec![false; HDC_DIM],
        }
    }

    pub fn set_bit(&mut self, index: usize) {
        if index < self.bits.len() {
            self.bits[index] = true;
        }
    }

    pub fn bundle(&self, other: &HdcVector) -> HdcVector {
        let mut result = HdcVector::zero();
        for i in 0..HDC_DIM {
            result.bits[i] = self.bits[i] || other.bits[i];
        }
        result
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
}
use super::{
    ImageEmbedding, OcrResult, VisualFeatures, RustCodeSemantics,
};

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

        Self {
            modality_weights: ModalityWeights::default(),
            adaptive_weighting: true,
            // 768D SigLIP embedding → 16,384D HDC
            image_projector: JLProjector::new(SIGLIP_DIM, HDC_DIM, IMAGE_PROJECTOR_SEED),
            // 1024D Qwen3 embedding → 16,384D HDC
            text_embedding_projector: JLProjector::new(QWEN3_DIM, HDC_DIM, TEXT_EMBEDDING_PROJECTOR_SEED),
            // 256D n-gram features → 16,384D HDC
            text_ngram_projector: JLProjector::new(256, HDC_DIM, TEXT_NGRAM_PROJECTOR_SEED),
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

    /// Project visual features into HDC space
    pub fn project_vision(&self, features: &VisualFeatures) -> Result<HdcVector> {
        // TODO: Actual projection using learned mapping
        // For now, create a placeholder HDC vector from visual features

        // Vision encoding strategy:
        // - Map color histogram to HDC dimensions 0-255
        // - Map edge patterns to dimensions 256-511
        // - Map texture features to dimensions 512-767
        // - Use binding and bundling for spatial relationships

        let mut hdc = HdcVector::zero();

        // Encode brightness (placeholder - will be actual feature mapping)
        let brightness_idx = (features.brightness * 255.0) as usize % 256;
        hdc.set_bit(brightness_idx);

        // Encode dominant colors if available
        if let Some(first_color) = features.dominant_colors.first() {
            let color_avg = ((first_color[0] as u32 + first_color[1] as u32 + first_color[2] as u32) / 3) as usize;
            hdc.set_bit(color_avg % 256);
        }

        // Encode edge density
        let edge_idx = 256 + ((features.edge_density * 255.0) as usize);
        hdc.set_bit(edge_idx);

        // More sophisticated encoding will come with actual training

        Ok(hdc)
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
            let hash = window.iter()
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
            let hash = window.iter()
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

    /// Project code semantics into HDC space
    pub fn project_code(&self, semantics: &RustCodeSemantics) -> Result<HdcVector> {
        // Code encoding strategy:
        // - Function signatures → structural patterns
        // - Control flow → execution paths
        // - Dependencies → relationship graph
        // - Complexity metrics → quality indicators

        let mut hdc = HdcVector::zero();

        // TODO: Implement AST-based encoding
        // For now, encode basic metrics

        // Encode function count
        let fn_idx = 1000 + (semantics.function_count.min(100));
        hdc.set_bit(fn_idx);

        // Encode complexity
        // Use function count as a proxy for complexity
        let complexity_idx = 1100 + (semantics.function_count.min(100));
        hdc.set_bit(complexity_idx);

        Ok(hdc)
    }

    /// Fuse multiple modality contributions into a unified perception
    pub fn fuse_modalities(&self, contributions: Vec<ModalityContribution>) -> Result<MultiModalPerception> {
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
            method: crate::perception::ocr::OcrMethod::RustOcr,
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

        let perception = integrator.perceive_image(
            &features,
            Some(&embedding),
            Some(&ocr),
        ).unwrap();

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
        let adaptive_perception = integrator.fuse_modalities(vec![high_conf.clone(), low_conf.clone()]).unwrap();

        // Without adaptive weighting
        integrator.set_adaptive_weighting(false);
        let non_adaptive_perception = integrator.fuse_modalities(vec![high_conf, low_conf]).unwrap();

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
}
