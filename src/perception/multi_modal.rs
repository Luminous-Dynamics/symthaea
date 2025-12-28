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

/// Multi-modal HDC vector type - Uses HDC_DIMENSION (16,384D) holographic vector
///
/// Note: This is a simplified boolean-based HDC implementation optimized for
/// multi-modal perception. For consciousness measurement, use `hdc::RealHV`.
/// For binary operations, use `hdc::HV16`. This type focuses on fast bundling
/// and projection for sensory fusion tasks.
#[derive(Debug, Clone)]
pub struct HdcVector {
    bits: Vec<bool>,
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
}

impl ModalityType {
    /// Get a human-readable name for this modality
    pub fn name(&self) -> &'static str {
        match self {
            ModalityType::Vision => "Vision",
            ModalityType::Voice => "Voice",
            ModalityType::Code => "Code",
            ModalityType::Ocr => "OCR",
        }
    }
}

/// Multi-modal integration system - the sensory fusion layer
pub struct MultiModalIntegrator {
    /// Default weights for each modality (can be adjusted dynamically)
    modality_weights: ModalityWeights,

    /// Whether to use adaptive weighting based on confidence
    adaptive_weighting: bool,
}

/// Weights for combining different modalities
#[derive(Debug, Clone)]
pub struct ModalityWeights {
    pub vision: f32,
    pub voice: f32,
    pub code: f32,
    pub ocr: f32,
}

impl Default for ModalityWeights {
    fn default() -> Self {
        Self {
            vision: 1.0,
            voice: 1.0,
            code: 1.0,
            ocr: 1.0,
        }
    }
}

impl Default for MultiModalIntegrator {
    fn default() -> Self {
        Self {
            modality_weights: ModalityWeights::default(),
            adaptive_weighting: true,
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
        Self {
            modality_weights: weights,
            adaptive_weighting: true,
        }
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

    /// Project image embedding into HDC space
    pub fn project_image_embedding(&self, embedding: &ImageEmbedding) -> Result<HdcVector> {
        // Map 768D SigLIP embedding to 10,000D HDC space
        // Strategy: Use random projection with consistent seed for determinism

        let mut hdc = HdcVector::zero();

        // TODO: Implement Johnson-Lindenstrauss random projection
        // For now, use a simple mapping
        for (i, &value) in embedding.vector.iter().enumerate() {
            if value > 0.0 {
                // Map positive values to corresponding HDC dimensions
                let base_idx = (i * 13) % HDC_DIM; // Prime number for better distribution
                hdc.set_bit(base_idx);
            }
        }

        Ok(hdc)
    }

    /// Project OCR text into HDC space
    pub fn project_ocr(&self, ocr: &OcrResult) -> Result<HdcVector> {
        // Text encoding strategy:
        // - Use n-gram encoding (character and word level)
        // - Position-aware encoding for spatial layout
        // - Confidence-weighted contributions

        let mut hdc = HdcVector::zero();

        // TODO: Implement proper text encoding
        // For now, simple character-based encoding
        for (i, ch) in ocr.text.chars().take(100).enumerate() {
            let char_idx = (ch as usize) % 256;
            let pos_idx = 768 + (i % 100); // Position encoding

            hdc.set_bit(char_idx);
            hdc.set_bit(pos_idx);
        }

        Ok(hdc)
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
