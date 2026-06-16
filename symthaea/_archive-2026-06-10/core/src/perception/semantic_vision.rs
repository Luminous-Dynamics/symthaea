// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Semantic Vision: Visual Understanding and Processing
//!
//! Provides semantic understanding of visual input including
//! image embedding, captioning, and OCR capabilities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use symthaea_core::hdc::ContinuousHV;

/// Configuration for semantic vision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Feature extraction layers
    pub feature_layers: usize,
    /// OCR enabled
    pub ocr_enabled: bool,
    /// Caption generation enabled
    pub caption_enabled: bool,
    /// Attention resolution
    pub attention_resolution: (usize, usize),
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            feature_layers: 4,
            ocr_enabled: true,
            caption_enabled: true,
            attention_resolution: (14, 14),
        }
    }
}

/// Image embedding result
#[derive(Debug, Clone)]
pub struct ImageEmbedding {
    /// Embedding vector
    pub embedding: ContinuousHV,
    /// Spatial features (if available)
    pub spatial_features: Option<Vec<ContinuousHV>>,
    /// Global features
    pub global_features: Vec<f32>,
    /// Confidence
    pub confidence: f32,
    /// Processing time (ms)
    pub processing_time_ms: f64,
}

impl ImageEmbedding {
    /// Create a new image embedding
    pub fn new(embedding: ContinuousHV) -> Self {
        Self {
            embedding,
            spatial_features: None,
            global_features: Vec::new(),
            confidence: 1.0,
            processing_time_ms: 0.0,
        }
    }

    /// With spatial features
    pub fn with_spatial(mut self, features: Vec<ContinuousHV>) -> Self {
        self.spatial_features = Some(features);
        self
    }
}

/// Image caption result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageCaption {
    /// Generated caption text
    pub text: String,
    /// Confidence score
    pub confidence: f32,
    /// Alternative captions
    pub alternatives: Vec<(String, f32)>,
    /// Detected objects/concepts
    pub detected_concepts: Vec<String>,
    /// Timestamp
    pub timestamp: u64,
}

impl ImageCaption {
    /// Create a new caption
    pub fn new(text: impl Into<String>, confidence: f32) -> Self {
        Self {
            text: text.into(),
            confidence,
            alternatives: Vec::new(),
            detected_concepts: Vec::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Add an alternative caption
    pub fn with_alternative(mut self, text: impl Into<String>, confidence: f32) -> Self {
        self.alternatives.push((text.into(), confidence));
        self
    }

    /// Add a detected concept
    pub fn with_concept(mut self, concept: impl Into<String>) -> Self {
        self.detected_concepts.push(concept.into());
        self
    }
}

/// Visual features extracted from an image
#[derive(Debug, Clone)]
pub struct VisualFeatures {
    /// Edge features
    pub edges: Vec<f32>,
    /// Color histogram
    pub color_histogram: Vec<f32>,
    /// Texture features
    pub texture: Vec<f32>,
    /// Shape features
    pub shapes: Vec<f32>,
    /// Detected regions of interest
    pub roi: Vec<BoundingBox>,
    /// Feature map (spatial)
    pub feature_map: Option<Vec<Vec<f32>>>,
}

impl VisualFeatures {
    /// Create empty features
    pub fn empty() -> Self {
        Self {
            edges: Vec::new(),
            color_histogram: Vec::new(),
            texture: Vec::new(),
            shapes: Vec::new(),
            roi: Vec::new(),
            feature_map: None,
        }
    }

    /// Create from raw feature vectors
    pub fn from_vectors(edges: Vec<f32>, colors: Vec<f32>, texture: Vec<f32>) -> Self {
        Self {
            edges,
            color_histogram: colors,
            texture,
            shapes: Vec::new(),
            roi: Vec::new(),
            feature_map: None,
        }
    }
}

/// A bounding box for detected objects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    /// X coordinate (normalized 0-1)
    pub x: f32,
    /// Y coordinate (normalized 0-1)
    pub y: f32,
    /// Width (normalized 0-1)
    pub width: f32,
    /// Height (normalized 0-1)
    pub height: f32,
    /// Label/class
    pub label: String,
    /// Confidence
    pub confidence: f32,
}

/// OCR (Optical Character Recognition) system
#[derive(Debug)]
pub struct OcrSystem {
    /// Configuration
    dimension: usize,
    /// Character embeddings
    char_embeddings: HashMap<char, ContinuousHV>,
    /// Statistics
    stats: OcrStats,
    /// Optional genesis-seeded RNG for deterministic confidence jitter
    seeded_rng: Option<symthaea_core::genesis::ShakeRng>,
}

/// OCR statistics
#[derive(Debug, Clone, Default)]
pub struct OcrStats {
    /// Total recognitions
    pub recognitions: u64,
    /// Characters recognized
    pub characters_recognized: u64,
    /// Average confidence
    pub avg_confidence: f32,
}

/// OCR result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    /// Recognized text
    pub text: String,
    /// Confidence per character
    pub char_confidences: Vec<f32>,
    /// Overall confidence
    pub confidence: f32,
    /// Bounding boxes for text regions
    pub text_regions: Vec<BoundingBox>,
}

impl OcrSystem {
    /// Create a new OCR system
    pub fn new(dimension: usize) -> Self {
        // Initialize character embeddings
        let mut char_embeddings = HashMap::new();
        for (i, c) in ('a'..='z').chain('A'..='Z').chain('0'..='9').enumerate() {
            char_embeddings.insert(c, ContinuousHV::random(dimension, i as u64));
        }
        // Add common punctuation
        for (i, c) in [' ', '.', ',', '!', '?', '-', ':', ';', '\'', '"']
            .iter()
            .enumerate()
        {
            char_embeddings.insert(*c, ContinuousHV::random(dimension, (1000 + i) as u64));
        }

        Self {
            dimension,
            char_embeddings,
            stats: OcrStats::default(),
            seeded_rng: None,
        }
    }

    /// Create an OCR system with deterministic RNG from a genesis seed.
    pub fn from_genesis(
        dimension: usize,
        genesis: &symthaea_core::genesis::GenesisSeed,
        label: &str,
    ) -> Self {
        let mut sys = Self::new(dimension);
        sys.seeded_rng = Some(genesis.domain(&format!("{label}::ocr")));
        sys
    }

    /// Recognize text from image features
    pub fn recognize(&mut self, visual_features: &VisualFeatures) -> OcrResult {
        self.stats.recognitions += 1;

        // Simplified OCR simulation (would use real model in production)
        // Extract text-like features
        let text = self.extract_text_from_features(visual_features);

        let char_confidences: Vec<f32> = text
            .chars()
            .map(|_| {
                let jitter: f32 = if let Some(ref mut rng) = self.seeded_rng {
                    rand::Rng::r#gen(rng)
                } else {
                    rand::random::<f32>()
                };
                0.9 + jitter * 0.1
            })
            .collect();

        let confidence = if char_confidences.is_empty() {
            0.0
        } else {
            char_confidences.iter().sum::<f32>() / char_confidences.len() as f32
        };

        self.stats.characters_recognized += text.len() as u64;
        let n = self.stats.recognitions as f32;
        self.stats.avg_confidence = (self.stats.avg_confidence * (n - 1.0) + confidence) / n;

        OcrResult {
            text,
            char_confidences,
            confidence,
            text_regions: Vec::new(),
        }
    }

    /// Extract text from visual features (simplified)
    fn extract_text_from_features(&self, _features: &VisualFeatures) -> String {
        // Placeholder - would use real OCR model
        String::new()
    }

    /// Get character embedding
    pub fn char_embedding(&self, c: char) -> Option<&ContinuousHV> {
        self.char_embeddings.get(&c)
    }

    /// Get statistics
    pub fn stats(&self) -> &OcrStats {
        &self.stats
    }
}

impl Default for OcrSystem {
    fn default() -> Self {
        Self::new(512)
    }
}

/// The semantic vision system
#[derive(Debug)]
pub struct SemanticVision {
    /// Configuration
    config: VisionConfig,
    /// OCR system
    ocr: OcrSystem,
    /// Visual concept embeddings
    concept_embeddings: HashMap<String, ContinuousHV>,
    /// Statistics
    stats: VisionStats,
}

/// Vision statistics
#[derive(Debug, Clone, Default)]
pub struct VisionStats {
    /// Total images processed
    pub images_processed: u64,
    /// Captions generated
    pub captions_generated: u64,
    /// OCR operations
    pub ocr_operations: u64,
}

impl SemanticVision {
    /// Create a new semantic vision system
    pub fn new(config: VisionConfig) -> Self {
        let dim = config.dimension;

        // Initialize common concept embeddings
        let mut concept_embeddings = HashMap::new();
        for (i, concept) in [
            "person", "animal", "vehicle", "building", "nature", "object", "text",
        ]
        .iter()
        .enumerate()
        {
            concept_embeddings.insert(
                concept.to_string(),
                ContinuousHV::random(dim, (2000 + i) as u64),
            );
        }

        Self {
            ocr: OcrSystem::new(dim),
            config,
            concept_embeddings,
            stats: VisionStats::default(),
        }
    }

    /// Create a semantic vision system with deterministic RNG from a genesis seed.
    pub fn from_genesis(
        config: VisionConfig,
        genesis: &symthaea_core::genesis::GenesisSeed,
        label: &str,
    ) -> Self {
        let dim = config.dimension;

        let mut concept_embeddings = HashMap::new();
        let mut rng = genesis.domain(&format!("{label}::vision::concepts"));
        for concept in [
            "person", "animal", "vehicle", "building", "nature", "object", "text",
        ] {
            let seed: u64 = rand::Rng::r#gen(&mut rng);
            concept_embeddings.insert(concept.to_string(), ContinuousHV::random(dim, seed));
        }

        Self {
            ocr: OcrSystem::from_genesis(dim, genesis, label),
            config,
            concept_embeddings,
            stats: VisionStats::default(),
        }
    }

    /// Process an image and get embedding
    pub fn embed_image(&mut self, image_data: &[u8]) -> ImageEmbedding {
        self.stats.images_processed += 1;
        let start = std::time::Instant::now();

        // Extract features
        let features = self.extract_features(image_data);

        // Create embedding from features
        let embedding = self.features_to_embedding(&features);

        let processing_time = start.elapsed().as_secs_f64() * 1000.0;

        ImageEmbedding {
            embedding,
            spatial_features: None,
            global_features: features.color_histogram.clone(),
            confidence: 0.9,
            processing_time_ms: processing_time,
        }
    }

    /// Extract visual features from raw image data
    fn extract_features(&self, image_data: &[u8]) -> VisualFeatures {
        // Simplified feature extraction (would use real CV in production)
        let mut features = VisualFeatures::empty();

        // Simple "feature extraction" based on byte values
        let mut color_hist = vec![0.0f32; 256];
        for &byte in image_data {
            color_hist[byte as usize] += 1.0;
        }

        // Normalize
        let total: f32 = color_hist.iter().sum();
        if total > 0.0 {
            for v in color_hist.iter_mut() {
                *v /= total;
            }
        }

        features.color_histogram = color_hist;
        features
    }

    /// Convert features to embedding
    fn features_to_embedding(&self, features: &VisualFeatures) -> ContinuousHV {
        let mut values = vec![0.0f32; self.config.dimension];

        // Map color histogram to embedding
        for (i, &v) in features.color_histogram.iter().enumerate() {
            let idx = i % self.config.dimension;
            values[idx] += v;
        }

        // Normalize
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in values.iter_mut() {
                *v /= magnitude;
            }
        }

        ContinuousHV::from_slice(&values)
    }

    /// Generate caption for image
    pub fn caption_image(&mut self, image_data: &[u8]) -> ImageCaption {
        self.stats.captions_generated += 1;

        let embedding = self.embed_image(image_data);

        // Find most similar concepts
        let mut concept_scores: Vec<_> = self
            .concept_embeddings
            .iter()
            .map(|(name, emb)| (name.clone(), embedding.embedding.similarity(emb)))
            .collect();

        concept_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Generate caption from top concepts
        let top_concepts: Vec<_> = concept_scores
            .iter()
            .take(3)
            .map(|(n, _)| n.clone())
            .collect();
        let caption_text = format!("An image containing {}", top_concepts.join(", "));

        let confidence = concept_scores.first().map(|(_, s)| *s).unwrap_or(0.5);

        ImageCaption::new(caption_text, confidence)
            .with_concept(top_concepts.first().cloned().unwrap_or_default())
    }

    /// Perform OCR on image
    pub fn ocr(&mut self, image_data: &[u8]) -> OcrResult {
        self.stats.ocr_operations += 1;

        let features = self.extract_features(image_data);
        self.ocr.recognize(&features)
    }

    /// Get statistics
    pub fn stats(&self) -> &VisionStats {
        &self.stats
    }
}

impl Default for SemanticVision {
    fn default() -> Self {
        Self::new(VisionConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_creation() {
        let vision = SemanticVision::default();
        assert_eq!(vision.stats.images_processed, 0);
    }

    #[test]
    fn test_image_embedding() {
        let mut vision = SemanticVision::default();
        let image_data = vec![0u8; 1024];

        let embedding = vision.embed_image(&image_data);
        assert_eq!(embedding.embedding.dim(), 512);
    }

    #[test]
    fn test_captioning() {
        let mut vision = SemanticVision::default();
        let image_data = vec![128u8; 1024];

        let caption = vision.caption_image(&image_data);
        assert!(!caption.text.is_empty());
    }

    #[test]
    fn test_ocr() {
        let mut vision = SemanticVision::default();
        let image_data = vec![64u8; 1024];

        let result = vision.ocr(&image_data);
        assert!(result.confidence >= 0.0);
    }
}
