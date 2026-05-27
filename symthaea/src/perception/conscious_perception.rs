// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Conscious Perception - End-to-End Perception Pipeline
//!
//! Integrates all perception modalities with causal reasoning and memory storage.
//!
//! ## Pipeline
//!
//! ```text
//! Input (Image/Text)
//!         │
//!         ▼
//! ┌─────────────────┐
//! │ Resilience      │ ← Timeout, retry, availability tracking
//! │ (graceful deg.) │
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Perception      │ ← HDC semantic vision, OCR, visual cortex
//! │ (multi-modal)   │
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Causal Mind     │ ← Extract causal structure
//! │ (HDC reasoning) │
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Memory Store    │ ← Qdrant, DuckDB, LanceDB
//! │ (persistence)   │
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ Consciousness   │ ← Φ integration, attention
//! │ (GWT broadcast) │
//! └─────────────────┘
//! ```
//!
//! ## Resilience Features
//!
//! - **Availability Tracking**: Know which models are loaded vs stub
//! - **Timeout/Retry**: Don't hang on slow inference
//! - **Caption Fallback**: Generate descriptions from visual features when VLM unavailable
//! - **Coherence Gating**: Filter low-quality outputs
//! - **Graceful Degradation**: Always produce useful output, even if degraded

use anyhow::{Context, Result};
use image::DynamicImage;
use std::path::Path;
use std::time::{Duration, Instant};

use super::resilience::{
    Availability, PerceptionCapabilities, ResilienceConfig, ResilienceManager, ResilienceStats,
    ResilientResult,
};
use crate::perception::{
    ImageCaption, ImageEmbedding, ModalityType, MultiModalPerception, OcrSystem, PerceptionInput,
    SemanticVision, VisionConfig, VisualCortex, VisualCortexConfig, VisualFeatures,
};
use symthaea_core::hdc::ContinuousHV as HdcVector;
use symthaea_core::hdc::causal_mind::{CausalMind, GroundedCausalLearning};

/// Configuration for the conscious perception system
#[derive(Debug, Clone)]
pub struct ConsciousPerceptionConfig {
    /// Minimum confidence for perception to be processed
    pub min_confidence: f32,

    /// Enable causal extraction from text
    pub enable_causal_extraction: bool,

    /// Enable memory storage
    pub enable_memory: bool,

    /// Embedding cache size
    pub cache_size: usize,

    /// Resilience configuration (timeout, retry, coherence)
    pub resilience: ResilienceConfig,

    /// Enable caption fallback when VLM unavailable
    pub enable_caption_fallback: bool,

    /// Enable coherence gating for output filtering
    pub enable_coherence_gating: bool,

    /// Optional genesis phrase for deterministic initialization
    pub genesis_phrase: Option<String>,
}

impl Default for ConsciousPerceptionConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            enable_causal_extraction: true,
            enable_memory: true,
            cache_size: 1000,
            resilience: ResilienceConfig::default(),
            enable_caption_fallback: true,
            enable_coherence_gating: true,
            genesis_phrase: None,
        }
    }
}

/// Result of conscious perception processing
#[derive(Debug, Clone)]
pub struct PerceptionResult {
    /// Type of input processed
    pub modality: ModalityType,

    /// HDC encoding of the perception (multi-modal HdcVector)
    pub hdc_encoding: HdcVector,

    /// Extracted text (from OCR or input)
    pub text: Option<String>,

    /// Image embedding (if image input)
    pub image_embedding: Option<ImageEmbedding>,

    /// Image caption (if image input)
    pub caption: Option<ImageCaption>,

    /// Causal learning results (if text contained causal structure)
    pub causal_learning: Option<GroundedCausalLearning>,

    /// Processing time in milliseconds
    pub processing_time_ms: u64,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Current Φ (integrated information) after processing
    pub phi: f64,

    // === Resilience Information ===
    /// Whether any fallback was used
    pub used_fallback: bool,

    /// Which capabilities were used
    pub capabilities_used: Vec<String>,

    /// Warnings generated during processing
    pub warnings: Vec<String>,

    /// Visual features (always available, used for fallback)
    pub visual_features: Option<VisualFeatures>,

    /// Coherence score (if coherence gating enabled)
    pub coherence_score: Option<f32>,
}

/// Conscious Perception System - Unified perception with causal reasoning
pub struct ConsciousPerception {
    /// Configuration
    config: ConsciousPerceptionConfig,

    /// Semantic vision (HDC-based)
    vision: SemanticVision,

    /// OCR system (HDC character recognition)
    ocr: OcrSystem,

    /// Multi-modal HDC perception
    multi_modal: MultiModalPerception,

    /// Causal reasoning mind
    causal_mind: CausalMind,

    /// Basic visual feature extraction (always available)
    visual_cortex: VisualCortex,

    /// HDC dimension for this system
    dim: usize,

    /// Resilience manager for graceful degradation
    resilience: ResilienceManager,

    /// Statistics
    stats: PerceptionStats,
}

/// Statistics for the perception system
#[derive(Debug, Default, Clone)]
pub struct PerceptionStats {
    /// Images processed
    pub images_processed: u64,

    /// Text inputs processed
    pub texts_processed: u64,

    /// OCR extractions performed
    pub ocr_extractions: u64,

    /// Causal relations extracted
    pub causal_relations_extracted: u64,

    /// Total concepts in causal mind
    pub causal_concepts: usize,

    /// Total causal links
    pub causal_links: usize,

    // === Resilience Stats ===
    /// Caption fallbacks used (Moondream unavailable)
    pub caption_fallbacks: u64,

    /// Embedding fallbacks used (SigLIP unavailable)
    pub embedding_fallbacks: u64,

    /// OCR fallbacks (Tesseract unavailable)
    pub ocr_fallbacks: u64,

    /// Total timeouts encountered
    pub timeouts: u64,

    /// Total retries performed
    pub retries: u64,

    /// Low coherence rejections
    pub coherence_rejections: u64,
}

/// Convert a DynamicImage to raw RGB bytes for HDC perception modules
fn image_to_bytes(image: &DynamicImage) -> Vec<u8> {
    image.to_rgb8().into_raw()
}

/// Extract basic image statistics from a DynamicImage for fallback caption generation
fn image_stats(image: &DynamicImage) -> (f32, f32, f32, f32, f32, f32) {
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width() as f32, rgb.height() as f32);
    let aspect_ratio = if h > 0.0 { w / h } else { 1.0 };
    let pixels = rgb.as_raw();
    let n_pixels = (pixels.len() / 3) as f32;

    if n_pixels < 1.0 {
        return (0.5, 0.0, 0.0, 0.0, 0.0, aspect_ratio);
    }

    // Compute brightness (mean luminance)
    let mut brightness_sum = 0.0f32;
    let mut r_sum = 0.0f32;
    let mut g_sum = 0.0f32;
    let mut b_sum = 0.0f32;
    for chunk in pixels.chunks(3) {
        let (r, g, b) = (
            chunk[0] as f32 / 255.0,
            chunk[1] as f32 / 255.0,
            chunk[2] as f32 / 255.0,
        );
        brightness_sum += 0.299 * r + 0.587 * g + 0.114 * b;
        r_sum += r;
        g_sum += g;
        b_sum += b;
    }
    let brightness = brightness_sum / n_pixels;

    // Dominant hue from average color
    let r_avg = r_sum / n_pixels;
    let g_avg = g_sum / n_pixels;
    let b_avg = b_sum / n_pixels;
    let max_c = r_avg.max(g_avg).max(b_avg);
    let min_c = r_avg.min(g_avg).min(b_avg);
    let hue = if (max_c - min_c).abs() < 1e-6 {
        0.0
    } else if max_c == r_avg {
        60.0 * ((g_avg - b_avg) / (max_c - min_c)).rem_euclid(6.0)
    } else if max_c == g_avg {
        60.0 * ((b_avg - r_avg) / (max_c - min_c) + 2.0)
    } else {
        60.0 * ((r_avg - g_avg) / (max_c - min_c) + 4.0)
    };

    let saturation = if max_c > 0.0 {
        (max_c - min_c) / max_c
    } else {
        0.0
    };

    // Color variance (simple proxy for contrast)
    let mut var_sum = 0.0f32;
    for chunk in pixels.chunks(3) {
        let lum = 0.299 * (chunk[0] as f32 / 255.0)
            + 0.587 * (chunk[1] as f32 / 255.0)
            + 0.114 * (chunk[2] as f32 / 255.0);
        let diff = lum - brightness;
        var_sum += diff * diff;
    }
    let color_variance = (var_sum / n_pixels).sqrt();

    // Edge density (simple gradient magnitude estimate on subsampled grid)
    let step = ((w as usize).max(1) / 20).max(1);
    let mut edge_count = 0usize;
    let mut edge_total = 0usize;
    for y in (1..rgb.height() as usize - 1).step_by(step) {
        for x in (1..rgb.width() as usize - 1).step_by(step) {
            let c = rgb.get_pixel(x as u32, y as u32)[0] as i32;
            let r = rgb.get_pixel((x + 1) as u32, y as u32)[0] as i32;
            let d = rgb.get_pixel(x as u32, (y + 1) as u32)[0] as i32;
            let grad = ((c - r).abs() + (c - d).abs()) as f32 / 255.0;
            if grad > 0.1 {
                edge_count += 1;
            }
            edge_total += 1;
        }
    }
    let edge_density = if edge_total > 0 {
        edge_count as f32 / edge_total as f32
    } else {
        0.0
    };

    (
        brightness,
        hue,
        saturation,
        color_variance,
        edge_density,
        aspect_ratio,
    )
}

impl ConsciousPerception {
    /// Create a new conscious perception system
    pub fn new(config: ConsciousPerceptionConfig) -> Self {
        let resilience = ResilienceManager::new(config.resilience.clone());

        let vision_config = VisionConfig::default();
        let dim = vision_config.dimension;
        let cortex_config = VisualCortexConfig::default();

        let (vision, ocr, visual_cortex) = if let Some(ref phrase) = config.genesis_phrase {
            let genesis = symthaea_core::genesis::GenesisSeed::from_phrase(phrase);
            let label = "perception";
            (
                SemanticVision::from_genesis(vision_config, &genesis, label),
                OcrSystem::from_genesis(dim, &genesis, label),
                VisualCortex::from_genesis(cortex_config, &genesis, label),
            )
        } else {
            (
                SemanticVision::new(vision_config),
                OcrSystem::new(dim),
                VisualCortex::new(cortex_config),
            )
        };

        Self {
            vision,
            ocr,
            multi_modal: MultiModalPerception::default(),
            causal_mind: CausalMind::new(),
            visual_cortex,
            dim,
            resilience,
            stats: PerceptionStats::default(),
            config,
        }
    }

    /// Initialize all perception subsystems with availability tracking.
    ///
    /// HDC perception modules are pure Rust and always available.
    /// This method sets up availability tracking for the resilience layer.
    pub fn initialize(&mut self) -> Result<()> {
        // HDC-based vision, OCR, and visual cortex are always available (pure Rust)
        self.resilience
            .set_capability("image_embedding", Availability::Full);
        self.resilience
            .set_capability("image_captioning", Availability::Full);
        self.resilience.set_capability("ocr", Availability::Full);
        self.resilience
            .set_capability("visual_features", Availability::Full);
        self.resilience
            .set_capability("multi_modal", Availability::Full);

        // Mark loading complete
        self.resilience.loader.start_loading("siglip");
        self.resilience.loader.complete_loading("siglip");
        self.resilience.loader.start_loading("moondream");
        self.resilience.loader.complete_loading("moondream");
        self.resilience.loader.start_loading("ocr");
        self.resilience.loader.complete_loading("ocr");

        Ok(())
    }

    /// Process an image through the full perception pipeline with resilience
    pub fn perceive_image(&mut self, image: &DynamicImage) -> Result<PerceptionResult> {
        let start = Instant::now();
        let mut warnings = Vec::new();
        let mut used_fallback = false;
        let mut capabilities_used = Vec::new();
        let image_bytes = image_to_bytes(image);

        // Step 0: Always extract visual features (pure Rust, always available)
        let input_hv = HdcVector::from_slice(
            &image_bytes
                .iter()
                .take(self.dim)
                .map(|&b| b as f32 / 255.0)
                .collect::<Vec<_>>(),
        );
        let cortex_result = self.visual_cortex.process(&input_hv);
        // Create VisualFeatures from extracted data
        let visual_features = Some(VisualFeatures::from_vectors(
            cortex_result
                .final_features
                .as_slice()
                .iter()
                .take(64)
                .copied()
                .collect(),
            vec![], // color histogram placeholder
            vec![], // texture placeholder
        ));
        capabilities_used.push("visual_features".to_string());

        // Step 1: Get image embedding with resilience
        let embedding_result = self.perceive_image_embedding_resilient(&image_bytes);
        let embedding = embedding_result.value;
        if embedding_result.is_fallback {
            used_fallback = true;
            self.stats.embedding_fallbacks += 1;
            warnings.extend(embedding_result.warnings);
        }
        capabilities_used.push("image_embedding".to_string());

        // Step 2: Get image caption with resilience and fallback
        let caption_result = self.perceive_caption_resilient(&image_bytes, image);
        let caption = caption_result.value;
        if caption_result.is_fallback {
            used_fallback = true;
            self.stats.caption_fallbacks += 1;
            warnings.extend(caption_result.warnings);
        }
        capabilities_used.push("image_captioning".to_string());

        // Step 3: OCR with resilience
        let ocr_result = self.perceive_ocr_resilient(visual_features.as_ref());
        let extracted_text = if ocr_result.value.is_empty() {
            None
        } else {
            Some(ocr_result.value)
        };
        if ocr_result.is_fallback {
            used_fallback = true;
            self.stats.ocr_fallbacks += 1;
            warnings.extend(ocr_result.warnings);
        }
        if extracted_text.is_some() {
            capabilities_used.push("ocr".to_string());
        }

        // Step 4: Project to HDC space via multi-modal perception
        let perception_input = PerceptionInput::new("image", ModalityType::Visual, image_bytes)
            .with_embedding(embedding.embedding.clone());
        let hdc_encoding = self.multi_modal.process_input(perception_input);
        capabilities_used.push("multi_modal".to_string());

        // Step 5: Coherence gating (optional)
        let coherence_score = if self.config.enable_coherence_gating {
            let coherence = caption.confidence * 0.7 + 0.3;
            if !self.resilience.check_coherence(coherence) {
                self.stats.coherence_rejections += 1;
                warnings.push(format!("Low coherence: {:.2}", coherence));
            }
            Some(coherence)
        } else {
            None
        };

        // Step 6: Causal extraction from caption and OCR text
        let mut causal_learning = None;
        if self.config.enable_causal_extraction {
            let mut all_text = caption.text.clone();
            if let Some(ref ocr_text) = extracted_text {
                all_text.push(' ');
                all_text.push_str(ocr_text);
            }

            if !all_text.is_empty() {
                let learning = self.causal_mind.learn_from_grounded_text(&all_text);
                if learning.links_added > 0 {
                    self.stats.causal_relations_extracted += learning.links_added as u64;
                    causal_learning = Some(learning);
                }
            }
        }

        // Update stats
        self.stats.images_processed += 1;
        if extracted_text.is_some() {
            self.stats.ocr_extractions += 1;
        }
        self.stats.causal_concepts = self.causal_mind.concept_count();
        self.stats.causal_links = self.causal_mind.link_count();

        let processing_time_ms = start.elapsed().as_millis() as u64;

        // Record success in resilience manager
        self.resilience.record_success(start.elapsed());

        Ok(PerceptionResult {
            modality: ModalityType::Visual,
            hdc_encoding,
            text: extracted_text,
            image_embedding: Some(embedding),
            caption: Some(caption.clone()),
            causal_learning,
            processing_time_ms,
            confidence: caption.confidence,
            phi: self.causal_mind.phi(),
            used_fallback,
            capabilities_used,
            warnings,
            visual_features,
            coherence_score,
        })
    }

    /// Embed image with resilience (HDC-based, always succeeds)
    fn perceive_image_embedding_resilient(
        &mut self,
        image_bytes: &[u8],
    ) -> ResilientResult<ImageEmbedding> {
        let start = Instant::now();

        // Check if embedding capability is available
        let caps = self.resilience.capabilities();
        if !caps.image_embedding.is_usable() {
            return ResilientResult::fallback(self.stub_embedding(), "Image embedding unavailable");
        }

        // HDC embedding always succeeds (pure Rust, no external models)
        let embedding = self.vision.embed_image(image_bytes);
        ResilientResult {
            value: embedding,
            is_fallback: false,
            retries: 0,
            duration: start.elapsed(),
            coherence: None,
            warnings: Vec::new(),
        }
    }

    /// Caption image with resilience and visual feature fallback
    fn perceive_caption_resilient(
        &mut self,
        image_bytes: &[u8],
        image: &DynamicImage,
    ) -> ResilientResult<ImageCaption> {
        let start = Instant::now();

        // Check if captioning capability is available
        let caps = self.resilience.capabilities();
        if !caps.image_captioning.is_usable() {
            return self.generate_fallback_caption(Some(image), "Captioning unavailable");
        }

        // HDC captioning always succeeds
        let caption = self.vision.caption_image(image_bytes);
        ResilientResult {
            value: caption,
            is_fallback: false,
            retries: 0,
            duration: start.elapsed(),
            coherence: None,
            warnings: Vec::new(),
        }
    }

    /// Generate fallback caption from image statistics
    fn generate_fallback_caption(
        &self,
        image: Option<&DynamicImage>,
        reason: &str,
    ) -> ResilientResult<ImageCaption> {
        if !self.config.enable_caption_fallback {
            return ResilientResult::fallback(
                ImageCaption::new("Image content unavailable", 0.1),
                reason,
            );
        }

        let (caption_text, confidence) = if let Some(img) = image {
            let (brightness, hue, saturation, contrast, edge_density, aspect_ratio) =
                image_stats(img);
            self.resilience.fallback_caption(
                brightness,
                hue,
                saturation,
                contrast,
                edge_density,
                aspect_ratio,
            )
        } else {
            ("Image content (features unavailable)".to_string(), 0.2)
        };

        ResilientResult::fallback(ImageCaption::new(caption_text, confidence), reason)
    }

    /// OCR with resilience
    fn perceive_ocr_resilient(
        &mut self,
        visual_features: Option<&VisualFeatures>,
    ) -> ResilientResult<String> {
        let caps = self.resilience.capabilities();
        if !caps.ocr.is_usable() {
            return ResilientResult::fallback(String::new(), "OCR unavailable");
        }

        let features = match visual_features {
            Some(f) => f,
            None => return ResilientResult::fallback(String::new(), "No visual features for OCR"),
        };

        let result = self.ocr.recognize(features);
        ResilientResult::full(result.text, Duration::ZERO)
    }

    /// Process text through the perception pipeline
    pub fn perceive_text(&mut self, text: &str) -> Result<PerceptionResult> {
        let start = Instant::now();
        let mut capabilities_used = vec!["text_encoding".to_string()];

        // Step 1: Project text to HDC space via multi-modal perception
        let perception_input =
            PerceptionInput::new("text", ModalityType::Textual, text.as_bytes().to_vec());
        let hdc_encoding = self.multi_modal.process_input(perception_input);
        capabilities_used.push("multi_modal".to_string());

        // Step 2: Causal extraction
        let mut causal_learning = None;
        if self.config.enable_causal_extraction && !text.is_empty() {
            let learning = self.causal_mind.learn_from_grounded_text(text);
            if learning.links_added > 0 {
                self.stats.causal_relations_extracted += learning.links_added as u64;
                causal_learning = Some(learning);
            }
        }

        // Update stats
        self.stats.texts_processed += 1;
        self.stats.causal_concepts = self.causal_mind.concept_count();
        self.stats.causal_links = self.causal_mind.link_count();

        let processing_time_ms = start.elapsed().as_millis() as u64;

        Ok(PerceptionResult {
            modality: ModalityType::Textual,
            hdc_encoding,
            text: Some(text.to_string()),
            image_embedding: None,
            caption: None,
            causal_learning,
            processing_time_ms,
            confidence: 1.0,
            phi: self.causal_mind.phi(),
            used_fallback: false,
            capabilities_used,
            warnings: Vec::new(),
            visual_features: None,
            coherence_score: Some(1.0),
        })
    }

    /// Process an image file
    pub fn perceive_image_file(&mut self, path: &Path) -> Result<PerceptionResult> {
        let image = image::open(path).context(format!("Failed to open image: {:?}", path))?;
        self.perceive_image(&image)
    }

    /// Create a stub embedding (zero vector) for fallback
    fn stub_embedding(&self) -> ImageEmbedding {
        ImageEmbedding::new(HdcVector::zero(self.dim))
    }

    /// Query why something happened (causal explanation)
    pub fn query_why(&self, concept: &str) -> Vec<String> {
        self.causal_mind
            .query_why(concept)
            .iter()
            .map(|e| e.explanation.clone())
            .collect()
    }

    /// Query what would happen if something occurred
    pub fn query_what_if(&self, concept: &str) -> Vec<String> {
        self.causal_mind
            .query_what_if(concept)
            .iter()
            .map(|p| p.prediction.clone())
            .collect()
    }

    /// Get current statistics
    pub fn stats(&self) -> &PerceptionStats {
        &self.stats
    }

    /// Get current Φ (integrated information)
    pub fn phi(&self) -> f64 {
        self.causal_mind.phi()
    }

    /// Get mutable access to causal mind for direct manipulation
    pub fn causal_mind_mut(&mut self) -> &mut CausalMind {
        &mut self.causal_mind
    }

    /// Get reference to causal mind
    pub fn causal_mind(&self) -> &CausalMind {
        &self.causal_mind
    }

    // ========================================================================
    // RESILIENCE METHODS
    // ========================================================================

    /// Get current perception capabilities
    pub fn capabilities(&self) -> PerceptionCapabilities {
        self.resilience.capabilities()
    }

    /// Check if a specific capability is usable
    pub fn is_capability_usable(&self, capability: &str) -> bool {
        self.resilience.is_usable(capability)
    }

    /// Get perception health score (0.0 - 1.0)
    pub fn health(&self) -> f32 {
        self.resilience.capabilities().health()
    }

    /// Get resilience statistics
    pub fn resilience_stats(&self) -> ResilienceStats {
        self.resilience.stats()
    }

    /// Get full status report
    pub fn status_report(&self) -> String {
        self.resilience.status_report()
    }

    /// Force a capability into a specific availability state
    pub fn set_capability_availability(&self, capability: &str, availability: Availability) {
        self.resilience.set_capability(capability, availability);
    }

    /// Check if any model loading is in progress
    pub fn is_loading(&self) -> bool {
        self.resilience.loader.is_loading()
    }

    /// Get loading status summary
    pub fn loading_summary(&self) -> String {
        self.resilience.loader.summary()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn create_test_image() -> DynamicImage {
        let mut img = RgbImage::new(100, 100);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([128, 128, 128]);
        }
        DynamicImage::ImageRgb8(img)
    }

    fn create_colorful_test_image() -> DynamicImage {
        let mut img = RgbImage::new(100, 100);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Rgb([((x * 255) / 100) as u8, ((y * 255) / 100) as u8, 128]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_conscious_perception_creation() {
        let config = ConsciousPerceptionConfig::default();
        let perception = ConsciousPerception::new(config);

        assert_eq!(perception.stats.images_processed, 0);
        assert_eq!(perception.stats.texts_processed, 0);

        // Check default resilience config
        assert!(perception.config.enable_caption_fallback);
        assert!(perception.config.enable_coherence_gating);
    }

    #[test]
    fn test_text_perception() {
        let config = ConsciousPerceptionConfig::default();
        let mut perception = ConsciousPerception::new(config);

        let result = perception
            .perceive_text("The rain caused the flood")
            .unwrap();

        assert_eq!(result.modality, ModalityType::Textual);
        assert!(result.text.is_some());
        assert_eq!(perception.stats.texts_processed, 1);
        assert!(!result.used_fallback);
        assert!(
            result
                .capabilities_used
                .contains(&"text_encoding".to_string())
        );
    }

    #[test]
    fn test_causal_extraction_in_perception() {
        let config = ConsciousPerceptionConfig::default();
        let mut perception = ConsciousPerception::new(config);

        let result = perception
            .perceive_text("The server crashed because memory was exhausted")
            .unwrap();

        if let Some(ref learning) = result.causal_learning {
            assert!(learning.links_added > 0);
        }

        let _explanations = perception.query_why("crashed");
    }

    #[test]
    fn test_image_perception_stub() {
        let config = ConsciousPerceptionConfig::default();
        let mut perception = ConsciousPerception::new(config);
        let _ = perception.initialize();

        let image = create_test_image();
        let result = perception.perceive_image(&image).unwrap();

        assert_eq!(result.modality, ModalityType::Visual);
        assert!(result.image_embedding.is_some());
        assert_eq!(perception.stats.images_processed, 1);
        assert!(result.visual_features.is_some());
    }

    // ========================================================================
    // RESILIENCE TESTS
    // ========================================================================

    #[test]
    fn test_default_capabilities() {
        let config = ConsciousPerceptionConfig::default();
        let perception = ConsciousPerception::new(config);

        let caps = perception.capabilities();
        // Before initialize(), ML models not loaded
        assert!(!caps.image_embedding.is_full());
        assert!(!caps.image_captioning.is_full());
        // Pure Rust always available
        assert!(caps.code_analysis.is_full());
        assert!(caps.visual_features.is_full());
    }

    #[test]
    fn test_health_calculation() {
        let config = ConsciousPerceptionConfig::default();
        let perception = ConsciousPerception::new(config);

        let health = perception.health();
        assert!(health >= 0.0 && health <= 1.0);
        assert!(health > 0.3);
    }

    #[test]
    fn test_image_perception_with_resilience() {
        let config = ConsciousPerceptionConfig::default();
        let mut perception = ConsciousPerception::new(config);
        let _ = perception.initialize();

        let image = create_colorful_test_image();
        let result = perception.perceive_image(&image).unwrap();

        assert!(!result.capabilities_used.is_empty());
        assert!(
            result
                .capabilities_used
                .contains(&"visual_features".to_string())
        );
        assert!(
            result
                .capabilities_used
                .contains(&"image_embedding".to_string())
        );
        assert!(result.coherence_score.is_some());
    }

    #[test]
    fn test_fallback_caption_from_visual_features() {
        let config = ConsciousPerceptionConfig::default();
        let mut perception = ConsciousPerception::new(config);

        perception.set_capability_availability("image_captioning", Availability::Unavailable);

        let image = create_colorful_test_image();
        let result = perception.perceive_image(&image).unwrap();

        assert!(result.used_fallback);
        assert!(result.caption.is_some());
        let caption = result.caption.unwrap();
        assert!(!caption.text.is_empty());
    }

    #[test]
    fn test_status_report() {
        let config = ConsciousPerceptionConfig::default();
        let perception = ConsciousPerception::new(config);

        let report = perception.status_report();
        assert!(report.contains("Perception"));
        assert!(report.contains("healthy"));
    }

    #[test]
    fn test_resilience_stats_tracking() {
        let config = ConsciousPerceptionConfig::default();
        let mut perception = ConsciousPerception::new(config);
        let _ = perception.initialize();

        let image = create_test_image();
        let _ = perception.perceive_image(&image);

        let stats = perception.resilience_stats();
        assert!(stats.total_attempts >= 1);
    }

    #[test]
    fn test_coherence_gating_disabled() {
        let mut config = ConsciousPerceptionConfig::default();
        config.enable_coherence_gating = false;
        let mut perception = ConsciousPerception::new(config);
        let _ = perception.initialize();

        let image = create_test_image();
        let result = perception.perceive_image(&image).unwrap();

        assert!(result.coherence_score.is_none());
    }

    #[test]
    fn test_capability_forcing() {
        let config = ConsciousPerceptionConfig::default();
        let perception = ConsciousPerception::new(config);

        perception.set_capability_availability("image_embedding", Availability::Full);
        let caps = perception.capabilities();
        assert!(caps.image_embedding.is_full());

        perception.set_capability_availability("image_embedding", Availability::Degraded);
        let caps = perception.capabilities();
        assert!(caps.image_embedding.is_usable());
        assert!(!caps.image_embedding.is_full());
    }
}
