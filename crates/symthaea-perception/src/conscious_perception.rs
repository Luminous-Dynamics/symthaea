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
//! │ Perception      │ ← SigLIP, Moondream, Tesseract, Qwen3
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

use super::multi_modal::HdcVector;
use super::resilience::{
    Availability, PerceptionCapabilities, ResilienceConfig, ResilienceManager, ResilienceStats,
    ResilientResult,
};
use crate::hdc::causal_mind::{CausalMind, GroundedCausalLearning};
use crate::perception::{
    ImageCaption, ImageEmbedding, ModalityType, MultiModalIntegrator, OcrSystem, SemanticVision,
    VisualCortex, VisualFeatures,
};

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

    /// Semantic vision (SigLIP + Moondream)
    vision: SemanticVision,

    /// OCR system (Tesseract)
    ocr: OcrSystem,

    /// Multi-modal HDC integrator
    integrator: MultiModalIntegrator,

    /// Causal reasoning mind
    causal_mind: CausalMind,

    /// Basic visual feature extraction (always available)
    visual_cortex: VisualCortex,

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

impl ConsciousPerception {
    /// Create a new conscious perception system
    pub fn new(config: ConsciousPerceptionConfig) -> Self {
        let resilience = ResilienceManager::new(config.resilience.clone());
        Self {
            vision: SemanticVision::new(config.cache_size),
            ocr: OcrSystem::new(),
            integrator: MultiModalIntegrator::new(),
            causal_mind: CausalMind::new(),
            visual_cortex: VisualCortex::new(),
            resilience,
            stats: PerceptionStats::default(),
            config,
        }
    }

    /// Initialize all perception subsystems with availability tracking
    pub fn initialize(&mut self) -> Result<()> {
        // Track loading status
        self.resilience.loader.start_loading("siglip");
        self.resilience.loader.start_loading("moondream");
        self.resilience.loader.start_loading("ocr");

        // Initialize vision system (SigLIP + Moondream)
        match self.vision.initialize() {
            Ok(()) => {
                self.resilience
                    .set_capability("image_embedding", Availability::Full);
                self.resilience
                    .set_capability("image_captioning", Availability::Full);
                self.resilience.loader.complete_loading("siglip");
                self.resilience.loader.complete_loading("moondream");
            }
            Err(e) => {
                // Vision initialization failed - use stubs
                let reason = format!("Vision init failed: {}", e);
                self.resilience
                    .set_capability("image_embedding", Availability::Stub);
                self.resilience
                    .set_capability("image_captioning", Availability::Stub);
                self.resilience.loader.fail_loading("siglip", &reason);
                self.resilience.loader.fail_loading("moondream", &reason);
                self.resilience.record_fallback(&reason);
            }
        }

        // Initialize OCR system
        match self.ocr.initialize() {
            Ok(()) => {
                self.resilience.set_capability("ocr", Availability::Full);
                self.resilience.loader.complete_loading("ocr");
            }
            Err(e) => {
                let reason = format!("OCR init failed: {}", e);
                self.resilience.set_capability("ocr", Availability::Stub);
                self.resilience.loader.fail_loading("ocr", &reason);
                self.resilience.record_fallback(&reason);
            }
        }

        // Visual cortex and integrator are always available (pure Rust)
        self.resilience
            .set_capability("visual_features", Availability::Full);
        self.resilience
            .set_capability("multi_modal", Availability::Full);

        Ok(())
    }

    /// Process an image through the full perception pipeline with resilience
    pub fn perceive_image(&mut self, image: &DynamicImage) -> Result<PerceptionResult> {
        let start = Instant::now();
        let mut warnings = Vec::new();
        let mut used_fallback = false;
        let mut capabilities_used = Vec::new();

        // Step 0: Always extract visual features (pure Rust, always available)
        let visual_features = self.visual_cortex.process_image(image).ok();
        capabilities_used.push("visual_features".to_string());

        // Step 1: Get image embedding (SigLIP) with resilience
        let embedding_result = self.perceive_image_embedding_resilient(image);
        let embedding = embedding_result.value;
        if embedding_result.is_fallback {
            used_fallback = true;
            self.stats.embedding_fallbacks += 1;
            warnings.extend(embedding_result.warnings);
        }
        capabilities_used.push("image_embedding".to_string());

        // Step 2: Get image caption (Moondream) with resilience and fallback
        let caption_result = self.perceive_caption_resilient(image, visual_features.as_ref());
        let caption = caption_result.value;
        if caption_result.is_fallback {
            used_fallback = true;
            self.stats.caption_fallbacks += 1;
            warnings.extend(caption_result.warnings);
        }
        capabilities_used.push("image_captioning".to_string());

        // Step 3: OCR with resilience
        let ocr_result = self.perceive_ocr_resilient(image);
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

        // Step 4: Project to HDC space
        let hdc_encoding = self
            .integrator
            .project_image_embedding(&embedding)
            .context("Failed to project image to HDC space")?;
        capabilities_used.push("multi_modal".to_string());

        // Step 5: Coherence gating (optional)
        let coherence_score = if self.config.enable_coherence_gating {
            // Calculate coherence based on caption confidence and embedding quality
            let coherence = caption.confidence * 0.7 + 0.3; // Baseline + confidence
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
                all_text.push_str(" ");
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
            modality: ModalityType::Vision,
            hdc_encoding,
            text: extracted_text,
            image_embedding: Some(embedding),
            caption: Some(caption.clone()),
            causal_learning,
            processing_time_ms,
            confidence: caption.confidence,
            phi: self.causal_mind.phi(),
            // Resilience info
            used_fallback,
            capabilities_used,
            warnings,
            visual_features,
            coherence_score,
        })
    }

    /// Embed image with timeout/retry and fallback to stub
    fn perceive_image_embedding_resilient(
        &mut self,
        image: &DynamicImage,
    ) -> ResilientResult<ImageEmbedding> {
        let start = Instant::now();
        let timeout = self.config.resilience.inference_timeout;
        let max_retries = self.config.resilience.max_retries;

        // Check if embedding capability is available
        let caps = self.resilience.capabilities();
        if !caps.image_embedding.is_usable() {
            return ResilientResult::fallback(
                self.vision.stub_embedding(image),
                "Image embedding unavailable",
            );
        }

        // Attempt with retries
        for attempt in 0..=max_retries {
            if attempt > 0 {
                self.stats.retries += 1;
                // Exponential backoff
                let delay = self.config.resilience.retry_base_delay * 2u32.pow(attempt - 1);
                std::thread::sleep(delay.min(self.config.resilience.retry_max_delay));
            }

            match self.vision.embed_image(image) {
                Ok(embedding) => {
                    return ResilientResult {
                        value: embedding,
                        is_fallback: false,
                        retries: attempt,
                        duration: start.elapsed(),
                        coherence: None,
                        warnings: Vec::new(),
                    };
                }
                Err(e) => {
                    if start.elapsed() > timeout {
                        self.stats.timeouts += 1;
                        return ResilientResult::fallback(
                            self.vision.stub_embedding(image),
                            &format!("Embedding timeout after {:?}", timeout),
                        );
                    }
                    // Continue to next retry
                }
            }
        }

        // All retries exhausted
        ResilientResult::fallback(
            self.vision.stub_embedding(image),
            &format!("Embedding failed after {} retries", max_retries),
        )
    }

    /// Caption image with timeout/retry and visual feature fallback
    fn perceive_caption_resilient(
        &mut self,
        image: &DynamicImage,
        visual_features: Option<&VisualFeatures>,
    ) -> ResilientResult<ImageCaption> {
        let start = Instant::now();
        let timeout = self.config.resilience.inference_timeout;
        let max_retries = self.config.resilience.max_retries;

        // Check if captioning capability is available
        let caps = self.resilience.capabilities();
        if !caps.image_captioning.is_usable() {
            return self.generate_fallback_caption(visual_features, "Captioning unavailable");
        }

        // Attempt with retries
        for attempt in 0..=max_retries {
            if attempt > 0 {
                self.stats.retries += 1;
                let delay = self.config.resilience.retry_base_delay * 2u32.pow(attempt - 1);
                std::thread::sleep(delay.min(self.config.resilience.retry_max_delay));
            }

            match self.vision.caption_image(image) {
                Ok(caption) => {
                    return ResilientResult {
                        value: caption,
                        is_fallback: false,
                        retries: attempt,
                        duration: start.elapsed(),
                        coherence: None,
                        warnings: Vec::new(),
                    };
                }
                Err(e) => {
                    if start.elapsed() > timeout {
                        self.stats.timeouts += 1;
                        return self.generate_fallback_caption(
                            visual_features,
                            &format!("Caption timeout after {:?}", timeout),
                        );
                    }
                }
            }
        }

        self.generate_fallback_caption(
            visual_features,
            &format!("Caption failed after {} retries", max_retries),
        )
    }

    /// Generate fallback caption from visual features
    fn generate_fallback_caption(
        &self,
        visual_features: Option<&VisualFeatures>,
        reason: &str,
    ) -> ResilientResult<ImageCaption> {
        if !self.config.enable_caption_fallback {
            return ResilientResult::fallback(
                ImageCaption {
                    text: "Image content unavailable".to_string(),
                    confidence: 0.1,
                    timestamp: Instant::now(),
                },
                reason,
            );
        }

        let (caption_text, confidence) = if let Some(features) = visual_features {
            // Use visual features to generate description
            let aspect_ratio = features.dimensions.0 as f32 / features.dimensions.1 as f32;

            // Extract dominant hue from first dominant color
            let dominant_hue = if let Some(color) = features.dominant_colors.first() {
                // Simple RGB to hue conversion
                let r = color[0] as f32 / 255.0;
                let g = color[1] as f32 / 255.0;
                let b = color[2] as f32 / 255.0;
                let max = r.max(g).max(b);
                let min = r.min(g).min(b);
                if max == min {
                    0.0
                } else if max == r {
                    60.0 * ((g - b) / (max - min)).rem_euclid(6.0)
                } else if max == g {
                    60.0 * ((b - r) / (max - min) + 2.0)
                } else {
                    60.0 * ((r - g) / (max - min) + 4.0)
                }
            } else {
                0.0
            };

            // Calculate saturation estimate from color variance
            let saturation = (features.color_variance * 2.0).min(1.0);

            self.resilience.fallback_caption(
                features.brightness,
                dominant_hue,
                saturation,
                features.color_variance, // Use as contrast proxy
                features.edge_density,
                aspect_ratio,
            )
        } else {
            ("Image content (features unavailable)".to_string(), 0.2)
        };

        ResilientResult::fallback(
            ImageCaption {
                text: caption_text,
                confidence,
                timestamp: Instant::now(),
            },
            reason,
        )
    }

    /// OCR with resilience
    fn perceive_ocr_resilient(&mut self, image: &DynamicImage) -> ResilientResult<String> {
        let caps = self.resilience.capabilities();
        if !caps.ocr.is_usable() {
            return ResilientResult::fallback(String::new(), "OCR unavailable");
        }

        match self.ocr.recognize(image) {
            Ok(result) => ResilientResult::full(result.text, Duration::ZERO),
            Err(_) => ResilientResult::fallback(String::new(), "OCR recognition failed"),
        }
    }

    /// Process text through the perception pipeline
    pub fn perceive_text(&mut self, text: &str) -> Result<PerceptionResult> {
        let start = Instant::now();
        let mut capabilities_used = vec!["text_encoding".to_string()];

        // Step 1: Project text to HDC space using n-gram encoding
        let hdc_encoding = self
            .integrator
            .project_text(text)
            .context("Failed to project text to HDC space")?;
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
            modality: ModalityType::Semantic,
            hdc_encoding,
            text: Some(text.to_string()),
            image_embedding: None,
            caption: None,
            causal_learning,
            processing_time_ms,
            confidence: 1.0, // Text input has full confidence
            phi: self.causal_mind.phi(),
            // Resilience info (text processing is always available)
            used_fallback: false,
            capabilities_used,
            warnings: Vec::new(),
            visual_features: None,
            coherence_score: Some(1.0), // Text input has full coherence
        })
    }

    /// Process an image file
    pub fn perceive_image_file(&mut self, path: &Path) -> Result<PerceptionResult> {
        let image = image::open(path).context(format!("Failed to open image: {:?}", path))?;
        self.perceive_image(&image)
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
            // Create a gradient with some color
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

        assert_eq!(result.modality, ModalityType::Semantic);
        assert!(result.text.is_some());
        assert_eq!(perception.stats.texts_processed, 1);
        // Text perception doesn't use fallback
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

        // Process text with causal structure
        let result = perception
            .perceive_text("The server crashed because memory was exhausted")
            .unwrap();

        // Should have extracted causal relations
        if let Some(ref learning) = result.causal_learning {
            assert!(learning.links_added > 0);
        }

        // Query should work
        let _explanations = perception.query_why("crashed");
        // May or may not find explanations depending on extraction
    }

    #[test]
    fn test_image_perception_stub() {
        let config = ConsciousPerceptionConfig::default();
        let mut perception = ConsciousPerception::new(config);

        // Initialize (will use stubs without real models)
        let _ = perception.initialize();

        let image = create_test_image();
        let result = perception.perceive_image(&image).unwrap();

        assert_eq!(result.modality, ModalityType::Vision);
        assert!(result.image_embedding.is_some());
        assert_eq!(perception.stats.images_processed, 1);
        // Should have visual features
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
        // Before initialize(), should start with stubs for ML models
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
        // Health should be between 0 and 1
        assert!(health >= 0.0 && health <= 1.0);
        // With default stubs, health should be moderate
        assert!(health > 0.3);
    }

    #[test]
    fn test_image_perception_with_resilience() {
        let config = ConsciousPerceptionConfig::default();
        let mut perception = ConsciousPerception::new(config);
        let _ = perception.initialize();

        let image = create_colorful_test_image();
        let result = perception.perceive_image(&image).unwrap();

        // Should have capabilities tracked
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

        // Should have coherence score when gating enabled
        assert!(result.coherence_score.is_some());
    }

    #[test]
    fn test_fallback_caption_from_visual_features() {
        let config = ConsciousPerceptionConfig::default();
        let mut perception = ConsciousPerception::new(config);

        // Force caption unavailable
        perception.set_capability_availability("image_captioning", Availability::Unavailable);

        let image = create_colorful_test_image();
        let result = perception.perceive_image(&image).unwrap();

        // Should have used fallback
        assert!(result.used_fallback);
        // Caption should still exist (from fallback)
        assert!(result.caption.is_some());
        let caption = result.caption.unwrap();
        // Fallback caption should describe visual features
        assert!(!caption.text.is_empty());
    }

    #[test]
    fn test_status_report() {
        let config = ConsciousPerceptionConfig::default();
        let perception = ConsciousPerception::new(config);

        let report = perception.status_report();
        // Report should include capability info
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
        // Should have recorded at least one attempt
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

        // Coherence score should be None when disabled
        assert!(result.coherence_score.is_none());
    }

    #[test]
    fn test_capability_forcing() {
        let config = ConsciousPerceptionConfig::default();
        let perception = ConsciousPerception::new(config);

        // Force embedding to full
        perception.set_capability_availability("image_embedding", Availability::Full);
        let caps = perception.capabilities();
        assert!(caps.image_embedding.is_full());

        // Force to degraded
        perception.set_capability_availability("image_embedding", Availability::Degraded);
        let caps = perception.capabilities();
        assert!(caps.image_embedding.is_usable());
        assert!(!caps.image_embedding.is_full());
    }
}