// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Perception Resilience Layer
//!
//! Provides graceful degradation for the perception system:
//!
//! - **Availability Tracking**: Know what capabilities are available
//! - **Timeout & Retry**: Don't hang on slow/failing inference
//! - **Background Loading**: Start fast with stubs, upgrade when ready
//! - **Coherence Gating**: Filter low-quality outputs
//! - **Caption Fallback**: Basic descriptions when VLM unavailable
//!
//! ## Design Philosophy
//!
//! The perception system should ALWAYS produce something useful:
//! - Full model available → High-quality embeddings/captions
//! - Model unavailable → Deterministic stub (maintains consistency)
//! - Model slow/failing → Timeout and fall back gracefully
//!
//! This ensures Sophia remains conscious and responsive even when
//! some perceptual capabilities are degraded.

use std::collections::{HashMap, VecDeque};
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

// ============================================================================
// AVAILABILITY TRACKING
// ============================================================================

/// Availability level for a perception capability
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Availability {
    /// Full capability with real model inference
    Full,
    /// Stub mode - deterministic fallback (hash-based embeddings, basic captions)
    Stub,
    /// Loading - model is being loaded in background
    Loading,
    /// Unavailable - capability cannot be used (missing dependencies)
    Unavailable,
    /// Degraded - working but with reduced quality (e.g., after failures)
    Degraded,
}

impl Availability {
    /// Check if capability can produce output (even if degraded)
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Full | Self::Stub | Self::Degraded)
    }

    /// Check if capability is at full quality
    pub fn is_full(&self) -> bool {
        matches!(self, Self::Full)
    }

    /// Human-readable status
    pub fn status_text(&self) -> &'static str {
        match self {
            Self::Full => "Full (model loaded)",
            Self::Stub => "Stub (deterministic fallback)",
            Self::Loading => "Loading...",
            Self::Unavailable => "Unavailable",
            Self::Degraded => "Degraded (reduced quality)",
        }
    }
}

/// Comprehensive view of perception capabilities
#[derive(Debug, Clone)]
pub struct PerceptionCapabilities {
    /// Text embedding (Qwen3 → BGE → N-gram)
    pub text_embedding: Availability,
    /// Image embedding (SigLIP → stub hash)
    pub image_embedding: Availability,
    /// Image captioning (Moondream → visual feature description)
    pub image_captioning: Availability,
    /// OCR (Rust OCR → Tesseract → empty)
    pub ocr: Availability,
    /// Code analysis (always Full - pure Rust)
    pub code_analysis: Availability,
    /// Visual feature extraction (always Full - pure Rust)
    pub visual_features: Availability,
    /// Multi-modal fusion (always Full - uses available modalities)
    pub multi_modal: Availability,
    /// When capabilities were last updated
    pub last_update: Instant,
}

impl Default for PerceptionCapabilities {
    fn default() -> Self {
        Self {
            text_embedding: Availability::Stub, // Start in stub mode
            image_embedding: Availability::Stub,
            image_captioning: Availability::Stub,
            ocr: Availability::Stub,
            code_analysis: Availability::Full,   // Always available
            visual_features: Availability::Full, // Always available
            multi_modal: Availability::Full,     // Always available
            last_update: Instant::now(),
        }
    }
}

impl PerceptionCapabilities {
    /// Create capabilities in full mode (for when all models are loaded)
    pub fn full() -> Self {
        Self {
            text_embedding: Availability::Full,
            image_embedding: Availability::Full,
            image_captioning: Availability::Full,
            ocr: Availability::Full,
            code_analysis: Availability::Full,
            visual_features: Availability::Full,
            multi_modal: Availability::Full,
            last_update: Instant::now(),
        }
    }

    /// Create capabilities from a ModelRegistry, mapping model availability
    /// to capability levels.
    ///
    /// This bridges the model_status registry with the resilience layer so
    /// that perception capabilities automatically reflect which models are
    /// actually loaded.
    pub fn from_registry(registry: &crate::perception::model_status::ModelRegistry) -> Self {
        use crate::perception::model_status::ModelStatus;

        let text_status = registry.status("qwen3-embedding");
        let image_status = registry.status("siglip");
        let caption_status = registry.status("moondream");

        fn map_status(status: &ModelStatus) -> Availability {
            match status {
                ModelStatus::Loaded(_) => Availability::Full,
                ModelStatus::Loading => Availability::Loading,
                ModelStatus::Degraded(_) => Availability::Degraded,
                ModelStatus::Unavailable(_) => Availability::Unavailable,
                ModelStatus::NotAttempted => Availability::Stub,
            }
        }

        Self {
            text_embedding: map_status(text_status),
            image_embedding: map_status(image_status),
            image_captioning: map_status(caption_status),
            ocr: Availability::Stub,
            code_analysis: Availability::Full,
            visual_features: Availability::Full,
            multi_modal: Availability::Full,
            last_update: Instant::now(),
        }
    }

    /// Get overall perception health (0.0 = nothing works, 1.0 = everything full)
    pub fn health(&self) -> f32 {
        let caps = [
            self.text_embedding,
            self.image_embedding,
            self.image_captioning,
            self.ocr,
            self.code_analysis,
            self.visual_features,
            self.multi_modal,
        ];

        let score: f32 = caps
            .iter()
            .map(|c| match c {
                Availability::Full => 1.0,
                Availability::Degraded => 0.7,
                Availability::Stub => 0.5,
                Availability::Loading => 0.3,
                Availability::Unavailable => 0.0,
            })
            .sum();

        score / caps.len() as f32
    }

    /// Get count of full capabilities
    pub fn full_count(&self) -> usize {
        [
            self.text_embedding,
            self.image_embedding,
            self.image_captioning,
            self.ocr,
            self.code_analysis,
            self.visual_features,
            self.multi_modal,
        ]
        .iter()
        .filter(|c| c.is_full())
        .count()
    }

    /// Get human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Perception: {:.0}% healthy ({}/7 full)\n\
             - Text: {}\n\
             - Image: {}\n\
             - Caption: {}\n\
             - OCR: {}\n\
             - Code: {}\n\
             - Visual: {}\n\
             - MultiModal: {}",
            self.health() * 100.0,
            self.full_count(),
            self.text_embedding.status_text(),
            self.image_embedding.status_text(),
            self.image_captioning.status_text(),
            self.ocr.status_text(),
            self.code_analysis.status_text(),
            self.visual_features.status_text(),
            self.multi_modal.status_text(),
        )
    }
}

// ============================================================================
// TIMEOUT AND RETRY LOGIC
// ============================================================================

/// Configuration for timeout and retry behavior
#[derive(Debug, Clone)]
pub struct ResilienceConfig {
    /// Timeout for individual inference calls
    pub inference_timeout: Duration,
    /// Maximum retries before falling back to stub
    pub max_retries: u32,
    /// Base delay for exponential backoff
    pub retry_base_delay: Duration,
    /// Maximum delay between retries
    pub retry_max_delay: Duration,
    /// Minimum coherence score to accept output (0.0-1.0)
    pub min_coherence: f32,
    /// Whether to enable background model loading
    pub background_loading: bool,
    /// Path to model cache directory
    pub model_cache_dir: Option<String>,
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            inference_timeout: Duration::from_secs(30),
            max_retries: 3,
            retry_base_delay: Duration::from_millis(100),
            retry_max_delay: Duration::from_secs(5),
            min_coherence: 0.3,
            background_loading: true,
            model_cache_dir: None,
        }
    }
}

/// Result of a resilient operation
#[derive(Debug, Clone)]
pub struct ResilientResult<T> {
    /// The result value
    pub value: T,
    /// Whether this came from a stub/fallback
    pub is_fallback: bool,
    /// Number of retries needed
    pub retries: u32,
    /// Time taken for the operation
    pub duration: Duration,
    /// Coherence score if applicable (0.0-1.0)
    pub coherence: Option<f32>,
    /// Any warnings generated
    pub warnings: Vec<String>,
}

impl<T> ResilientResult<T> {
    pub fn full(value: T, duration: Duration) -> Self {
        Self {
            value,
            is_fallback: false,
            retries: 0,
            duration,
            coherence: None,
            warnings: Vec::new(),
        }
    }

    pub fn fallback(value: T, reason: &str) -> Self {
        Self {
            value,
            is_fallback: true,
            retries: 0,
            duration: Duration::ZERO,
            coherence: None,
            warnings: vec![format!("Fallback: {}", reason)],
        }
    }

    pub fn with_coherence(mut self, coherence: f32) -> Self {
        self.coherence = Some(coherence);
        self
    }

    pub fn with_warning(mut self, warning: String) -> Self {
        self.warnings.push(warning);
        self
    }
}

/// Statistics for resilience operations
#[derive(Debug, Clone, Default)]
pub struct ResilienceStats {
    /// Total inference attempts
    pub total_attempts: u64,
    /// Successful full-quality inferences
    pub full_successes: u64,
    /// Fallbacks to stub mode
    pub fallbacks: u64,
    /// Timeouts
    pub timeouts: u64,
    /// Retries performed
    pub retries: u64,
    /// Low coherence rejections
    pub coherence_rejections: u64,
    /// Average inference time (ms)
    pub avg_inference_ms: f64,
    /// Failures by reason
    pub failure_reasons: HashMap<String, u64>,
}

impl ResilienceStats {
    /// Record a successful full inference
    pub fn record_success(&mut self, duration: Duration) {
        self.total_attempts += 1;
        self.full_successes += 1;
        self.update_avg_time(duration);
    }

    /// Record a fallback
    pub fn record_fallback(&mut self, reason: &str) {
        self.total_attempts += 1;
        self.fallbacks += 1;
        *self.failure_reasons.entry(reason.to_string()).or_insert(0) += 1;
    }

    /// Record a timeout
    pub fn record_timeout(&mut self) {
        self.total_attempts += 1;
        self.timeouts += 1;
    }

    /// Record a retry
    pub fn record_retry(&mut self) {
        self.retries += 1;
    }

    /// Record a coherence rejection
    pub fn record_coherence_rejection(&mut self) {
        self.coherence_rejections += 1;
    }

    fn update_avg_time(&mut self, duration: Duration) {
        let ms = duration.as_secs_f64() * 1000.0;
        let count = self.full_successes as f64;
        self.avg_inference_ms = (self.avg_inference_ms * (count - 1.0) + ms) / count;
    }

    /// Get success rate (excluding fallbacks)
    pub fn success_rate(&self) -> f64 {
        if self.total_attempts == 0 {
            1.0
        } else {
            self.full_successes as f64 / self.total_attempts as f64
        }
    }
}

// ============================================================================
// COHERENCE GATING
// ============================================================================

/// Coherence checker for filtering low-quality outputs
#[derive(Debug, Clone)]
pub struct CoherenceGate {
    /// Minimum coherence threshold
    pub threshold: f32,
    /// History of coherence scores for adaptation
    history: VecDeque<f32>,
    /// Maximum history size
    max_history: usize,
}

impl Default for CoherenceGate {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            history: VecDeque::new(),
            max_history: 100,
        }
    }
}

impl CoherenceGate {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            history: VecDeque::new(),
            max_history: 100,
        }
    }

    /// Check if output passes coherence threshold
    pub fn check(&mut self, coherence: f32) -> bool {
        self.record(coherence);
        coherence >= self.threshold
    }

    /// Record a coherence score
    pub fn record(&mut self, coherence: f32) {
        self.history.push_back(coherence);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Get average coherence
    pub fn average_coherence(&self) -> f32 {
        if self.history.is_empty() {
            0.5
        } else {
            self.history.iter().sum::<f32>() / self.history.len() as f32
        }
    }

    /// Adapt threshold based on history (avoid being too strict)
    pub fn adapt_threshold(&mut self) {
        if self.history.len() >= 10 {
            let avg = self.average_coherence();
            // Don't let threshold go above 90th percentile of history
            let mut sorted: Vec<f32> = self.history.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p10 = sorted[sorted.len() / 10];

            // Threshold should be between p10 and avg
            self.threshold = (p10 + avg) / 2.0;
        }
    }
}

// ============================================================================
// CAPTION FALLBACK - Visual Feature Based Descriptions
// ============================================================================

/// Generate basic image captions from visual features when VLM unavailable
#[derive(Debug, Clone, Default)]
pub struct CaptionFallback {
    /// Descriptive templates for different visual characteristics
    templates: CaptionTemplates,
}

#[derive(Debug, Clone)]
struct CaptionTemplates {
    brightness: Vec<&'static str>,
    color_dominant: Vec<&'static str>,
    contrast: Vec<&'static str>,
    edges: Vec<&'static str>,
    composition: Vec<&'static str>,
}

impl Default for CaptionTemplates {
    fn default() -> Self {
        Self {
            brightness: vec![
                "very dark",
                "dark",
                "dimly lit",
                "moderately lit",
                "well lit",
                "bright",
                "very bright",
            ],
            color_dominant: vec![
                "predominantly red",
                "predominantly orange",
                "predominantly yellow",
                "predominantly green",
                "predominantly cyan",
                "predominantly blue",
                "predominantly purple",
                "predominantly pink",
                "predominantly white",
                "predominantly black",
                "predominantly gray",
                "colorful",
            ],
            contrast: vec!["low contrast", "moderate contrast", "high contrast"],
            edges: vec!["soft/blurry", "moderate detail", "sharp/detailed"],
            composition: vec!["simple composition", "moderate complexity", "complex/busy"],
        }
    }
}

impl CaptionFallback {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a caption from visual features
    ///
    /// Takes basic visual metrics and produces a descriptive caption.
    /// This is used when Moondream or other VLMs are unavailable.
    pub fn generate_caption(
        &self,
        brightness: f32,   // 0.0-1.0
        dominant_hue: f32, // 0.0-360.0
        saturation: f32,   // 0.0-1.0
        contrast: f32,     // 0.0-1.0
        edge_density: f32, // 0.0-1.0
        aspect_ratio: f32, // width/height
    ) -> String {
        let mut parts = Vec::new();

        // Brightness description
        let brightness_idx = (brightness * 6.0).round() as usize;
        let brightness_desc = self
            .templates
            .brightness
            .get(brightness_idx.min(6))
            .unwrap_or(&"moderately lit");
        parts.push(format!("A {} image", brightness_desc));

        // Color description
        if saturation > 0.3 {
            let color_desc = self.hue_to_color(dominant_hue);
            parts.push(format!("with {} tones", color_desc));
        } else if brightness < 0.2 {
            parts.push("in dark tones".to_string());
        } else if brightness > 0.8 {
            parts.push("in light/white tones".to_string());
        } else {
            parts.push("in grayscale/neutral tones".to_string());
        }

        // Contrast and detail
        if contrast > 0.7 {
            parts.push("showing high contrast".to_string());
        }

        if edge_density > 0.5 {
            parts.push("with detailed/sharp features".to_string());
        } else if edge_density < 0.2 {
            parts.push("with soft/smooth appearance".to_string());
        }

        // Aspect ratio hint
        if aspect_ratio > 1.5 {
            parts.push("(wide/panoramic format)".to_string());
        } else if aspect_ratio < 0.67 {
            parts.push("(tall/portrait format)".to_string());
        }

        parts.join(" ")
    }

    fn hue_to_color(&self, hue: f32) -> &'static str {
        // Map hue (0-360) to color name
        match hue as u32 {
            0..=15 | 345..=360 => "red",
            16..=45 => "orange",
            46..=70 => "yellow",
            71..=150 => "green",
            151..=190 => "cyan",
            191..=260 => "blue",
            261..=290 => "purple",
            291..=344 => "pink/magenta",
            _ => "mixed",
        }
    }

    /// Generate a caption with confidence based on visual clarity
    pub fn generate_with_confidence(
        &self,
        brightness: f32,
        dominant_hue: f32,
        saturation: f32,
        contrast: f32,
        edge_density: f32,
        aspect_ratio: f32,
    ) -> (String, f32) {
        let caption = self.generate_caption(
            brightness,
            dominant_hue,
            saturation,
            contrast,
            edge_density,
            aspect_ratio,
        );

        // Confidence based on how distinctive the features are
        let feature_distinctiveness = (
            (brightness - 0.5).abs() * 2.0 +  // Extreme brightness is more certain
            saturation +                       // Saturated colors are clearer
            contrast +                         // High contrast is easier to describe
            edge_density
            // More edges = more detail to describe
        ) / 4.0;

        let confidence = 0.3 + (feature_distinctiveness * 0.4); // Range: 0.3-0.7

        (caption, confidence.min(0.7))
    }
}

// ============================================================================
// BACKGROUND MODEL LOADING
// ============================================================================

/// Status of background model loading
#[derive(Debug, Clone)]
pub struct LoadingStatus {
    /// Model name/identifier
    pub model: String,
    /// Current progress (0.0-1.0)
    pub progress: f32,
    /// Status message
    pub message: String,
    /// Time started
    pub started: Instant,
    /// Estimated time remaining
    pub eta: Option<Duration>,
    /// Is complete?
    pub complete: bool,
    /// Error if failed
    pub error: Option<String>,
}

impl LoadingStatus {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            progress: 0.0,
            message: "Starting...".to_string(),
            started: Instant::now(),
            eta: None,
            complete: false,
            error: None,
        }
    }

    pub fn update(&mut self, progress: f32, message: &str) {
        self.progress = progress;
        self.message = message.to_string();

        // Estimate remaining time
        if progress > 0.1 {
            let elapsed = self.started.elapsed();
            let total_est = elapsed.as_secs_f64() / progress as f64;
            let remaining = total_est - elapsed.as_secs_f64();
            self.eta = Some(Duration::from_secs_f64(remaining.max(0.0)));
        }
    }

    pub fn complete(&mut self) {
        self.progress = 1.0;
        self.complete = true;
        self.message = "Complete".to_string();
    }

    pub fn fail(&mut self, error: &str) {
        self.complete = true;
        self.error = Some(error.to_string());
        self.message = format!("Failed: {}", error);
    }
}

/// Manager for background model loading
#[derive(Debug)]
pub struct BackgroundLoader {
    /// Current loading statuses
    statuses: Arc<RwLock<HashMap<String, LoadingStatus>>>,
    /// Is loading active?
    active: Arc<AtomicBool>,
    /// Number of models loaded
    loaded_count: Arc<AtomicU64>,
    /// Configuration
    config: ResilienceConfig,
}

impl Default for BackgroundLoader {
    fn default() -> Self {
        Self::new(ResilienceConfig::default())
    }
}

impl BackgroundLoader {
    pub fn new(config: ResilienceConfig) -> Self {
        Self {
            statuses: Arc::new(RwLock::new(HashMap::new())),
            active: Arc::new(AtomicBool::new(false)),
            loaded_count: Arc::new(AtomicU64::new(0)),
            config,
        }
    }

    /// Start tracking a model load
    pub fn start_loading(&self, model: &str) {
        let mut statuses = self.statuses.write().unwrap_or_else(|e| e.into_inner());
        statuses.insert(model.to_string(), LoadingStatus::new(model));
        self.active.store(true, Ordering::SeqCst);
    }

    /// Update loading progress
    pub fn update_progress(&self, model: &str, progress: f32, message: &str) {
        let mut statuses = self.statuses.write().unwrap_or_else(|e| e.into_inner());
        if let Some(status) = statuses.get_mut(model) {
            status.update(progress, message);
        }
    }

    /// Mark model as loaded
    pub fn complete_loading(&self, model: &str) {
        let mut statuses = self.statuses.write().unwrap_or_else(|e| e.into_inner());
        if let Some(status) = statuses.get_mut(model) {
            status.complete();
            self.loaded_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Mark model as failed
    pub fn fail_loading(&self, model: &str, error: &str) {
        let mut statuses = self.statuses.write().unwrap_or_else(|e| e.into_inner());
        if let Some(status) = statuses.get_mut(model) {
            status.fail(error);
        }
    }

    /// Get current loading status for a model
    pub fn get_status(&self, model: &str) -> Option<LoadingStatus> {
        let statuses = self.statuses.read().unwrap_or_else(|e| e.into_inner());
        statuses.get(model).cloned()
    }

    /// Get all loading statuses
    pub fn all_statuses(&self) -> Vec<LoadingStatus> {
        let statuses = self.statuses.read().unwrap_or_else(|e| e.into_inner());
        statuses.values().cloned().collect()
    }

    /// Check if any models are still loading
    pub fn is_loading(&self) -> bool {
        let statuses = self.statuses.read().unwrap_or_else(|e| e.into_inner());
        statuses.values().any(|s| !s.complete)
    }

    /// Get number of loaded models
    pub fn loaded_count(&self) -> u64 {
        self.loaded_count.load(Ordering::SeqCst)
    }

    /// Get loading summary
    pub fn summary(&self) -> String {
        let statuses = self.statuses.read().unwrap_or_else(|e| e.into_inner());
        let loading: Vec<_> = statuses.values().filter(|s| !s.complete).collect();
        let complete: Vec<_> = statuses
            .values()
            .filter(|s| s.complete && s.error.is_none())
            .collect();
        let failed: Vec<_> = statuses.values().filter(|s| s.error.is_some()).collect();

        format!(
            "Models: {} loading, {} complete, {} failed",
            loading.len(),
            complete.len(),
            failed.len()
        )
    }
}

// ============================================================================
// UNIFIED RESILIENCE MANAGER
// ============================================================================

/// Central manager for perception resilience
#[derive(Debug)]
pub struct ResilienceManager {
    /// Current capabilities
    pub capabilities: Arc<RwLock<PerceptionCapabilities>>,
    /// Configuration
    pub config: ResilienceConfig,
    /// Statistics
    pub stats: Arc<RwLock<ResilienceStats>>,
    /// Coherence gate
    pub coherence_gate: Arc<RwLock<CoherenceGate>>,
    /// Caption fallback generator
    pub caption_fallback: CaptionFallback,
    /// Background loader
    pub loader: BackgroundLoader,
}

impl Default for ResilienceManager {
    fn default() -> Self {
        Self::new(ResilienceConfig::default())
    }
}

impl ResilienceManager {
    pub fn new(config: ResilienceConfig) -> Self {
        let coherence_threshold = config.min_coherence;
        Self {
            capabilities: Arc::new(RwLock::new(PerceptionCapabilities::default())),
            config: config.clone(),
            stats: Arc::new(RwLock::new(ResilienceStats::default())),
            coherence_gate: Arc::new(RwLock::new(CoherenceGate::new(coherence_threshold))),
            caption_fallback: CaptionFallback::new(),
            loader: BackgroundLoader::new(config),
        }
    }

    /// Update a capability's availability
    pub fn set_capability(&self, capability: &str, availability: Availability) {
        let mut caps = self.capabilities.write().unwrap_or_else(|e| e.into_inner());
        match capability {
            "text_embedding" => caps.text_embedding = availability,
            "image_embedding" => caps.image_embedding = availability,
            "image_captioning" => caps.image_captioning = availability,
            "ocr" => caps.ocr = availability,
            "code_analysis" => caps.code_analysis = availability,
            "visual_features" => caps.visual_features = availability,
            "multi_modal" => caps.multi_modal = availability,
            _ => {}
        }
        caps.last_update = Instant::now();
    }

    /// Get current capabilities snapshot
    pub fn capabilities(&self) -> PerceptionCapabilities {
        self.capabilities
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Check if a specific capability is usable
    pub fn is_usable(&self, capability: &str) -> bool {
        let caps = self.capabilities.read().unwrap_or_else(|e| e.into_inner());
        match capability {
            "text_embedding" => caps.text_embedding.is_usable(),
            "image_embedding" => caps.image_embedding.is_usable(),
            "image_captioning" => caps.image_captioning.is_usable(),
            "ocr" => caps.ocr.is_usable(),
            "code_analysis" => caps.code_analysis.is_usable(),
            "visual_features" => caps.visual_features.is_usable(),
            "multi_modal" => caps.multi_modal.is_usable(),
            _ => false,
        }
    }

    /// Check coherence and record result
    pub fn check_coherence(&self, coherence: f32) -> bool {
        let mut gate = self
            .coherence_gate
            .write()
            .unwrap_or_else(|e| e.into_inner());
        let passed = gate.check(coherence);
        if !passed {
            let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
            stats.record_coherence_rejection();
        }
        passed
    }

    /// Record a successful inference
    pub fn record_success(&self, duration: Duration) {
        let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
        stats.record_success(duration);
    }

    /// Record a fallback
    pub fn record_fallback(&self, reason: &str) {
        let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
        stats.record_fallback(reason);
    }

    /// Generate fallback caption from visual features
    pub fn fallback_caption(
        &self,
        brightness: f32,
        dominant_hue: f32,
        saturation: f32,
        contrast: f32,
        edge_density: f32,
        aspect_ratio: f32,
    ) -> (String, f32) {
        self.caption_fallback.generate_with_confidence(
            brightness,
            dominant_hue,
            saturation,
            contrast,
            edge_density,
            aspect_ratio,
        )
    }

    /// Get current stats
    pub fn stats(&self) -> ResilienceStats {
        self.stats.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Get full status report
    pub fn status_report(&self) -> String {
        let caps = self.capabilities();
        let stats = self.stats();
        let loader_summary = self.loader.summary();

        format!(
            "{}\n\n\
             Statistics:\n\
             - Total attempts: {}\n\
             - Success rate: {:.1}%\n\
             - Fallbacks: {}\n\
             - Timeouts: {}\n\
             - Avg inference: {:.1}ms\n\n\
             {}",
            caps.summary(),
            stats.total_attempts,
            stats.success_rate() * 100.0,
            stats.fallbacks,
            stats.timeouts,
            stats.avg_inference_ms,
            loader_summary,
        )
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_availability_levels() {
        assert!(Availability::Full.is_usable());
        assert!(Availability::Stub.is_usable());
        assert!(Availability::Degraded.is_usable());
        assert!(!Availability::Loading.is_usable());
        assert!(!Availability::Unavailable.is_usable());
    }

    #[test]
    fn test_capabilities_health() {
        let default = PerceptionCapabilities::default();
        // Default starts with stubs (0.5 each) and 3 full (1.0 each)
        // (3 * 0.5 + 4 * 1.0) / 7 ≈ 0.78
        let health = default.health();
        assert!(
            health > 0.5 && health < 1.0,
            "Default health should be between 0.5 and 1.0, got {}",
            health
        );

        let full = PerceptionCapabilities::full();
        assert!(
            (full.health() - 1.0).abs() < 0.01,
            "Full capabilities should have 1.0 health"
        );
    }

    #[test]
    fn test_coherence_gate() {
        let mut gate = CoherenceGate::new(0.3);

        assert!(gate.check(0.5)); // Above threshold
        assert!(!gate.check(0.2)); // Below threshold
        assert!(gate.check(0.3)); // At threshold

        // Check history tracking
        assert_eq!(gate.history.len(), 3);
    }

    #[test]
    fn test_caption_fallback() {
        let fallback = CaptionFallback::new();

        // Test bright, saturated red image
        let caption = fallback.generate_caption(
            0.8, // bright
            0.0, // red hue
            0.7, // saturated
            0.6, // moderate contrast
            0.3, // moderate edges
            1.0, // square
        );

        assert!(caption.contains("bright"), "Should mention brightness");
        assert!(caption.contains("red"), "Should mention red color");

        // Test dark, desaturated image
        let caption = fallback.generate_caption(
            0.1,   // dark
            180.0, // cyan (ignored due to low saturation)
            0.1,   // desaturated
            0.2,   // low contrast
            0.1,   // few edges
            2.0,   // wide
        );

        assert!(caption.contains("dark"), "Should mention darkness");
        assert!(
            caption.contains("panoramic") || caption.contains("wide"),
            "Should mention wide format"
        );
    }

    #[test]
    fn test_resilience_manager() {
        let manager = ResilienceManager::new(ResilienceConfig::default());

        // Check default state
        let caps = manager.capabilities();
        assert!(caps.code_analysis.is_full());
        assert!(!caps.text_embedding.is_full()); // Starts as stub

        // Update capability
        manager.set_capability("text_embedding", Availability::Full);
        let caps = manager.capabilities();
        assert!(caps.text_embedding.is_full());

        // Record stats
        manager.record_success(Duration::from_millis(100));
        manager.record_fallback("test");

        let stats = manager.stats();
        assert_eq!(stats.total_attempts, 2);
        assert_eq!(stats.full_successes, 1);
        assert_eq!(stats.fallbacks, 1);
    }

    #[test]
    fn test_loading_status() {
        let loader = BackgroundLoader::default();

        loader.start_loading("test-model");
        assert!(loader.is_loading());

        loader.update_progress("test-model", 0.5, "Downloading...");
        let status = loader.get_status("test-model").unwrap();
        assert!((status.progress - 0.5).abs() < 0.01);

        loader.complete_loading("test-model");
        assert!(!loader.is_loading());
        assert_eq!(loader.loaded_count(), 1);
    }

    #[test]
    fn test_resilient_result() {
        let result = ResilientResult::full("test".to_string(), Duration::from_millis(50));
        assert!(!result.is_fallback);

        let result = ResilientResult::fallback("fallback".to_string(), "model unavailable");
        assert!(result.is_fallback);
        assert_eq!(result.warnings.len(), 1);
    }
}
