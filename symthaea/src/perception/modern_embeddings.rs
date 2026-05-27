// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Modern Embedding Models - Unified Interface
//!
//! Provides a unified interface for modern embedding models, replacing the
//! problematic BERT layer extraction with models that properly support
//! intermediate layer access.
//!
//! ## Supported Models
//!
//! | Model | Dimension | Layers | Layer Access | Notes |
//! |-------|-----------|--------|--------------|-------|
//! | BGE-M3 | 1024 | 24 | Full | Primary - validated for phenomenal corridor |
//! | ModernBERT | 768/1024 | 22/28 | Full | Flash attention, faster |
//! | Nomic Embed v1.5 | 768 | 12 | Full | Matryoshka, long context |
//! | GTE-large | 1024 | 24 | Full | Good multilingual |
//! | E5-large-v2 | 1024 | 24 | Partial | Instruction-tuned |
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     EmbeddingModel Trait                            │
//! │  embed() / embed_batch() / dimension() / extract_layer()           │
//! └─────────────────────────────────────────────────────────────────────┘
//!          │                    │                    │
//!          ▼                    ▼                    ▼
//! ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
//! │    BgeM3Model   │  │ ModernBertModel │  │  FallbackModel  │
//! │  (Primary)      │  │  (Secondary)    │  │  (Graceful)     │
//! └─────────────────┘  └─────────────────┘  └─────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::perception::modern_embeddings::{
//!     UnifiedEmbedder, EmbeddingConfig, ModelBackend
//! };
//!
//! // Auto-select best available model
//! let embedder = UnifiedEmbedder::auto()?;
//!
//! // Get embedding
//! let embedding = embedder.embed("What is consciousness?")?;
//!
//! // Layer-wise extraction for H2 hypothesis testing
//! let layer_22 = embedder.extract_layer("The redness of the sunset", 22)?;
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

// Device is used in BgeM3Backend::new() under the neural-bridge-cuda feature
#[allow(unused_imports)]
#[cfg(feature = "neural-bridge")]
use candle_core::Device;

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::{HDC_DIMENSION, PackedBipolar, binary_hv::BinaryHV};

// ============================================================================
// Core Traits
// ============================================================================

/// Unified embedding model trait
///
/// All embedding backends implement this trait, providing a consistent
/// interface for encoding text to vectors.
pub trait EmbeddingModel: Send + Sync {
    /// Encode a single text to an embedding vector
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Encode multiple texts in a batch (more efficient)
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Get the embedding dimension
    fn dimension(&self) -> usize;

    /// Get the number of transformer layers
    fn num_layers(&self) -> usize;

    /// Extract activation from a specific layer (for phenomenal corridor analysis)
    fn extract_layer(&self, text: &str, layer_idx: usize) -> Result<LayerOutput>;

    /// Extract activations from multiple layers
    fn extract_layers(&self, text: &str, layer_indices: &[usize]) -> Result<Vec<LayerOutput>>;

    /// Get model name/identifier
    fn model_name(&self) -> &str;

    /// Check if the model supports full layer-wise extraction
    fn supports_layer_extraction(&self) -> bool;

    /// Get the phenomenal corridor layer (~92% depth)
    fn phenomenal_corridor_layer(&self) -> usize {
        ((self.num_layers() as f64) * 0.92) as usize
    }

    /// Check if using GPU
    fn is_gpu(&self) -> bool {
        false
    }
}

/// Output from layer extraction
#[derive(Debug, Clone)]
pub struct LayerOutput {
    /// Layer index (0-indexed)
    pub layer_idx: usize,
    /// Mean-pooled activation vector
    pub activation: Vec<f32>,
    /// Optional: per-token activations for detailed analysis
    pub token_activations: Option<Vec<Vec<f32>>>,
    /// Processing time in microseconds
    pub processing_time_us: u64,
}

impl LayerOutput {
    /// Convert to BinaryHV for HDC processing
    #[cfg(feature = "neural-bridge")]
    pub fn to_hv16(&self) -> BinaryHV {
        activation_to_hv16(&self.activation)
    }
}

/// Convert an activation vector to BinaryHV
#[cfg(feature = "neural-bridge")]
pub fn activation_to_hv16(activation: &[f32]) -> BinaryHV {
    if activation.is_empty() {
        return BinaryHV::random(0);
    }
    let mut expanded = Vec::with_capacity(HDC_DIMENSION);
    let tiles = HDC_DIMENSION / activation.len();
    let remainder = HDC_DIMENSION % activation.len();

    for tile in 0..tiles {
        for (i, &val) in activation.iter().enumerate() {
            let perturbation = ((tile * activation.len() + i) as f32 * 0.001).sin() * 0.01;
            expanded.push(val + perturbation);
        }
    }

    for i in 0..remainder {
        expanded.push(activation[i]);
    }

    BinaryHV::from_bipolar(&expanded)
}

// ============================================================================
// Configuration
// ============================================================================

/// Model backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelBackend {
    /// BGE-M3 (BAAI/bge-m3) - Primary, validated
    BgeM3,
    /// ModernBERT (answerdotai/ModernBERT-base or -large)
    ModernBert,
    /// ModernBERT Large variant
    ModernBertLarge,
    /// Nomic Embed v1.5 (nomic-ai/nomic-embed-text-v1.5)
    NomicEmbed,
    /// GTE-large (thenlper/gte-large)
    GteLarge,
    /// E5-large-v2 (intfloat/e5-large-v2)
    E5LargeV2,
    /// Auto-select best available
    Auto,
}

impl Default for ModelBackend {
    fn default() -> Self {
        Self::Auto
    }
}

impl ModelBackend {
    /// Get the HuggingFace model ID
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::BgeM3 => "BAAI/bge-m3",
            Self::ModernBert => "answerdotai/ModernBERT-base",
            Self::ModernBertLarge => "answerdotai/ModernBERT-large",
            Self::NomicEmbed => "nomic-ai/nomic-embed-text-v1.5",
            Self::GteLarge => "thenlper/gte-large",
            Self::E5LargeV2 => "intfloat/e5-large-v2",
            Self::Auto => "BAAI/bge-m3", // Default to BGE-M3
        }
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        match self {
            Self::BgeM3 | Self::GteLarge | Self::E5LargeV2 | Self::ModernBertLarge => 1024,
            Self::ModernBert | Self::NomicEmbed => 768,
            Self::Auto => 1024,
        }
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        match self {
            Self::BgeM3 | Self::GteLarge | Self::E5LargeV2 => 24,
            Self::ModernBert | Self::NomicEmbed => 12,
            Self::ModernBertLarge => 28,
            Self::Auto => 24,
        }
    }

    /// Whether this backend supports full layer extraction
    pub fn supports_layer_extraction(&self) -> bool {
        match self {
            Self::BgeM3 => true, // Implemented in our custom XLM-RoBERTa
            Self::ModernBert | Self::ModernBertLarge => true, // When implemented
            Self::NomicEmbed => true, // When implemented
            Self::GteLarge => true,
            Self::E5LargeV2 => false, // Instruction-following format makes this complex
            Self::Auto => true,
        }
    }

    /// Architecture family
    pub fn architecture(&self) -> ModelArchitecture {
        match self {
            Self::BgeM3 => ModelArchitecture::XlmRoberta,
            Self::ModernBert | Self::ModernBertLarge => ModelArchitecture::ModernBert,
            Self::NomicEmbed => ModelArchitecture::NomicBert,
            Self::GteLarge => ModelArchitecture::Bert,
            Self::E5LargeV2 => ModelArchitecture::Bert,
            Self::Auto => ModelArchitecture::XlmRoberta,
        }
    }
}

/// Model architecture family
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// XLM-RoBERTa (BGE-M3, mGTE)
    XlmRoberta,
    /// ModernBERT (Flash Attention, RoPE)
    ModernBert,
    /// Nomic BERT (ALiBi attention)
    NomicBert,
    /// Standard BERT
    Bert,
}

/// Configuration for the unified embedder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Primary model backend
    pub backend: ModelBackend,
    /// Fallback backends (tried in order if primary fails)
    pub fallbacks: Vec<ModelBackend>,
    /// Whether to use GPU if available
    pub use_gpu: bool,
    /// GPU device index
    pub gpu_device: usize,
    /// Enable embedding cache
    pub enable_cache: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Batch size for batch encoding
    pub batch_size: usize,
    /// Whether to keep token-level activations
    pub keep_token_activations: bool,
    /// Pooling method
    pub pooling: PoolingMethod,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            backend: ModelBackend::Auto,
            fallbacks: vec![ModelBackend::BgeM3],
            use_gpu: true,
            gpu_device: 0,
            enable_cache: true,
            max_cache_size: 1000,
            batch_size: 32,
            keep_token_activations: false,
            pooling: PoolingMethod::Mean,
        }
    }
}

/// Pooling method for sequence to vector conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PoolingMethod {
    /// Mean pooling over all tokens
    #[default]
    Mean,
    /// Use `[CLS]` token
    Cls,
    /// Use last token (for causal models)
    LastToken,
    /// Max pooling
    Max,
    /// Weighted mean with attention weights
    WeightedMean,
}

// ============================================================================
// BGE-M3 Implementation (Primary Backend)
// ============================================================================

#[cfg(feature = "neural-bridge")]
pub struct BgeM3Backend {
    /// Layer extractor with direct layer access
    extractor: super::layer_extractor::LayerExtractor,
    /// Embedding cache
    cache: std::sync::RwLock<HashMap<String, Vec<f32>>>,
    /// Configuration
    config: EmbeddingConfig,
}

#[cfg(feature = "neural-bridge")]
impl BgeM3Backend {
    /// Create a new BGE-M3 backend
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        use super::layer_extractor::{LayerExtractorConfig, PoolingMethod as ExtractorPooling};

        let pooling = match config.pooling {
            PoolingMethod::Mean => ExtractorPooling::Mean,
            PoolingMethod::Cls => ExtractorPooling::Cls,
            PoolingMethod::LastToken => ExtractorPooling::Last,
            PoolingMethod::Max => ExtractorPooling::Max,
            PoolingMethod::WeightedMean => ExtractorPooling::Mean, // Fallback
        };

        let extractor_config = LayerExtractorConfig {
            model_id: config.backend.model_id().to_string(),
            keep_sequence_activations: config.keep_token_activations,
            pooling,
        };

        let extractor = if config.use_gpu {
            #[cfg(feature = "neural-bridge-cuda")]
            {
                if candle_core::utils::cuda_is_available() {
                    let device = Device::new_cuda(config.gpu_device)
                        .context("Failed to create CUDA device")?;
                    super::layer_extractor::LayerExtractor::load_with_device(
                        extractor_config,
                        device,
                    )?
                } else {
                    super::layer_extractor::LayerExtractor::load(extractor_config)?
                }
            }
            #[cfg(not(feature = "neural-bridge-cuda"))]
            {
                super::layer_extractor::LayerExtractor::load(extractor_config)?
            }
        } else {
            super::layer_extractor::LayerExtractor::load(extractor_config)?
        };

        Ok(Self {
            extractor,
            cache: std::sync::RwLock::new(HashMap::new()),
            config,
        })
    }

    /// Get final layer embedding (standard use case)
    fn get_final_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache
        if self.config.enable_cache {
            if let Ok(cache) = self.cache.read() {
                if let Some(cached) = cache.get(text) {
                    return Ok(cached.clone());
                }
            }
        }

        // Extract from final layer
        let final_layer = self.extractor.num_layers() - 1;
        let activation = self.extractor.extract_layer(text, final_layer)?;

        // Cache result
        if self.config.enable_cache {
            if let Ok(mut cache) = self.cache.write() {
                if cache.len() >= self.config.max_cache_size {
                    // Simple eviction
                    let keys: Vec<_> = cache
                        .keys()
                        .take(self.config.max_cache_size / 2)
                        .cloned()
                        .collect();
                    for key in keys {
                        cache.remove(&key);
                    }
                }
                cache.insert(text.to_string(), activation.activation.clone());
            }
        }

        Ok(activation.activation)
    }
}

#[cfg(feature = "neural-bridge")]
impl EmbeddingModel for BgeM3Backend {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.get_final_embedding(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Sequential processing with caching — true batched forward pass deferred
        // until candle supports batched BertModel inference (upstream limitation)
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dimension(&self) -> usize {
        self.extractor.hidden_dim()
    }

    fn num_layers(&self) -> usize {
        self.extractor.num_layers()
    }

    fn extract_layer(&self, text: &str, layer_idx: usize) -> Result<LayerOutput> {
        let start = Instant::now();
        let activation = self.extractor.extract_layer(text, layer_idx)?;

        Ok(LayerOutput {
            layer_idx,
            activation: activation.activation,
            token_activations: activation.sequence_activations,
            processing_time_us: start.elapsed().as_micros() as u64,
        })
    }

    fn extract_layers(&self, text: &str, layer_indices: &[usize]) -> Result<Vec<LayerOutput>> {
        let start = Instant::now();
        let activations = self.extractor.extract_layers(text, layer_indices)?;
        let elapsed = start.elapsed().as_micros() as u64;
        let time_per_layer = elapsed / layer_indices.len().max(1) as u64;

        Ok(activations
            .into_iter()
            .map(|a| LayerOutput {
                layer_idx: a.layer_idx,
                activation: a.activation,
                token_activations: a.sequence_activations,
                processing_time_us: time_per_layer,
            })
            .collect())
    }

    fn model_name(&self) -> &str {
        "BGE-M3"
    }

    fn supports_layer_extraction(&self) -> bool {
        true
    }

    fn is_gpu(&self) -> bool {
        self.extractor.is_cuda()
    }
}

// ============================================================================
// Unified Embedder (Main Interface)
// ============================================================================

/// Unified embedding interface with automatic fallback
#[cfg(feature = "neural-bridge")]
pub struct UnifiedEmbedder {
    /// Active backend
    backend: Arc<dyn EmbeddingModel>,
    /// Configuration
    config: EmbeddingConfig,
    /// Statistics
    stats: EmbedderStats,
}

/// Statistics for the embedder
#[derive(Debug, Default, Clone)]
pub struct EmbedderStats {
    /// Total embedding calls
    pub total_calls: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Total processing time (microseconds)
    pub total_time_us: u64,
    /// Layer extraction calls
    pub layer_extractions: u64,
}

#[cfg(feature = "neural-bridge")]
impl UnifiedEmbedder {
    /// Create with automatic model selection
    pub fn auto() -> Result<Self> {
        Self::new(EmbeddingConfig::default())
    }

    /// Create with specific configuration
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        let backend = Self::create_backend(&config)?;

        Ok(Self {
            backend,
            config,
            stats: EmbedderStats::default(),
        })
    }

    /// Create with specific backend
    pub fn with_backend(backend: ModelBackend) -> Result<Self> {
        let config = EmbeddingConfig {
            backend,
            ..Default::default()
        };
        Self::new(config)
    }

    fn create_backend(config: &EmbeddingConfig) -> Result<Arc<dyn EmbeddingModel>> {
        let backend = match config.backend {
            ModelBackend::Auto | ModelBackend::BgeM3 => {
                Arc::new(BgeM3Backend::new(config.clone())?) as Arc<dyn EmbeddingModel>
            }
            ModelBackend::ModernBert | ModelBackend::ModernBertLarge => {
                // ModernBERT not yet implemented - fall back to BGE-M3
                tracing::warn!("ModernBERT not yet implemented, falling back to BGE-M3");
                let fallback_config = EmbeddingConfig {
                    backend: ModelBackend::BgeM3,
                    ..config.clone()
                };
                Arc::new(BgeM3Backend::new(fallback_config)?)
            }
            ModelBackend::NomicEmbed => {
                tracing::warn!("Nomic Embed not yet implemented, falling back to BGE-M3");
                let fallback_config = EmbeddingConfig {
                    backend: ModelBackend::BgeM3,
                    ..config.clone()
                };
                Arc::new(BgeM3Backend::new(fallback_config)?)
            }
            ModelBackend::GteLarge | ModelBackend::E5LargeV2 => {
                tracing::warn!(
                    "{:?} not yet implemented, falling back to BGE-M3",
                    config.backend
                );
                let fallback_config = EmbeddingConfig {
                    backend: ModelBackend::BgeM3,
                    ..config.clone()
                };
                Arc::new(BgeM3Backend::new(fallback_config)?)
            }
        };

        Ok(backend)
    }

    /// Get embedding for text
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        self.stats.total_calls += 1;
        let start = Instant::now();
        let result = self.backend.embed(text);
        self.stats.total_time_us += start.elapsed().as_micros() as u64;
        result
    }

    /// Get embeddings for multiple texts
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.stats.total_calls += texts.len() as u64;
        let start = Instant::now();
        let result = self.backend.embed_batch(texts);
        self.stats.total_time_us += start.elapsed().as_micros() as u64;
        result
    }

    /// Extract activation from specific layer
    pub fn extract_layer(&mut self, text: &str, layer_idx: usize) -> Result<LayerOutput> {
        self.stats.layer_extractions += 1;
        self.backend.extract_layer(text, layer_idx)
    }

    /// Extract activations from multiple layers
    pub fn extract_layers(
        &mut self,
        text: &str,
        layer_indices: &[usize],
    ) -> Result<Vec<LayerOutput>> {
        self.stats.layer_extractions += layer_indices.len() as u64;
        self.backend.extract_layers(text, layer_indices)
    }

    /// Extract all layers for full H2 hypothesis analysis
    pub fn extract_all_layers(&mut self, text: &str) -> Result<Vec<LayerOutput>> {
        let indices: Vec<usize> = (0..self.backend.num_layers()).collect();
        self.extract_layers(text, &indices)
    }

    /// Get the phenomenal corridor layer
    pub fn phenomenal_corridor_layer(&self) -> usize {
        self.backend.phenomenal_corridor_layer()
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.backend.dimension()
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.backend.num_layers()
    }

    /// Get model name
    pub fn model_name(&self) -> &str {
        self.backend.model_name()
    }

    /// Check if layer extraction is supported
    pub fn supports_layer_extraction(&self) -> bool {
        self.backend.supports_layer_extraction()
    }

    /// Check if using GPU
    pub fn is_gpu(&self) -> bool {
        self.backend.is_gpu()
    }

    /// Get statistics
    pub fn stats(&self) -> &EmbedderStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = EmbedderStats::default();
    }

    /// Get configuration
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

// ============================================================================
// H2 Hypothesis Testing Support
// ============================================================================

/// Result of layer-by-layer phenomenal analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAnalysisResult {
    /// Text being analyzed
    pub text: String,
    /// Per-layer phenomenal scores
    pub layer_scores: Vec<LayerScore>,
    /// Detected phenomenal corridor
    pub phenomenal_corridor: PhenomenalCorridor,
    /// Total analysis time (microseconds)
    pub total_time_us: u64,
}

/// Score for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerScore {
    /// Layer index
    pub layer_idx: usize,
    /// Depth fraction (0.0 = first layer, 1.0 = last layer)
    pub depth_fraction: f64,
    /// Phenomenal score (higher = more phenomenal)
    pub phenomenal_score: f64,
    /// Topological unity
    pub unity: f64,
    /// Phi loading
    pub phi_loading: f64,
}

/// Detected phenomenal corridor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhenomenalCorridor {
    /// Peak layer index
    pub peak_layer: usize,
    /// Peak depth fraction
    pub peak_depth: f64,
    /// Peak phenomenal score
    pub peak_score: f64,
    /// Corridor width (number of layers with high scores)
    pub width: usize,
    /// Whether this matches the expected ~92% depth
    pub matches_h2_prediction: bool,
}

/// Analyzer for H2 hypothesis testing
#[cfg(feature = "neural-bridge")]
pub struct PhenomenalLayerAnalyzer {
    /// Unified embedder
    embedder: UnifiedEmbedder,
    /// Phi signature for scoring
    phi_signature: Vec<f64>,
}

#[cfg(feature = "neural-bridge")]
impl PhenomenalLayerAnalyzer {
    /// Create analyzer with default settings
    pub fn new() -> Result<Self> {
        let embedder = UnifiedEmbedder::auto()?;
        let phi_signature = Self::default_phi_signature();

        Ok(Self {
            embedder,
            phi_signature,
        })
    }

    /// Create with custom embedder
    pub fn with_embedder(embedder: UnifiedEmbedder) -> Self {
        Self {
            embedder,
            phi_signature: Self::default_phi_signature(),
        }
    }

    /// Load pre-computed Phi signature
    fn default_phi_signature() -> Vec<f64> {
        let mut phi = vec![0.0; 1024];

        // Top contributing dimensions from phenomenal corridor research
        phi[297] = -0.2636;
        phi[616] = 0.1194;
        phi[428] = 0.1098;
        phi[122] = -0.1085;
        phi[743] = -0.1036;
        phi[520] = 0.0982;
        phi[549] = 0.0955;
        phi[917] = -0.0931;
        phi[813] = -0.0880;
        phi[346] = -0.0854;
        phi[89] = 0.0812;
        phi[445] = -0.0798;
        phi[672] = 0.0756;
        phi[201] = -0.0734;
        phi[888] = 0.0712;

        // Normalize
        let norm: f64 = phi.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in phi.iter_mut() {
                *x /= norm;
            }
        }

        phi
    }

    /// Set custom Phi signature
    pub fn set_phi_signature(&mut self, phi: Vec<f64>) {
        self.phi_signature = phi;
    }

    /// Analyze layer-by-layer phenomenal content
    pub fn analyze(&mut self, text: &str) -> Result<LayerAnalysisResult> {
        let start = Instant::now();
        let num_layers = self.embedder.num_layers();

        // Extract all layers
        let layers = self.embedder.extract_all_layers(text)?;

        // Score each layer
        let mut layer_scores = Vec::with_capacity(num_layers);
        let mut max_score = f64::MIN;
        let mut peak_layer = 0;

        for output in &layers {
            let depth_fraction =
                output.layer_idx as f64 / num_layers.saturating_sub(1).max(1) as f64;

            // Compute Phi loading
            let phi_loading = self.compute_phi_loading(&output.activation);

            // Compute unity (simplified version)
            let unity = self.compute_unity(&output.activation);

            // Combined phenomenal score
            let phenomenal_score = 0.2 * unity + 0.8 * phi_loading;

            if phenomenal_score > max_score {
                max_score = phenomenal_score;
                peak_layer = output.layer_idx;
            }

            layer_scores.push(LayerScore {
                layer_idx: output.layer_idx,
                depth_fraction,
                phenomenal_score,
                unity,
                phi_loading,
            });
        }

        // Detect corridor
        let peak_depth = peak_layer as f64 / (num_layers - 1) as f64;
        let threshold = max_score * 0.8; // 80% of peak
        let width = layer_scores
            .iter()
            .filter(|s| s.phenomenal_score >= threshold)
            .count();

        // Check H2 prediction (peak should be around 92% depth)
        let matches_h2 = (peak_depth - 0.92).abs() < 0.1;

        let corridor = PhenomenalCorridor {
            peak_layer,
            peak_depth,
            peak_score: max_score,
            width,
            matches_h2_prediction: matches_h2,
        };

        Ok(LayerAnalysisResult {
            text: text.to_string(),
            layer_scores,
            phenomenal_corridor: corridor,
            total_time_us: start.elapsed().as_micros() as u64,
        })
    }

    fn compute_phi_loading(&self, activation: &[f32]) -> f64 {
        let mut loading = 0.0;
        let len = activation.len().min(self.phi_signature.len());

        for i in 0..len {
            loading += activation[i] as f64 * self.phi_signature[i];
        }

        // Normalize to 0-10 scale
        (loading.abs() * 10.0).min(10.0)
    }

    fn compute_unity(&self, activation: &[f32]) -> f64 {
        if activation.is_empty() {
            return 0.5;
        }
        // Simplified unity measure based on activation distribution
        let n = activation.len() as f32;
        let mean: f32 = activation.iter().sum::<f32>() / n;
        let variance: f32 = activation.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;

        // Lower variance = higher unity (more coherent representation)
        let normalized_var = (variance / 10.0).min(1.0);
        1.0 - normalized_var as f64
    }

    /// Get underlying embedder
    pub fn embedder(&self) -> &UnifiedEmbedder {
        &self.embedder
    }

    /// Get mutable embedder
    pub fn embedder_mut(&mut self) -> &mut UnifiedEmbedder {
        &mut self.embedder
    }
}

// ============================================================================
// HDC Integration
// ============================================================================

/// Project embedding to BinaryHV space
#[cfg(feature = "neural-bridge")]
pub fn project_to_hv16(embedding: &[f32]) -> BinaryHV {
    activation_to_hv16(embedding)
}

/// Project embedding to PackedBipolar via linear probe
#[cfg(feature = "neural-bridge")]
pub fn project_to_packed(
    embedding: &[f32],
    probe: &super::neural_bridge::NeuralBridge,
) -> Result<PackedBipolar> {
    probe.project_to_packed(embedding)
}

// ============================================================================
// Model Info and Status
// ============================================================================

/// Information about available models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub backend: ModelBackend,
    pub model_id: String,
    pub dimension: usize,
    pub num_layers: usize,
    pub supports_layer_extraction: bool,
    pub phenomenal_corridor_layer: usize,
    pub architecture: ModelArchitecture,
    pub status: ModelStatus,
}

/// Model implementation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStatus {
    /// Fully implemented and tested
    Production,
    /// Implemented but needs more testing
    Beta,
    /// Not yet implemented, will fall back
    NotImplemented,
}

impl ModelBackend {
    /// Get full model info
    pub fn info(&self) -> ModelInfo {
        let status = match self {
            Self::BgeM3 | Self::Auto => ModelStatus::Production,
            Self::ModernBert | Self::ModernBertLarge => ModelStatus::NotImplemented,
            Self::NomicEmbed => ModelStatus::NotImplemented,
            Self::GteLarge => ModelStatus::NotImplemented,
            Self::E5LargeV2 => ModelStatus::NotImplemented,
        };

        ModelInfo {
            backend: *self,
            model_id: self.model_id().to_string(),
            dimension: self.dimension(),
            num_layers: self.num_layers(),
            supports_layer_extraction: self.supports_layer_extraction(),
            phenomenal_corridor_layer: ((self.num_layers() as f64) * 0.92) as usize,
            architecture: self.architecture(),
            status,
        }
    }
}

/// Get info for all available models
pub fn all_model_info() -> Vec<ModelInfo> {
    vec![
        ModelBackend::BgeM3.info(),
        ModelBackend::ModernBert.info(),
        ModelBackend::ModernBertLarge.info(),
        ModelBackend::NomicEmbed.info(),
        ModelBackend::GteLarge.info(),
        ModelBackend::E5LargeV2.info(),
    ]
}

/// Print summary of model support
pub fn print_model_summary() {
    println!("========================================================================");
    println!("                    MODERN EMBEDDING MODELS STATUS                       ");
    println!("========================================================================\n");

    println!("Model               | Dim  | Layers | Layer Extract | Corridor | Status");
    println!("--------------------|------|--------|---------------|----------|--------");

    for info in all_model_info() {
        let status_str = match info.status {
            ModelStatus::Production => "READY",
            ModelStatus::Beta => "BETA",
            ModelStatus::NotImplemented => "PENDING",
        };

        let extract_str = if info.supports_layer_extraction {
            "Yes"
        } else {
            "No"
        };

        println!(
            "{:18} | {:4} | {:6} | {:13} | {:8} | {}",
            info.model_id.split('/').last().unwrap_or(&info.model_id),
            info.dimension,
            info.num_layers,
            extract_str,
            info.phenomenal_corridor_layer,
            status_str
        );
    }

    println!("\n");
    println!("Primary Backend: BGE-M3 (BAAI/bge-m3)");
    println!("  - 1024 dimensions, 24 layers");
    println!("  - Full layer-wise extraction implemented");
    println!("  - Validated for phenomenal corridor detection (Layer 22, d=+0.69)");
    println!("\n");
    println!("Fallback: All other models fall back to BGE-M3 until implemented");
    println!("========================================================================");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_backend_info() {
        let info = ModelBackend::BgeM3.info();
        assert_eq!(info.dimension, 1024);
        assert_eq!(info.num_layers, 24);
        assert!(info.supports_layer_extraction);
        assert_eq!(info.phenomenal_corridor_layer, 22);
    }

    #[test]
    fn test_model_backend_dimension() {
        assert_eq!(ModelBackend::BgeM3.dimension(), 1024);
        assert_eq!(ModelBackend::ModernBert.dimension(), 768);
        assert_eq!(ModelBackend::ModernBertLarge.dimension(), 1024);
    }

    #[test]
    fn test_model_backend_layers() {
        assert_eq!(ModelBackend::BgeM3.num_layers(), 24);
        assert_eq!(ModelBackend::ModernBert.num_layers(), 12);
        assert_eq!(ModelBackend::ModernBertLarge.num_layers(), 28);
    }

    #[test]
    fn test_default_config() {
        let config = EmbeddingConfig::default();
        assert!(config.use_gpu);
        assert!(config.enable_cache);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_all_model_info() {
        let infos = all_model_info();
        assert!(!infos.is_empty());
        assert!(infos.iter().any(|i| i.backend == ModelBackend::BgeM3));
    }

    #[test]
    #[ignore = "requires embedding model files (BGE-M3 etc.) on disk"]
    fn test_unified_embedder_creation() {
        let embedder = UnifiedEmbedder::auto();
        assert!(embedder.is_ok());
    }
}
