// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neural Bridge v2: Unified Text → HDC Pipeline
//!
//! Complete pipeline from raw text to 16,384-dimensional HDC vectors
//! using BGE-M3 (pure Rust via Candle) + trained linear probe.
//!
//! ## Key Improvements over v1
//!
//! - **Pure Rust**: No Python/Ollama dependency, self-contained binary
//! - **Low Latency**: ~50-200ms vs 5-18s (25-100x faster)
//! - **GPU Support**: Optional CUDA acceleration
//! - **Multilingual**: 100+ languages via BGE-M3
//! - **Epistemic Metadata**: Optional uncertainty/stability tracking
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
//! │    Text      │     │   BGE-M3     │     │   Linear     │     │    HDC       │
//! │  "Justice"   │────>│  Encoder     │────>│   Probe      │────>│   Vector     │
//! │              │     │  [1024-dim]  │     │  [16384-dim] │     │  PackedBip.  │
//! └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::perception::NeuralBridgeV2;
//!
//! // Load model + probe
//! let bridge = NeuralBridgeV2::load_default()?;
//!
//! // Encode text directly to HDC
//! let hdc = bridge.encode_to_hdc("What is consciousness?")?;
//!
//! // Compare concepts
//! let justice = bridge.encode_to_hdc("justice")?;
//! let fairness = bridge.encode_to_hdc("fairness")?;
//! let similarity = justice.xor_similarity(&fairness);
//! println!("Similarity: {:.4}", similarity); // High similarity expected
//! ```

#[cfg(feature = "neural-bridge")]
use anyhow::{Context, Result, bail};

#[cfg(feature = "neural-bridge")]
use std::time::Instant;

#[cfg(feature = "neural-bridge")]
use symthaea_core::hdc::{HDC_DIMENSION, PackedBipolar};

#[cfg(feature = "neural-bridge")]
use super::bge_m3::{BGE_M3_DIM, BGE_M3_MODEL_ID, BgeM3};

#[cfg(feature = "neural-bridge")]
use super::neural_bridge::NeuralBridge;

#[cfg(feature = "neural-bridge")]
use super::epistemic_vector::{EpistemicSemanticVector, UncertaintySource};

/// Expected probe input dimension for BGE-M3
#[cfg(feature = "neural-bridge")]
pub const PROBE_INPUT_DIM: usize = BGE_M3_DIM; // 1024

/// Default probe weights path
#[cfg(feature = "neural-bridge")]
pub const DEFAULT_PROBE_PATH: &str = "models/neural_bridge/probe_weights_bge_m3.npy";

/// Neural Bridge v2 configuration
#[cfg(feature = "neural-bridge")]
#[derive(Debug, Clone)]
pub struct NeuralBridgeV2Config {
    /// Model ID for BGE-M3 (HuggingFace Hub)
    pub model_id: String,

    /// Path to trained probe weights
    pub probe_path: Option<String>,

    /// Enable epistemic metadata tracking
    pub track_epistemic: bool,

    /// Cache embeddings for repeated concepts
    pub enable_cache: bool,

    /// Maximum cache size (number of entries)
    pub max_cache_size: usize,
}

#[cfg(feature = "neural-bridge")]
impl Default for NeuralBridgeV2Config {
    fn default() -> Self {
        Self {
            model_id: BGE_M3_MODEL_ID.to_string(),
            probe_path: None,
            track_epistemic: true,
            enable_cache: true,
            max_cache_size: 1000,
        }
    }
}

/// Neural Bridge v2: Complete text → HDC pipeline
#[cfg(feature = "neural-bridge")]
pub struct NeuralBridgeV2 {
    /// BGE-M3 encoder
    encoder: BgeM3,

    /// Linear probe: 1024 → 16384
    probe: NeuralBridge,

    /// Configuration
    config: NeuralBridgeV2Config,

    /// Embedding cache (text → 1024-dim embedding)
    embedding_cache: std::collections::HashMap<String, Vec<f32>>,

    /// HDC cache (text → PackedBipolar) - saves ~30ms probe time per hit
    hdc_cache: std::collections::HashMap<String, PackedBipolar>,

    /// Statistics for monitoring
    stats: BridgeStats,
}

/// Runtime statistics for monitoring and optimization
#[cfg(feature = "neural-bridge")]
#[derive(Debug, Default, Clone)]
pub struct BridgeStats {
    /// Total encode calls
    pub total_calls: u64,

    /// Embedding cache hits (saves ~350ms encoder time)
    pub embedding_cache_hits: u64,

    /// HDC cache hits (saves full ~380ms pipeline)
    pub hdc_cache_hits: u64,

    /// Total encoding time (microseconds)
    pub total_encode_us: u64,

    /// Total probe time (microseconds)
    pub total_probe_us: u64,
}

#[cfg(feature = "neural-bridge")]
impl BridgeStats {
    /// HDC cache hit rate (full pipeline cached)
    pub fn hdc_cache_hit_rate(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.hdc_cache_hits as f64 / self.total_calls as f64
        }
    }

    /// Embedding cache hit rate (encoder cached, probe still runs)
    pub fn embedding_cache_hit_rate(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.embedding_cache_hits as f64 / self.total_calls as f64
        }
    }

    /// Combined cache hit rate (any cache hit)
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            (self.hdc_cache_hits + self.embedding_cache_hits) as f64 / self.total_calls as f64
        }
    }

    /// Average encode latency in milliseconds (excluding cache hits)
    pub fn avg_encode_ms(&self) -> f64 {
        let uncached_calls = self
            .total_calls
            .saturating_sub(self.hdc_cache_hits + self.embedding_cache_hits);
        if uncached_calls == 0 {
            0.0
        } else {
            (self.total_encode_us as f64 / uncached_calls as f64) / 1000.0
        }
    }

    /// Average probe latency in milliseconds (excluding HDC cache hits)
    pub fn avg_probe_ms(&self) -> f64 {
        let probe_calls = self.total_calls.saturating_sub(self.hdc_cache_hits);
        if probe_calls == 0 {
            0.0
        } else {
            (self.total_probe_us as f64 / probe_calls as f64) / 1000.0
        }
    }
}

#[cfg(feature = "neural-bridge")]
impl NeuralBridgeV2 {
    /// Load with default settings.
    ///
    /// Downloads BGE-M3 from HuggingFace Hub and loads probe from default path.
    pub fn load_default() -> Result<Self> {
        Self::load(NeuralBridgeV2Config::default())
    }

    /// Load with custom configuration.
    pub fn load(config: NeuralBridgeV2Config) -> Result<Self> {
        // Load BGE-M3 encoder
        let encoder =
            BgeM3::load_from_hub(&config.model_id).context("Failed to load BGE-M3 encoder")?;

        // Load probe
        let probe_path = config.probe_path.as_deref().unwrap_or(DEFAULT_PROBE_PATH);

        let probe = NeuralBridge::load(probe_path).context("Failed to load probe weights")?;

        // Validate dimensions match
        if probe.input_dim() != encoder.dim() {
            bail!(
                "Dimension mismatch: encoder produces {} dims, probe expects {}",
                encoder.dim(),
                probe.input_dim()
            );
        }

        Ok(Self {
            encoder,
            probe,
            config,
            embedding_cache: std::collections::HashMap::new(),
            hdc_cache: std::collections::HashMap::new(),
            stats: BridgeStats::default(),
        })
    }

    /// Load encoder only (no probe - useful for training new probes).
    pub fn encoder_only(model_id: &str) -> Result<Self> {
        let encoder = BgeM3::load_from_hub(model_id).context("Failed to load BGE-M3 encoder")?;

        // Create dummy identity probe (for API compatibility)
        let identity_weights = vec![0.0f32; encoder.dim() * HDC_DIMENSION];
        let probe = NeuralBridge::from_weights(identity_weights, encoder.dim(), HDC_DIMENSION)?;

        Ok(Self {
            encoder,
            probe,
            config: NeuralBridgeV2Config::default(),
            embedding_cache: std::collections::HashMap::new(),
            hdc_cache: std::collections::HashMap::new(),
            stats: BridgeStats::default(),
        })
    }

    /// Encode text to BGE-M3 embedding (1024-dim).
    ///
    /// This is the raw encoder output before probe projection.
    /// For most use cases, prefer `encode_to_hdc` which includes caching
    /// of the final HDC vectors.
    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>> {
        let start = Instant::now();

        // Check embedding cache
        if self.config.enable_cache {
            if let Some(cached) = self.embedding_cache.get(text) {
                self.stats.embedding_cache_hits += 1;
                return Ok(cached.clone());
            }
        }

        // Encode via BGE-M3
        let embedding = self.encoder.encode(text)?;

        // Update embedding cache
        if self.config.enable_cache {
            if self.embedding_cache.len() >= self.config.max_cache_size {
                // Simple eviction: clear half the cache
                let keys: Vec<_> = self
                    .embedding_cache
                    .keys()
                    .take(self.config.max_cache_size / 2)
                    .cloned()
                    .collect();
                for key in keys {
                    self.embedding_cache.remove(&key);
                }
            }
            self.embedding_cache
                .insert(text.to_string(), embedding.clone());
        }

        self.stats.total_encode_us += start.elapsed().as_micros() as u64;

        Ok(embedding)
    }

    /// Encode text directly to HDC PackedBipolar.
    ///
    /// This is the main entry point for consciousness processing.
    /// Results are cached - repeated calls with the same text return in <1ms.
    pub fn encode_to_hdc(&mut self, text: &str) -> Result<PackedBipolar> {
        self.stats.total_calls += 1;

        // Check HDC cache first (saves full ~380ms pipeline)
        if self.config.enable_cache {
            if let Some(cached) = self.hdc_cache.get(text) {
                self.stats.hdc_cache_hits += 1;
                return Ok(cached.clone());
            }
        }

        // Get embedding (may be cached)
        let embedding = self.encode(text)?;

        // Project to HDC
        let probe_start = Instant::now();
        let packed = self.probe.project_to_packed(&embedding)?;
        self.stats.total_probe_us += probe_start.elapsed().as_micros() as u64;

        // Cache the HDC result
        if self.config.enable_cache {
            if self.hdc_cache.len() >= self.config.max_cache_size {
                // Simple eviction: clear half the cache
                let keys: Vec<_> = self
                    .hdc_cache
                    .keys()
                    .take(self.config.max_cache_size / 2)
                    .cloned()
                    .collect();
                for key in keys {
                    self.hdc_cache.remove(&key);
                }
            }
            self.hdc_cache.insert(text.to_string(), packed.clone());
        }

        Ok(packed)
    }

    /// Encode with epistemic metadata.
    ///
    /// Returns the HDC vector along with uncertainty estimates.
    pub fn encode_epistemic(&mut self, text: &str) -> Result<EpistemicSemanticVector> {
        // Get main embedding
        let embedding = self.encode(text)?;

        // Project to HDC
        let probe_start = Instant::now();
        let continuous = self.probe.project(&embedding)?;
        let packed = PackedBipolar::from_continuous(&continuous);
        let probe_time = probe_start.elapsed();
        self.stats.total_probe_us += probe_time.as_micros() as u64;

        if !self.config.track_epistemic {
            // Return without epistemic metadata
            return Ok(EpistemicSemanticVector::from_packed(packed));
        }

        // Calculate epistemic metadata from projection margins
        let mut uncertainty_sources = Vec::new();

        // 1. Projection margin uncertainty (how close to decision boundary)
        let margins: Vec<f32> = continuous.iter().map(|x| x.abs()).collect();
        let n = margins.len().max(1) as f32;
        let avg_margin = margins.iter().sum::<f32>() / n;
        let low_margin_count = margins.iter().filter(|&&m| m < 0.1).count();
        let margin_uncertainty = low_margin_count as f32 / n;

        if margin_uncertainty > 0.1 {
            uncertainty_sources.push(UncertaintySource::ProjectionMargin {
                avg_margin,
                low_margin_ratio: margin_uncertainty,
            });
        }

        // 2. Embedding norm (unusual norm might indicate OOD)
        let embedding_norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        // BGE-M3 produces L2-normalized embeddings, norm should be ~1.0
        let norm_deviation = (embedding_norm - 1.0).abs();
        if norm_deviation > 0.1 {
            uncertainty_sources.push(UncertaintySource::EmbeddingNorm {
                norm: embedding_norm,
                expected: 1.0,
            });
        }

        // Overall confidence based on margin
        let confidence = 1.0 - margin_uncertainty.min(1.0);

        Ok(EpistemicSemanticVector {
            vector: packed,
            confidence,
            stability: Some(avg_margin),
            uncertainty_sources,
            source_text: Some(text.to_string()),
            encode_time_us: (self.stats.total_encode_us / self.stats.total_calls.max(1)) as u32,
        })
    }

    /// Batch encode multiple texts.
    pub fn encode_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let start = Instant::now();
        let embeddings = self.encoder.encode_batch(texts)?;

        self.stats.total_calls += texts.len() as u64;
        self.stats.total_encode_us += start.elapsed().as_micros() as u64;

        Ok(embeddings)
    }

    /// Batch encode to HDC vectors.
    pub fn encode_batch_to_hdc(&mut self, texts: &[&str]) -> Result<Vec<PackedBipolar>> {
        let embeddings = self.encode_batch(texts)?;

        let probe_start = Instant::now();
        let results: Result<Vec<_>> = embeddings
            .iter()
            .map(|emb| self.probe.project_to_packed(emb))
            .collect();
        self.stats.total_probe_us += probe_start.elapsed().as_micros() as u64;

        results
    }

    /// Get runtime statistics.
    pub fn stats(&self) -> &BridgeStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = BridgeStats::default();
    }

    /// Clear all caches (embedding + HDC).
    pub fn clear_cache(&mut self) {
        self.embedding_cache.clear();
        self.hdc_cache.clear();
    }

    /// Get combined cache size (embedding + HDC entries).
    pub fn cache_size(&self) -> usize {
        self.embedding_cache.len() + self.hdc_cache.len()
    }

    /// Get HDC cache size.
    pub fn hdc_cache_size(&self) -> usize {
        self.hdc_cache.len()
    }

    /// Get embedding cache size.
    pub fn embedding_cache_size(&self) -> usize {
        self.embedding_cache.len()
    }

    /// Check if using CUDA.
    pub fn is_cuda(&self) -> bool {
        self.encoder.is_cuda()
    }

    /// Get encoder dimension.
    pub fn encoder_dim(&self) -> usize {
        self.encoder.dim()
    }

    /// Get probe input dimension.
    pub fn probe_input_dim(&self) -> usize {
        self.probe.input_dim()
    }

    /// Get probe output dimension (always HDC_DIMENSION).
    pub fn probe_output_dim(&self) -> usize {
        self.probe.output_dim()
    }

    /// Get reference to underlying probe (for advanced use).
    pub fn probe(&self) -> &NeuralBridge {
        &self.probe
    }
}

/// Builder for NeuralBridgeV2
#[cfg(feature = "neural-bridge")]
pub struct NeuralBridgeV2Builder {
    config: NeuralBridgeV2Config,
}

#[cfg(feature = "neural-bridge")]
impl NeuralBridgeV2Builder {
    pub fn new() -> Self {
        Self {
            config: NeuralBridgeV2Config::default(),
        }
    }

    /// Set model ID for BGE-M3.
    pub fn model_id(mut self, model_id: &str) -> Self {
        self.config.model_id = model_id.to_string();
        self
    }

    /// Set probe weights path.
    pub fn probe_path(mut self, path: &str) -> Self {
        self.config.probe_path = Some(path.to_string());
        self
    }

    /// Enable/disable epistemic tracking.
    pub fn track_epistemic(mut self, enable: bool) -> Self {
        self.config.track_epistemic = enable;
        self
    }

    /// Enable/disable caching.
    pub fn enable_cache(mut self, enable: bool) -> Self {
        self.config.enable_cache = enable;
        self
    }

    /// Set maximum cache size.
    pub fn max_cache_size(mut self, size: usize) -> Self {
        self.config.max_cache_size = size;
        self
    }

    /// Build the NeuralBridgeV2.
    pub fn build(self) -> Result<NeuralBridgeV2> {
        NeuralBridgeV2::load(self.config)
    }
}

#[cfg(feature = "neural-bridge")]
impl Default for NeuralBridgeV2Builder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for quick single-text encoding.
///
/// Creates a temporary bridge and encodes. Use NeuralBridgeV2 directly
/// for repeated calls to avoid model reload overhead.
#[cfg(feature = "neural-bridge")]
pub fn quick_encode(text: &str) -> Result<PackedBipolar> {
    let mut bridge = NeuralBridgeV2::load_default()?;
    bridge.encode_to_hdc(text)
}

#[cfg(test)]
#[cfg(feature = "neural-bridge")]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = NeuralBridgeV2Config::default();
        assert_eq!(config.model_id, BGE_M3_MODEL_ID);
        assert!(config.track_epistemic);
        assert!(config.enable_cache);
    }

    #[test]
    fn test_builder() {
        let config = NeuralBridgeV2Builder::new()
            .model_id("custom/model")
            .probe_path("custom/probe.npy")
            .track_epistemic(false)
            .enable_cache(false)
            .max_cache_size(500)
            .config;

        assert_eq!(config.model_id, "custom/model");
        assert_eq!(config.probe_path, Some("custom/probe.npy".to_string()));
        assert!(!config.track_epistemic);
        assert!(!config.enable_cache);
        assert_eq!(config.max_cache_size, 500);
    }

    #[test]
    fn test_stats_calculations() {
        let mut stats = BridgeStats::default();
        assert_eq!(stats.cache_hit_rate(), 0.0);
        assert_eq!(stats.avg_encode_ms(), 0.0);

        stats.total_calls = 100;
        stats.hdc_cache_hits = 15;
        stats.embedding_cache_hits = 10;
        stats.total_encode_us = 75_000; // 75ms total for 75 uncached calls

        // Combined hit rate: (15+10)/100 = 0.25
        assert_eq!(stats.cache_hit_rate(), 0.25);
        // HDC hit rate: 15/100 = 0.15
        assert_eq!(stats.hdc_cache_hit_rate(), 0.15);
        // Avg encode time: 75ms / 75 uncached calls = 1.0ms
        assert_eq!(stats.avg_encode_ms(), 1.0);
    }
}
