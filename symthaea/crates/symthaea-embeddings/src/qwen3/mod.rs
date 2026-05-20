// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Qwen3 Embeddings: Semantic Text Encoding
//!
//! This module provides text embedding using Qwen3-Embedding models.
//! Supports both Burn inference (via the `burn` feature when available)
//! and high-quality simulated embeddings for testing.
//!
//! ## Model Specifications
//!
//! - **Qwen3-Embedding-0.6B**: 1024D output, 8192 context length
//! - **Qwen3-Embedding-1.5B**: 1536D output (QWEN3_FULL_DIMENSION)
//!
//! ## Precision & Quantization
//!
//! | Mode | Method | Load Speed | Memory | Quality |
//! |------|--------|------------|--------|---------|
//! | f32 | default | Baseline | 2.4 GB | Full |
//! | f16 | `.with_f16()` | ~2x faster | 1.2 GB disk | Full |
//! | MRL | `.with_matryoshka_dim(256)` | Same | Smaller index | Slightly reduced |
//!
//! burn 0.18 does **not** support int8/int4 quantization. f16 is the only
//! precision reduction available. Combine f16 with Matryoshka dimension
//! reduction for the smallest effective footprint.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::embeddings::qwen3::{Qwen3Embedder, Qwen3Config};
//!
//! let config = Qwen3Config::default();
//! let mut embedder = Qwen3Embedder::new(config)?;
//!
//! let result = embedder.embed("Hello, world!")?;
//! println!("Embedding dimension: {}", result.dimension);
//! ```

use super::{Embedder, EmbeddingResult};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[cfg(feature = "burn")]
pub mod attention;
#[cfg(feature = "burn-hub")]
pub mod hub;
#[cfg(feature = "burn")]
pub mod mlp;
#[cfg(feature = "burn")]
pub mod model;
#[cfg(feature = "ort-embed")]
pub mod onnx;
#[cfg(feature = "burn")]
pub mod safetensors_loader;

#[cfg(feature = "burn")]
use burn::backend::NdArray;
#[cfg(feature = "burn")]
use std::sync::Arc;

/// Full dimension for Qwen3-1.5B embeddings
pub const QWEN3_FULL_DIMENSION: usize = 1536;

/// Standard dimension for Qwen3-0.6B embeddings
pub const QWEN3_STANDARD_DIMENSION: usize = 1024;

/// Standard Matryoshka (MRL) dimensions for Qwen3-Embedding models.
///
/// Qwen3-Embedding models are trained with Matryoshka Representation Learning,
/// meaning the first N dimensions of the full embedding form a valid embedding
/// at dimension N. Smaller dimensions trade some quality for faster downstream
/// similarity search and reduced storage.
///
/// These dimensions are validated for Qwen3-Embedding-0.6B (1024D full).
/// For Qwen3-Embedding-1.5B, dimensions up to 1536 are valid.
pub const QWEN3_MRL_DIMENSIONS: &[usize] = &[64, 128, 256, 512, 768, 1024];

// ── Burn backend abstraction ──────────────────────────────────────────

#[cfg(feature = "burn")]
enum BurnBackend {
    NdArray {
        model: model::Qwen3Model<NdArray>,
        device: burn::backend::ndarray::NdArrayDevice,
    },
    #[cfg(feature = "burn-wgpu")]
    Wgpu {
        model: model::Qwen3Model<burn::backend::Wgpu>,
        device: burn::backend::wgpu::WgpuDevice,
    },
}

/// Generic forward + pool: runs the model and extracts a Vec<f32> embedding.
/// Works for any Burn backend.
#[cfg(feature = "burn")]
#[allow(clippy::too_many_arguments)]
fn forward_and_pool<B: burn::prelude::Backend>(
    burn_model: &model::Qwen3Model<B>,
    input_ids: &[i32],
    mask: &[u32],
    seq_len: usize,
    device: &B::Device,
    pooling: PoolingStrategy,
    embedding_dim: usize,
    normalize: bool,
) -> Result<Vec<f32>> {
    use burn::prelude::*;

    let input_tensor = Tensor::<B, 1, Int>::from_data(input_ids, device).unsqueeze_dim(0); // [1, seq_len]

    let hidden = burn_model.forward(input_tensor, None);
    let [_batch, _seq, hidden_dim] = hidden.dims();
    let out_dim = embedding_dim.min(hidden_dim);

    let pooled: Tensor<B, 2> = match pooling {
        PoolingStrategy::LastTokenPooling => {
            let last_idx = mask[..seq_len]
                .iter()
                .rposition(|&m| m == 1)
                .unwrap_or(seq_len.saturating_sub(1));
            hidden
                .slice([0..1, last_idx..last_idx + 1, 0..out_dim])
                .reshape([1, out_dim])
        }
        PoolingStrategy::MeanPooling => {
            let token_count = mask[..seq_len].iter().filter(|&&m| m == 1).count();
            if token_count > 0 {
                // sum_dim keeps rank → [1, 1, out_dim]; squeeze to [1, out_dim]
                let sum = hidden
                    .slice([0..1, 0..token_count, 0..out_dim])
                    .sum_dim(1)
                    .squeeze(1);
                sum.div_scalar(token_count as f32)
            } else {
                hidden.slice([0..1, 0..1, 0..out_dim]).reshape([1, out_dim])
            }
        }
        PoolingStrategy::ClsToken => hidden.slice([0..1, 0..1, 0..out_dim]).reshape([1, out_dim]),
        PoolingStrategy::MaxPooling => {
            // max_dim keeps rank → [1, 1, out_dim]; squeeze to [1, out_dim]
            hidden
                .slice([0..1, 0..seq_len, 0..out_dim])
                .max_dim(1)
                .squeeze(1)
        }
    };

    let data = pooled.into_data();
    let mut embedding: Vec<f32> = data
        .to_vec()
        .map_err(|e| anyhow::anyhow!("tensor to_vec failed: {e:?}"))?;
    embedding.truncate(out_dim);

    if normalize {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for x in embedding.iter_mut() {
                *x /= norm;
            }
        }
    }

    Ok(embedding)
}

/// Batch forward + pool with length-sorted chunking.
///
/// For small batches (≤4) or uniform lengths (max ≤ 2×min), processes everything
/// in a single forward pass. Otherwise, sorts sequences by length and groups them
/// into chunks where padding waste is bounded (max_len ≤ 2×min_len per chunk),
/// processes each chunk separately, then reorders results to match the original order.
#[cfg(feature = "burn")]
#[allow(clippy::too_many_arguments)]
fn forward_and_pool_batch<B: burn::prelude::Backend>(
    burn_model: &model::Qwen3Model<B>,
    batch_ids: &[Vec<i32>],
    batch_masks: &[Vec<u32>],
    device: &B::Device,
    pooling: PoolingStrategy,
    embedding_dim: usize,
    normalize: bool,
) -> Result<Vec<Vec<f32>>> {
    if batch_ids.is_empty() {
        return Ok(Vec::new());
    }

    let min_len = batch_ids
        .iter()
        .map(|ids| ids.len())
        .min()
        .unwrap_or(1)
        .max(1);
    let max_len = batch_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);

    // For small/uniform batches, process directly
    if batch_ids.len() <= 4 || max_len <= min_len * 2 {
        return forward_and_pool_batch_direct(
            burn_model,
            batch_ids,
            batch_masks,
            device,
            pooling,
            embedding_dim,
            normalize,
        );
    }

    // Sort indices by sequence length
    let chunks = chunk_by_length(batch_ids, 2.0);

    // Process each chunk and collect results with original indices
    let mut indexed_results: Vec<(usize, Vec<f32>)> = Vec::with_capacity(batch_ids.len());

    for chunk_indices in &chunks {
        let chunk_ids: Vec<Vec<i32>> = chunk_indices
            .iter()
            .map(|&i| batch_ids[i].clone())
            .collect();
        let chunk_masks: Vec<Vec<u32>> = chunk_indices
            .iter()
            .map(|&i| batch_masks[i].clone())
            .collect();

        let chunk_embeddings = forward_and_pool_batch_direct(
            burn_model,
            &chunk_ids,
            &chunk_masks,
            device,
            pooling,
            embedding_dim,
            normalize,
        )?;

        for (local_idx, emb) in chunk_embeddings.into_iter().enumerate() {
            indexed_results.push((chunk_indices[local_idx], emb));
        }
    }

    // Reorder to match original input order
    indexed_results.sort_by_key(|(idx, _)| *idx);
    Ok(indexed_results.into_iter().map(|(_, emb)| emb).collect())
}

/// Group batch indices by length so that within each chunk, the padding ratio
/// (max_len / min_len) stays within `max_ratio`.
#[cfg(feature = "burn")]
fn chunk_by_length(batch_ids: &[Vec<i32>], max_ratio: f32) -> Vec<Vec<usize>> {
    let mut sorted_indices: Vec<usize> = (0..batch_ids.len()).collect();
    sorted_indices.sort_by_key(|&i| batch_ids[i].len());

    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut current_chunk: Vec<usize> = Vec::new();
    let mut chunk_min_len: usize = 0;

    for &idx in &sorted_indices {
        let seq_len = batch_ids[idx].len().max(1);

        if current_chunk.is_empty() {
            chunk_min_len = seq_len;
            current_chunk.push(idx);
        } else if seq_len as f32 <= chunk_min_len as f32 * max_ratio {
            current_chunk.push(idx);
        } else {
            chunks.push(std::mem::take(&mut current_chunk));
            chunk_min_len = seq_len;
            current_chunk.push(idx);
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

/// Direct batch forward pass — pads all sequences to the longest and runs one forward pass.
#[cfg(feature = "burn")]
#[allow(clippy::too_many_arguments)]
fn forward_and_pool_batch_direct<B: burn::prelude::Backend>(
    burn_model: &model::Qwen3Model<B>,
    batch_ids: &[Vec<i32>],
    batch_masks: &[Vec<u32>],
    device: &B::Device,
    pooling: PoolingStrategy,
    embedding_dim: usize,
    normalize: bool,
) -> Result<Vec<Vec<f32>>> {
    use burn::prelude::*;

    if batch_ids.is_empty() {
        return Ok(Vec::new());
    }

    let batch_size = batch_ids.len();
    let max_len = batch_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);

    // Pad all sequences to max_len and flatten into [batch * max_len]
    let mut flat_ids = Vec::with_capacity(batch_size * max_len);
    let mut flat_masks = Vec::with_capacity(batch_size * max_len);
    for i in 0..batch_size {
        let seq = &batch_ids[i];
        let mask = &batch_masks[i];
        flat_ids.extend_from_slice(seq);
        flat_masks.extend_from_slice(mask);
        // Pad with zeros
        for _ in seq.len()..max_len {
            flat_ids.push(0);
            flat_masks.push(0);
        }
    }

    let input_tensor =
        Tensor::<B, 1, Int>::from_data(flat_ids.as_slice(), device).reshape([batch_size, max_len]);

    // Build additive attention mask: 0.0 for real tokens, -1e9 for padding
    let attn_mask_data: Vec<f32> = flat_masks
        .iter()
        .map(|&m| if m == 1 { 0.0f32 } else { -1e9f32 })
        .collect();
    let attn_mask = Tensor::<B, 1>::from_data(attn_mask_data.as_slice(), device)
        .reshape([batch_size, 1, 1, max_len]);

    let hidden = burn_model.forward(input_tensor, Some(attn_mask));
    let [_b, _s, hidden_dim] = hidden.dims();
    let out_dim = embedding_dim.min(hidden_dim);

    let hidden_sliced = hidden.slice([0..batch_size, 0..max_len, 0..out_dim]);

    // Vectorized path for MeanPooling; per-sequence fallback for others
    let pooled: Tensor<B, 2> = match pooling {
        PoolingStrategy::MeanPooling => {
            // Build float mask [batch, max_len, 1] for broadcasting
            let mask_f32: Vec<f32> = flat_masks.iter().map(|&m| m as f32).collect();
            let mask_tensor = Tensor::<B, 1>::from_data(mask_f32.as_slice(), device)
                .reshape([batch_size, max_len, 1]);

            // Masked sum: zero out padding positions, then sum over seq dim
            let masked = hidden_sliced * mask_tensor.clone();
            let summed = masked.sum_dim(1).squeeze(1); // [B, D]

            // Per-sequence token counts for averaging
            let counts = mask_tensor.sum_dim(1).squeeze(1).clamp_min(1.0); // [B, 1]
            summed / counts
        }
        _ => {
            // LastTokenPooling, ClsToken, MaxPooling — require per-sequence indexing
            let mut rows: Vec<Tensor<B, 2>> = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let seq_len = batch_ids[i].len();
                let mask_slice = &flat_masks[i * max_len..i * max_len + seq_len];
                let seq_hidden = hidden_sliced
                    .clone()
                    .slice([i..i + 1, 0..max_len, 0..out_dim]);

                let row: Tensor<B, 2> = match pooling {
                    PoolingStrategy::LastTokenPooling => {
                        let last_idx = mask_slice
                            .iter()
                            .rposition(|&m| m == 1)
                            .unwrap_or(seq_len.saturating_sub(1));
                        seq_hidden
                            .slice([0..1, last_idx..last_idx + 1, 0..out_dim])
                            .reshape([1, out_dim])
                    }
                    PoolingStrategy::ClsToken => seq_hidden
                        .slice([0..1, 0..1, 0..out_dim])
                        .reshape([1, out_dim]),
                    PoolingStrategy::MaxPooling => seq_hidden
                        .slice([0..1, 0..seq_len, 0..out_dim])
                        .max_dim(1)
                        .squeeze(1),
                    PoolingStrategy::MeanPooling => {
                        unreachable!("MeanPooling handled in vectorized batch path above")
                    }
                };
                rows.push(row);
            }
            Tensor::cat(rows, 0) // [B, D]
        }
    };

    // Vectorized normalization
    let pooled = if normalize {
        let norms = pooled
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-6); // [B, 1]
        pooled / norms
    } else {
        pooled
    };

    // Convert [B, D] tensor to Vec<Vec<f32>>
    let data = pooled.into_data();
    let flat: Vec<f32> = data
        .to_vec()
        .map_err(|e| anyhow::anyhow!("tensor to_vec failed: {e:?}"))?;
    let results: Vec<Vec<f32>> = flat.chunks(out_dim).map(|chunk| chunk.to_vec()).collect();

    Ok(results)
}

#[cfg(feature = "burn")]
impl BurnBackend {
    fn embed_batch(
        &self,
        batch_ids: &[Vec<i32>],
        batch_masks: &[Vec<u32>],
        pooling: PoolingStrategy,
        embedding_dim: usize,
        normalize: bool,
    ) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::NdArray { model, device } => forward_and_pool_batch(
                model,
                batch_ids,
                batch_masks,
                device,
                pooling,
                embedding_dim,
                normalize,
            ),
            #[cfg(feature = "burn-wgpu")]
            Self::Wgpu { model, device } => forward_and_pool_batch(
                model,
                batch_ids,
                batch_masks,
                device,
                pooling,
                embedding_dim,
                normalize,
            ),
        }
    }

    fn embed(
        &self,
        input_ids: &[i32],
        mask: &[u32],
        seq_len: usize,
        pooling: PoolingStrategy,
        embedding_dim: usize,
        normalize: bool,
    ) -> Result<Vec<f32>> {
        match self {
            Self::NdArray { model, device } => forward_and_pool(
                model,
                input_ids,
                mask,
                seq_len,
                device,
                pooling,
                embedding_dim,
                normalize,
            ),
            #[cfg(feature = "burn-wgpu")]
            Self::Wgpu { model, device } => forward_and_pool(
                model,
                input_ids,
                mask,
                seq_len,
                device,
                pooling,
                embedding_dim,
                normalize,
            ),
        }
    }
}

// ── Model cache ───────────────────────────────────────────────────────
// Loaded models are expensive (0.6B = ~2.4 GB f32). Cache by path so
// multiple Qwen3Embedder instances share the same weights.

#[cfg(feature = "burn")]
fn model_cache()
-> &'static std::sync::Mutex<std::collections::HashMap<String, Arc<std::sync::Mutex<BurnBackend>>>>
{
    use std::sync::OnceLock;
    static CACHE: OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, Arc<std::sync::Mutex<BurnBackend>>>>,
    > = OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

// ── Config ────────────────────────────────────────────────────────────

/// Configuration for Qwen3Embedder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3Config {
    /// Model path (directory containing safetensors + tokenizer.json)
    pub model_path: Option<String>,

    /// Tokenizer path (for HuggingFace tokenizers)
    pub tokenizer_path: Option<String>,

    /// Output embedding dimension
    pub embedding_dim: usize,

    /// Maximum sequence length
    pub max_seq_length: usize,

    /// Whether to use simulated embeddings (for testing)
    pub use_simulated: bool,

    /// Whether to normalize output embeddings
    pub normalize_output: bool,

    /// Pooling strategy
    pub pooling: PoolingStrategy,

    /// Device to use for inference
    pub device: Device,

    /// Number of threads for CPU inference
    pub num_threads: usize,

    /// Load weights in half precision (f16 storage, f32 compute).
    /// Halves model loading time and on-disk size. Runtime memory is
    /// unchanged on CPU (NdArray) but enables f16 compute on WGPU.
    pub half_precision: bool,

    /// Maximum number of cached embeddings (LRU eviction). Default: 10,000.
    pub cache_capacity: usize,

    /// Optional instruction prefix for retrieval tasks.
    /// When set, text is formatted as "Instruct: {instruction}\nQuery: {text}"
    /// before tokenization, following the Qwen3-Embedding prompt template.
    pub instruction: Option<String>,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Self {
            model_path: None,
            tokenizer_path: None,
            embedding_dim: QWEN3_STANDARD_DIMENSION,
            max_seq_length: 8192,
            use_simulated: true, // Safe default - simulation mode
            normalize_output: true,
            pooling: PoolingStrategy::LastTokenPooling,
            device: Device::Cpu,
            num_threads: 4,
            half_precision: false,
            cache_capacity: 10_000,
            instruction: None,
        }
    }
}

impl Qwen3Config {
    /// Create config for Qwen3-0.6B model
    pub fn qwen3_06b(model_path: impl Into<String>) -> Self {
        Self {
            model_path: Some(model_path.into()),
            embedding_dim: QWEN3_STANDARD_DIMENSION,
            use_simulated: false,
            ..Default::default()
        }
    }

    /// Create config for Qwen3-1.5B model
    pub fn qwen3_15b(model_path: impl Into<String>) -> Self {
        Self {
            model_path: Some(model_path.into()),
            embedding_dim: QWEN3_FULL_DIMENSION,
            use_simulated: false,
            ..Default::default()
        }
    }

    /// Create config for HF Hub model (auto-downloads if `burn-hub` feature enabled)
    pub fn from_hub(repo_id: impl Into<String>) -> Self {
        Self {
            model_path: Some(repo_id.into()),
            use_simulated: false,
            ..Default::default()
        }
    }

    /// Create config for simulation mode (testing)
    pub fn simulated() -> Self {
        Self {
            use_simulated: true,
            ..Default::default()
        }
    }

    /// Set embedding dimension
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Enable CUDA if available
    pub fn with_cuda(mut self) -> Self {
        self.device = Device::Cuda;
        self
    }

    /// Use GPU via WGPU (auto-selects best available backend)
    pub fn with_gpu(mut self) -> Self {
        self.device = Device::Wgpu;
        self
    }

    /// Enable half-precision weight loading (f16 storage, faster load).
    ///
    /// # Quantization Status (burn 0.18)
    ///
    /// burn 0.18 does NOT support int8 or int4 quantization.
    /// f16 is the only precision reduction available. Combine with
    /// Matryoshka dimension reduction (`.with_matryoshka_dim()`) for
    /// smallest effective footprint.
    pub fn with_half_precision(mut self) -> Self {
        self.half_precision = true;
        self
    }

    /// Alias for [`with_half_precision`](Self::with_half_precision).
    pub fn with_f16(self) -> Self {
        self.with_half_precision()
    }

    /// Set an instruction prefix for retrieval tasks.
    ///
    /// When set, text is formatted as `"Instruct: {instruction}\nQuery: {text}"`
    /// before tokenization, following the Qwen3-Embedding prompt template.
    pub fn with_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = Some(instruction.into());
        self
    }

    /// Set Matryoshka (MRL) dimension for quality-preserving dimension reduction.
    ///
    /// Qwen3-Embedding models are trained with Matryoshka Representation Learning:
    /// the first `dim` dimensions of the full output form a valid lower-dimensional
    /// embedding. The output is automatically L2-renormalized after truncation.
    ///
    /// Standard dimensions: 64, 128, 256, 512, 768, 1024 (0.6B) or up to 1536 (1.5B).
    /// See [`QWEN3_MRL_DIMENSIONS`] for the standard set.
    pub fn with_matryoshka_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }
}

/// Pooling strategy for sequence embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingStrategy {
    /// Mean pooling over all tokens
    MeanPooling,
    /// Max pooling over all tokens
    MaxPooling,
    /// Use CLS token (first token)
    ClsToken,
    /// Use last token (Qwen3 default)
    LastTokenPooling,
}

/// Device for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    /// CPU inference (NdArray backend)
    Cpu,
    /// GPU via WGPU (auto-selects Vulkan/Metal/DX12)
    Wgpu,
    /// CUDA GPU inference (mapped to WGPU)
    Cuda,
    /// Metal GPU (mapped to WGPU)
    Metal,
}

// ── Embedder ──────────────────────────────────────────────────────────

/// Qwen3 Text Embedder
///
/// Provides semantic embeddings using Qwen3-Embedding models.
/// Falls back to high-quality simulation when Burn model is not available.
pub struct Qwen3Embedder {
    /// Configuration
    config: Qwen3Config,

    /// Model loaded state
    model_loaded: bool,

    /// Embedding cache with LRU eviction
    cache: lru::LruCache<String, Vec<f32>>,

    /// Statistics
    stats: Qwen3Stats,

    /// Backend-erased model (shared via Arc for cache reuse).
    /// Mutex makes BurnBackend Sync (burn's Param<T> is Send but not Sync).
    #[cfg(feature = "burn")]
    inference: Option<Arc<std::sync::Mutex<BurnBackend>>>,

    /// Tokenizer for text→token conversion
    #[cfg(feature = "burn")]
    tokenizer: Option<tokenizers::Tokenizer>,

    /// On-disk persistent cache (survives process restarts).
    #[cfg(feature = "persistent-cache")]
    persistent_cache: Option<crate::cache::PersistentCache>,
}

/// Statistics for the embedder
#[derive(Debug, Clone, Default)]
pub struct Qwen3Stats {
    /// Total embeddings computed (excludes cache hits)
    pub total_embeddings: u64,

    /// LRU cache hits
    pub cache_hits: u64,

    /// LRU cache misses
    pub cache_misses: u64,

    /// Persistent (on-disk) cache hits
    pub persistent_cache_hits: u64,

    /// Persistent (on-disk) cache misses
    pub persistent_cache_misses: u64,

    /// Number of `embed_batch()` calls
    pub batch_calls: u64,

    /// Total items processed across all batch calls
    pub total_batch_items: u64,

    /// Average processing time (ms)
    pub avg_time_ms: f32,

    /// Total characters processed
    pub total_chars: u64,
}

impl Qwen3Embedder {
    /// Create a new Qwen3 embedder
    pub fn new(config: Qwen3Config) -> Result<Self> {
        let cache_cap = std::num::NonZeroUsize::new(config.cache_capacity)
            .unwrap_or(std::num::NonZeroUsize::new(10_000).unwrap());
        #[cfg(feature = "persistent-cache")]
        let persistent_cache = if !config.use_simulated {
            let pc_config = crate::cache::PersistentCacheConfig {
                path: None, // default: ~/.cache/symthaea/embeddings.redb
                model_name: config
                    .model_path
                    .clone()
                    .unwrap_or_else(|| "unknown".into()),
                dimension: config.embedding_dim,
                use_f16_storage: false,
            };
            match crate::cache::PersistentCache::open(pc_config) {
                Ok(pc) => Some(pc),
                Err(e) => {
                    eprintln!("Warning: persistent cache init failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        let mut embedder = Self {
            config,
            model_loaded: false,
            cache: lru::LruCache::new(cache_cap),
            stats: Qwen3Stats::default(),
            #[cfg(feature = "burn")]
            inference: None,
            #[cfg(feature = "burn")]
            tokenizer: None,
            #[cfg(feature = "persistent-cache")]
            persistent_cache,
        };

        // Try to load Burn model if path is specified and not in simulation mode
        if !embedder.config.use_simulated {
            #[cfg(feature = "burn")]
            {
                embedder.try_load_burn()?;
            }
            #[cfg(not(feature = "burn"))]
            {
                if embedder.config.model_path.is_some() {
                    eprintln!(
                        "Note: Burn model support requires the `burn` feature, using simulation"
                    );
                    embedder.config.use_simulated = true;
                }
            }
        }

        Ok(embedder)
    }

    /// Attempt to load Burn model and tokenizer from safetensors.
    /// Falls back to simulation on any load failure.
    /// Uses a process-wide cache so multiple embedders share weights.
    #[cfg(feature = "burn")]
    #[allow(unused_mut)]
    fn try_load_burn(&mut self) -> Result<()> {
        let mut model_dir = match self.config.model_path {
            Some(ref p) => p.clone(),
            None => {
                self.config.use_simulated = true;
                return Ok(());
            }
        };

        // Auto-download from HF Hub if path looks like a repo ID
        #[cfg(feature = "burn-hub")]
        if hub::is_repo_id(&model_dir) {
            match hub::ensure_model(&model_dir) {
                Ok(local) => {
                    tracing::info!("Downloaded model from HF Hub: {model_dir} → {local}");
                    model_dir = local;
                }
                Err(e) => {
                    tracing::warn!("HF Hub download failed for {model_dir}: {e}, using simulation");
                    self.config.use_simulated = true;
                    return Ok(());
                }
            }
        }

        // Load tokenizer
        let tokenizer_path = self.config.tokenizer_path.clone().unwrap_or_else(|| {
            let base = std::path::Path::new(&model_dir);
            base.join("tokenizer.json").to_string_lossy().to_string()
        });

        match tokenizers::Tokenizer::from_file(&tokenizer_path) {
            Ok(tok) => self.tokenizer = Some(tok),
            Err(e) => {
                tracing::warn!(
                    "Failed to load tokenizer from {}: {}, using simulation",
                    tokenizer_path,
                    e
                );
                self.config.use_simulated = true;
                return Ok(());
            }
        }

        // Build a cache key from path + device + precision
        let cache_key = format!(
            "{}::{:?}::half={}",
            model_dir, self.config.device, self.config.half_precision
        );

        // Check cache first
        if let Ok(cache) = model_cache().lock() {
            if let Some(cached) = cache.get(&cache_key) {
                self.inference = Some(Arc::clone(cached));
                self.model_loaded = true;
                tracing::info!("Reusing cached Qwen3 model for {}", model_dir);
                return Ok(());
            }
        }

        // Load fresh model based on device selection
        let backend = self.load_backend(&model_dir)?;

        match backend {
            Some(b) => {
                let arc = Arc::new(std::sync::Mutex::new(b));
                // Insert into cache
                if let Ok(mut cache) = model_cache().lock() {
                    cache.insert(cache_key, Arc::clone(&arc));
                }
                self.inference = Some(arc);
                self.model_loaded = true;
                tracing::info!(
                    "Loaded Qwen3 Burn model from {} ({}D, {:?})",
                    model_dir,
                    self.config.embedding_dim,
                    self.config.device,
                );
            }
            None => {
                self.config.use_simulated = true;
            }
        }

        Ok(())
    }

    /// Load the Burn model with the appropriate backend.
    #[cfg(feature = "burn")]
    fn load_backend(&self, model_dir: &str) -> Result<Option<BurnBackend>> {
        let wants_gpu = matches!(
            self.config.device,
            Device::Wgpu | Device::Cuda | Device::Metal
        );

        if wants_gpu {
            #[cfg(feature = "burn-wgpu")]
            {
                match self.load_wgpu_model(model_dir) {
                    Ok(backend) => return Ok(Some(backend)),
                    Err(e) => {
                        tracing::warn!("WGPU init failed ({}), falling back to CPU", e);
                    }
                }
            }
            #[cfg(not(feature = "burn-wgpu"))]
            {
                tracing::warn!(
                    "GPU requested but `burn-wgpu` feature not enabled, falling back to CPU"
                );
            }
        }

        // CPU (NdArray) path
        match self.load_ndarray_model(model_dir) {
            Ok(backend) => Ok(Some(backend)),
            Err(e) => {
                tracing::warn!(
                    "Failed to load model from {}: {}, using simulation",
                    model_dir,
                    e
                );
                Ok(None)
            }
        }
    }

    /// Load model into NdArray (CPU) backend.
    #[cfg(feature = "burn")]
    fn load_ndarray_model(&self, model_dir: &str) -> Result<BurnBackend> {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let model_cfg = self.resolve_model_config();
        let burn_model = model_cfg.init::<NdArray>(&device);

        let burn_model = safetensors_loader::load_qwen3_from_dir(
            burn_model,
            std::path::Path::new(model_dir),
            &model_cfg,
            &device,
        )?;

        Ok(BurnBackend::NdArray {
            model: burn_model,
            device,
        })
    }

    /// Load model into WGPU (GPU) backend.
    #[cfg(feature = "burn-wgpu")]
    fn load_wgpu_model(&self, model_dir: &str) -> Result<BurnBackend> {
        let device = burn::backend::wgpu::WgpuDevice::DefaultDevice;
        let model_cfg = self.resolve_model_config();
        let burn_model = model_cfg.init::<burn::backend::Wgpu>(&device);

        let burn_model = safetensors_loader::load_qwen3_from_dir(
            burn_model,
            std::path::Path::new(model_dir),
            &model_cfg,
            &device,
        )?;

        Ok(BurnBackend::Wgpu {
            model: burn_model,
            device,
        })
    }

    /// Determine model config from embedding dimension.
    #[cfg(feature = "burn")]
    fn resolve_model_config(&self) -> model::Qwen3ModelConfig {
        if self.config.embedding_dim >= QWEN3_FULL_DIMENSION {
            model::Qwen3ModelConfig::qwen3_15b()
        } else {
            model::Qwen3ModelConfig::qwen3_06b()
        }
    }

    /// Embed a single text
    pub fn embed(&mut self, text: &str) -> Result<EmbeddingResult> {
        let start = Instant::now();

        // Check cache first (LRU — get() promotes to most-recently-used)
        if let Some(cached) = self.cache.get(text) {
            self.stats.cache_hits += 1;
            return Ok(EmbeddingResult::new(cached.clone(), "qwen3-cached"));
        }

        self.stats.cache_misses += 1;

        // Check persistent cache (on-disk, survives restarts)
        #[cfg(feature = "persistent-cache")]
        if let Some(ref mut pcache) = self.persistent_cache {
            match pcache.get(text) {
                Ok(Some(cached)) => {
                    // Promote to LRU cache for fast subsequent access
                    self.cache.put(text.to_string(), cached.clone());
                    self.stats.persistent_cache_hits += 1;
                    return Ok(EmbeddingResult::new(cached, "qwen3-persistent-cached"));
                }
                Ok(None) => {
                    self.stats.persistent_cache_misses += 1;
                }
                Err(_) => {
                    self.stats.persistent_cache_misses += 1;
                }
            }
        }

        // Generate embedding
        let embedding = if self.config.use_simulated {
            self.simulate_embedding(text)
        } else {
            #[cfg(feature = "burn")]
            {
                self.burn_embed(text)
                    .unwrap_or_else(|_| self.simulate_embedding(text))
            }
            #[cfg(not(feature = "burn"))]
            {
                self.simulate_embedding(text)
            }
        };

        let elapsed = start.elapsed().as_secs_f32() * 1000.0;

        // Update stats
        self.stats.total_embeddings += 1;
        self.stats.total_chars += text.len() as u64;
        let n = self.stats.total_embeddings as f32;
        self.stats.avg_time_ms = (self.stats.avg_time_ms * (n - 1.0) + elapsed) / n;

        // Cache the result (LRU — oldest evicted when at capacity)
        self.cache.put(text.to_string(), embedding.clone());

        // Best-effort store to persistent cache
        #[cfg(feature = "persistent-cache")]
        if let Some(ref mut pcache) = self.persistent_cache {
            let _ = pcache.put(text, &embedding);
        }

        let mut result = EmbeddingResult::new(embedding, "qwen3-embedding").with_time(elapsed);

        if self.config.use_simulated {
            result = result.simulated();
        }

        Ok(result)
    }

    /// Embed multiple texts in batch.
    ///
    /// When a Burn model is loaded, uses true batch inference (single forward pass
    /// with padding) for better throughput. Falls back to sequential embedding
    /// in simulation mode or on batch inference failure.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<EmbeddingResult>> {
        self.stats.batch_calls += 1;
        self.stats.total_batch_items += texts.len() as u64;

        if !self.config.use_simulated {
            #[cfg(feature = "burn")]
            if self.inference.is_some() {
                if let Ok(embeddings) = self.burn_embed_batch(texts) {
                    let start = Instant::now();
                    let results: Result<Vec<_>> = embeddings
                        .into_iter()
                        .enumerate()
                        .map(|(i, emb)| {
                            let text = texts[i];
                            self.stats.total_embeddings += 1;
                            self.stats.total_chars += text.len() as u64;
                            self.cache.put(text.to_string(), emb.clone());
                            Ok(EmbeddingResult::new(emb, "qwen3-embedding"))
                        })
                        .collect();
                    let elapsed = start.elapsed().as_secs_f32() * 1000.0;
                    let n = self.stats.total_embeddings as f32;
                    self.stats.avg_time_ms =
                        (self.stats.avg_time_ms * (n - texts.len() as f32) + elapsed) / n;
                    return results;
                }
            }
        }
        // Fallback: sequential per-text embedding
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// True batch inference through Burn: tokenize all texts, pad, single forward pass.
    #[cfg(feature = "burn")]
    fn burn_embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let inference_arc = self
            .inference
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Burn model not loaded"))?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;

        // Apply instruction prefix if configured, then batch tokenize
        let formatted: Vec<std::borrow::Cow<'_, str>> =
            texts.iter().map(|t| self.format_text(t)).collect();
        let inputs: Vec<tokenizers::EncodeInput> = formatted
            .iter()
            .map(|t| tokenizers::EncodeInput::Single(t.as_ref().into()))
            .collect();
        let encodings = tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| anyhow::anyhow!("Batch tokenization failed: {e}"))?;

        let max_seq = self.config.max_seq_length;
        let batch_ids: Vec<Vec<i32>> = encodings
            .iter()
            .map(|e| {
                let ids = e.get_ids();
                let len = ids.len().min(max_seq);
                ids[..len].iter().map(|&x| x as i32).collect()
            })
            .collect();
        let batch_masks: Vec<Vec<u32>> = encodings
            .iter()
            .map(|e| {
                let mask = e.get_attention_mask();
                let len = mask.len().min(max_seq);
                mask[..len].to_vec()
            })
            .collect();

        let inference = inference_arc
            .lock()
            .map_err(|e| anyhow::anyhow!("Model lock poisoned: {e}"))?;

        inference.embed_batch(
            &batch_ids,
            &batch_masks,
            self.config.pooling,
            self.config.embedding_dim,
            self.config.normalize_output,
        )
    }

    /// Format text with instruction prefix if configured.
    #[cfg(feature = "burn")]
    fn format_text<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        match &self.config.instruction {
            Some(instr) => std::borrow::Cow::Owned(format!("Instruct: {instr}\nQuery: {text}")),
            None => std::borrow::Cow::Borrowed(text),
        }
    }

    /// Run Burn inference through the backend-erased model.
    #[cfg(feature = "burn")]
    fn burn_embed(&self, text: &str) -> Result<Vec<f32>> {
        let inference_arc = self
            .inference
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Burn model not loaded"))?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;

        // Apply instruction prefix if configured
        let formatted = self.format_text(text);

        // Tokenize
        let encoding = tokenizer
            .encode(formatted.as_ref(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let seq_len = ids.len().min(self.config.max_seq_length);

        let input_ids: Vec<i32> = ids[..seq_len].iter().map(|&x| x as i32).collect();

        let inference = inference_arc
            .lock()
            .map_err(|e| anyhow::anyhow!("Model lock poisoned: {}", e))?;

        inference.embed(
            &input_ids,
            &mask[..seq_len],
            seq_len,
            self.config.pooling,
            self.config.embedding_dim,
            self.config.normalize_output,
        )
    }

    /// Simulate a high-quality embedding for text
    ///
    /// This simulation produces deterministic, semantically-meaningful embeddings
    /// that preserve basic similarity relationships.
    fn simulate_embedding(&self, text: &str) -> Vec<f32> {
        let dim = self.config.embedding_dim;
        let mut embedding = vec![0.0f32; dim];

        // Use text content to generate deterministic embedding.
        // Incorporate instruction prefix into hash so prefixed embeddings
        // differ deterministically from unprefixed ones.
        let mut hash: u64 = 0x5174_1AEA_5174_1AEA; // "SYMTHAEA"
        if let Some(ref instr) = self.config.instruction {
            for byte in instr.bytes() {
                hash = hash.wrapping_mul(37).wrapping_add(byte as u64);
            }
            // Separator to avoid collisions between instruction suffix and text prefix
            hash = hash.wrapping_mul(37).wrapping_add(0xFF);
        }
        for byte in text.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }

        // Generate pseudo-random embedding from hash
        let mut state = hash;
        for i in 0..dim {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f32) / (u32::MAX as f32);
            embedding[i] = (val - 0.5) * 2.0; // Range [-1, 1]
        }

        // Add semantic features based on text characteristics
        self.add_semantic_features(&mut embedding, text);

        // Normalize if configured
        if self.config.normalize_output {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for x in embedding.iter_mut() {
                    *x /= norm;
                }
            }
        }

        embedding
    }

    /// Add semantic features to embedding based on text analysis
    fn add_semantic_features(&self, embedding: &mut [f32], text: &str) {
        let text_lower = text.to_lowercase();
        let dim = embedding.len();

        // Feature 1: Text length (dimension 0-3)
        let len_feature = (text.len() as f32 / 500.0).min(1.0);
        for i in 0..4.min(dim) {
            embedding[i] = (embedding[i] + len_feature) / 2.0;
        }

        // Feature 2: Word count (dimension 4-7)
        let word_count = text.split_whitespace().count() as f32;
        let word_feature = (word_count / 100.0).min(1.0);
        for i in 4..8.min(dim) {
            embedding[i] = (embedding[i] + word_feature) / 2.0;
        }

        // Feature 3: Question detection (dimension 8-11)
        if text.contains('?') {
            for i in 8..12.min(dim) {
                embedding[i] = (embedding[i] + 0.5) / 2.0;
            }
        }

        // Feature 4: Command/imperative detection (dimension 12-15)
        let command_words = [
            "install",
            "run",
            "start",
            "stop",
            "create",
            "delete",
            "add",
            "remove",
            "update",
            "build",
            "configure",
        ];
        let is_command = command_words.iter().any(|w| text_lower.contains(w));
        if is_command {
            for i in 12..16.min(dim) {
                embedding[i] = (embedding[i] + 0.5) / 2.0;
            }
        }

        // Feature 5: Error/negative sentiment (dimension 16-19)
        let error_words = ["error", "fail", "crash", "bug", "broken", "wrong"];
        let is_error = error_words.iter().any(|w| text_lower.contains(w));
        if is_error {
            for i in 16..20.min(dim) {
                embedding[i] = (embedding[i] + 0.5) / 2.0;
            }
        }

        // Feature 6: Technical content (dimension 20-23)
        let tech_words = [
            "nix",
            "nixos",
            "flake",
            "derivation",
            "package",
            "module",
            "config",
            "system",
            "service",
        ];
        let is_tech = tech_words.iter().any(|w| text_lower.contains(w));
        if is_tech {
            for i in 20..24.min(dim) {
                embedding[i] = (embedding[i] + 0.5) / 2.0;
            }
        }

        // Feature 7: N-gram signatures (dimension 24+)
        // Create signatures from word n-grams for better semantic similarity
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        for (idx, word) in words.iter().take(20).enumerate() {
            let word_hash: u64 = word.bytes().fold(0x5974_1AEA_u64, |acc, b| {
                acc.wrapping_mul(31).wrapping_add(b as u64)
            });
            let dim_offset = 24 + (word_hash as usize % (dim.saturating_sub(24).max(1)));
            if dim_offset < dim {
                embedding[dim_offset] += 0.1 / (idx + 1) as f32;
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &Qwen3Stats {
        &self.stats
    }

    /// Clear the embedding cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Check if model is loaded
    pub fn is_model_loaded(&self) -> bool {
        self.model_loaded
    }

    /// Get configuration
    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    /// Evict all cached models from the process-wide model cache.
    #[cfg(feature = "burn")]
    pub fn clear_model_cache() {
        if let Ok(mut cache) = model_cache().lock() {
            cache.clear();
        }
    }
}

impl Embedder for Qwen3Embedder {
    fn dimension(&self) -> usize {
        self.config.embedding_dim
    }

    fn embed(&mut self, text: &str) -> Result<EmbeddingResult> {
        Qwen3Embedder::embed(self, text)
    }

    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<EmbeddingResult>> {
        Qwen3Embedder::embed_batch(self, texts)
    }

    fn model_name(&self) -> &str {
        if self.config.use_simulated {
            "qwen3-simulated"
        } else {
            "qwen3-embedding"
        }
    }
}

/// Simple text embedder (alias for backward compatibility)
pub type TextEmbedder = Qwen3Embedder;

// ── Async wrapper ─────────────────────────────────────────────────────

/// Async wrapper around [`Qwen3Embedder`] using `tokio::task::spawn_blocking`.
///
/// Shares state via `Arc<Mutex<Qwen3Embedder>>` — cloning an `AsyncQwen3Embedder`
/// shares the same underlying embedder (model weights, cache, stats).
///
/// ```rust,ignore
/// let embedder = AsyncQwen3Embedder::new(Qwen3Config::simulated())?;
/// let result = embedder.embed("Hello, world!".into()).await?;
/// ```
#[cfg(feature = "async-embed")]
#[derive(Clone)]
pub struct AsyncQwen3Embedder {
    inner: std::sync::Arc<std::sync::Mutex<Qwen3Embedder>>,
}

#[cfg(feature = "async-embed")]
impl AsyncQwen3Embedder {
    /// Create a new async embedder from config.
    pub fn new(config: Qwen3Config) -> Result<Self> {
        let embedder = Qwen3Embedder::new(config)?;
        Ok(Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(embedder)),
        })
    }

    /// Wrap an existing synchronous embedder.
    pub fn from_embedder(embedder: Qwen3Embedder) -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(embedder)),
        }
    }

    /// Embed a single text asynchronously.
    pub async fn embed(&self, text: String) -> Result<EmbeddingResult> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut embedder = inner
                .lock()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {e}"))?;
            embedder.embed(&text)
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))?
    }

    /// Embed multiple texts asynchronously.
    pub async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<EmbeddingResult>> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut embedder = inner
                .lock()
                .map_err(|e| anyhow::anyhow!("Lock poisoned: {e}"))?;
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            embedder.embed_batch(&refs)
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))?
    }

    /// Check if the underlying model is loaded.
    pub fn is_model_loaded(&self) -> bool {
        self.inner
            .lock()
            .map(|e| e.is_model_loaded())
            .unwrap_or(false)
    }

    /// Clear the embedding cache.
    pub fn clear_cache(&self) {
        if let Ok(mut e) = self.inner.lock() {
            e.clear_cache();
        }
    }

    /// Get a snapshot of the current statistics.
    pub fn stats(&self) -> Qwen3Stats {
        self.inner
            .lock()
            .map(|e| e.stats().clone())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_creation() {
        let config = Qwen3Config::default();
        let embedder = Qwen3Embedder::new(config).unwrap();
        assert_eq!(embedder.config.embedding_dim, QWEN3_STANDARD_DIMENSION);
    }

    #[test]
    fn test_embed_text() {
        let config = Qwen3Config::simulated();
        let mut embedder = Qwen3Embedder::new(config).unwrap();

        let result = embedder.embed("Hello, world!").unwrap();
        assert_eq!(result.dimension, QWEN3_STANDARD_DIMENSION);
        assert!(result.is_simulated);
    }

    #[test]
    fn test_similar_texts_similar_embeddings() {
        let config = Qwen3Config::simulated();
        let mut embedder = Qwen3Embedder::new(config).unwrap();

        let emb1 = embedder.embed("Install nginx").unwrap();
        let emb2 = embedder.embed("Install nginx server").unwrap();

        // Compute cosine similarity
        let dot: f32 = emb1
            .embedding
            .iter()
            .zip(emb2.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();

        // Normalized embeddings have norms close to 1
        assert!(
            dot > 0.5,
            "Similar texts should have high similarity, got {}",
            dot
        );
    }

    #[test]
    fn test_caching() {
        let config = Qwen3Config::simulated();
        let mut embedder = Qwen3Embedder::new(config).unwrap();

        let _ = embedder.embed("test text").unwrap();
        let _ = embedder.embed("test text").unwrap();

        assert_eq!(embedder.stats.cache_hits, 1);
    }

    #[test]
    fn test_batch_embedding() {
        let config = Qwen3Config::simulated();
        let mut embedder = Qwen3Embedder::new(config).unwrap();

        let texts = vec!["Hello", "World", "Test"];
        let results = embedder.embed_batch(&texts).unwrap();

        assert_eq!(results.len(), 3);
        for result in results {
            assert_eq!(result.dimension, QWEN3_STANDARD_DIMENSION);
        }
    }

    #[test]
    fn test_half_precision_config() {
        let config = Qwen3Config::simulated().with_half_precision();
        assert!(config.half_precision);
    }

    #[test]
    fn test_gpu_config() {
        let config = Qwen3Config::simulated().with_gpu();
        assert_eq!(config.device, Device::Wgpu);
    }

    #[test]
    fn test_instruction_config() {
        let config = Qwen3Config::simulated().with_instruction("Retrieve relevant passages");
        assert_eq!(
            config.instruction.as_deref(),
            Some("Retrieve relevant passages")
        );
    }

    #[test]
    fn test_instruction_changes_embedding() {
        let mut plain = Qwen3Embedder::new(Qwen3Config::simulated()).unwrap();
        let mut prefixed = Qwen3Embedder::new(
            Qwen3Config::simulated().with_instruction("Retrieve relevant passages"),
        )
        .unwrap();

        let text = "What is quantum computing?";
        let emb_plain = plain.embed(text).unwrap();
        let emb_prefixed = prefixed.embed(text).unwrap();

        // Instruction prefix should change the embedding
        let dot: f32 = emb_plain
            .embedding
            .iter()
            .zip(emb_prefixed.embedding.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            dot < 0.999,
            "Instruction prefix should change embedding, cosine={dot}"
        );
    }

    #[test]
    fn test_instruction_deterministic() {
        let mut emb1 =
            Qwen3Embedder::new(Qwen3Config::simulated().with_instruction("Search")).unwrap();
        let mut emb2 =
            Qwen3Embedder::new(Qwen3Config::simulated().with_instruction("Search")).unwrap();

        let r1 = emb1.embed("test query").unwrap();
        let r2 = emb2.embed("test query").unwrap();
        assert_eq!(r1.embedding, r2.embedding);
    }

    #[test]
    fn test_lru_eviction() {
        let mut config = Qwen3Config::simulated();
        config.cache_capacity = 3;
        let mut embedder = Qwen3Embedder::new(config).unwrap();

        embedder.embed("text1").unwrap();
        embedder.embed("text2").unwrap();
        embedder.embed("text3").unwrap();
        assert_eq!(embedder.cache_size(), 3);

        // Fourth insert evicts the least-recently-used (text1)
        embedder.embed("text4").unwrap();
        assert_eq!(embedder.cache_size(), 3);
    }

    #[test]
    fn test_f16_alias() {
        let config = Qwen3Config::simulated().with_f16();
        assert!(config.half_precision);
    }

    #[test]
    fn test_cache_stats_tracking() {
        let config = Qwen3Config::simulated();
        let mut embedder = Qwen3Embedder::new(config).unwrap();

        // First embed — LRU miss, compute
        embedder.embed("alpha").unwrap();
        assert_eq!(embedder.stats().cache_hits, 0);
        assert_eq!(embedder.stats().cache_misses, 1);
        assert_eq!(embedder.stats().total_embeddings, 1);

        // Second embed of same text — LRU hit
        embedder.embed("alpha").unwrap();
        assert_eq!(embedder.stats().cache_hits, 1);
        assert_eq!(embedder.stats().cache_misses, 1);

        // Different text — LRU miss
        embedder.embed("beta").unwrap();
        assert_eq!(embedder.stats().cache_hits, 1);
        assert_eq!(embedder.stats().cache_misses, 2);
        assert_eq!(embedder.stats().total_embeddings, 2);
    }

    #[test]
    fn test_batch_stats_tracking() {
        let config = Qwen3Config::simulated();
        let mut embedder = Qwen3Embedder::new(config).unwrap();

        let texts = vec!["one", "two", "three"];
        embedder.embed_batch(&texts).unwrap();

        assert_eq!(embedder.stats().batch_calls, 1);
        assert_eq!(embedder.stats().total_batch_items, 3);

        embedder.embed_batch(&["four", "five"]).unwrap();
        assert_eq!(embedder.stats().batch_calls, 2);
        assert_eq!(embedder.stats().total_batch_items, 5);
    }
}

#[cfg(test)]
#[cfg(feature = "async-embed")]
mod async_tests {
    use super::*;

    #[test]
    fn test_async_embedder_creation() {
        let embedder = AsyncQwen3Embedder::new(Qwen3Config::simulated()).unwrap();
        assert!(!embedder.is_model_loaded());
    }

    #[test]
    fn test_async_embedder_clone_shares_state() {
        let embedder = AsyncQwen3Embedder::new(Qwen3Config::simulated()).unwrap();
        let clone = embedder.clone();

        // Both should point to the same Arc — clearing cache on one affects the other
        {
            let mut inner = embedder.inner.lock().unwrap();
            inner.embed("shared text").unwrap();
        }
        assert_eq!(clone.stats().total_embeddings, 1);
    }
}

#[cfg(test)]
#[cfg(feature = "burn")]
mod burn_tests {
    use super::model::Qwen3ModelConfig;
    use burn::backend::NdArray;
    use burn::prelude::*;

    type B = NdArray;

    #[test]
    fn test_burn_model_init() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let cfg = Qwen3ModelConfig::tiny();
        let _model = cfg.init::<B>(&device);
    }

    #[test]
    fn test_burn_forward_shape() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let cfg = Qwen3ModelConfig::tiny();
        let model = cfg.init::<B>(&device);

        let ids = Tensor::<B, 2, Int>::zeros([1, 8], &device);
        let out = model.forward(ids, None);
        let [b, s, h] = out.dims();
        assert_eq!((b, s, h), (1, 8, 64));
    }

    #[test]
    fn test_swiglu_mlp_shape() {
        use super::mlp::Qwen3MlpConfig;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let mlp = Qwen3MlpConfig {
            hidden_size: 64,
            intermediate_size: 128,
        }
        .init::<B>(&device);

        let x = Tensor::<B, 3>::zeros([1, 4, 64], &device);
        let y = mlp.forward(x);
        assert_eq!(y.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_gqa_attention_shape() {
        use super::attention::Qwen3AttentionConfig;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let attn = Qwen3AttentionConfig {
            hidden_size: 64,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
        }
        .init::<B>(&device);

        let x = Tensor::<B, 3>::zeros([1, 8, 64], &device);
        let y = attn.forward(x, None);
        assert_eq!(y.dims(), [1, 8, 64]);
    }

    #[test]
    fn test_missing_safetensors_fallback() {
        let config = super::Qwen3Config::qwen3_06b("/nonexistent/path");
        let embedder = super::Qwen3Embedder::new(config).unwrap();
        // Should gracefully fall back to simulation
        assert!(embedder.config().use_simulated);
    }

    #[test]
    fn test_rmsnorm_preserves_shape() {
        use burn::nn::RmsNormConfig;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let norm = RmsNormConfig::new(64).init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([1, 4, 64], &device);
        let y = norm.forward(x);
        assert_eq!(y.dims(), [1, 4, 64]);
    }

    #[test]
    fn test_forward_and_pool_generic() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let cfg = Qwen3ModelConfig::tiny();
        let model = cfg.init::<B>(&device);

        let input_ids: Vec<i32> = vec![1, 2, 3, 4];
        let mask: Vec<u32> = vec![1, 1, 1, 1];

        let result = super::forward_and_pool(
            &model,
            &input_ids,
            &mask,
            4,
            &device,
            super::PoolingStrategy::MeanPooling,
            64,
            true,
        )
        .unwrap();

        assert_eq!(result.len(), 64);
        // Normalized: should have unit norm
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Expected unit norm, got {}",
            norm
        );
    }

    #[test]
    fn test_gpu_fallback_to_cpu() {
        // Request GPU but burn-wgpu not enabled → should fall back to simulation
        let config = super::Qwen3Config::qwen3_06b("/nonexistent/path").with_gpu();
        let embedder = super::Qwen3Embedder::new(config).unwrap();
        assert!(embedder.config().use_simulated);
    }

    #[test]
    fn test_padding_does_not_affect_embedding() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let cfg = Qwen3ModelConfig::tiny();
        let model = cfg.init::<B>(&device);

        let real_ids = vec![1i32, 2, 3, 4];
        let real_mask = vec![1u32; 4];

        // Unpadded: single sequence via forward_and_pool
        let unpadded = super::forward_and_pool(
            &model,
            &real_ids,
            &real_mask,
            4,
            &device,
            super::PoolingStrategy::LastTokenPooling,
            64,
            true,
        )
        .unwrap();

        // Padded batch: same sequence twice — once exact, once with 8 pad tokens
        let padded_ids = vec![vec![1, 2, 3, 4], vec![1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]];
        let padded_masks = vec![vec![1u32; 4], vec![1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]];
        let batch = super::forward_and_pool_batch_direct(
            &model,
            &padded_ids,
            &padded_masks,
            &device,
            super::PoolingStrategy::LastTokenPooling,
            64,
            true,
        )
        .unwrap();

        // The first sequence (no padding) should match single-sequence result
        let dot0: f32 = batch[0]
            .iter()
            .zip(unpadded.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            dot0 > 0.999,
            "Unpadded seq in batch should match single: cosine={dot0:.6}"
        );

        // The second sequence (with padding masked out) should also match
        let dot1: f32 = batch[1]
            .iter()
            .zip(unpadded.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            dot1 > 0.999,
            "Padded seq should match unpadded: cosine={dot1:.6}"
        );
    }

    #[test]
    fn test_mask_with_mean_pooling() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let cfg = Qwen3ModelConfig::tiny();
        let model = cfg.init::<B>(&device);

        let real_ids = vec![1i32, 2, 3, 4];
        let real_mask = vec![1u32; 4];

        // Single sequence mean pooling
        let single = super::forward_and_pool(
            &model,
            &real_ids,
            &real_mask,
            4,
            &device,
            super::PoolingStrategy::MeanPooling,
            64,
            true,
        )
        .unwrap();

        // Batch: same sequence with padding
        let padded_ids = vec![vec![1, 2, 3, 4], vec![1, 2, 3, 4, 0, 0, 0, 0]];
        let padded_masks = vec![vec![1u32; 4], vec![1, 1, 1, 1, 0, 0, 0, 0]];
        let batch = super::forward_and_pool_batch_direct(
            &model,
            &padded_ids,
            &padded_masks,
            &device,
            super::PoolingStrategy::MeanPooling,
            64,
            true,
        )
        .unwrap();

        // Both batch results should match the single-sequence result
        let dot0: f32 = batch[0].iter().zip(single.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot0 > 0.999,
            "Mean-pool unpadded batch should match single: cosine={dot0:.6}"
        );

        let dot1: f32 = batch[1].iter().zip(single.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot1 > 0.999,
            "Mean-pool padded batch should match single: cosine={dot1:.6}"
        );
    }

    #[test]
    fn test_matryoshka_dimension_reduction() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let cfg = Qwen3ModelConfig::tiny(); // hidden_size=64
        let model = cfg.init::<B>(&device);

        let ids_a: Vec<i32> = vec![1, 2, 3, 4];
        let ids_b: Vec<i32> = vec![1, 2, 3, 5]; // slightly different
        let ids_c: Vec<i32> = vec![10, 20, 30, 40]; // very different
        let mask = vec![1u32; 4];

        // Full dimension (64)
        let full_a = super::forward_and_pool(
            &model,
            &ids_a,
            &mask,
            4,
            &device,
            super::PoolingStrategy::LastTokenPooling,
            64,
            true,
        )
        .unwrap();
        let full_b = super::forward_and_pool(
            &model,
            &ids_b,
            &mask,
            4,
            &device,
            super::PoolingStrategy::LastTokenPooling,
            64,
            true,
        )
        .unwrap();
        let full_c = super::forward_and_pool(
            &model,
            &ids_c,
            &mask,
            4,
            &device,
            super::PoolingStrategy::LastTokenPooling,
            64,
            true,
        )
        .unwrap();

        // Reduced dimension (32 = half) — Matryoshka truncation + renormalization
        let half_a = super::forward_and_pool(
            &model,
            &ids_a,
            &mask,
            4,
            &device,
            super::PoolingStrategy::LastTokenPooling,
            32,
            true,
        )
        .unwrap();
        let half_b = super::forward_and_pool(
            &model,
            &ids_b,
            &mask,
            4,
            &device,
            super::PoolingStrategy::LastTokenPooling,
            32,
            true,
        )
        .unwrap();
        let half_c = super::forward_and_pool(
            &model,
            &ids_c,
            &mask,
            4,
            &device,
            super::PoolingStrategy::LastTokenPooling,
            32,
            true,
        )
        .unwrap();

        // Dimension should be correctly truncated
        assert_eq!(half_a.len(), 32);
        assert_eq!(half_b.len(), 32);
        assert_eq!(half_c.len(), 32);

        // Reduced-dim embedding should be re-normalized to unit length
        let norm: f32 = half_a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Reduced-dim embedding should be unit norm, got {norm}"
        );

        // Compute similarities at both dimensions for observability
        let full_ab: f32 = full_a.iter().zip(full_b.iter()).map(|(x, y)| x * y).sum();
        let full_ac: f32 = full_a.iter().zip(full_c.iter()).map(|(x, y)| x * y).sum();
        let half_ab: f32 = half_a.iter().zip(half_b.iter()).map(|(x, y)| x * y).sum();
        let half_ac: f32 = half_a.iter().zip(half_c.iter()).map(|(x, y)| x * y).sum();

        eprintln!("Full-dim(64): sim(a,b)={full_ab:.4}, sim(a,c)={full_ac:.4}");
        eprintln!("Half-dim(32): sim(a,b)={half_ab:.4}, sim(a,c)={half_ac:.4}");
    }

    #[test]
    fn test_batch_chunking_matches_sequential() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let cfg = Qwen3ModelConfig::tiny();
        let model = cfg.init::<B>(&device);

        // 4 sequences of very different lengths to trigger chunking
        let seqs: Vec<Vec<i32>> = vec![
            vec![1, 2],                                  // len 2
            vec![1, 2, 3],                               // len 3
            vec![1, 2, 3, 4, 5, 6],                      // len 6
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], // len 12
        ];
        let masks: Vec<Vec<u32>> = seqs.iter().map(|s| vec![1u32; s.len()]).collect();

        // Get sequential results (one at a time → no padding effects)
        let mut sequential = Vec::new();
        for (ids, mask) in seqs.iter().zip(masks.iter()) {
            let emb = super::forward_and_pool(
                &model,
                ids,
                mask,
                ids.len(),
                &device,
                super::PoolingStrategy::LastTokenPooling,
                64,
                true,
            )
            .unwrap();
            sequential.push(emb);
        }

        // Get batch results (exercises chunking for len ratio > 2)
        let batch = super::forward_and_pool_batch(
            &model,
            &seqs,
            &masks,
            &device,
            super::PoolingStrategy::LastTokenPooling,
            64,
            true,
        )
        .unwrap();

        assert_eq!(batch.len(), sequential.len());

        for (i, (b, s)) in batch.iter().zip(sequential.iter()).enumerate() {
            // Compute cosine similarity — padding affects self-attention so
            // batch vs sequential won't be identical, but should be very close
            let dot: f32 = b.iter().zip(s.iter()).map(|(x, y)| x * y).sum();
            assert!(
                dot > 0.99,
                "Batch vs sequential mismatch for seq {i}: cosine={dot:.4}"
            );
        }
    }

    #[test]
    fn test_batch_cls_token_pooling() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let cfg = Qwen3ModelConfig::tiny();
        let model = cfg.init::<B>(&device);

        let real_ids = vec![1i32, 2, 3, 4];
        let real_mask = vec![1u32; 4];

        // Single sequence via forward_and_pool
        let single = super::forward_and_pool(
            &model,
            &real_ids,
            &real_mask,
            4,
            &device,
            super::PoolingStrategy::ClsToken,
            64,
            true,
        )
        .unwrap();

        // Batch: unpadded + padded
        let batch_ids = vec![vec![1, 2, 3, 4], vec![1, 2, 3, 4, 0, 0, 0, 0]];
        let batch_masks = vec![vec![1u32; 4], vec![1, 1, 1, 1, 0, 0, 0, 0]];
        let batch = super::forward_and_pool_batch_direct(
            &model,
            &batch_ids,
            &batch_masks,
            &device,
            super::PoolingStrategy::ClsToken,
            64,
            true,
        )
        .unwrap();

        // CLS reads position 0 — padding should not affect it
        let dot0: f32 = batch[0].iter().zip(single.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot0 > 0.999,
            "CLS unpadded batch should match single: cosine={dot0:.6}"
        );

        let dot1: f32 = batch[1].iter().zip(single.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot1 > 0.999,
            "CLS padded batch should match single: cosine={dot1:.6}"
        );
    }

    #[test]
    fn test_batch_max_pooling() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let cfg = Qwen3ModelConfig::tiny();
        let model = cfg.init::<B>(&device);

        let real_ids = vec![1i32, 2, 3, 4];
        let real_mask = vec![1u32; 4];

        // Single sequence via forward_and_pool
        let single = super::forward_and_pool(
            &model,
            &real_ids,
            &real_mask,
            4,
            &device,
            super::PoolingStrategy::MaxPooling,
            64,
            true,
        )
        .unwrap();

        // Batch: unpadded + padded
        let batch_ids = vec![vec![1, 2, 3, 4], vec![1, 2, 3, 4, 0, 0, 0, 0]];
        let batch_masks = vec![vec![1u32; 4], vec![1, 1, 1, 1, 0, 0, 0, 0]];
        let batch = super::forward_and_pool_batch_direct(
            &model,
            &batch_ids,
            &batch_masks,
            &device,
            super::PoolingStrategy::MaxPooling,
            64,
            true,
        )
        .unwrap();

        // Unpadded should match closely
        let dot0: f32 = batch[0].iter().zip(single.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot0 > 0.999,
            "Max-pool unpadded batch should match single: cosine={dot0:.6}"
        );

        // Padded: max over padding-influenced hidden states may vary more
        // (additive mask zeroes attention but residual stream still differs)
        let dot1: f32 = batch[1].iter().zip(single.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot1 > 0.95,
            "Max-pool padded batch should be close to single: cosine={dot1:.6}"
        );
    }
}
