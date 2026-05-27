// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BGE-M3 Embedding Model via Candle
//!
//! Pure Rust implementation of BAAI/bge-m3 using HuggingFace Candle.
//! Produces 1024-dimensional dense embeddings from text input.
//!
//! ## Features
//!
//! - **Multilingual**: 100+ languages with consistent quality
//! - **Long Context**: Up to 8192 tokens (vs typical 512)
//! - **Multi-Functionality**: Dense, sparse, and ColBERT embeddings
//! - **CUDA Support**: Optional GPU acceleration via `neural-bridge-cuda` feature
//!
//! ## Architecture
//!
//! BGE-M3 uses XLM-RoBERTa-large as its backbone:
//! - 24 transformer layers
//! - 1024 hidden dimensions
//! - 16 attention heads
//! - 560M parameters
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::perception::bge_m3::BgeM3;
//!
//! let model = BgeM3::load_from_hub("BAAI/bge-m3")?;
//! let embedding = model.encode("What is consciousness?")?;
//! // embedding: [1024] f32 vector
//! ```

#[cfg(feature = "neural-bridge")]
use anyhow::{Context, Result};

#[cfg(feature = "neural-bridge")]
use candle_core::{DType, Device, Tensor};

#[cfg(feature = "neural-bridge")]
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm, linear};

#[cfg(feature = "neural-bridge")]
use tokenizers::Tokenizer;

#[cfg(feature = "neural-bridge")]
use hf_hub::{Repo, RepoType, api::sync::Api};

#[cfg(feature = "neural-bridge")]
use std::path::PathBuf;

/// BGE-M3 hidden dimension (XLM-RoBERTa-large)
pub const BGE_M3_DIM: usize = 1024;

/// Maximum sequence length for BGE-M3
pub const BGE_M3_MAX_SEQ_LEN: usize = 8192;

/// Default model ID on HuggingFace Hub
pub const BGE_M3_MODEL_ID: &str = "BAAI/bge-m3";

/// XLM-RoBERTa configuration for BGE-M3
#[cfg(feature = "neural-bridge")]
#[derive(Debug, Clone)]
pub struct XlmRobertaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
}

#[cfg(feature = "neural-bridge")]
impl Default for XlmRobertaConfig {
    fn default() -> Self {
        // BGE-M3 / XLM-RoBERTa-large configuration
        Self {
            vocab_size: 250002,
            hidden_size: BGE_M3_DIM,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: BGE_M3_MAX_SEQ_LEN,
            type_vocab_size: 1,
            layer_norm_eps: 1e-5,
            pad_token_id: 1,
        }
    }
}

/// Self-attention layer for XLM-RoBERTa
#[cfg(feature = "neural-bridge")]
struct XlmRobertaSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
}

#[cfg(feature = "neural-bridge")]
impl XlmRobertaSelfAttention {
    fn new(config: &XlmRobertaConfig, vb: VarBuilder) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        Ok(Self {
            query: linear(config.hidden_size, config.hidden_size, vb.pp("query"))?,
            key: linear(config.hidden_size, config.hidden_size, vb.pp("key"))?,
            value: linear(config.hidden_size, config.hidden_size, vb.pp("value"))?,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Project Q, K, V
        let query = self.query.forward(hidden_states)?;
        let key = self.key.forward(hidden_states)?;
        let value = self.value.forward(hidden_states)?;

        // Reshape for multi-head attention: [batch, seq, heads, head_dim]
        // Note: .contiguous() is needed for batched matmul after transpose
        let query = query
            .reshape((
                batch_size,
                seq_len,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .transpose(1, 2)? // [batch, heads, seq, head_dim]
            .contiguous()?;
        let key = key
            .reshape((
                batch_size,
                seq_len,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let value = value
            .reshape((
                batch_size,
                seq_len,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention
        let scale = (self.attention_head_size as f64).sqrt();
        let key_t = key.transpose(2, 3)?.contiguous()?;
        let attention_scores = (query.matmul(&key_t)? / scale)?;

        // Apply attention mask if provided
        let attention_scores = match attention_mask {
            Some(mask) => {
                // Explicit broadcast: [batch, 1, 1, seq] -> [batch, heads, seq, seq]
                let mask = mask.broadcast_as(attention_scores.shape())?;
                (attention_scores + mask)?
            }
            None => attention_scores,
        };

        // Softmax and apply to values
        let attention_probs = candle_nn::ops::softmax(&attention_scores, candle_core::D::Minus1)?;
        let context = attention_probs.matmul(&value)?;

        // Reshape back: [batch, seq, hidden_size]
        let context = context.transpose(1, 2)?.contiguous()?.reshape((
            batch_size,
            seq_len,
            self.num_attention_heads * self.attention_head_size,
        ))?;

        Ok(context)
    }
}

/// Self-attention output layer (dense + LayerNorm + residual)
#[cfg(feature = "neural-bridge")]
struct XlmRobertaSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

#[cfg(feature = "neural-bridge")]
impl XlmRobertaSelfOutput {
    fn new(config: &XlmRobertaConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            dense: linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?,
            layer_norm: layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("LayerNorm"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = (hidden_states + input_tensor)?;
        Ok(self.layer_norm.forward(&hidden_states)?)
    }
}

/// Complete attention block
#[cfg(feature = "neural-bridge")]
struct XlmRobertaAttention {
    self_attention: XlmRobertaSelfAttention,
    output: XlmRobertaSelfOutput,
}

#[cfg(feature = "neural-bridge")]
impl XlmRobertaAttention {
    fn new(config: &XlmRobertaConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attention: XlmRobertaSelfAttention::new(config, vb.pp("self"))?,
            output: XlmRobertaSelfOutput::new(config, vb.pp("output"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let self_output = self.self_attention.forward(hidden_states, attention_mask)?;
        self.output.forward(&self_output, hidden_states)
    }
}

/// Feed-forward intermediate layer
#[cfg(feature = "neural-bridge")]
struct XlmRobertaIntermediate {
    dense: Linear,
}

#[cfg(feature = "neural-bridge")]
impl XlmRobertaIntermediate {
    fn new(config: &XlmRobertaConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            dense: linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        // GELU activation
        Ok(hidden_states.gelu()?)
    }
}

/// Feed-forward output layer
#[cfg(feature = "neural-bridge")]
struct XlmRobertaOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

#[cfg(feature = "neural-bridge")]
impl XlmRobertaOutput {
    fn new(config: &XlmRobertaConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            dense: linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?,
            layer_norm: layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("LayerNorm"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = (hidden_states + input_tensor)?;
        Ok(self.layer_norm.forward(&hidden_states)?)
    }
}

/// Complete transformer layer
#[cfg(feature = "neural-bridge")]
pub(crate) struct XlmRobertaLayer {
    attention: XlmRobertaAttention,
    intermediate: XlmRobertaIntermediate,
    output: XlmRobertaOutput,
}

#[cfg(feature = "neural-bridge")]
impl XlmRobertaLayer {
    pub(crate) fn new(config: &XlmRobertaConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attention: XlmRobertaAttention::new(config, vb.pp("attention"))?,
            intermediate: XlmRobertaIntermediate::new(config, vb.pp("intermediate"))?,
            output: XlmRobertaOutput::new(config, vb.pp("output"))?,
        })
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        self.output.forward(&intermediate_output, &attention_output)
    }
}

/// XLM-RoBERTa encoder (stack of transformer layers)
#[cfg(feature = "neural-bridge")]
struct XlmRobertaEncoder {
    layers: Vec<XlmRobertaLayer>,
}

#[cfg(feature = "neural-bridge")]
impl XlmRobertaEncoder {
    fn new(config: &XlmRobertaConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(XlmRobertaLayer::new(config, vb.pp(format!("layer.{}", i)))?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }
}

/// XLM-RoBERTa embeddings (token + position)
#[cfg(feature = "neural-bridge")]
pub(crate) struct XlmRobertaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
    padding_idx: usize,
}

#[cfg(feature = "neural-bridge")]
impl XlmRobertaEmbeddings {
    pub(crate) fn new(config: &XlmRobertaConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            word_embeddings: embedding(
                config.vocab_size,
                config.hidden_size,
                vb.pp("word_embeddings"),
            )?,
            position_embeddings: embedding(
                config.max_position_embeddings,
                config.hidden_size,
                vb.pp("position_embeddings"),
            )?,
            layer_norm: layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("LayerNorm"),
            )?,
            padding_idx: config.pad_token_id,
        })
    }

    pub(crate) fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_batch_size, seq_len) = input_ids.dims2()?;

        // Get word embeddings
        let word_embeds = self.word_embeddings.forward(input_ids)?;

        // Create position IDs: [0, 1, 2, ..., seq_len-1] + padding_idx + 1
        // XLM-RoBERTa uses offset positions
        let position_ids: Vec<u32> = (0..seq_len)
            .map(|i| (i + self.padding_idx + 1) as u32)
            .collect();
        let position_ids = Tensor::new(&position_ids[..], input_ids.device())?;
        let position_ids = position_ids.unsqueeze(0)?; // [1, seq_len]

        let position_embeds = self.position_embeddings.forward(&position_ids)?;

        // Broadcast position embeddings to match batch size
        let position_embeds = position_embeds.broadcast_as(word_embeds.shape())?;

        // Sum embeddings
        let embeddings = (word_embeds + position_embeds)?;

        // Layer norm
        Ok(self.layer_norm.forward(&embeddings)?)
    }
}

/// Complete XLM-RoBERTa model
#[cfg(feature = "neural-bridge")]
struct XlmRobertaModel {
    embeddings: XlmRobertaEmbeddings,
    encoder: XlmRobertaEncoder,
}

#[cfg(feature = "neural-bridge")]
impl XlmRobertaModel {
    fn new(config: &XlmRobertaConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embeddings: XlmRobertaEmbeddings::new(config, vb.pp("embeddings"))?,
            encoder: XlmRobertaEncoder::new(config, vb.pp("encoder"))?,
        })
    }

    fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward(input_ids)?;

        // Convert attention mask to additive mask for attention
        let extended_attention_mask = match attention_mask {
            Some(mask) => {
                // [batch, seq] -> [batch, 1, 1, seq]
                let mask = mask.unsqueeze(1)?.unsqueeze(1)?;
                // Convert 0/1 mask to -inf/0 additive mask
                let mask = mask.to_dtype(DType::F32)?;
                let mask = ((mask - 1.0)? * 1e9)?; // 0 -> -1e9, 1 -> 0
                Some(mask)
            }
            None => None,
        };

        self.encoder
            .forward(&hidden_states, extended_attention_mask.as_ref())
    }
}

/// BGE-M3 Embedding Model
///
/// Wraps XLM-RoBERTa with mean pooling to produce 1024-dim dense embeddings.
#[cfg(feature = "neural-bridge")]
pub struct BgeM3 {
    model: XlmRobertaModel,
    tokenizer: Tokenizer,
    device: Device,
    config: XlmRobertaConfig,
}

#[cfg(feature = "neural-bridge")]
impl BgeM3 {
    /// Load BGE-M3 from HuggingFace Hub.
    ///
    /// Downloads and caches model weights and tokenizer automatically.
    ///
    /// # Arguments
    ///
    /// * `model_id` - HuggingFace model ID (default: "BAAI/bge-m3")
    ///
    /// # Returns
    ///
    /// * `Result<Self>` - Loaded model or error
    pub fn load_from_hub(model_id: &str) -> Result<Self> {
        Self::load_from_hub_with_device(model_id, Self::default_device()?)
    }

    /// Load with explicit device selection.
    pub fn load_from_hub_with_device(model_id: &str, device: Device) -> Result<Self> {
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        // Download tokenizer and config first (these are always available)
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer")?;

        let config_path = repo
            .get("config.json")
            .context("Failed to download config")?;

        // Try to get safetensors first, handling the case where the hub
        // might return a different format in the default revision
        let weights_path = match repo.get("model.safetensors") {
            Ok(path)
                if path
                    .extension()
                    .map(|e| e == "safetensors")
                    .unwrap_or(false) =>
            {
                // Got actual safetensors file
                path
            }
            _ => {
                // Safetensors not available in default revision
                // Check if there's a safetensors blob in the local cache
                if let Some(cached) = Self::find_cached_safetensors(model_id)? {
                    cached
                } else {
                    // Fall back to pytorch_model.bin
                    repo.get("pytorch_model.bin")
                        .context("Failed to download model weights (tried model.safetensors and pytorch_model.bin)")?
                }
            }
        };

        Self::load_from_files(&weights_path, &tokenizer_path, &config_path, device)
    }

    /// Search for a cached safetensors file in the HuggingFace cache.
    fn find_cached_safetensors(model_id: &str) -> Result<Option<PathBuf>> {
        let cache_dir = dirs::home_dir()
            .map(|h| h.join(".cache/huggingface/hub"))
            .context("Could not find home directory")?;

        // Convert model_id to cache directory name (e.g., "BAAI/bge-m3" -> "models--BAAI--bge-m3")
        let model_cache_name = format!("models--{}", model_id.replace('/', "--"));
        let model_cache_dir = cache_dir.join(&model_cache_name);

        if !model_cache_dir.exists() {
            return Ok(None);
        }

        // Look for safetensors blob by checking all blobs
        let blobs_dir = model_cache_dir.join("blobs");
        if !blobs_dir.exists() {
            return Ok(None);
        }

        // Find the safetensors file by scanning snapshots for symlinks
        let snapshots_dir = model_cache_dir.join("snapshots");
        if snapshots_dir.exists() {
            for entry in std::fs::read_dir(&snapshots_dir)? {
                let entry = entry?;
                let snapshot_dir = entry.path();
                let safetensors_link = snapshot_dir.join("model.safetensors");

                if safetensors_link.exists() {
                    // This snapshot has safetensors - resolve the symlink
                    let resolved = std::fs::canonicalize(&safetensors_link)?;
                    if resolved.exists() {
                        return Ok(Some(resolved));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Detect if a file is in safetensors format by checking magic bytes.
    ///
    /// Safetensors files start with a little-endian u64 header size, followed by JSON.
    /// The header size is typically small (< 1MB), so we check for reasonable values.
    fn is_safetensors_file(path: &PathBuf) -> Result<bool> {
        use std::io::Read;

        let mut file = std::fs::File::open(path)?;
        let mut header = [0u8; 8];

        if file.read_exact(&mut header).is_err() {
            return Ok(false);
        }

        // Safetensors starts with a little-endian u64 header size
        let header_size = u64::from_le_bytes(header);

        // Valid safetensors have header size < 100MB and > 0
        // Most are a few KB to a few hundred KB
        if header_size == 0 || header_size > 100_000_000 {
            return Ok(false);
        }

        // Try to read the first few bytes of the header to verify it's JSON
        let mut json_start = [0u8; 2];
        if file.read_exact(&mut json_start).is_err() {
            return Ok(false);
        }

        // JSON header should start with '{'
        Ok(json_start[0] == b'{')
    }

    /// Load PyTorch .bin weights using candle's pickle support.
    fn load_pytorch_weights(path: &PathBuf, device: &Device) -> Result<VarBuilder<'static>> {
        use std::collections::HashMap;

        // Read the pickle file - returns Vec<(String, Tensor)>
        let weights =
            candle_core::pickle::read_all(path).context("Failed to read PyTorch weights file")?;

        // Convert to HashMap and move tensors to target device
        let mut tensors_map: HashMap<String, Tensor> = HashMap::new();

        for (name, tensor) in weights {
            // Move tensor to target device and convert to f32
            let tensor = tensor.to_device(device)?.to_dtype(DType::F32)?;
            tensors_map.insert(name, tensor);
        }

        // Create VarBuilder from the loaded tensors
        Ok(VarBuilder::from_tensors(tensors_map, DType::F32, device))
    }

    /// Load from local files.
    pub fn load_from_files(
        weights_path: &PathBuf,
        tokenizer_path: &PathBuf,
        config_path: &PathBuf,
        device: Device,
    ) -> Result<Self> {
        // Load config
        let config_str = std::fs::read_to_string(config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_str)?;

        let xlm_config = XlmRobertaConfig {
            vocab_size: config["vocab_size"].as_u64().unwrap_or(250002) as usize,
            hidden_size: config["hidden_size"].as_u64().unwrap_or(1024) as usize,
            num_hidden_layers: config["num_hidden_layers"].as_u64().unwrap_or(24) as usize,
            num_attention_heads: config["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            intermediate_size: config["intermediate_size"].as_u64().unwrap_or(4096) as usize,
            hidden_act: config["hidden_act"].as_str().unwrap_or("gelu").to_string(),
            hidden_dropout_prob: config["hidden_dropout_prob"].as_f64().unwrap_or(0.1),
            attention_probs_dropout_prob: config["attention_probs_dropout_prob"]
                .as_f64()
                .unwrap_or(0.1),
            max_position_embeddings: config["max_position_embeddings"].as_u64().unwrap_or(8194)
                as usize,
            type_vocab_size: config["type_vocab_size"].as_u64().unwrap_or(1) as usize,
            layer_norm_eps: config["layer_norm_eps"].as_f64().unwrap_or(1e-5),
            pad_token_id: config["pad_token_id"].as_u64().unwrap_or(1) as usize,
        };

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Detect file format by magic bytes (HF cache uses hash names without extensions)
        let is_safetensors = Self::is_safetensors_file(weights_path)?;

        // Load weights
        let vb = if is_safetensors {
            super::model_integrity::verified_mmap_safetensors(
                &[weights_path],
                DType::F32,
                &device,
                None,
            )?
        } else {
            // PyTorch pickle format - try to load using candle's pickle support
            Self::load_pytorch_weights(weights_path, &device)?
        };

        // BGE-M3 safetensors uses flat names (no "roberta" prefix)
        // e.g., "embeddings.word_embeddings.weight" not "roberta.embeddings.word_embeddings.weight"
        let model = XlmRobertaModel::new(&xlm_config, vb)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            config: xlm_config,
        })
    }

    /// Get the default device (CUDA if available and working, else CPU).
    fn default_device() -> Result<Device> {
        #[cfg(feature = "neural-bridge-cuda")]
        {
            if candle_core::utils::cuda_is_available() {
                // Try to actually create a CUDA device - this validates driver compatibility
                match Device::new_cuda(0) {
                    Ok(device) => {
                        // Validate by creating a small tensor - catches driver mismatch errors
                        match candle_core::Tensor::zeros((1,), candle_core::DType::F32, &device) {
                            Ok(_) => {
                                tracing::info!("CUDA device initialized successfully");
                                return Ok(device);
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "CUDA tensor creation failed, falling back to CPU: {}",
                                    e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("CUDA device creation failed, falling back to CPU: {}", e);
                    }
                }
            }
        }
        tracing::info!("Using CPU device for BGE-M3 inference");
        Ok(Device::Cpu)
    }

    /// Encode text to a 1024-dimensional embedding.
    ///
    /// Uses mean pooling over non-padding tokens.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to encode
    ///
    /// # Returns
    ///
    /// * `Result<Vec<f32>>` - 1024-dimensional embedding
    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.encode_batch(&[text])?;
        Ok(embeddings
            .into_iter()
            .next()
            .expect("embeddings must not be empty"))
    }

    /// Encode multiple texts in a batch.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of texts to encode
    ///
    /// # Returns
    ///
    /// * `Result<Vec<Vec<f32>>>` - Batch of 1024-dim embeddings
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        // Find max length for padding
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Prepare input tensors
        let mut input_ids_vec = Vec::new();
        let mut attention_mask_vec = Vec::new();

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            // Pad to max_len
            let mut padded_ids = ids.to_vec();
            let mut padded_mask = mask.to_vec();

            padded_ids.resize(max_len, self.config.pad_token_id as u32);
            padded_mask.resize(max_len, 0);

            input_ids_vec.extend(padded_ids);
            attention_mask_vec.extend(padded_mask);
        }

        let batch_size = texts.len();
        let input_ids =
            Tensor::new(&input_ids_vec[..], &self.device)?.reshape((batch_size, max_len))?;
        let attention_mask = Tensor::new(&attention_mask_vec[..], &self.device)?
            .reshape((batch_size, max_len))?
            .to_dtype(DType::F32)?;

        // Forward pass
        let hidden_states = self.model.forward(&input_ids, Some(&attention_mask))?;

        // Mean pooling: sum(hidden * mask) / sum(mask)
        let mask_expanded = attention_mask.unsqueeze(2)?; // [batch, seq, 1]
        let mask_expanded = mask_expanded.broadcast_as(hidden_states.shape())?;

        let sum_hidden = (hidden_states * &mask_expanded)?.sum(1)?; // [batch, hidden]
        let sum_mask = mask_expanded.sum(1)?; // [batch, hidden]

        let embeddings = (sum_hidden / (sum_mask + 1e-9)?)?;

        // L2 normalize
        let norms = embeddings.sqr()?.sum(1)?.sqrt()?.unsqueeze(1)?;
        let norms = (norms + 1e-9)?.broadcast_as(embeddings.shape())?;
        let embeddings = (embeddings / norms)?;

        // Extract to Vec<Vec<f32>>
        let embeddings = embeddings.to_vec2::<f32>()?;

        Ok(embeddings)
    }

    /// Get the embedding dimension (always 1024 for BGE-M3).
    pub fn dim(&self) -> usize {
        self.config.hidden_size
    }

    /// Get the device being used.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if using CUDA.
    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }
}

/// Lightweight BGE-M3 client for when you don't need the full model in memory.
///
/// Loads model on-demand and caches it.
#[cfg(feature = "neural-bridge")]
pub struct BgeM3Client {
    model: Option<BgeM3>,
    model_id: String,
}

#[cfg(feature = "neural-bridge")]
impl BgeM3Client {
    /// Create a new client with the default model.
    pub fn new() -> Self {
        Self {
            model: None,
            model_id: BGE_M3_MODEL_ID.to_string(),
        }
    }

    /// Create with a custom model ID.
    pub fn with_model(model_id: &str) -> Self {
        Self {
            model: None,
            model_id: model_id.to_string(),
        }
    }

    /// Encode text, loading the model if needed.
    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>> {
        if self.model.is_none() {
            self.model = Some(BgeM3::load_from_hub(&self.model_id)?);
        }
        self.model
            .as_ref()
            .expect("model must be loaded")
            .encode(text)
    }

    /// Encode batch, loading the model if needed.
    pub fn encode_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if self.model.is_none() {
            self.model = Some(BgeM3::load_from_hub(&self.model_id)?);
        }
        self.model
            .as_ref()
            .expect("model must be loaded")
            .encode_batch(texts)
    }

    /// Check if model is loaded.
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Unload the model to free memory.
    pub fn unload(&mut self) {
        self.model = None;
    }
}

#[cfg(feature = "neural-bridge")]
impl Default for BgeM3Client {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[cfg(feature = "neural-bridge")]
mod tests {
    use super::*;

    // =========================================================================
    // XlmRobertaConfig Tests
    // =========================================================================

    #[test]
    fn test_config_defaults() {
        let config = XlmRobertaConfig::default();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 16);
    }

    #[test]
    fn test_config_default_vocab() {
        let config = XlmRobertaConfig::default();
        assert_eq!(config.vocab_size, 250002);
    }

    #[test]
    fn test_config_default_intermediate() {
        let config = XlmRobertaConfig::default();
        assert_eq!(config.intermediate_size, 4096);
    }

    #[test]
    fn test_config_default_max_position() {
        let config = XlmRobertaConfig::default();
        assert_eq!(config.max_position_embeddings, BGE_M3_MAX_SEQ_LEN);
    }

    #[test]
    fn test_config_default_hidden_act() {
        let config = XlmRobertaConfig::default();
        assert_eq!(config.hidden_act, "gelu");
    }

    #[test]
    fn test_config_default_dropout() {
        let config = XlmRobertaConfig::default();
        assert!((config.hidden_dropout_prob - 0.1).abs() < 0.01);
        assert!((config.attention_probs_dropout_prob - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_config_default_layer_norm_eps() {
        let config = XlmRobertaConfig::default();
        assert!((config.layer_norm_eps - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_config_default_type_vocab() {
        let config = XlmRobertaConfig::default();
        assert_eq!(config.type_vocab_size, 1);
    }

    #[test]
    fn test_config_default_pad_token() {
        let config = XlmRobertaConfig::default();
        assert_eq!(config.pad_token_id, 1);
    }

    #[test]
    fn test_config_clone() {
        let config = XlmRobertaConfig::default();
        let cloned = config.clone();
        assert_eq!(config.hidden_size, cloned.hidden_size);
        assert_eq!(config.num_hidden_layers, cloned.num_hidden_layers);
    }

    // =========================================================================
    // BgeM3Client Tests
    // =========================================================================

    #[test]
    fn test_client_creation() {
        let client = BgeM3Client::new();
        assert!(!client.is_loaded());
    }

    #[test]
    fn test_client_default() {
        let client = BgeM3Client::default();
        assert!(!client.is_loaded());
    }

    #[test]
    fn test_client_with_model() {
        let client = BgeM3Client::with_model("custom/model");
        assert!(!client.is_loaded());
        assert_eq!(client.model_id, "custom/model");
    }

    #[test]
    fn test_client_unload() {
        let mut client = BgeM3Client::new();
        client.unload();
        assert!(!client.is_loaded());
    }

    // =========================================================================
    // Constants Tests
    // =========================================================================

    #[test]
    fn test_bge_m3_dim_constant() {
        assert_eq!(BGE_M3_DIM, 1024);
    }

    #[test]
    fn test_bge_m3_max_seq_len_constant() {
        assert_eq!(BGE_M3_MAX_SEQ_LEN, 8192);
    }

    #[test]
    fn test_bge_m3_model_id_constant() {
        assert_eq!(BGE_M3_MODEL_ID, "BAAI/bge-m3");
    }

    // =========================================================================
    // Config Consistency Tests
    // =========================================================================

    #[test]
    fn test_attention_head_size_calculation() {
        let config = XlmRobertaConfig::default();
        // Head size should be hidden_size / num_attention_heads
        let expected_head_size = config.hidden_size / config.num_attention_heads;
        assert_eq!(expected_head_size, 64); // 1024 / 16 = 64
    }

    #[test]
    fn test_config_is_debug() {
        let config = XlmRobertaConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("XlmRobertaConfig"));
        assert!(debug_str.contains("1024"));
    }
}

// Tests that don't require neural-bridge feature
#[cfg(test)]
mod feature_independent_tests {
    use super::*;

    #[test]
    fn test_constants_are_accessible() {
        assert_eq!(BGE_M3_DIM, 1024);
        assert_eq!(BGE_M3_MAX_SEQ_LEN, 8192);
        assert_eq!(BGE_M3_MODEL_ID, "BAAI/bge-m3");
    }
}
