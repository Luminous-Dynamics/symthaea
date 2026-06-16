// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Layer-wise Activation Extraction for Transformer Models
//!
//! This module provides infrastructure for extracting activations from
//! specific transformer layers, enabling research into which layers
//! encode phenomenal vs functional structure.
//!
//! ## Research Question
//!
//! At which layer does the phenomenal/functional distinction emerge?
//! - Early layers encode syntax and surface features
//! - Middle layers encode semantics
//! - Late layers encode task-specific representations
//!
//! Where does "qualia-like" structure appear?
//!
//! ## Architecture
//!
//! ```text
//! Text → Tokenizer → Embeddings → Layer 0 → Layer 1 → ... → Layer 23 → Pooling
//!                                    ↓         ↓              ↓
//!                              Extract    Extract        Extract
//!                                    ↓         ↓              ↓
//!                              Probe W₀   Probe W₁      Probe W₂₃
//!                                    ↓         ↓              ↓
//!                               HDC₀       HDC₁          HDC₂₃
//!                                    ↓         ↓              ↓
//!                             Topology₀  Topology₁     Topology₂₃
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::perception::LayerExtractor;
//!
//! let mut extractor = LayerExtractor::load_default()?;
//!
//! // Extract from all layers
//! let all_layers = extractor.extract_all_layers("The experience of seeing red")?;
//! println!("Extracted {} layers", all_layers.len());
//!
//! // Extract from specific layer
//! let layer_12 = extractor.extract_layer("consciousness", 12)?;
//! ```

#[cfg(feature = "neural-bridge")]
use anyhow::{Context, Result};

#[cfg(feature = "neural-bridge")]
use candle_core::{DType, Device, IndexOp, Tensor};

#[cfg(feature = "neural-bridge")]
use tokenizers::Tokenizer;

#[cfg(feature = "neural-bridge")]
use hf_hub::{Repo, RepoType, api::sync::Api};

#[cfg(feature = "neural-bridge")]
use std::path::PathBuf;

#[cfg(feature = "neural-bridge")]
use serde_json;

/// Number of layers in BGE-M3 (XLM-RoBERTa-large)
pub const BGE_M3_NUM_LAYERS: usize = 24;

/// Hidden dimension of BGE-M3
pub const BGE_M3_HIDDEN_DIM: usize = 1024;

/// Result of extracting from a single layer
#[cfg(feature = "neural-bridge")]
#[derive(Debug, Clone)]
pub struct LayerActivation {
    /// Layer index (0-23 for BGE-M3)
    pub layer_idx: usize,
    /// Mean-pooled activation vector
    pub activation: Vec<f32>,
    /// Raw sequence activations (optional, for advanced analysis)
    pub sequence_activations: Option<Vec<Vec<f32>>>,
}

/// Result of extracting from all layers
#[cfg(feature = "neural-bridge")]
#[derive(Debug, Clone)]
pub struct AllLayerActivations {
    /// Text that was encoded
    pub text: String,
    /// Activations from each layer (index = layer number)
    pub layers: Vec<LayerActivation>,
    /// Embedding layer activation (before any transformer layers)
    pub embedding_activation: Vec<f32>,
}

/// Configuration for layer extraction
#[cfg(feature = "neural-bridge")]
#[derive(Debug, Clone)]
pub struct LayerExtractorConfig {
    /// Model ID on HuggingFace Hub
    pub model_id: String,
    /// Whether to keep sequence-level activations (memory intensive)
    pub keep_sequence_activations: bool,
    /// Pooling method for converting sequence to single vector
    pub pooling: PoolingMethod,
}

/// Pooling method for sequence → vector
#[cfg(feature = "neural-bridge")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingMethod {
    /// Average all token activations
    Mean,
    /// Use only the `[CLS]` token (first token)
    Cls,
    /// Use only the last token
    Last,
    /// Max pooling across tokens
    Max,
}

#[cfg(feature = "neural-bridge")]
impl Default for LayerExtractorConfig {
    fn default() -> Self {
        Self {
            model_id: super::bge_m3::BGE_M3_MODEL_ID.to_string(),
            keep_sequence_activations: false,
            pooling: PoolingMethod::Mean,
        }
    }
}

/// Layer-wise activation extractor for transformer models
///
/// Extracts intermediate activations from each transformer layer,
/// enabling analysis of how representations evolve through the network.
#[cfg(feature = "neural-bridge")]
pub struct LayerExtractor {
    /// Loaded model with layer access
    model: LayerAccessModel,
    /// Tokenizer
    tokenizer: Tokenizer,
    /// Device (CPU or CUDA)
    device: Device,
    /// Configuration
    config: LayerExtractorConfig,
}

/// Internal model struct with layer-by-layer access
#[cfg(feature = "neural-bridge")]
struct LayerAccessModel {
    embeddings: super::bge_m3::XlmRobertaEmbeddings,
    layers: Vec<super::bge_m3::XlmRobertaLayer>,
}

#[cfg(feature = "neural-bridge")]
impl LayerExtractor {
    /// Load with default BGE-M3 configuration
    pub fn load_default() -> Result<Self> {
        Self::load(LayerExtractorConfig::default())
    }

    /// Load with custom configuration
    pub fn load(config: LayerExtractorConfig) -> Result<Self> {
        let device = Self::default_device()?;
        Self::load_with_device(config, device)
    }

    /// Load with explicit device selection
    pub fn load_with_device(config: LayerExtractorConfig, device: Device) -> Result<Self> {
        use candle_nn::VarBuilder;

        let api = Api::new()?;
        let repo = api.repo(Repo::new(config.model_id.clone(), RepoType::Model));

        // Download tokenizer and config
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer")?;
        let config_path = repo
            .get("config.json")
            .context("Failed to download config")?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Download weights
        let weights_path = Self::get_weights_path(&repo)?;

        // Load model weights
        // BGE-M3 safetensors uses flat names (no "roberta" prefix)
        // Detect file format by checking magic bytes (HF cache uses hash names without extensions)
        let is_safetensors = Self::is_safetensors_file(&weights_path)?;

        let vb = if is_safetensors {
            super::model_integrity::verified_mmap_safetensors(
                &[weights_path],
                DType::F32,
                &device,
                None,
            )?
        } else {
            VarBuilder::from_pth(&weights_path, DType::F32, &device)?
        };

        // Load config from model's config.json to get correct dimensions
        let config_str = std::fs::read_to_string(&config_path)?;
        let model_config: serde_json::Value = serde_json::from_str(&config_str)?;

        let xlm_config = super::bge_m3::XlmRobertaConfig {
            vocab_size: model_config["vocab_size"].as_u64().unwrap_or(250002) as usize,
            hidden_size: model_config["hidden_size"].as_u64().unwrap_or(1024) as usize,
            num_hidden_layers: model_config["num_hidden_layers"].as_u64().unwrap_or(24) as usize,
            num_attention_heads: model_config["num_attention_heads"].as_u64().unwrap_or(16)
                as usize,
            intermediate_size: model_config["intermediate_size"].as_u64().unwrap_or(4096) as usize,
            hidden_act: model_config["hidden_act"]
                .as_str()
                .unwrap_or("gelu")
                .to_string(),
            hidden_dropout_prob: model_config["hidden_dropout_prob"].as_f64().unwrap_or(0.1),
            attention_probs_dropout_prob: model_config["attention_probs_dropout_prob"]
                .as_f64()
                .unwrap_or(0.1),
            max_position_embeddings: model_config["max_position_embeddings"]
                .as_u64()
                .unwrap_or(8194) as usize,
            type_vocab_size: model_config["type_vocab_size"].as_u64().unwrap_or(1) as usize,
            layer_norm_eps: model_config["layer_norm_eps"].as_f64().unwrap_or(1e-5),
            pad_token_id: model_config["pad_token_id"].as_u64().unwrap_or(1) as usize,
        };

        // Build model components with layer access
        // Note: BGE-M3 uses flat tensor names without "roberta" prefix
        let embeddings =
            super::bge_m3::XlmRobertaEmbeddings::new(&xlm_config, vb.pp("embeddings"))?;

        let mut layers = Vec::with_capacity(xlm_config.num_hidden_layers);
        for i in 0..xlm_config.num_hidden_layers {
            let layer = super::bge_m3::XlmRobertaLayer::new(
                &xlm_config,
                vb.pp(format!("encoder.layer.{}", i)),
            )?;
            layers.push(layer);
        }

        let model = LayerAccessModel { embeddings, layers };

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }

    fn get_weights_path(repo: &hf_hub::api::sync::ApiRepo) -> Result<PathBuf> {
        // Try safetensors first
        if let Ok(path) = repo.get("model.safetensors") {
            if path
                .extension()
                .map(|e| e == "safetensors")
                .unwrap_or(false)
            {
                return Ok(path);
            }
        }
        // Fall back to pytorch
        repo.get("pytorch_model.bin")
            .context("Failed to download model weights")
    }

    fn default_device() -> Result<Device> {
        #[cfg(feature = "neural-bridge-cuda")]
        {
            if candle_core::utils::cuda_is_available() {
                return Ok(Device::new_cuda(0)?);
            }
        }
        Ok(Device::Cpu)
    }

    /// Detect if a file is in safetensors format by checking magic bytes.
    ///
    /// Safetensors files start with a little-endian u64 header size, followed by JSON.
    /// HF cache uses hash names without extensions, so we can't rely on file extension.
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
        Ok(header_size > 0 && header_size < 100_000_000)
    }

    /// Extract activation from a specific layer
    ///
    /// # Arguments
    /// * `text` - Input text to encode
    /// * `layer_idx` - Layer index (0 to 23 for BGE-M3)
    pub fn extract_layer(&self, text: &str, layer_idx: usize) -> Result<LayerActivation> {
        if layer_idx >= self.model.layers.len() {
            anyhow::bail!(
                "Layer index {} out of range (model has {} layers)",
                layer_idx,
                self.model.layers.len()
            );
        }

        let (input_ids, attention_mask) = self.tokenize(text)?;
        let hidden_states = self.forward_to_layer(&input_ids, &attention_mask, layer_idx)?;
        let activation = self.pool(&hidden_states, &attention_mask)?;

        let sequence_activations = if self.config.keep_sequence_activations {
            Some(tensor_to_vec2d(&hidden_states)?)
        } else {
            None
        };

        Ok(LayerActivation {
            layer_idx,
            activation,
            sequence_activations,
        })
    }

    /// Extract activations from all layers
    ///
    /// Returns activations from embedding layer + all 24 transformer layers.
    pub fn extract_all_layers(&self, text: &str) -> Result<AllLayerActivations> {
        let (input_ids, attention_mask) = self.tokenize(text)?;

        // Get embedding layer output
        let embedding_hidden = self.model.embeddings.forward(&input_ids)?;
        let embedding_activation = self.pool(&embedding_hidden, &attention_mask)?;

        // Forward through each layer, collecting activations
        let mut layers = Vec::with_capacity(self.model.layers.len());
        let mut hidden_states = embedding_hidden;

        // Create attention mask for transformer layers
        let extended_mask = self.create_extended_attention_mask(&attention_mask)?;

        for (idx, layer) in self.model.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, Some(&extended_mask))?;
            let activation = self.pool(&hidden_states, &attention_mask)?;

            let sequence_activations = if self.config.keep_sequence_activations {
                Some(tensor_to_vec2d(&hidden_states)?)
            } else {
                None
            };

            layers.push(LayerActivation {
                layer_idx: idx,
                activation,
                sequence_activations,
            });
        }

        Ok(AllLayerActivations {
            text: text.to_string(),
            layers,
            embedding_activation,
        })
    }

    /// Extract activations from specific layers only
    pub fn extract_layers(
        &self,
        text: &str,
        layer_indices: &[usize],
    ) -> Result<Vec<LayerActivation>> {
        let max_layer = layer_indices.iter().max().copied().unwrap_or(0);
        if max_layer >= self.model.layers.len() {
            anyhow::bail!(
                "Layer index {} out of range (model has {} layers)",
                max_layer,
                self.model.layers.len()
            );
        }

        let (input_ids, attention_mask) = self.tokenize(text)?;
        let embedding_hidden = self.model.embeddings.forward(&input_ids)?;

        let mut results = Vec::with_capacity(layer_indices.len());
        let mut hidden_states = embedding_hidden;
        let extended_mask = self.create_extended_attention_mask(&attention_mask)?;

        for idx in 0..=max_layer {
            hidden_states = self.model.layers[idx].forward(&hidden_states, Some(&extended_mask))?;

            if layer_indices.contains(&idx) {
                let activation = self.pool(&hidden_states, &attention_mask)?;
                results.push(LayerActivation {
                    layer_idx: idx,
                    activation,
                    sequence_activations: None,
                });
            }
        }

        Ok(results)
    }

    fn tokenize(&self, text: &str) -> Result<(Tensor, Tensor)> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();

        let input_ids = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::new(&attention_mask[..], &self.device)?.unsqueeze(0)?;

        Ok((input_ids, attention_mask))
    }

    fn forward_to_layer(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        target_layer: usize,
    ) -> Result<Tensor> {
        let mut hidden_states = self.model.embeddings.forward(input_ids)?;
        let extended_mask = self.create_extended_attention_mask(attention_mask)?;

        for idx in 0..=target_layer {
            hidden_states = self.model.layers[idx].forward(&hidden_states, Some(&extended_mask))?;
        }

        Ok(hidden_states)
    }

    fn create_extended_attention_mask(&self, attention_mask: &Tensor) -> Result<Tensor> {
        // [batch, seq] -> [batch, 1, 1, seq]
        let mask = attention_mask.unsqueeze(1)?.unsqueeze(1)?;
        let mask = mask.to_dtype(DType::F32)?;
        // Convert 0/1 mask to -inf/0 additive mask
        Ok(((mask - 1.0)? * 1e9)?)
    }

    fn pool(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Vec<f32>> {
        match self.config.pooling {
            PoolingMethod::Mean => {
                // Mean pooling with attention mask
                let mask = attention_mask.unsqueeze(2)?.to_dtype(DType::F32)?;
                let mask_expanded = mask.broadcast_as(hidden_states.shape())?;
                let sum_hidden = (hidden_states * &mask_expanded)?.sum(1)?;
                let sum_mask = mask_expanded.sum(1)?.clamp(1e-9, f64::MAX)?;
                let mean = (sum_hidden / sum_mask)?;
                let mean = mean.squeeze(0)?;
                Ok(mean.to_vec1()?)
            }
            PoolingMethod::Cls => {
                // First token (CLS)
                let cls = hidden_states.i((0, 0))?;
                Ok(cls.to_vec1()?)
            }
            PoolingMethod::Last => {
                // Last non-padding token
                let seq_len = hidden_states.dim(1)?;
                let last = hidden_states.i((0, seq_len - 1))?;
                Ok(last.to_vec1()?)
            }
            PoolingMethod::Max => {
                // Max pooling
                let max_pooled = hidden_states.max(1)?;
                let max_pooled = max_pooled.squeeze(0)?;
                Ok(max_pooled.to_vec1()?)
            }
        }
    }

    /// Get number of layers in the model
    pub fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> usize {
        BGE_M3_HIDDEN_DIM
    }

    /// Check if using CUDA
    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }
}

/// Convert tensor to 2D Vec
#[cfg(feature = "neural-bridge")]
fn tensor_to_vec2d(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    let tensor = tensor.squeeze(0)?; // Remove batch dimension
    let (seq_len, hidden_dim) = tensor.dims2()?;

    let flat: Vec<f32> = tensor.to_vec1()?;
    let mut result = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        let start = i * hidden_dim;
        let end = start + hidden_dim;
        result.push(flat[start..end].to_vec());
    }

    Ok(result)
}

#[cfg(test)]
#[cfg(feature = "neural-bridge")]
mod tests {
    use super::*;

    // =========================================================================
    // LayerExtractorConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        let config = LayerExtractorConfig::default();
        assert_eq!(config.pooling, PoolingMethod::Mean);
        assert!(!config.keep_sequence_activations);
    }

    #[test]
    fn test_config_default_model_id() {
        let config = LayerExtractorConfig::default();
        assert_eq!(config.model_id, super::super::bge_m3::BGE_M3_MODEL_ID);
    }

    #[test]
    fn test_config_custom() {
        let config = LayerExtractorConfig {
            model_id: "custom/model".to_string(),
            keep_sequence_activations: true,
            pooling: PoolingMethod::Cls,
        };
        assert_eq!(config.model_id, "custom/model");
        assert!(config.keep_sequence_activations);
        assert_eq!(config.pooling, PoolingMethod::Cls);
    }

    #[test]
    fn test_config_clone() {
        let config = LayerExtractorConfig::default();
        let cloned = config.clone();
        assert_eq!(config.model_id, cloned.model_id);
        assert_eq!(config.pooling, cloned.pooling);
    }

    // =========================================================================
    // PoolingMethod Tests
    // =========================================================================

    #[test]
    fn test_pooling_method_equality() {
        assert_eq!(PoolingMethod::Mean, PoolingMethod::Mean);
        assert_eq!(PoolingMethod::Cls, PoolingMethod::Cls);
        assert_eq!(PoolingMethod::Last, PoolingMethod::Last);
        assert_eq!(PoolingMethod::Max, PoolingMethod::Max);
    }

    #[test]
    fn test_pooling_method_inequality() {
        assert_ne!(PoolingMethod::Mean, PoolingMethod::Cls);
        assert_ne!(PoolingMethod::Cls, PoolingMethod::Last);
        assert_ne!(PoolingMethod::Last, PoolingMethod::Max);
        assert_ne!(PoolingMethod::Max, PoolingMethod::Mean);
    }

    #[test]
    fn test_pooling_method_debug() {
        let method = PoolingMethod::Mean;
        let debug_str = format!("{:?}", method);
        assert!(debug_str.contains("Mean"));
    }

    #[test]
    fn test_pooling_method_clone() {
        let method = PoolingMethod::Cls;
        let cloned = method;
        assert_eq!(method, cloned);
    }

    // =========================================================================
    // Constants Tests
    // =========================================================================

    #[test]
    fn test_bge_m3_num_layers_constant() {
        assert_eq!(BGE_M3_NUM_LAYERS, 24);
    }

    #[test]
    fn test_bge_m3_hidden_dim_constant() {
        assert_eq!(BGE_M3_HIDDEN_DIM, 1024);
    }

    // =========================================================================
    // LayerActivation Tests (struct creation without model)
    // =========================================================================

    #[test]
    fn test_layer_activation_creation() {
        let activation = LayerActivation {
            layer_idx: 5,
            activation: vec![0.1, 0.2, 0.3],
            sequence_activations: None,
        };
        assert_eq!(activation.layer_idx, 5);
        assert_eq!(activation.activation.len(), 3);
        assert!(activation.sequence_activations.is_none());
    }

    #[test]
    fn test_layer_activation_with_sequence() {
        let activation = LayerActivation {
            layer_idx: 10,
            activation: vec![0.5; 1024],
            sequence_activations: Some(vec![vec![0.1; 1024], vec![0.2; 1024]]),
        };
        assert!(activation.sequence_activations.is_some());
        assert_eq!(activation.sequence_activations.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_layer_activation_clone() {
        let activation = LayerActivation {
            layer_idx: 3,
            activation: vec![1.0, 2.0, 3.0],
            sequence_activations: None,
        };
        let cloned = activation.clone();
        assert_eq!(activation.layer_idx, cloned.layer_idx);
        assert_eq!(activation.activation, cloned.activation);
    }

    // =========================================================================
    // AllLayerActivations Tests
    // =========================================================================

    #[test]
    fn test_all_layer_activations_creation() {
        let all_activations = AllLayerActivations {
            text: "test text".to_string(),
            layers: vec![
                LayerActivation {
                    layer_idx: 0,
                    activation: vec![0.1; 1024],
                    sequence_activations: None,
                },
                LayerActivation {
                    layer_idx: 1,
                    activation: vec![0.2; 1024],
                    sequence_activations: None,
                },
            ],
            embedding_activation: vec![0.0; 1024],
        };
        assert_eq!(all_activations.text, "test text");
        assert_eq!(all_activations.layers.len(), 2);
        assert_eq!(all_activations.embedding_activation.len(), 1024);
    }

    #[test]
    fn test_all_layer_activations_clone() {
        let all_activations = AllLayerActivations {
            text: "clone test".to_string(),
            layers: vec![],
            embedding_activation: vec![1.0, 2.0],
        };
        let cloned = all_activations.clone();
        assert_eq!(all_activations.text, cloned.text);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_empty_activation_vector() {
        let activation = LayerActivation {
            layer_idx: 0,
            activation: vec![],
            sequence_activations: None,
        };
        assert!(activation.activation.is_empty());
    }

    #[test]
    fn test_layer_idx_boundary() {
        // Test layer 23 (last layer in BGE-M3)
        let activation = LayerActivation {
            layer_idx: BGE_M3_NUM_LAYERS - 1,
            activation: vec![0.0; BGE_M3_HIDDEN_DIM],
            sequence_activations: None,
        };
        assert_eq!(activation.layer_idx, 23);
    }
}

// Tests that don't require neural-bridge feature
#[cfg(test)]
mod feature_independent_tests {
    use super::*;

    #[test]
    fn test_constants_accessible_without_feature() {
        assert_eq!(BGE_M3_NUM_LAYERS, 24);
        assert_eq!(BGE_M3_HIDDEN_DIM, 1024);
    }
}
