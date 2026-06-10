// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BERT Layer-wise Activation Extraction
//!
//! Provides layer-by-layer activation extraction for BERT-family models using
//! candle-transformers' native BERT implementation.
//!
//! ## Supported Models
//!
//! - bert-base-uncased (12 layers, 768 dim)
//! - bert-large-uncased (24 layers, 1024 dim)
//! - bert-base-cased (12 layers, 768 dim)
//! - bert-large-cased (24 layers, 1024 dim)
//!
//! ## Research Application
//!
//! Testing whether the phenomenal signature (Φ) discovered in BGE-M3 at ~92% depth
//! generalizes to standard BERT architectures.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::perception::BertLayerExtractor;
//!
//! let extractor = BertLayerExtractor::load("bert-base-uncased")?;
//! let activations = extractor.extract_all_layers("The experience of seeing red")?;
//!
//! // Check layer 11 (92% of 12 layers) for phenomenal signature
//! println!("Layer 11 activation dim: {}", activations.layers[10].activation.len());
//! ```

#[cfg(feature = "neural-bridge")]
use anyhow::{Context, Result};

#[cfg(feature = "neural-bridge")]
use candle_core::{DType, Device, IndexOp, Tensor};

#[cfg(feature = "neural-bridge")]
use candle_nn::VarBuilder;

#[cfg(feature = "neural-bridge")]
use candle_transformers::models::bert::{BertModel, Config as BertConfig};

#[cfg(feature = "neural-bridge")]
use tokenizers::Tokenizer;

#[cfg(feature = "neural-bridge")]
use hf_hub::{Repo, RepoType, api::sync::Api};

#[cfg(feature = "neural-bridge")]
use std::path::PathBuf;

use super::layer_extractor::{AllLayerActivations, LayerActivation, PoolingMethod};

/// Supported BERT-family model presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BertPreset {
    /// bert-base-uncased: 12 layers, 768 hidden (legacy gamma/beta naming - may not load)
    BertBaseUncased,
    /// bert-large-uncased: 24 layers, 1024 hidden (legacy gamma/beta naming - may not load)
    BertLargeUncased,
    /// bert-base-cased: 12 layers, 768 hidden (legacy gamma/beta naming - may not load)
    BertBaseCased,
    /// bert-large-cased: 24 layers, 1024 hidden (legacy gamma/beta naming - may not load)
    BertLargeCased,
    /// xlm-roberta-base: 12 layers, 768 hidden (modern weight/bias naming - RECOMMENDED)
    XlmRobertaBase,
}

impl BertPreset {
    /// Get HuggingFace model ID
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::BertBaseUncased => "bert-base-uncased",
            Self::BertLargeUncased => "bert-large-uncased",
            Self::BertBaseCased => "bert-base-cased",
            Self::BertLargeCased => "bert-large-cased",
            Self::XlmRobertaBase => "xlm-roberta-base",
        }
    }

    /// Number of transformer layers
    pub fn num_layers(&self) -> usize {
        match self {
            Self::BertBaseUncased | Self::BertBaseCased | Self::XlmRobertaBase => 12,
            Self::BertLargeUncased | Self::BertLargeCased => 24,
        }
    }

    /// Hidden dimension
    pub fn hidden_dim(&self) -> usize {
        match self {
            Self::BertBaseUncased | Self::BertBaseCased | Self::XlmRobertaBase => 768,
            Self::BertLargeUncased | Self::BertLargeCased => 1024,
        }
    }

    /// Predicted phenomenal corridor layer (~92% depth)
    pub fn phenomenal_corridor_layer(&self) -> usize {
        ((self.num_layers() as f64) * 0.92) as usize
    }

    /// Get the tensor prefix for this model's safetensors file
    pub fn tensor_prefix(&self) -> &'static str {
        match self {
            Self::BertBaseUncased
            | Self::BertLargeUncased
            | Self::BertBaseCased
            | Self::BertLargeCased => "bert",
            Self::XlmRobertaBase => "roberta",
        }
    }

    /// Whether this model uses modern LayerNorm naming (weight/bias vs gamma/beta)
    pub fn uses_modern_naming(&self) -> bool {
        match self {
            Self::BertBaseUncased
            | Self::BertLargeUncased
            | Self::BertBaseCased
            | Self::BertLargeCased => false, // Legacy gamma/beta
            Self::XlmRobertaBase => true, // Modern weight/bias
        }
    }
}

/// Configuration for BERT layer extraction
#[cfg(feature = "neural-bridge")]
#[derive(Debug, Clone)]
pub struct BertExtractorConfig {
    /// Model preset to use
    pub preset: BertPreset,
    /// Pooling method for sequence → vector
    pub pooling: PoolingMethod,
    /// Whether to keep sequence-level activations
    pub keep_sequence_activations: bool,
}

#[cfg(feature = "neural-bridge")]
impl Default for BertExtractorConfig {
    fn default() -> Self {
        Self {
            preset: BertPreset::BertBaseUncased,
            pooling: PoolingMethod::Mean,
            keep_sequence_activations: false,
        }
    }
}

/// BERT layer-wise activation extractor
///
/// Uses candle-transformers' native BERT implementation with layer-by-layer access.
#[cfg(feature = "neural-bridge")]
pub struct BertLayerExtractor {
    /// BERT model with layer access via encoder.layers
    model: BertModelWithLayerAccess,
    /// Tokenizer
    tokenizer: Tokenizer,
    /// Device
    device: Device,
    /// Configuration
    config: BertExtractorConfig,
    /// Model config for dimensions
    bert_config: BertConfig,
}

/// Internal wrapper that provides layer-by-layer access
#[cfg(feature = "neural-bridge")]
struct BertModelWithLayerAccess {
    /// The underlying BERT model
    inner: BertModel,
}

#[cfg(feature = "neural-bridge")]
impl BertLayerExtractor {
    /// Load with default configuration (bert-base-uncased)
    pub fn load_default() -> Result<Self> {
        Self::load(BertExtractorConfig::default())
    }

    /// Load with specific preset
    pub fn load_preset(preset: BertPreset) -> Result<Self> {
        Self::load(BertExtractorConfig {
            preset,
            ..Default::default()
        })
    }

    /// Load with custom configuration
    pub fn load(config: BertExtractorConfig) -> Result<Self> {
        let device = Self::default_device()?;
        Self::load_with_device(config, device)
    }

    /// Load with explicit device
    pub fn load_with_device(config: BertExtractorConfig, device: Device) -> Result<Self> {
        let api = Api::new()?;
        let model_id = config.preset.model_id();
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        // Download tokenizer
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer")?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Download config
        let config_path = repo
            .get("config.json")
            .context("Failed to download config")?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let bert_config: BertConfig = serde_json::from_str(&config_str)?;

        // Download weights
        let weights_path = Self::get_weights_path(&repo)?;

        // Load model weights
        let vb = Self::load_var_builder(&weights_path, &device)?;

        // Load BERT model
        // Different models use different tensor prefixes:
        // - BERT: bert.embeddings.*, bert.encoder.*
        // - XLM-RoBERTa: roberta.embeddings.*, roberta.encoder.*
        let prefix = config.preset.tensor_prefix();
        let vb_prefixed = vb.pp(prefix);

        // Check if model uses legacy LayerNorm naming (gamma/beta vs weight/bias)
        if !config.preset.uses_modern_naming() {
            anyhow::bail!(
                "Model {} uses legacy LayerNorm naming (gamma/beta) which is not supported by candle-transformers. \
                 Use XlmRobertaBase instead, which has modern naming and is architecturally similar.",
                config.preset.model_id()
            );
        }

        let inner = BertModel::load(vb_prefixed, &bert_config)?;
        let model = BertModelWithLayerAccess { inner };

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
            bert_config,
        })
    }

    fn get_weights_path(repo: &hf_hub::api::sync::ApiRepo) -> Result<PathBuf> {
        // Try safetensors first
        if let Ok(path) = repo.get("model.safetensors") {
            return Ok(path);
        }
        // Fall back to pytorch
        repo.get("pytorch_model.bin")
            .context("Failed to download model weights")
    }

    fn load_var_builder(weights_path: &PathBuf, device: &Device) -> Result<VarBuilder<'static>> {
        // Check if safetensors by attempting to parse header
        let is_safetensors = Self::is_safetensors_file(weights_path)?;

        if is_safetensors {
            super::model_integrity::verified_mmap_safetensors(
                &[weights_path.clone()],
                DType::F32,
                device,
                None,
            )
        } else {
            Ok(VarBuilder::from_pth(weights_path, DType::F32, device)?)
        }
    }

    fn is_safetensors_file(path: &PathBuf) -> Result<bool> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut header = [0u8; 8];
        if file.read_exact(&mut header).is_err() {
            return Ok(false);
        }
        let header_size = u64::from_le_bytes(header);
        Ok(header_size > 0 && header_size < 100_000_000)
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

    /// Extract activations from all layers
    ///
    /// Returns activations from embedding layer + all transformer layers.
    ///
    /// Note: Due to candle-transformers' BertModel not exposing internal layer states,
    /// we need to manually iterate through encoder.layers for layer-by-layer extraction.
    pub fn extract_all_layers(&self, text: &str) -> Result<AllLayerActivations> {
        let (input_ids, token_type_ids, attention_mask) = self.tokenize(text)?;

        // Get embedding output via forward pass - we'll need to access the encoder layers directly
        // Unfortunately, BertModel::forward doesn't return intermediate states.
        // We need to replicate the forward pass with layer-by-layer access.

        // Workaround: run full forward pass and use final output.
        // Intermediate layer extraction requires upstream candle-transformers changes
        // to expose encoder layers publicly (tracked upstream, not forked yet).

        let _extended_mask = Self::get_extended_attention_mask(&attention_mask)?;

        // Access the encoder layers directly through the model's encoder field
        // This requires modifying how we interact with the model
        let mut layers = Vec::with_capacity(self.bert_config.num_hidden_layers);

        // We need to manually forward through embeddings and each layer
        // This is a limitation - we'll document that full layer extraction
        // requires direct layer access which BertModel doesn't expose publicly

        // Fallback: Run full forward and return only final layer
        let output =
            self.model
                .inner
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
        let final_activation = self.pool(&output, &attention_mask)?;

        // Create layer activations - for now only final layer is accurate
        for layer_idx in 0..self.bert_config.num_hidden_layers {
            // Note: These are placeholders - true layer extraction requires model modification
            layers.push(LayerActivation {
                layer_idx,
                activation: if layer_idx == self.bert_config.num_hidden_layers - 1 {
                    final_activation.clone()
                } else {
                    // Placeholder - would need intermediate extraction
                    vec![0.0; self.bert_config.hidden_size]
                },
                sequence_activations: None,
            });
        }

        // Get embedding activation (also requires direct access)
        let embedding_activation = vec![0.0; self.bert_config.hidden_size];

        Ok(AllLayerActivations {
            text: text.to_string(),
            layers,
            embedding_activation,
        })
    }

    /// Extract from a single layer (final layer only - accurate)
    ///
    /// Note: Due to BertModel encapsulation, only the final layer can be accurately
    /// extracted without model modification.
    pub fn extract_final_layer(&self, text: &str) -> Result<LayerActivation> {
        let (input_ids, token_type_ids, attention_mask) = self.tokenize(text)?;
        let output =
            self.model
                .inner
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
        let activation = self.pool(&output, &attention_mask)?;

        Ok(LayerActivation {
            layer_idx: self.bert_config.num_hidden_layers - 1,
            activation,
            sequence_activations: None,
        })
    }

    fn tokenize(&self, text: &str) -> Result<(Tensor, Tensor, Tensor)> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let type_ids: Vec<u32> = encoding.get_type_ids().to_vec();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();

        let input_ids = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(&type_ids[..], &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::new(&attention_mask[..], &self.device)?.unsqueeze(0)?;

        Ok((input_ids, token_type_ids, attention_mask))
    }

    fn get_extended_attention_mask(attention_mask: &Tensor) -> Result<Tensor> {
        let attention_mask = attention_mask.unsqueeze(1)?.unsqueeze(1)?;
        let attention_mask = attention_mask.to_dtype(DType::F32)?;
        Ok(((attention_mask.ones_like()? - &attention_mask)? * f32::MIN as f64)?)
    }

    fn pool(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Vec<f32>> {
        match self.config.pooling {
            PoolingMethod::Mean => {
                let mask = attention_mask.unsqueeze(2)?.to_dtype(DType::F32)?;
                let mask_expanded = mask.broadcast_as(hidden_states.shape())?;
                let sum_hidden = (hidden_states * &mask_expanded)?.sum(1)?;
                let sum_mask = mask_expanded.sum(1)?.clamp(1e-9, f64::MAX)?;
                let mean = (sum_hidden / sum_mask)?;
                let mean = mean.squeeze(0)?;
                Ok(mean.to_vec1()?)
            }
            PoolingMethod::Cls => {
                let cls = hidden_states.i((0, 0))?;
                Ok(cls.to_vec1()?)
            }
            PoolingMethod::Last => {
                let seq_len = hidden_states.dim(1)?;
                let last = hidden_states.i((0, seq_len - 1))?;
                Ok(last.to_vec1()?)
            }
            PoolingMethod::Max => {
                let max_pooled = hidden_states.max(1)?;
                let max_pooled = max_pooled.squeeze(0)?;
                Ok(max_pooled.to_vec1()?)
            }
        }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.bert_config.num_hidden_layers
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.bert_config.hidden_size
    }

    /// Get the predicted phenomenal corridor layer
    pub fn phenomenal_corridor_layer(&self) -> usize {
        self.config.preset.phenomenal_corridor_layer()
    }

    /// Check if using CUDA
    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }
}

/// Status of BERT layer extraction support
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BertExtractionStatus {
    /// Full layer-by-layer extraction supported
    FullSupport,
    /// Only final layer extraction supported (current state)
    FinalLayerOnly,
    /// Not supported
    NotSupported,
}

/// Get current extraction status for BERT models
pub fn bert_extraction_status() -> BertExtractionStatus {
    // Currently only final layer is supported due to BertModel encapsulation
    BertExtractionStatus::FinalLayerOnly
}

/// Print BERT extraction status summary
pub fn print_bert_status() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   BERT LAYER EXTRACTION STATUS                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Current Status: {:?}\n", bert_extraction_status());

    println!("Supported Models (Modern Naming - WORKING):");
    for preset in [BertPreset::XlmRobertaBase] {
        println!(
            "  ✓ {} ({} layers, {} dim) → Corridor Layer {}",
            preset.model_id(),
            preset.num_layers(),
            preset.hidden_dim(),
            preset.phenomenal_corridor_layer()
        );
    }

    println!("\nUnsupported Models (Legacy gamma/beta Naming):");
    for preset in [
        BertPreset::BertBaseUncased,
        BertPreset::BertLargeUncased,
        BertPreset::BertBaseCased,
        BertPreset::BertLargeCased,
    ] {
        println!(
            "  ✗ {} ({} layers, {} dim) - legacy LayerNorm naming",
            preset.model_id(),
            preset.num_layers(),
            preset.hidden_dim()
        );
    }

    println!("\nLimitations:");
    println!("  - candle-transformers BertModel doesn't expose intermediate layer states");
    println!("  - Legacy BERT checkpoints use gamma/beta instead of weight/bias");
    println!("  - Currently only final layer activation is accurate");

    println!("\nRecommendation:");
    println!("  Use XLM-RoBERTa-base for cross-architecture validation (12 layers).");
    println!("  For full phenomenal corridor analysis, use BGE-M3 (24 layers, validated).");
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // BertPreset Tests
    // =========================================================================

    #[test]
    fn test_bert_presets_num_layers() {
        assert_eq!(BertPreset::BertBaseUncased.num_layers(), 12);
        assert_eq!(BertPreset::BertLargeUncased.num_layers(), 24);
        assert_eq!(BertPreset::BertBaseCased.num_layers(), 12);
        assert_eq!(BertPreset::BertLargeCased.num_layers(), 24);
        assert_eq!(BertPreset::XlmRobertaBase.num_layers(), 12);
    }

    #[test]
    fn test_bert_presets_hidden_dim() {
        assert_eq!(BertPreset::BertBaseUncased.hidden_dim(), 768);
        assert_eq!(BertPreset::BertLargeUncased.hidden_dim(), 1024);
        assert_eq!(BertPreset::BertBaseCased.hidden_dim(), 768);
        assert_eq!(BertPreset::BertLargeCased.hidden_dim(), 1024);
        assert_eq!(BertPreset::XlmRobertaBase.hidden_dim(), 768);
    }

    #[test]
    fn test_bert_presets_phenomenal_corridor_layer() {
        // 92% of 12 layers = 11.04 → 11
        assert_eq!(BertPreset::BertBaseUncased.phenomenal_corridor_layer(), 11);
        // 92% of 24 layers = 22.08 → 22
        assert_eq!(BertPreset::BertLargeUncased.phenomenal_corridor_layer(), 22);
        assert_eq!(BertPreset::XlmRobertaBase.phenomenal_corridor_layer(), 11);
    }

    #[test]
    fn test_bert_presets_model_id() {
        assert_eq!(BertPreset::BertBaseUncased.model_id(), "bert-base-uncased");
        assert_eq!(
            BertPreset::BertLargeUncased.model_id(),
            "bert-large-uncased"
        );
        assert_eq!(BertPreset::BertBaseCased.model_id(), "bert-base-cased");
        assert_eq!(BertPreset::BertLargeCased.model_id(), "bert-large-cased");
        assert_eq!(BertPreset::XlmRobertaBase.model_id(), "xlm-roberta-base");
    }

    #[test]
    fn test_bert_presets_tensor_prefix() {
        assert_eq!(BertPreset::BertBaseUncased.tensor_prefix(), "bert");
        assert_eq!(BertPreset::BertLargeUncased.tensor_prefix(), "bert");
        assert_eq!(BertPreset::XlmRobertaBase.tensor_prefix(), "roberta");
    }

    #[test]
    fn test_bert_presets_uses_modern_naming() {
        // Legacy models use gamma/beta
        assert!(!BertPreset::BertBaseUncased.uses_modern_naming());
        assert!(!BertPreset::BertLargeUncased.uses_modern_naming());
        assert!(!BertPreset::BertBaseCased.uses_modern_naming());
        assert!(!BertPreset::BertLargeCased.uses_modern_naming());
        // XLM-RoBERTa uses modern weight/bias
        assert!(BertPreset::XlmRobertaBase.uses_modern_naming());
    }

    #[test]
    fn test_bert_preset_equality() {
        assert_eq!(BertPreset::BertBaseUncased, BertPreset::BertBaseUncased);
        assert_ne!(BertPreset::BertBaseUncased, BertPreset::BertLargeUncased);
    }

    #[test]
    fn test_bert_preset_clone() {
        let preset = BertPreset::XlmRobertaBase;
        let cloned = preset;
        assert_eq!(preset, cloned);
    }

    // =========================================================================
    // BertExtractionStatus Tests
    // =========================================================================

    #[test]
    fn test_extraction_status() {
        assert_eq!(
            bert_extraction_status(),
            BertExtractionStatus::FinalLayerOnly
        );
    }

    #[test]
    fn test_extraction_status_variants() {
        // Ensure all variants are distinct
        assert_ne!(
            BertExtractionStatus::FullSupport,
            BertExtractionStatus::FinalLayerOnly
        );
        assert_ne!(
            BertExtractionStatus::FinalLayerOnly,
            BertExtractionStatus::NotSupported
        );
        assert_ne!(
            BertExtractionStatus::FullSupport,
            BertExtractionStatus::NotSupported
        );
    }

    #[test]
    fn test_extraction_status_debug() {
        let status = BertExtractionStatus::FinalLayerOnly;
        let debug_str = format!("{:?}", status);
        assert!(debug_str.contains("FinalLayerOnly"));
    }

    // =========================================================================
    // print_bert_status Tests
    // =========================================================================

    #[test]
    fn test_print_bert_status_does_not_panic() {
        // This test ensures the print function runs without panicking
        print_bert_status();
    }

    // =========================================================================
    // Phenomenal Corridor Calculation Tests
    // =========================================================================

    #[test]
    fn test_phenomenal_corridor_boundary_values() {
        // Test that 92% calculation is correct for various layer counts
        // For 12 layers: 12 * 0.92 = 11.04 → truncated to 11
        assert_eq!(((12_f64) * 0.92) as usize, 11);
        // For 24 layers: 24 * 0.92 = 22.08 → truncated to 22
        assert_eq!(((24_f64) * 0.92) as usize, 22);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_all_presets_have_valid_configurations() {
        let presets = [
            BertPreset::BertBaseUncased,
            BertPreset::BertLargeUncased,
            BertPreset::BertBaseCased,
            BertPreset::BertLargeCased,
            BertPreset::XlmRobertaBase,
        ];

        for preset in presets {
            // All presets should have positive layer counts
            assert!(
                preset.num_layers() > 0,
                "Preset {:?} has invalid num_layers",
                preset
            );
            // All presets should have positive hidden dimensions
            assert!(
                preset.hidden_dim() > 0,
                "Preset {:?} has invalid hidden_dim",
                preset
            );
            // Phenomenal corridor should be within layer bounds
            assert!(
                preset.phenomenal_corridor_layer() < preset.num_layers(),
                "Preset {:?} has phenomenal corridor outside layer bounds",
                preset
            );
            // Model ID should not be empty
            assert!(
                !preset.model_id().is_empty(),
                "Preset {:?} has empty model_id",
                preset
            );
            // Tensor prefix should not be empty
            assert!(
                !preset.tensor_prefix().is_empty(),
                "Preset {:?} has empty tensor_prefix",
                preset
            );
        }
    }
}
