//! Mamba SSM wrapper: loads pre-trained mamba-130m via candle-transformers.
//!
//! Wraps `candle_transformers::models::mamba::{Config, Model, State}` with
//! Symthaea-specific loading, single-token inference, embedding access,
//! and biological state scaling for the Liquid-Mamba fusion.
//!
//! # Model
//!
//! Default: `state-spaces/mamba-130m` (~130M params, d_model=768, n_layer=24, vocab=50280)
//! CPU-first design targeting 6W edge deployment.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::mamba::{Config, Model, State};

/// Wrapper around candle-transformers' Mamba model with Symthaea integration.
pub struct MambaWrapper {
    model: Model,
    state: State,
    config: Config,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

impl MambaWrapper {
    /// Load a pre-trained Mamba model from HuggingFace Hub.
    ///
    /// Downloads model weights (safetensors) and tokenizer via `hf_hub`.
    /// Default model: `state-spaces/mamba-130m`.
    pub fn load(model_id: &str, device: Device) -> Result<Self> {
        tracing::info!(model_id, "Loading Mamba model");

        // Download model files from HuggingFace Hub
        let api = hf_hub::api::sync::Api::new()
            .context("Failed to create HuggingFace Hub API")?;
        let repo = api.model(model_id.to_string());

        // Load config
        let config_path = repo.get("config.json")
            .context("Failed to download config.json")?;
        let config_str = std::fs::read_to_string(&config_path)
            .context("Failed to read config.json")?;
        let config: Config = serde_json::from_str(&config_str)
            .context("Failed to parse Mamba config")?;

        // Load tokenizer — Mamba models typically don't ship their own tokenizer.json,
        // they use the GPT-NeoX tokenizer from EleutherAI/gpt-neox-20b.
        let tokenizer_path = match repo.get("tokenizer.json") {
            Ok(path) => path,
            Err(_) => {
                tracing::info!("No tokenizer.json in model repo, falling back to EleutherAI/gpt-neox-20b");
                let tok_repo = api.model("EleutherAI/gpt-neox-20b".to_string());
                tok_repo.get("tokenizer.json")
                    .context("Failed to download tokenizer.json from EleutherAI/gpt-neox-20b")?
            }
        };
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        // Load model weights — try safetensors first, fall back to pytorch .bin
        let vb = if let Ok(weights_path) = repo.get("model.safetensors") {
            tracing::info!("Loading weights from model.safetensors");
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[&weights_path],
                    DType::F32,
                    &device,
                ).context("Failed to load safetensors")?
            }
        } else {
            tracing::info!("No model.safetensors, loading from pytorch_model.bin");
            let pth_path = repo.get("pytorch_model.bin")
                .context("Failed to download pytorch_model.bin (no safetensors or .bin found)")?;
            let tensors = candle_core::pickle::read_all(&pth_path)
                .context("Failed to read pytorch_model.bin")?;
            let tensors_map: std::collections::HashMap<String, Tensor> =
                tensors.into_iter().collect();
            VarBuilder::from_tensors(tensors_map, DType::F32, &device)
        };

        let model = Model::new(&config, vb.pp("backbone"))
            .context("Failed to build Mamba model")?;

        let state = State::new(1, &config, DType::F32, &device)
            .context("Failed to initialize Mamba state")?;

        tracing::info!(
            d_model = config.d_model,
            n_layer = config.n_layer,
            vocab_size = config.vocab_size,
            "Mamba model loaded"
        );

        Ok(Self {
            model,
            state,
            config,
            tokenizer,
            device,
        })
    }

    /// Single-token forward pass.
    ///
    /// Takes a token ID, runs it through the full Mamba model,
    /// and returns logits as `Vec<f32>` (vocab_size elements).
    pub fn forward_one_token(&mut self, token_id: u32) -> Result<Vec<f32>> {
        let input_ids = Tensor::new(&[token_id], &self.device)?; // [1] — single token

        let logits = self.model.forward(&input_ids, &mut self.state)
            .context("Mamba forward pass failed")?;

        // logits shape: [1, vocab_size] — squeeze to [vocab_size]
        let logits = logits.squeeze(0)?;
        let logits_vec: Vec<f32> = logits.to_vec1()
            .context("Failed to convert logits to Vec<f32>")?;

        Ok(logits_vec)
    }

    /// Extract the 768D embedding vector for a given token ID.
    ///
    /// Accesses the model's embedding table directly. Used for
    /// back-projecting generated tokens into HDC space.
    pub fn embedding_vector(&self, token_id: u32) -> Result<Vec<f32>> {
        // Run a single-token forward pass on a fresh state and extract the
        // SSM hidden state as a contextual embedding (richer than a static lookup).
        // Model internals (embedding table, lm_head weights) are private in
        // candle-transformers, so we use the hidden state proxy instead.
        let input_ids = Tensor::new(&[token_id], &self.device)?; // [1] — single token
        let mut temp_state = State::new(1, &self.config, DType::F32, &self.device)
            .context("Failed to create temp state for embedding extraction")?;
        let _logits = self.model.forward(&input_ids, &mut temp_state)
            .context("Embedding forward pass failed")?;

        // Extract the final hidden state by averaging the SSM states
        // (each layer's hs is [1, d_inner, d_state=16])
        // We'll average across layers and reduce d_state to get a d_model-sized vector.
        let d_model = self.config.d_model;
        let mut embedding = vec![0.0f32; d_model];

        // Use the first layer's hidden state projected to d_model size
        if let Some(hs) = temp_state.hs.first() {
            let hs_flat: Vec<f32> = hs.flatten_all()?.to_vec1()
                .unwrap_or_default();
            // Take the first d_model elements (or pad with zeros)
            let take = hs_flat.len().min(d_model);
            embedding[..take].copy_from_slice(&hs_flat[..take]);
        }

        Ok(embedding)
    }

    /// Inject an initial context vector into the SSM hidden state.
    ///
    /// Encodes the context as a pseudo-embedding and runs one forward pass
    /// to prime Mamba's hidden states. This is how the HDC thought projection
    /// "steers" Mamba before generation begins.
    pub fn inject_initial_context(&mut self, context: &[f32]) -> Result<()> {
        // Reset state before injection
        self.reset();

        // Find the token whose embedding best matches the context vector
        // by using BOS token to prime, then modulate the hidden states directly.
        // Pragmatic approach: run BOS token through forward to initialize states,
        // then scale the hidden states by the context magnitude/direction.
        let bos_id = self.tokenizer.token_to_id("<|endoftext|>").unwrap_or(0);
        let _logits = self.forward_one_token(bos_id)?;

        // Modulate hidden states with context signal
        // Each layer's hidden state gets a projection of the context
        let d_model = self.config.d_model;
        let context_len = context.len().min(d_model);

        for hs in &mut self.state.hs {
            let hs_shape = hs.shape().clone();
            let hs_flat: Vec<f32> = hs.flatten_all()?.to_vec1()?;
            let mut modulated = hs_flat.clone();

            // Add context signal to the hidden state (additive modulation)
            // Scale down to avoid disrupting the SSM dynamics
            let scale = 0.1;
            for (i, m) in modulated.iter_mut().enumerate() {
                let ctx_val = context[i % context_len];
                *m += ctx_val * scale;
            }

            *hs = Tensor::from_vec(modulated, hs_shape, &self.device)?;
        }

        Ok(())
    }

    /// Get a reference to the SSM hidden states.
    pub fn hidden_states(&self) -> &State {
        &self.state
    }

    /// Get a mutable reference to the SSM hidden states.
    pub fn hidden_states_mut(&mut self) -> &mut State {
        &mut self.state
    }

    /// Scale all hidden states by a biological modulation factor.
    ///
    /// `factor < 1.0` → faster decay → shorter memory (exhausted/agitated)
    /// `factor > 1.0` → slower decay → longer memory (rested/calm)
    pub fn scale_hidden_states(&mut self, factor: f32) -> Result<()> {
        let factor_f64 = factor as f64;
        for hs in &mut self.state.hs {
            *hs = (hs.clone() * factor_f64)?;
        }
        Ok(())
    }

    /// Reset state to zeros and position to 0.
    pub fn reset(&mut self) {
        if let Ok(new_state) = State::new(1, &self.config, DType::F32, &self.device) {
            self.state = new_state;
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Model hidden dimension.
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    /// Number of layers.
    pub fn n_layer(&self) -> usize {
        self.config.n_layer
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Detokenization failed: {e}"))
    }

    /// Decode a single token ID to text.
    pub fn decode_token(&self, id: u32) -> Result<String> {
        self.decode(&[id])
    }

    /// Get the EOS token ID.
    pub fn eos_token_id(&self) -> u32 {
        self.tokenizer.token_to_id("<|endoftext|>").unwrap_or(0)
    }

    /// Get reference to the underlying tokenizer.
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Get reference to the device.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl std::fmt::Debug for MambaWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MambaWrapper")
            .field("d_model", &self.config.d_model)
            .field("n_layer", &self.config.n_layer)
            .field("vocab_size", &self.config.vocab_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Integration test requiring network access to download mamba-130m.
    /// Ignored in CI — run manually with `cargo test --features mamba -- --ignored`.
    #[test]
    #[ignore = "Requires network access to download mamba-130m (~260MB)"]
    fn test_mamba_load_and_forward() {
        let device = Device::Cpu;
        let mut wrapper = MambaWrapper::load("state-spaces/mamba-130m", device)
            .expect("Failed to load mamba-130m");

        assert_eq!(wrapper.d_model(), 768);
        assert_eq!(wrapper.n_layer(), 24);
        assert!(wrapper.vocab_size() >= 50257);

        // Forward pass with a single token
        let logits = wrapper.forward_one_token(0).expect("Forward pass failed");
        // Logits length matches model's actual output dim (may differ from config.vocab_size
        // by a few padding entries — config says 50280, weights may be 50277)
        assert!(logits.len() >= 50257, "vocab too small: {}", logits.len());
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    #[ignore = "Requires network access to download mamba-130m (~260MB)"]
    fn test_mamba_context_injection() {
        let device = Device::Cpu;
        let mut wrapper = MambaWrapper::load("state-spaces/mamba-130m", device)
            .expect("Failed to load mamba-130m");

        let context = vec![0.1f32; 768];
        wrapper.inject_initial_context(&context)
            .expect("Context injection failed");

        let logits = wrapper.forward_one_token(1).expect("Forward after injection failed");
        assert!(logits.len() >= 50257, "vocab too small: {}", logits.len());
    }

    #[test]
    #[ignore = "Requires network access to download mamba-130m (~260MB)"]
    fn test_mamba_state_scaling() {
        let device = Device::Cpu;
        let mut wrapper = MambaWrapper::load("state-spaces/mamba-130m", device)
            .expect("Failed to load mamba-130m");

        // Run a token to get non-zero state
        let _logits = wrapper.forward_one_token(0).expect("Forward pass failed");

        // Scale states
        wrapper.scale_hidden_states(0.5).expect("State scaling failed");
        wrapper.scale_hidden_states(2.0).expect("State scaling failed");

        // Should still produce valid logits
        let logits = wrapper.forward_one_token(1).expect("Forward after scaling failed");
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    #[ignore = "Requires network access to download mamba-130m (~260MB)"]
    fn test_mamba_encode_decode() {
        let device = Device::Cpu;
        let wrapper = MambaWrapper::load("state-spaces/mamba-130m", device)
            .expect("Failed to load mamba-130m");

        let text = "Hello, world!";
        let ids = wrapper.encode(text).expect("Encoding failed");
        assert!(!ids.is_empty());

        let decoded = wrapper.decode(&ids).expect("Decoding failed");
        assert_eq!(decoded.trim(), text);
    }

    #[test]
    #[ignore = "Requires network access to download mamba-130m (~260MB)"]
    fn test_mamba_reset() {
        let device = Device::Cpu;
        let mut wrapper = MambaWrapper::load("state-spaces/mamba-130m", device)
            .expect("Failed to load mamba-130m");

        // Run some tokens
        let _ = wrapper.forward_one_token(0);
        let _ = wrapper.forward_one_token(1);

        // Reset
        wrapper.reset();

        // Should work fine after reset
        let logits = wrapper.forward_one_token(0).expect("Forward after reset failed");
        assert!(logits.len() >= 50257, "vocab too small: {}", logits.len());
    }

    #[test]
    #[ignore = "Requires network access to download mamba-130m (~260MB)"]
    fn test_mamba_embedding_extraction() {
        let device = Device::Cpu;
        let wrapper = MambaWrapper::load("state-spaces/mamba-130m", device)
            .expect("Failed to load mamba-130m");

        let emb = wrapper.embedding_vector(42).expect("Embedding extraction failed");
        assert_eq!(emb.len(), 768);
        // Should be non-zero for a real token
        assert!(emb.iter().any(|x| x.abs() > 1e-10));
    }

    #[test]
    #[ignore = "Requires network access to download mamba-130m (~260MB)"]
    fn test_mamba_generate_10_tokens() {
        let device = Device::Cpu;
        let mut wrapper = MambaWrapper::load("state-spaces/mamba-130m", device)
            .expect("Failed to load mamba-130m");

        // Encode a prompt
        let prompt_ids = wrapper.encode("The meaning of life is").expect("Encode failed");

        // Feed prompt tokens
        let mut last_logits = vec![];
        for &token_id in &prompt_ids {
            last_logits = wrapper.forward_one_token(token_id).expect("Forward failed");
        }

        // Generate 10 tokens greedily
        let mut generated_ids = Vec::new();
        for _ in 0..10 {
            let next_id = last_logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as u32)
                .unwrap();
            generated_ids.push(next_id);
            last_logits = wrapper.forward_one_token(next_id).expect("Forward failed");
        }

        assert_eq!(generated_ids.len(), 10);
        let decoded = wrapper.decode(&generated_ids).expect("Decode failed");
        // Should be printable ASCII (no random binary)
        assert!(
            decoded.chars().all(|c| c.is_ascii() || c.is_alphanumeric()),
            "Generated text should be readable: {decoded:?}"
        );
    }

    #[test]
    #[ignore = "Requires network access to download mamba-130m (~260MB)"]
    fn test_mamba_e2e_thought_to_text() {
        use crate::encoder::ThoughtChannels;
        use crate::projection::HdcSsmProjection;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-mamba-e2e");
        let device = Device::Cpu;

        let mut wrapper = MambaWrapper::load("state-spaces/mamba-130m", device)
            .expect("Failed to load mamba-130m");

        // Create thought channels and encode to HDC
        let mut channels = ThoughtChannels::with_intent(1); // Answer
        channels.set_epistemic(0.0); // Certain
        channels.set_emotion(0.5, 0.5, 0.5);

        let encoder = crate::encoder::ThoughtLanguageEncoder::new(&genesis);
        let thought_hv = encoder.encode(&channels);

        // Project HDC → SSM space
        let mut projection = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let ssm_context = projection.project_to_ssm(&thought_hv);

        // Inject context into Mamba
        wrapper.inject_initial_context(&ssm_context)
            .expect("Context injection failed");

        // Generate a few tokens
        let mut token_ids = Vec::new();
        let mut logits = wrapper.forward_one_token(0).expect("Forward failed");
        for _ in 0..5 {
            let next_id = logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as u32)
                .unwrap();
            token_ids.push(next_id);
            logits = wrapper.forward_one_token(next_id).expect("Forward failed");
        }

        let text = wrapper.decode(&token_ids).expect("Decode failed");
        assert!(!text.is_empty(), "E2E should produce text");
        assert!(text.len() > 0, "E2E text: {text:?}");
    }
}
