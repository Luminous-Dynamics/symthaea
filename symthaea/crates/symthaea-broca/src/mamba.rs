// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
//!
//! # Trait Abstraction
//!
//! The [`MambaBackend`] trait provides a mockable interface for all Mamba
//! operations used by [`crate::liquid_mamba::LiquidMambaGenerator`]. This
//! enables deterministic testing without network access or model downloads.

use crate::mamba_model::{Config, Model, State};
use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;

/// Select the best available compute device.
///
/// Returns CUDA GPU 0 if available (requires the `mamba` feature),
/// otherwise falls back to CPU.
///
/// # CUDA Requirements
///
/// - Compile with `--features mamba` (enables `candle-core/cuda`)
/// - NVIDIA GPU with CUDA toolkit installed
/// - On NixOS: `LD_LIBRARY_PATH="/run/opengl-driver/lib"` required
///
/// # Fallback Behavior
///
/// If CUDA is unavailable (feature disabled, no GPU, or driver error),
/// silently falls back to CPU. Mamba-130m runs at ~430ms/forward on
/// CPU (with `RUSTFLAGS="-C target-cpu=native"` for AVX2+FMA).
pub fn best_device() -> Device {
    #[cfg(feature = "mamba")]
    {
        match Device::cuda_if_available(0) {
            Ok(dev) if dev.is_cuda() => {
                tracing::info!("Using CUDA device (GPU 0)");
                return dev;
            }
            _ => {}
        }
    }
    tracing::info!("Using CPU device");
    Device::Cpu
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAMBA BACKEND TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait abstracting Mamba SSM operations for mockability.
///
/// All methods used by [`crate::liquid_mamba::LiquidMambaGenerator`] are
/// included. Methods that expose candle-specific types (`State`, `Device`,
/// `Tokenizer`) remain as inherent methods on [`MambaWrapper`] only.
pub trait MambaBackend: Send {
    fn clone_box(&self) -> Box<dyn MambaBackend>;

    /// Single-token forward pass. Returns logits as `Vec<f32>` (vocab_size elements).
    fn forward_one_token(&mut self, token_id: u32) -> Result<Vec<f32>>;

    /// Extract the embedding vector for a given token ID (d_model dimensions).
    fn embedding_vector(&self, token_id: u32) -> Result<Vec<f32>>;

    /// Inject an initial context vector into the SSM hidden state.
    fn inject_initial_context(&mut self, context: &[f32]) -> Result<()>;

    /// Scale all hidden states by a biological modulation factor.
    fn scale_hidden_states(&mut self, factor: f32) -> Result<()>;

    /// Reset state to zeros.
    fn reset(&mut self);

    /// Vocabulary size.
    /// Forward pass through one token, returning (logits, intermediate_hidden_state).
    fn forward_with_state(&mut self, token_id: u32) -> Result<(Vec<f32>, Vec<f32>)>;

    fn vocab_size(&self) -> usize;

    /// Model hidden dimension.
    fn d_model(&self) -> usize;

    /// Number of layers.
    fn n_layer(&self) -> usize;

    /// Encode text to token IDs.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode token IDs to text.
    fn decode(&self, ids: &[u32]) -> Result<String>;

    /// Decode a single token ID to text.
    fn decode_token(&self, id: u32) -> Result<String>;

    /// Get the EOS token ID.
    fn eos_token_id(&self) -> u32;

    /// Feed a sequence of continuous embedding vectors through Mamba's layers.
    ///
    /// Each vector is processed as a soft token (bypasses token embedding lookup).
    /// SSM state accumulates across the sequence, integrating all context.
    /// The sequence is typically 64 chunks of 768D from temporal projection.
    fn inject_context_sequence(&mut self, sequence: &[Vec<f32>]) -> Result<()>;

    /// Compute E2E token loss at a specific position in the SSM sequence (Improvement B).
    ///
    /// Injects [0..pos) as context (no grad), injects [pos] as Var, detaches state,
    /// injects [pos+1..N) as context, teacher-forces, backward, and extracts gradient
    /// at position `pos`.
    ///
    /// Default implementation delegates to the full `compute_e2e_token_loss` with
    /// all positions, returning a zero gradient vector.
    fn compute_e2e_token_loss_at(
        &mut self,
        _sequence: &[Vec<f32>],
        _tokens: &[u32],
        _pos: usize,
    ) -> Result<Vec<f32>> {
        // Default: return zero gradient
        let d = self.d_model();
        Ok(vec![0.0f32; d])
    }

    /// Compute E2E token loss at multiple positions simultaneously.
    ///
    /// Returns `Vec<(position, gradient)>` for each requested position.
    /// The SSM state carries gradient across positions: we detach once before
    /// the first Var position, then all K Vars share one connected computation graph.
    ///
    /// Default: delegates to single-position for each (backward compat).
    fn compute_e2e_token_loss_multi(
        &mut self,
        sequence: &[Vec<f32>],
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<(usize, Vec<f32>)>> {
        let mut results = Vec::with_capacity(positions.len());
        for &pos in positions {
            let grad = self.compute_e2e_token_loss_at(sequence, tokens, pos)?;
            results.push((pos, grad));
        }
        Ok(results)
    }

    /// Enable LoRA adapters on the model layers (Improvement D).
    /// Default: no-op.
    fn enable_lora(&mut self, _rank: usize, _alpha: f32, _lr: f32) {}

    /// Set per-token CfC modulation for the next forward pass.
    ///
    /// `delta_scale` modulates the SSM step-size Δ (temporal evolution speed).
    /// `b_scale` modulates the input matrix B (input sensitivity).
    /// Both default to 1.0 (no modulation). Consumed after each forward pass.
    fn set_cfc_modulation(&mut self, _delta_scale: f32, _b_scale: f32) {}

    /// Set per-layer delta modulation conditioned on HDC coherence (Phase 5).
    ///
    /// Each entry in `modulation` scales the Δ (step-size) for the corresponding
    /// Mamba layer. Early layers (syntax) get stronger modulation, later layers
    /// (semantics) get less. Length must equal `n_layer()`.
    ///
    /// Default: no-op (all layers use the global delta_scale).
    fn set_per_layer_delta_modulation(&mut self, _modulation: &[f32]) {}

    /// Inject a hidden state to warm-start the next generation.
    /// Returns `Ok(())` on success.
    fn inject_hidden_state(&mut self, hidden: &[f32]) -> Result<()>;

    /// Extract the current hidden state after generation.
    /// Used for semantic chunk carry-over in Phase 3.
    fn extract_hidden_state(&self) -> Result<Vec<f32>>;
}

impl Clone for Box<dyn MambaBackend> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Wrapper around candle-transformers' Mamba model with Symthaea integration.
pub struct MambaWrapper {
    model: Model,
    state: State,
    config: Config,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    /// LoRA learning rate (0 = disabled, Improvement D)
    lora_lr: f32,
}

impl MambaWrapper {
    /// Load a pre-trained Mamba model from HuggingFace Hub.
    ///
    /// Downloads model weights (safetensors) and tokenizer via `hf_hub`.
    /// Default model: `state-spaces/mamba-130m`.
    pub fn load(model_id: &str, device: Device) -> Result<Self> {
        tracing::info!(model_id, "Loading Mamba model");

        // Download model files from HuggingFace Hub
        let api = hf_hub::api::sync::Api::new().context("Failed to create HuggingFace Hub API")?;
        let repo = api.model(model_id.to_string());

        // Load config
        let config_path = repo
            .get("config.json")
            .context("Failed to download config.json")?;
        let config_str =
            std::fs::read_to_string(&config_path).context("Failed to read config.json")?;
        let config: Config =
            serde_json::from_str(&config_str).context("Failed to parse Mamba config")?;

        // Load tokenizer — Mamba models typically don't ship their own tokenizer.json,
        // they use the GPT-NeoX tokenizer from EleutherAI/gpt-neox-20b.
        let tokenizer_path = match repo.get("tokenizer.json") {
            Ok(path) => path,
            Err(_) => {
                tracing::info!(
                    "No tokenizer.json in model repo, falling back to EleutherAI/gpt-neox-20b"
                );
                let tok_repo = api.model("EleutherAI/gpt-neox-20b".to_string());
                tok_repo
                    .get("tokenizer.json")
                    .context("Failed to download tokenizer.json from EleutherAI/gpt-neox-20b")?
            }
        };
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        // Load model weights — try safetensors first, fall back to pytorch .bin
        let vb = if let Ok(weights_path) = repo.get("model.safetensors") {
            tracing::info!("Loading weights from model.safetensors");
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)
                    .context("Failed to load safetensors")?
            }
        } else {
            tracing::info!("No model.safetensors, loading from pytorch_model.bin");
            let pth_path = repo
                .get("pytorch_model.bin")
                .context("Failed to download pytorch_model.bin (no safetensors or .bin found)")?;
            let tensors = candle_core::pickle::read_all(&pth_path)
                .context("Failed to read pytorch_model.bin")?;
            let tensors_map: std::collections::HashMap<String, Tensor> =
                tensors.into_iter().collect();
            VarBuilder::from_tensors(tensors_map, DType::F32, &device)
        };

        let model =
            Model::new(&config, vb.pp("backbone")).context("Failed to build Mamba model")?;

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
            lora_lr: 0.0,
        })
    }

    /// Single-token forward pass.
    ///
    /// Takes a token ID, runs it through the full Mamba model,
    /// and returns logits as `Vec<f32>` (vocab_size elements).
    pub fn forward_one_token(&mut self, token_id: u32) -> Result<Vec<f32>> {
        let input_ids = Tensor::new(&[token_id], &self.device)?; // [1] — single token

        let logits = self
            .model
            .forward(&input_ids, &mut self.state)
            .map_err(|e| {
                tracing::error!("Mamba forward error (device={:?}): {e:#}", self.device);
                e
            })
            .context("Mamba forward pass failed")?;

        // logits shape: [1, vocab_size] — squeeze to [vocab_size]
        let logits = logits.squeeze(0)?;
        let logits_vec: Vec<f32> = logits
            .to_vec1()
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
        let _logits = self
            .model
            .forward(&input_ids, &mut temp_state)
            .context("Embedding forward pass failed")?;

        // Extract the final hidden state by averaging the SSM states
        // (each layer's hs is [1, d_inner, d_state=16])
        // We'll average across layers and reduce d_state to get a d_model-sized vector.
        let d_model = self.config.d_model;
        let mut embedding = vec![0.0f32; d_model];

        // Use the first layer's hidden state projected to d_model size
        if let Some(hs) = temp_state.hs.first() {
            let hs_flat: Vec<f32> = hs.flatten_all()?.to_vec1().unwrap_or_default();
            // Take the first d_model elements (or pad with zeros)
            let take = hs_flat.len().min(d_model);
            embedding[..take].copy_from_slice(&hs_flat[..take]);
        }

        Ok(embedding)
    }

    /// Inject an initial context vector into the SSM hidden state.
    ///
    /// Feeds the 768D context vector directly through Mamba's layers as a
    /// continuous embedding (soft token), using `forward_embeds`. This is
    /// the proper way to initialize Mamba's SSM state from the HDC thought
    /// projection — the context vector flows through all 24 layers, building
    /// up hidden state (h₀) naturally via the SSM equations.
    ///
    /// Optionally warms the conv1d history first for better initial dynamics.
    pub fn inject_initial_context(&mut self, context: &[f32]) -> Result<()> {
        self.reset();

        let d_model = self.config.d_model;
        let mut ctx = context.to_vec();
        ctx.resize(d_model, 0.0); // Pad or truncate to d_model

        let embed = Tensor::from_vec(ctx, (1, d_model), &self.device)?.to_dtype(DType::F32)?;

        // Warm conv1d history with the context summary
        self.model.warmstart_conv_history(&embed, &mut self.state)?;

        // Run the context through all layers as a soft token
        // This builds h₀ naturally via the SSM equations: h = exp(ΔA)h + ΔBx
        let _logits = self.model.forward_embeds(&embed, &mut self.state)?;

        Ok(())
    }

    /// Set per-token CfC modulation for the next forward pass.
    ///
    /// The CfC cognitive loop calls this before each token generation to
    /// steer Mamba's dynamics. `delta_scale` modulates the SSM step-size
    /// (how much temporal evolution per token), `b_scale` modulates how
    /// strongly new input enters the hidden state.
    ///
    /// Modulation is consumed (reset to None) after each forward pass.
    pub fn set_cfc_modulation(&mut self, delta_scale: f32, b_scale: f32) {
        self.state.cfc_modulation = Some(crate::mamba_model::CfcModulation {
            delta_scale,
            b_scale,
        });
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
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
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

    /// Single-token forward pass returning (logits, hidden_state).
    pub fn internal_forward_with_state(&mut self, token_id: u32) -> Result<(Vec<f32>, Vec<f32>)> {
        let logits = self.forward_one_token(token_id)?;
        let hidden = self.internal_extract_hidden_state()?;
        Ok((logits, hidden))
    }

    /// Extract the current average hidden state across all layers.
    pub fn internal_extract_hidden_state(&self) -> Result<Vec<f32>> {
        let mut avg_h = vec![0.0f32; self.config.d_model];
        let n = self.state.hs.len() as f32;

        for h_tensor in &self.state.hs {
            // h_tensor is [batch, d_inner, d_state]
            // We want a [d_model] summary. Sum over d_state, then average.
            let h_sum = h_tensor.sum(D::Minus1)?.squeeze(D::Minus1)?;
            // h_sum is [batch, d_inner] = [1, 1536]
            let h_vec = h_sum.squeeze(0)?.to_vec1::<f32>()?;

            // Down-sample d_inner (1536) to d_model (768) via simple average of pairs
            for i in 0..self.config.d_model {
                avg_h[i] += (h_vec[i * 2] + h_vec[i * 2 + 1]) / (2.0 * n);
            }
        }

        Ok(avg_h)
    }

    /// Access the raw embedding table `[vocab_size, d_model]`.
    ///
    /// Used to compute per-dimension mean/variance for manifold moment matching
    /// (whitening adapter initialization).
    pub fn embedding_table(&self) -> &Tensor {
        self.model.embedding_table()
    }

    /// Enable LoRA adapters on all Mamba layers (Improvement D).
    pub fn enable_lora(&mut self, rank: usize, alpha: f32, lr: f32) -> Result<()> {
        self.lora_lr = lr;
        Ok(self.model.enable_lora(rank, alpha, &self.device)?)
    }

    /// Compute E2E token loss at a specific position (Improvement B).
    ///
    /// Injects [0..pos) as context (no grad), makes [pos] a Var, detaches state,
    /// injects [pos+1..N) as context, teacher-forces, backward, returns gradient at pos.
    pub fn compute_e2e_token_loss_at(
        &mut self,
        sequence: &[Vec<f32>],
        tokens: &[u32],
        pos: usize,
    ) -> Result<Vec<f32>> {
        let d_model = self.config.d_model;
        let n = sequence.len();
        if n == 0 || pos >= n || tokens.is_empty() {
            return Ok(vec![0.0f32; d_model]);
        }

        // Reset state
        self.reset();

        // 1. Inject [0..pos) as context (no autograd)
        for i in 0..pos {
            let embed = Tensor::from_vec(sequence[i].clone(), (1, d_model), &self.device)?;
            self.model.forward_embeds_no_head(&embed, &mut self.state)?;
        }

        // 2. Make sequence[pos] a Var for gradient tracking
        let var_embed = candle_core::Var::from_tensor(&Tensor::from_vec(
            sequence[pos].clone(),
            (1, d_model),
            &self.device,
        )?)?;
        let _logits = self
            .model
            .forward_embeds(&var_embed.as_tensor(), &mut self.state)?;

        // 3. Detach state to prevent gradient flow through warmup
        self.state.detach()?;

        // 4. Inject [pos+1..N) as context (no autograd needed for these)
        for i in (pos + 1)..n {
            let embed = Tensor::from_vec(sequence[i].clone(), (1, d_model), &self.device)?;
            self.model.forward_embeds_no_head(&embed, &mut self.state)?;
        }

        // 5. Teacher-force: run one token to get loss
        // Use the first token as the teacher signal
        let teacher_token = tokens[0.min(tokens.len() - 1)];
        let input = Tensor::new(&[teacher_token], &self.device)?;
        let logits = self.model.forward(&input, &mut self.state)?;
        let logits = logits.squeeze(0)?;

        // Cross-entropy loss at target token
        let target_idx = tokens.get(1).copied().unwrap_or(teacher_token) as usize;
        let vocab = logits.dim(0)?;
        if target_idx >= vocab {
            return Ok(vec![0.0f32; d_model]);
        }

        let log_softmax = candle_nn::ops::log_softmax(&logits, 0)?;
        let loss = log_softmax.i(target_idx)?.neg()?;

        // 6. Backward pass
        let grads = loss.backward()?;

        // 7. Apply LoRA grads if enabled
        if self.lora_lr > 0.0 && self.model.has_lora() {
            self.model.apply_lora_grads(&grads, self.lora_lr)?;
        }

        // 8. Extract gradient at the Var position
        let grad = match grads.get(var_embed.as_tensor()) {
            Some(g) => g.flatten_all()?.to_vec1::<f32>()?,
            None => vec![0.0f32; d_model],
        };

        Ok(grad)
    }

    /// Compute E2E token loss at multiple positions simultaneously.
    ///
    /// Creates K Vars at the requested positions and builds a single connected
    /// computation graph. The SSM state carries gradient between Var positions
    /// because `State.hs` contains autograd-tracked tensors.
    ///
    /// Steps:
    /// 1. Inject `[0..first_var_pos)` as context (no autograd)
    /// 2. Detach state (cut gradient flow from warmup)
    /// 3. Inject `[first_var_pos..N)`: Var at requested positions, Tensor elsewhere
    /// 4. Teacher-force 1 token, compute cross-entropy loss
    /// 5. Backward → extract gradients from all K Vars
    pub fn compute_e2e_token_loss_multi(
        &mut self,
        sequence: &[Vec<f32>],
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<(usize, Vec<f32>)>> {
        let d_model = self.config.d_model;
        let n = sequence.len();
        if n == 0 || positions.is_empty() || tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Sort positions for sequential processing
        let mut sorted_positions = positions.to_vec();
        sorted_positions.sort_unstable();
        sorted_positions.dedup();

        let first_var_pos = sorted_positions[0];

        // Reset state
        self.reset();

        // 1. Inject [0..first_var_pos) as context (no autograd)
        for i in 0..first_var_pos.min(n) {
            let embed = Tensor::from_vec(sequence[i].clone(), (1, d_model), &self.device)?;
            self.model.forward_embeds_no_head(&embed, &mut self.state)?;
        }

        // 2. Detach state to cut gradient flow from warmup
        self.state.detach()?;

        // 3. Inject [first_var_pos..N): Var at requested positions, Tensor elsewhere
        let pos_set: std::collections::HashSet<usize> = sorted_positions.iter().copied().collect();
        let mut vars: Vec<(usize, candle_core::Var)> = Vec::with_capacity(sorted_positions.len());

        for i in first_var_pos..n {
            if pos_set.contains(&i) {
                let var_embed = candle_core::Var::from_tensor(&Tensor::from_vec(
                    sequence[i].clone(),
                    (1, d_model),
                    &self.device,
                )?)?;
                self.model
                    .forward_embeds_no_head(var_embed.as_tensor(), &mut self.state)?;
                vars.push((i, var_embed));
            } else {
                let embed = Tensor::from_vec(sequence[i].clone(), (1, d_model), &self.device)?;
                self.model.forward_embeds_no_head(&embed, &mut self.state)?;
            }
        }

        // 4. Teacher-force: run one token to get loss
        let teacher_token = tokens[0.min(tokens.len() - 1)];
        let input = Tensor::new(&[teacher_token], &self.device)?;
        let logits = self.model.forward(&input, &mut self.state)?;
        let logits = logits.squeeze(0)?;

        let target_idx = tokens.get(1).copied().unwrap_or(teacher_token) as usize;
        let vocab = logits.dim(0)?;
        if target_idx >= vocab {
            return Ok(vars
                .iter()
                .map(|(pos, _)| (*pos, vec![0.0f32; d_model]))
                .collect());
        }

        let log_softmax = candle_nn::ops::log_softmax(&logits, 0)?;
        let loss = log_softmax.i(target_idx)?.neg()?;

        // 5. Backward pass
        let grads = loss.backward()?;

        // 6. Apply LoRA grads if enabled
        if self.lora_lr > 0.0 && self.model.has_lora() {
            self.model.apply_lora_grads(&grads, self.lora_lr)?;
        }

        // 7. Extract gradients from all Vars
        let results: Vec<(usize, Vec<f32>)> = vars
            .iter()
            .map(|(pos, var)| {
                let grad = match grads.get(var.as_tensor()) {
                    Some(g) => g
                        .flatten_all()
                        .and_then(|t| t.to_vec1::<f32>())
                        .unwrap_or_else(|_| vec![0.0f32; d_model]),
                    None => vec![0.0f32; d_model],
                };
                (*pos, grad)
            })
            .collect();

        Ok(results)
    }

    /// Feed a sequence of continuous embedding vectors through Mamba's layers.
    ///
    /// Resets state, then processes each vector as a soft token via `forward_embeds()`,
    /// bypassing the token embedding lookup. SSM state accumulates across the sequence.
    pub fn inject_context_sequence(&mut self, sequence: &[Vec<f32>]) -> Result<()> {
        self.reset();
        let d_model = self.config.d_model;
        let seq_len = sequence.len();

        // Pre-stack all chunks into a single (seq_len, d_model) tensor to avoid
        // per-chunk allocation overhead. The model processes them sequentially
        // (SSM scan is inherently sequential) but we save 64 separate Tensor::from_vec calls.
        let mut flat = Vec::with_capacity(seq_len * d_model);
        for chunk in sequence {
            assert_eq!(
                chunk.len(),
                d_model,
                "Each chunk must be d_model={d_model} dimensions"
            );
            flat.extend_from_slice(chunk);
        }
        let stacked = Tensor::from_vec(flat, (seq_len, d_model), &self.device)?;

        // Warmstart conv1d history with the mean of the first chunk,
        // so the first few tokens see meaningful convolution context.
        if seq_len > 0 {
            let summary = stacked.i(0)?.unsqueeze(0)?; // (1, d_model)
            self.model
                .warmstart_conv_history(&summary, &mut self.state)?;
        }

        // Forward through all layers using inject_sequence (skips lm_head)
        self.model.inject_sequence(&stacked, &mut self.state)?;
        Ok(())
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

impl MambaWrapper {
    pub fn internal_inject_hidden_state(&mut self, hidden: &[f32]) -> Result<()> {
        let d_model = self.config.d_model;
        let n_layers = self.config.n_layer;
        if hidden.len() != d_model {
            anyhow::bail!(
                "Hidden state dimension mismatch: expected {}, got {}",
                d_model,
                hidden.len()
            );
        }

        // Broadcast the 768D semantic vector to all 64 layers as a warm-start
        let h_tensor = Tensor::from_slice(hidden, (1, d_model), self.model.device())?;
        // Expand to [1, d_inner, d_state] = [1, 1536, 16]
        let h_inner = h_tensor.repeat((1, 2))?; // [1, 1536]
        let h_state = h_inner.unsqueeze(D::Minus1)?.repeat((1, 1, 16))?;

        for i in 0..n_layers {
            self.state.hs[i] = h_state.clone();
        }

        Ok(())
    }
}

impl Clone for MambaWrapper {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            state: self.state.clone(),
            config: self.config.clone(),
            tokenizer: self.tokenizer.clone(),
            device: self.device.clone(),
            lora_lr: self.lora_lr,
        }
    }
}

impl MambaBackend for MambaWrapper {
    fn clone_box(&self) -> Box<dyn MambaBackend> {
        Box::new(self.clone())
    }

    fn forward_one_token(&mut self, token_id: u32) -> Result<Vec<f32>> {
        MambaWrapper::forward_one_token(self, token_id)
    }

    fn embedding_vector(&self, token_id: u32) -> Result<Vec<f32>> {
        MambaWrapper::embedding_vector(self, token_id)
    }

    fn inject_initial_context(&mut self, context: &[f32]) -> Result<()> {
        MambaWrapper::inject_initial_context(self, context)
    }

    fn scale_hidden_states(&mut self, factor: f32) -> Result<()> {
        MambaWrapper::scale_hidden_states(self, factor)
    }

    fn reset(&mut self) {
        MambaWrapper::reset(self)
    }

    fn vocab_size(&self) -> usize {
        MambaWrapper::vocab_size(self)
    }

    fn d_model(&self) -> usize {
        MambaWrapper::d_model(self)
    }

    fn n_layer(&self) -> usize {
        MambaWrapper::n_layer(self)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        MambaWrapper::encode(self, text)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        MambaWrapper::decode(self, ids)
    }

    fn decode_token(&self, id: u32) -> Result<String> {
        MambaWrapper::decode_token(self, id)
    }

    fn eos_token_id(&self) -> u32 {
        MambaWrapper::eos_token_id(self)
    }

    fn inject_context_sequence(&mut self, sequence: &[Vec<f32>]) -> Result<()> {
        MambaWrapper::inject_context_sequence(self, sequence)
    }

    fn compute_e2e_token_loss_at(
        &mut self,
        sequence: &[Vec<f32>],
        tokens: &[u32],
        pos: usize,
    ) -> Result<Vec<f32>> {
        MambaWrapper::compute_e2e_token_loss_at(self, sequence, tokens, pos)
    }

    fn compute_e2e_token_loss_multi(
        &mut self,
        sequence: &[Vec<f32>],
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<(usize, Vec<f32>)>> {
        MambaWrapper::compute_e2e_token_loss_multi(self, sequence, tokens, positions)
    }

    fn enable_lora(&mut self, rank: usize, alpha: f32, lr: f32) {
        if let Err(e) = MambaWrapper::enable_lora(self, rank, alpha, lr) {
            tracing::error!("Failed to enable LoRA: {e}");
        }
    }

    fn set_cfc_modulation(&mut self, delta_scale: f32, b_scale: f32) {
        MambaWrapper::set_cfc_modulation(self, delta_scale, b_scale)
    }

    fn forward_with_state(&mut self, token_id: u32) -> Result<(Vec<f32>, Vec<f32>)> {
        self.internal_forward_with_state(token_id)
    }

    fn inject_hidden_state(&mut self, hidden: &[f32]) -> Result<()> {
        self.internal_inject_hidden_state(hidden)
    }

    fn extract_hidden_state(&self) -> Result<Vec<f32>> {
        self.internal_extract_hidden_state()
    }
}

/// Mock Mamba backend for testing without network access.
#[cfg(any(test, feature = "test-helpers", feature = "mamba-cpu"))]
pub mod mock {
    use super::*;

    // ═══════════════════════════════════════════════════════════════════════
    // MOCK MAMBA
    // ═══════════════════════════════════════════════════════════════════════

    /// Deterministic mock Mamba backend for testing without network access.
    ///
    /// Produces repeatable, seeded outputs using xorshift64 PRNG. All operations
    /// that would require a real model (forward, embedding, encode/decode) are
    /// replaced with lightweight deterministic computations.
    #[derive(Debug, Clone)]
    pub struct MockMamba {
        /// Number of forward passes executed.
        pub forward_count: usize,
        /// Cumulative hidden state scale factor (starts at 1.0).
        pub scale_factor: f32,
        /// Number of context injections performed.
        pub injection_count: usize,
        /// Last injected context magnitude (L2 norm).
        pub last_context_magnitude: f32,
        /// Number of resets performed.
        pub reset_count: usize,
        /// Phase 3: Simulated last hidden state.
        pub last_hidden: Option<Vec<f32>>,
    }

    impl MockMamba {
        /// Create a new mock Mamba with default state.
        pub fn new() -> Self {
            Self {
                forward_count: 0,
                scale_factor: 1.0,
                injection_count: 0,
                last_context_magnitude: 0.0,
                reset_count: 0,
                last_hidden: None,
            }
        }

        /// xorshift64 PRNG step. Seed 0 is a fixed point, so we XOR with a
        /// golden-ratio constant before starting.
        fn xorshift64(state: &mut u64) -> u64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        }
    }

    impl MambaBackend for MockMamba {
        fn clone_box(&self) -> Box<dyn MambaBackend> {
            Box::new(self.clone())
        }

        fn forward_one_token(&mut self, token_id: u32) -> Result<Vec<f32>> {
            self.forward_count += 1;
            let vocab = 50280usize;
            let mut logits = Vec::with_capacity(vocab);

            // Seed from token_id × golden ratio + forward_count for determinism that varies per call
            let mut state = (token_id as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(self.forward_count as u64)
                .wrapping_mul(0x517CC1B727220A95);
            if state == 0 {
                state = 0x9E3779B97F4A7C15;
            }

            for _ in 0..vocab {
                let r = Self::xorshift64(&mut state);
                // Map u64 to [-2.0, 2.0] range for logit-like distribution
                let val = (r as f64 / u64::MAX as f64) * 4.0 - 2.0;
                logits.push(val as f32 * self.scale_factor);
            }
            Ok(logits)
        }

        fn forward_with_state(&mut self, token_id: u32) -> Result<(Vec<f32>, Vec<f32>)> {
            let logits = self.forward_one_token(token_id)?;
            let d_model = 768;
            Ok((logits, vec![0.001f32; d_model]))
        }

        fn embedding_vector(&self, token_id: u32) -> Result<Vec<f32>> {
            let d = 768;
            let mut emb = Vec::with_capacity(d);
            let mut state = (token_id as u64 ^ 0x517CC1B727220A95).wrapping_add(1);
            if state == 0 {
                state = 0x517CC1B727220A95;
            }

            for _ in 0..d {
                let r = Self::xorshift64(&mut state);
                let val = (r as f64 / u64::MAX as f64) * 2.0 - 1.0;
                emb.push(val as f32);
            }
            Ok(emb)
        }

        fn inject_initial_context(&mut self, context: &[f32]) -> Result<()> {
            self.injection_count += 1;
            // Record L2 magnitude
            let mag: f32 = context.iter().map(|x| x * x).sum::<f32>().sqrt();
            self.last_context_magnitude = mag;
            Ok(())
        }

        fn scale_hidden_states(&mut self, factor: f32) -> Result<()> {
            self.scale_factor *= factor;
            Ok(())
        }

        fn reset(&mut self) {
            self.reset_count += 1;
            self.forward_count = 0;
            self.scale_factor = 1.0;
            self.last_context_magnitude = 0.0;
        }

        fn vocab_size(&self) -> usize {
            50280
        }

        fn d_model(&self) -> usize {
            768
        }

        fn n_layer(&self) -> usize {
            24
        }

        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            // Simple byte-to-id mapping: each UTF-8 byte becomes a token ID
            Ok(text.as_bytes().iter().map(|&b| b as u32).collect())
        }

        fn decode(&self, ids: &[u32]) -> Result<String> {
            // Reverse: id → byte (mod 128 for ASCII safety)
            let bytes: Vec<u8> = ids.iter().map(|&id| (id % 128) as u8).collect();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }

        fn decode_token(&self, id: u32) -> Result<String> {
            self.decode(&[id])
        }

        fn eos_token_id(&self) -> u32 {
            0 // Same convention as MambaWrapper (fallback when <|endoftext|> not found)
        }

        fn inject_context_sequence(&mut self, sequence: &[Vec<f32>]) -> Result<()> {
            self.injection_count += 1;
            // Record average L2 magnitude across chunks for diagnostics
            let avg_mag: f32 = sequence
                .iter()
                .map(|c| c.iter().map(|x| x * x).sum::<f32>().sqrt())
                .sum::<f32>()
                / sequence.len().max(1) as f32;
            self.last_context_magnitude = avg_mag;
            Ok(())
        }

        fn compute_e2e_token_loss_at(
            &mut self,
            _sequence: &[Vec<f32>],
            _tokens: &[u32],
            _pos: usize,
        ) -> Result<Vec<f32>> {
            // Mock: return small deterministic gradient
            let d = self.d_model();
            Ok(vec![0.001f32; d])
        }

        fn compute_e2e_token_loss_multi(
            &mut self,
            _sequence: &[Vec<f32>],
            _tokens: &[u32],
            positions: &[usize],
        ) -> Result<Vec<(usize, Vec<f32>)>> {
            // Mock: return small deterministic gradient at each requested position
            let d = self.d_model();
            Ok(positions
                .iter()
                .map(|&pos| {
                    // Vary gradient slightly by position for testing
                    let grad_val = 0.001 * (1.0 + pos as f32 * 0.01);
                    (pos, vec![grad_val; d])
                })
                .collect())
        }

        fn enable_lora(&mut self, _rank: usize, _alpha: f32, _lr: f32) {
            // No-op in mock
        }

        fn inject_hidden_state(&mut self, hidden: &[f32]) -> Result<()> {
            // Store for next forward pass (simulated warm start)
            self.last_hidden = Some(hidden.to_vec());
            Ok(())
        }

        fn extract_hidden_state(&self) -> Result<Vec<f32>> {
            if let Some(ref h) = self.last_hidden {
                Ok(h.clone())
            } else {
                // Return a default hidden state if none exists
                Ok(vec![0.01; 768])
            }
        }
    }
}

#[cfg(any(test, feature = "test-helpers"))]
pub(crate) mod tests {
    use super::mock::MockMamba;
    use super::*;

    // ═══════════════════════════════════════════════════════════════════════
    // ORIGINAL INTEGRATION TESTS (require network — kept for manual runs)
    // ═══════════════════════════════════════════════════════════════════════

    /// Integration test requiring network access to download mamba-130m.
    /// Ignored in CI — run manually with `cargo test --features mamba -- --ignored`.
    #[test]
    #[ignore = "Requires network access to download mamba-130m (~260MB)"]
    fn test_mamba_load_and_forward() {
        let device = best_device();
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
        let device = best_device();
        let mut wrapper = MambaWrapper::load("state-spaces/mamba-130m", device)
            .expect("Failed to load mamba-130m");

        let context = vec![0.1f32; 768];
        wrapper
            .inject_initial_context(&context)
            .expect("Context injection failed");

        let logits = wrapper
            .forward_one_token(1)
            .expect("Forward after injection failed");
        assert!(logits.len() >= 50257, "vocab too small: {}", logits.len());
    }

    #[test]
    #[ignore = "Requires network access to download mamba-130m (~260MB)"]
    fn test_mamba_state_scaling() {
        let device = best_device();
        let mut wrapper = MambaWrapper::load("state-spaces/mamba-130m", device)
            .expect("Failed to load mamba-130m");

        // Run a token to get non-zero state
        let _logits = wrapper.forward_one_token(0).expect("Forward pass failed");

        // Scale states
        wrapper
            .scale_hidden_states(0.5)
            .expect("State scaling failed");
        wrapper
            .scale_hidden_states(2.0)
            .expect("State scaling failed");

        // Should still produce valid logits
        let logits = wrapper
            .forward_one_token(1)
            .expect("Forward after scaling failed");
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    #[ignore = "Requires network access to download mamba-130m (~260MB)"]
    fn test_mamba_encode_decode() {
        let device = best_device();
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
        let device = best_device();
        let mut wrapper = MambaWrapper::load("state-spaces/mamba-130m", device)
            .expect("Failed to load mamba-130m");

        // Run some tokens
        let _ = wrapper.forward_one_token(0);
        let _ = wrapper.forward_one_token(1);

        // Reset
        wrapper.reset();

        // Should work fine after reset
        let logits = wrapper
            .forward_one_token(0)
            .expect("Forward after reset failed");
        assert!(logits.len() >= 50257, "vocab too small: {}", logits.len());
    }

    #[test]
    #[ignore = "Requires network access to download mamba-130m (~260MB)"]
    fn test_mamba_embedding_extraction() {
        let device = best_device();
        let wrapper = MambaWrapper::load("state-spaces/mamba-130m", device)
            .expect("Failed to load mamba-130m");

        let emb = wrapper
            .embedding_vector(42)
            .expect("Embedding extraction failed");
        assert_eq!(emb.len(), 768);
        // Should be non-zero for a real token
        assert!(emb.iter().any(|x| x.abs() > 1e-10));
    }

    #[test]
    #[ignore = "Requires network access to download mamba-130m (~260MB)"]
    fn test_mamba_generate_10_tokens() {
        let device = best_device();
        let mut wrapper = MambaWrapper::load("state-spaces/mamba-130m", device)
            .expect("Failed to load mamba-130m");

        // Encode a prompt
        let prompt_ids = wrapper
            .encode("The meaning of life is")
            .expect("Encode failed");

        // Feed prompt tokens
        let mut last_logits = vec![];
        for &token_id in &prompt_ids {
            last_logits = wrapper.forward_one_token(token_id).expect("Forward failed");
        }

        // Generate 10 tokens greedily
        let mut generated_ids = Vec::new();
        for _ in 0..10 {
            let next_id = last_logits
                .iter()
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
        let device = best_device();

        let mut wrapper = MambaWrapper::load("state-spaces/mamba-130m", device)
            .expect("Failed to load mamba-130m");

        // Create thought channels and encode to HDC
        let mut channels = ThoughtChannels::with_intent(1); // Answer
        channels.set_epistemic(0.0); // Certain
        channels.set_emotion(0.5, 0.5, 0.5);

        let encoder = crate::encoder::ThoughtLanguageEncoder::new(&genesis);
        let thought_hv = encoder.encode(&channels);

        // Project HDC → SSM space
        let projection = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let ssm_context = projection.project_to_ssm(&thought_hv);

        // Inject context into Mamba
        wrapper
            .inject_initial_context(&ssm_context)
            .expect("Context injection failed");

        // Generate a few tokens
        let mut token_ids = Vec::new();
        let mut logits = wrapper.forward_one_token(0).expect("Forward failed");
        for _ in 0..5 {
            let next_id = logits
                .iter()
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

    // ═══════════════════════════════════════════════════════════════════════
    // MOCK-BASED TESTS (no network required)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_mock_mamba_load_and_forward() {
        let mut mock = MockMamba::new();

        assert_eq!(mock.d_model(), 768);
        assert_eq!(mock.n_layer(), 24);
        assert_eq!(mock.vocab_size(), 50280);

        // Forward pass with a single token
        let logits = mock.forward_one_token(0).expect("Forward pass failed");
        assert_eq!(logits.len(), 50280);
        assert!(logits.iter().all(|x| x.is_finite()));
        assert_eq!(mock.forward_count, 1);
    }

    #[test]
    fn test_mock_mamba_context_injection() {
        let mut mock = MockMamba::new();

        let context = vec![0.1f32; 768];
        mock.inject_initial_context(&context)
            .expect("Context injection failed");

        assert_eq!(mock.injection_count, 1);
        // L2 norm of 768 elements of 0.1 = sqrt(768 * 0.01) = sqrt(7.68)
        let expected_mag = (768.0f32 * 0.01).sqrt();
        assert!(
            (mock.last_context_magnitude - expected_mag).abs() < 0.01,
            "Expected magnitude ~{expected_mag}, got {}",
            mock.last_context_magnitude
        );

        let logits = mock
            .forward_one_token(1)
            .expect("Forward after injection failed");
        assert_eq!(logits.len(), 50280);
    }

    #[test]
    fn test_mock_mamba_state_scaling() {
        let mut mock = MockMamba::new();

        // Run a token to get non-zero state
        let _logits = mock.forward_one_token(0).expect("Forward pass failed");

        // Scale states
        mock.scale_hidden_states(0.5).expect("State scaling failed");
        assert!((mock.scale_factor - 0.5).abs() < 1e-6);

        mock.scale_hidden_states(2.0).expect("State scaling failed");
        assert!((mock.scale_factor - 1.0).abs() < 1e-6);

        // Scale factor affects logit magnitude
        mock.scale_hidden_states(0.1).expect("Scale failed");
        let logits = mock
            .forward_one_token(1)
            .expect("Forward after scaling failed");
        assert!(logits.iter().all(|x| x.is_finite()));
        // Logits should be scaled down (max magnitude should be < 0.3 since range is [-2,2]*0.1)
        let max_abs = logits.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 0.3,
            "Scaled logits should be small, max_abs = {max_abs}"
        );
    }

    #[test]
    fn test_mock_mamba_encode_decode() {
        let mock = MockMamba::new();

        let text = "Hello, world!";
        let ids = mock.encode(text).expect("Encoding failed");
        assert_eq!(ids.len(), text.len()); // byte-per-token
        assert_eq!(ids[0], b'H' as u32);
        assert_eq!(ids[1], b'e' as u32);

        let decoded = mock.decode(&ids).expect("Decoding failed");
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_mock_mamba_reset() {
        let mut mock = MockMamba::new();

        // Run some tokens
        let _ = mock.forward_one_token(0);
        let _ = mock.forward_one_token(1);
        assert_eq!(mock.forward_count, 2);

        mock.scale_hidden_states(0.5).unwrap();
        assert!((mock.scale_factor - 0.5).abs() < 1e-6);

        // Reset
        mock.reset();

        assert_eq!(mock.forward_count, 0);
        assert!((mock.scale_factor - 1.0).abs() < 1e-6);
        assert_eq!(mock.reset_count, 1);

        // Should work fine after reset
        let logits = mock
            .forward_one_token(0)
            .expect("Forward after reset failed");
        assert_eq!(logits.len(), 50280);
    }

    #[test]
    fn test_mock_mamba_embedding_extraction() {
        let mock = MockMamba::new();

        let emb = mock
            .embedding_vector(42)
            .expect("Embedding extraction failed");
        assert_eq!(emb.len(), 768);
        // Should be non-zero
        assert!(emb.iter().any(|x| x.abs() > 1e-10));
        // Should be finite
        assert!(emb.iter().all(|x| x.is_finite()));

        // Deterministic: same token_id should produce same embedding
        let emb2 = mock.embedding_vector(42).expect("Embedding failed");
        assert_eq!(emb, emb2);

        // Different token_id should produce different embedding
        let emb3 = mock.embedding_vector(43).expect("Embedding failed");
        assert_ne!(emb, emb3);
    }

    #[test]
    fn test_mock_mamba_generate_10_tokens() {
        let mut mock = MockMamba::new();

        // Encode a prompt
        let prompt_ids = mock
            .encode("The meaning of life is")
            .expect("Encode failed");
        assert!(!prompt_ids.is_empty());

        // Feed prompt tokens
        let mut last_logits = vec![];
        for &token_id in &prompt_ids {
            last_logits = mock.forward_one_token(token_id).expect("Forward failed");
        }

        // Generate 10 tokens greedily
        let mut generated_ids = Vec::new();
        for _ in 0..10 {
            let next_id = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as u32)
                .unwrap();
            generated_ids.push(next_id);
            last_logits = mock.forward_one_token(next_id).expect("Forward failed");
        }

        assert_eq!(generated_ids.len(), 10);
        // All IDs should be within vocab range
        assert!(generated_ids.iter().all(|&id| (id as usize) < 50280));

        let decoded = mock.decode(&generated_ids).expect("Decode failed");
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_mock_mamba_e2e_thought_to_text() {
        use crate::encoder::ThoughtChannels;
        use crate::projection::HdcSsmProjection;
        use symthaea_core::genesis::GenesisSeed;

        let genesis = GenesisSeed::from_phrase("test-mock-mamba-e2e");
        let mut mock: Box<dyn MambaBackend> = Box::new(MockMamba::new());

        // Create thought channels and encode to HDC
        let mut channels = ThoughtChannels::with_intent(1); // Answer
        channels.set_epistemic(0.0); // Certain
        channels.set_emotion(0.5, 0.5, 0.5);

        let encoder = crate::encoder::ThoughtLanguageEncoder::new(&genesis);
        let thought_hv = encoder.encode(&channels);

        // Project HDC → SSM space
        let projection = HdcSsmProjection::new(&genesis, 16384, 256, 768);
        let ssm_context = projection.project_to_ssm(&thought_hv);

        // Inject context into Mamba (via trait)
        mock.inject_initial_context(&ssm_context)
            .expect("Context injection failed");

        // Generate a few tokens
        let mut token_ids = Vec::new();
        let mut logits = mock.forward_one_token(0).expect("Forward failed");
        for _ in 0..5 {
            let next_id = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i as u32)
                .unwrap();
            token_ids.push(next_id);
            logits = mock.forward_one_token(next_id).expect("Forward failed");
        }

        assert_eq!(token_ids.len(), 5);
        assert!(token_ids.iter().all(|&id| (id as usize) < 50280));
    }

    #[test]
    fn test_mock_mamba_determinism() {
        // Verify that MockMamba produces deterministic results across runs
        let mut mock1 = MockMamba::new();
        let mut mock2 = MockMamba::new();

        let logits1 = mock1.forward_one_token(42).expect("Forward failed");
        let logits2 = mock2.forward_one_token(42).expect("Forward failed");
        assert_eq!(logits1, logits2, "Same token_id should produce same logits");

        // Different token IDs produce different logits
        let logits3 = mock1.forward_one_token(43).expect("Forward failed");
        assert_ne!(
            logits1, logits3,
            "Different token_ids should produce different logits"
        );
    }

    #[test]
    fn test_mock_mamba_trait_object() {
        // Verify MockMamba works through Box<dyn MambaBackend>
        let mut backend: Box<dyn MambaBackend> = Box::new(MockMamba::new());

        assert_eq!(backend.d_model(), 768);
        assert_eq!(backend.n_layer(), 24);
        assert_eq!(backend.vocab_size(), 50280);
        assert_eq!(backend.eos_token_id(), 0);

        let logits = backend.forward_one_token(100).expect("Forward failed");
        assert_eq!(logits.len(), 50280);

        backend.reset();

        let emb = backend.embedding_vector(0).expect("Embedding failed");
        assert_eq!(emb.len(), 768);

        backend
            .inject_initial_context(&[1.0; 768])
            .expect("Inject failed");
        backend.scale_hidden_states(0.5).expect("Scale failed");

        let ids = backend.encode("test").expect("Encode failed");
        assert_eq!(ids.len(), 4);
        let text = backend.decode(&ids).expect("Decode failed");
        assert_eq!(text, "test");
    }

    #[test]
    fn test_mock_inject_context_sequence() {
        let mut mock = MockMamba::new();

        // 64 chunks of 768D (simulating temporal projection output)
        let sequence: Vec<Vec<f32>> = (0..64).map(|i| vec![0.1 * (i as f32 + 1.0); 768]).collect();

        mock.inject_context_sequence(&sequence)
            .expect("inject_context_sequence failed");

        assert_eq!(mock.injection_count, 1);
        assert!(mock.last_context_magnitude > 0.0, "Should record magnitude");
    }

    #[test]
    fn test_mock_context_sequence_then_generate() {
        let mut mock = MockMamba::new();

        // Inject 64 chunks as soft tokens
        let sequence: Vec<Vec<f32>> = (0..64).map(|_| vec![0.5; 768]).collect();
        mock.inject_context_sequence(&sequence)
            .expect("inject failed");

        // Should still be able to do normal forward passes after
        let logits = mock.forward_one_token(42).expect("Forward failed");
        assert_eq!(logits.len(), 50280);
        assert!(logits.iter().all(|x| x.is_finite()));
    }
}
