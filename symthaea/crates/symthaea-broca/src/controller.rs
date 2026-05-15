// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Language controller: HdcLtcUnifiedNetwork + weight-tied output projection.
//!
//! Direct analog of `VocalTractController` (`crates/symthaea-vocal-tract/src/controller.rs`).
//! Instead of projecting to 9D formant parameters, this projects to vocab-sized logits
//! via weight-tied similarity with token embeddings.
//!
//! # Forward step (one token)
//!
//! ```text
//! 1. input_hv = thought_hv ⊗ token_emb[prev_token] ⊗ permute(position_base, pos)
//! 2. network.evolve_closed_form(dt, &input_hv)
//! 3. output_hv = network.output().normalize()
//! 4. logits[i] = cosine_similarity(output_hv, token_emb[i])  // weight-tied
//! ```

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{
    ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig, HDC_DIMENSION,
};

/// GPU-resident pre-normalized embedding matrix for accelerated logit computation.
/// Converts 4,096 × 16,384 individual cosine similarities into a single matrix-vector
/// multiply on GPU (or optimized CPU tensor ops via candle).
#[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
pub struct GpuEmbeddingCache {
    /// Pre-normalized embedding matrix [vocab_size, HDC_DIMENSION] on device.
    embeddings: candle_core::Tensor,
    /// Pre-computed L2 norms for each embedding row (for cosine sim).
    /// Not needed if rows are pre-normalized, but kept for numerical stability.
    device: candle_core::Device,
}

#[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
impl std::fmt::Debug for GpuEmbeddingCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuEmbeddingCache")
            .field("shape", &self.embeddings.shape())
            .field("device", &self.device)
            .finish()
    }
}

#[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
impl Clone for GpuEmbeddingCache {
    fn clone(&self) -> Self {
        Self {
            embeddings: self.embeddings.clone(),
            device: self.device.clone(),
        }
    }
}

#[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
impl GpuEmbeddingCache {
    pub fn device(&self) -> &candle_core::Device {
        &self.device
    }

    pub fn embeddings(&self) -> &candle_core::Tensor {
        &self.embeddings
    }
}

/// Configuration for the language controller.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LanguageControllerConfig {
    /// Number of layers in the HdcLtcUnifiedNetwork.
    pub network_layers: usize,
    /// Neurons per layer.
    pub neurons_per_layer: usize,
    /// Learning rate for BPTT training.
    pub learning_rate: f32,
    /// Vocabulary size (number of token embeddings).
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Time step per token (seconds). Can be dynamically adjusted for
    /// thermodynamic subjective time.
    pub dt_per_token: f32,
    /// Temperature scaling for logits before softmax/sampling.
    pub logit_temperature: f32,
    /// Multiplicative scale for cosine-similarity logits (CLIP-style).
    /// Cosine similarities are in [-1, 1]; this scales them to [-scale, scale]
    /// so softmax can discriminate between tokens in high-dimensional space.
    /// Default 20.0 (like CLIP's learned temperature).
    #[serde(default = "default_logit_scale")]
    pub logit_scale: f32,
    /// CfC backbone time constant — controls state dependency for sequence coherence.
    #[serde(default = "default_backbone_tau")]
    pub backbone_tau: f32,
    /// Gradient attenuation factor for earlier layers during BPTT (stability).
    #[serde(default = "default_gradient_attenuation")]
    pub gradient_attenuation: f32,
    /// Vocab size threshold above which logit computation is parallelized.
    #[serde(default = "default_parallel_threshold")]
    pub parallel_threshold: usize,

    // ── Direct Thought-Logit Residual ──
    /// Blend direct thought-token logits into the recurrent decoder logits.
    ///
    /// This is a lightweight thought-to-structure bridge: the CfC recurrent
    /// output remains primary, but the encoded thought can directly bias token
    /// selection when the recurrent state is still undertrained.
    #[serde(default, skip)]
    pub thought_logit_residual_weight: f32,

    // ── Phase 1: Compositional Logit Refinement ──
    /// Enable compositional logit refinement via HDC bind.
    /// When enabled, `output_hv` is blended with `output_hv.bind(thought_hv)`
    /// to create a thought-conditioned output that amplifies on-topic tokens.
    #[serde(default)]
    pub enable_compositional_logits: bool,
    /// Blend weight for compositional correction (0.0 = no-op, 1.0 = full bind).
    /// Default 0.15: gentle correction that preserves autoregressive dynamics
    /// while amplifying dimensions where output aligns with thought intent.
    /// When `adaptive_compositional_alpha` is true, this becomes the *maximum*
    /// alpha, scaled by `(1 - coherence)` so correction is strongest when drifting.
    #[serde(default = "default_compositional_alpha")]
    pub compositional_alpha: f32,
    /// Enable coherence-conditioned alpha: `effective_alpha = alpha * (1 - coherence)`.
    /// When coherence is high (on-track), alpha → 0 (no correction needed).
    /// When coherence is low (drifting), alpha → max (full correction).
    #[serde(default)]
    pub adaptive_compositional_alpha: bool,

    // ── Phase 3: Adaptive dt from Coherence ──
    /// Enable adaptive dt modulation from coherence feedback.
    /// Low coherence → large dt → CfC "listens harder" to thought input.
    /// High coherence → small dt → preserves autoregressive momentum.
    #[serde(default)]
    pub enable_adaptive_dt: bool,
    /// Minimum dt (seconds) used when coherence is high (on-track). Default 0.005.
    #[serde(default = "default_dt_min")]
    pub dt_min: f32,
    /// Maximum dt (seconds) used when coherence is low (drifting). Default 0.08.
    #[serde(default = "default_dt_max")]
    pub dt_max: f32,

    // ── W3.1: Orthogonal Positional Encoding ──
    /// Replace cyclic permutation with Gram-Schmidt orthogonal position bases.
    #[serde(default)]
    pub enable_orthogonal_positions: bool,
    /// Number of orthogonal position vectors to generate. Default 512.
    #[serde(default = "default_orthogonal_position_count")]
    pub orthogonal_position_count: usize,

    // ── Thought-Seeded CfC State ──
    /// When true, `seed_from_thought()` projects the thought HV into each
    /// CfC neuron's initial state instead of starting from zeros.
    /// This makes early tokens thought-dependent (different intents → different starts)
    /// and gives coherence monitoring a non-zero baseline.
    #[serde(default = "default_true")]
    pub enable_thought_seeding: bool,
    /// Scaling factor for thought seed injection (0.0-1.0).
    /// Controls how strongly the thought HV influences the initial CfC state.
    /// Default 0.3 — gentle seeding that doesn't overwhelm learned dynamics.
    #[serde(default = "default_thought_seed_scale")]
    pub thought_seed_scale: f32,
}

fn default_logit_scale() -> f32 {
    250.0
}

fn default_backbone_tau() -> f32 {
    0.3
}

fn default_gradient_attenuation() -> f32 {
    0.5
}

fn default_parallel_threshold() -> usize {
    128
}

fn default_compositional_alpha() -> f32 {
    0.15
}

fn default_dt_min() -> f32 {
    0.005
}

fn default_dt_max() -> f32 {
    0.08
}

fn default_orthogonal_position_count() -> usize {
    512
}

fn default_true() -> bool {
    true
}

fn default_thought_seed_scale() -> f32 {
    0.3
}

impl Default for LanguageControllerConfig {
    fn default() -> Self {
        Self {
            network_layers: 3,
            neurons_per_layer: 8,
            learning_rate: 0.001,
            vocab_size: 256, // Minimal default, override for real use
            max_seq_len: 512,
            dt_per_token: 0.02, // 20ms per token step
            logit_temperature: 1.0,
            logit_scale: default_logit_scale(),
            backbone_tau: default_backbone_tau(),
            gradient_attenuation: default_gradient_attenuation(),
            parallel_threshold: default_parallel_threshold(),
            thought_logit_residual_weight: 0.0,
            // Benchmark-validated (Mar 20): -1.83 perplexity solo, -3.98 in combo.
            enable_compositional_logits: true,
            compositional_alpha: default_compositional_alpha(),
            // Benchmark-validated (Mar 21, release): +0.075 coherence, zero ppl cost.
            adaptive_compositional_alpha: true,
            enable_adaptive_dt: false,
            dt_min: default_dt_min(),
            dt_max: default_dt_max(),
            enable_orthogonal_positions: false,
            orthogonal_position_count: default_orthogonal_position_count(),
            enable_thought_seeding: true,
            thought_seed_scale: default_thought_seed_scale(),
        }
    }
}

/// Snapshot of CfC network state for soft veto temporal rewind (Phase 4).
///
/// Captures the output HV at a known-good coherence point. On veto,
/// the network can interpolate toward this snapshot instead of full reset.
#[derive(Clone)]
pub struct NetworkSnapshot {
    /// Saved per-neuron states for each layer.
    layer_states: Vec<Vec<ContinuousHV>>,
}

/// Language controller wrapping an HdcLtcUnifiedNetwork + weight-tied token output.
///
/// Forward pass:
/// 1. Compose input: thought_hv ⊗ token_emb ⊗ pos_emb
/// 2. `network.evolve_closed_form(dt, &input_hv)`
/// 3. `output_hv = network.output().normalize()`
/// 4. Weight-tied logits: `logits[i] = similarity(output_hv, token_emb[i])`
pub struct LanguageController {
    /// The temporal dynamics engine — full 16,384D HDC-LTC.
    network: HdcLtcUnifiedNetwork,
    /// Token embeddings (vocab_size × 16,384D). Weight-tied with output.
    /// Growable for swarm vocabulary extension.
    token_embeddings: Vec<ContinuousHV>,
    /// Position base vector — permute(base, pos) for positional encoding.
    position_base: ContinuousHV,
    /// Orthogonal position vectors (Gram-Schmidt). Used when enable_orthogonal_positions.
    orthogonal_positions: Option<Vec<ContinuousHV>>,
    /// Current learning rate.
    learning_rate: f32,
    /// Configuration.
    config: LanguageControllerConfig,
    /// Current sequence position (reset between generations).
    current_pos: usize,
    /// Coherence feedback from the generator (Phase 3: adaptive dt).
    coherence_for_dt: f32,
    /// Last effective dt used in forward_step_with_dt (for BPTT consistency).
    last_effective_dt: f32,
    /// Cached refined output HV from last forward_step (for coherence measurement).
    last_logit_hv: Option<ContinuousHV>,
    /// GPU-resident embedding matrix for accelerated logit computation.
    /// Shape: [vocab_size, HDC_DIMENSION]. Pre-normalized rows.
    #[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
    gpu_embeddings: Option<GpuEmbeddingCache>,
}

impl LanguageController {
    /// Create a new controller from a genesis seed and config.
    pub fn new(genesis: &GenesisSeed, config: &LanguageControllerConfig) -> Self {
        let neuron_config = UnifiedConfig {
            tau_base: 0.02,                    // 20ms — matches token generation rate
            backbone_tau: config.backbone_tau, // Configurable state dependency for coherent sequences
            dimension: HDC_DIMENSION,
            learning_rate: config.learning_rate,
            ..UnifiedConfig::default()
        };

        let net_config = UnifiedNetworkConfig {
            layer_sizes: vec![config.neurons_per_layer; config.network_layers],
            neuron_config,
            use_layer_binding: true,
            skip_connections: false,
        };

        let network = HdcLtcUnifiedNetwork::from_genesis(net_config, genesis);

        // Genesis-seeded token embeddings
        let token_embeddings: Vec<ContinuousHV> = (0..config.vocab_size)
            .map(|i| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("broca::token_emb::{i}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        let position_base =
            ContinuousHV::from_genesis(genesis, "broca::position_base", HDC_DIMENSION);

        let orthogonal_positions = if config.enable_orthogonal_positions {
            let count = config.orthogonal_position_count.max(1);
            use rand::RngCore;
            let seed: u64 = genesis.domain("broca::orthogonal_positions").next_u64();
            Some(ContinuousHV::orthogonal_set(HDC_DIMENSION, count, seed))
        } else {
            None
        };

        #[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
        let gpu_embeddings = Self::build_gpu_cache(&token_embeddings);

        Self {
            network,
            token_embeddings,
            position_base,
            orthogonal_positions,
            learning_rate: config.learning_rate,
            config: config.clone(),
            current_pos: 0,
            coherence_for_dt: 1.0,
            last_effective_dt: config.dt_per_token,
            last_logit_hv: None,
            #[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
            gpu_embeddings,
        }
    }

    /// Build GPU/candle embedding cache from token embeddings.
    /// Falls back to None if device initialization fails.
    #[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
    fn build_gpu_cache(token_embeddings: &[ContinuousHV]) -> Option<GpuEmbeddingCache> {
        use candle_core::{Device, Tensor};

        // Try CUDA first, fall back to CPU tensor ops
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        let vocab_size = token_embeddings.len();
        let dim = if vocab_size > 0 {
            token_embeddings[0].as_slice().len()
        } else {
            return None;
        };

        // Flatten all embeddings into a contiguous [vocab_size * dim] buffer
        let mut flat = Vec::with_capacity(vocab_size * dim);
        for emb in token_embeddings {
            let s = emb.as_slice();
            // Pre-normalize each row for cosine similarity
            let norm = s.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            flat.extend(s.iter().map(|x| x / norm));
        }

        match Tensor::from_vec(flat, (vocab_size, dim), &device) {
            Ok(embeddings) => {
                tracing::info!("GPU embedding cache built: [{vocab_size}, {dim}] on {device:?}");
                Some(GpuEmbeddingCache { embeddings, device })
            }
            Err(e) => {
                tracing::warn!("Failed to build GPU embedding cache: {e}");
                None
            }
        }
    }

    /// Compute logits for all tokens via weight-tied similarity.
    ///
    /// `logits[i] = logit_scale * cosine_similarity(output_hv, token_embeddings[i])`
    /// The scale factor (default 20.0, CLIP-style) spreads cosine similarities from
    /// [-1, 1] to a range where softmax can discriminate between tokens.
    ///
    /// When the `mamba-cpu` feature is enabled, uses candle tensor ops (GPU if CUDA
    /// available, otherwise optimized CPU BLAS) for the matrix-vector multiply.
    /// This replaces 4,096 individual 16,384D cosine similarities with a single
    /// batched operation.
    pub fn compute_logits(&self, output_hv: &ContinuousHV) -> Vec<f32> {
        let scale = self.config.logit_scale;

        // Try GPU/candle accelerated path
        #[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
        if let Some(ref cache) = self.gpu_embeddings {
            if let Ok(logits) = self.compute_logits_candle(output_hv, cache, scale) {
                return logits;
            }
            // Fall through to CPU path on error
        }

        #[cfg(feature = "parallel")]
        if self.token_embeddings.len() > self.config.parallel_threshold {
            return self
                .token_embeddings
                .par_iter()
                .map(|emb| scale * output_hv.similarity(emb))
                .collect();
        }
        self.token_embeddings
            .iter()
            .map(|emb| scale * output_hv.similarity(emb))
            .collect()
    }

    /// Candle-accelerated logit computation via batched matrix-vector multiply.
    /// Embeddings are pre-normalized, so: cosine_sim = dot(emb_normalized, query_normalized).
    /// Result: logits = scale * (E_norm @ q_norm), single BLAS gemv.
    #[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
    fn compute_logits_candle(
        &self,
        output_hv: &ContinuousHV,
        cache: &GpuEmbeddingCache,
        scale: f32,
    ) -> Result<Vec<f32>, candle_core::Error> {
        use candle_core::Tensor;

        let s = output_hv.as_slice();
        let norm = s.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);

        // Normalize query vector and move to device
        let q_norm: Vec<f32> = s.iter().map(|x| x / norm).collect();
        let q_tensor = Tensor::from_vec(q_norm, (1, s.len()), &cache.device)?;

        // matmul: [1, dim] × [dim, vocab] = [1, vocab]
        // Since embeddings are [vocab, dim], we do q @ E^T
        let similarities = q_tensor.matmul(&cache.embeddings.t()?)?;

        // Scale and extract
        let scaled = (similarities * scale as f64)?;
        let logits: Vec<f32> = scaled.squeeze(0)?.to_vec1()?;
        Ok(logits)
    }

    /// Forward step: evolve network with composed input, return logits.
    ///
    /// - `thought_hv`: encoded thought (16,384D)
    /// - `prev_token_id`: previous token ID (for autoregressive input)
    /// - `pos`: current sequence position
    pub fn forward_step(
        &mut self,
        thought_hv: &ContinuousHV,
        prev_token_id: u32,
        pos: usize,
    ) -> Vec<f32> {
        self.forward_step_with_dt(thought_hv, prev_token_id, pos, self.config.dt_per_token)
    }

    /// Forward step with explicit dt (for thermodynamic subjective time).
    pub fn forward_step_with_dt(
        &mut self,
        thought_hv: &ContinuousHV,
        prev_token_id: u32,
        pos: usize,
        dt: f32,
    ) -> Vec<f32> {
        // 1. Compose input: thought_hv ⊗ token_emb[prev_token] ⊗ permute(pos_base, pos)
        let token_emb = self.get_token_embedding(prev_token_id);
        let pos_emb = self.position_base.permute(pos);
        let input_hv = thought_hv.bind(&token_emb).bind(&pos_emb);

        // Guard: if dt is non-finite, use the configured default
        let mut dt = if dt.is_finite() && dt > 0.0 {
            dt
        } else {
            self.config.dt_per_token
        };

        // Phase 3: Adaptive dt from coherence feedback.
        // Low coherence → large dt → CfC sigma approaches 1 → state snaps to input-driven
        // equilibrium (the network "listens harder" to thought_hv).
        // High coherence → small dt → preserves autoregressive momentum.
        if self.config.enable_adaptive_dt {
            let c = self.coherence_for_dt.clamp(0.0, 1.0);
            dt = self.config.dt_max - c * (self.config.dt_max - self.config.dt_min);
            dt = dt.clamp(self.config.dt_min, self.config.dt_max);
        }

        // 2. Evolve network dynamics
        self.network.evolve_closed_form(dt, &input_hv);

        // 3. Get normalized output
        let output_hv = self.network.output().normalize();

        // Phase 1: Compositional logit refinement via HDC bind.
        // output_hv.bind(thought_hv) amplifies dimensions where output aligns with
        // thought intent (Hadamard product: both positive → large positive).
        // The weighted bundle blends raw output with this thought-conditioned signal.
        let logit_hv = if self.config.enable_compositional_logits {
            let base_alpha = self.config.compositional_alpha.clamp(0.0, 1.0);
            // Coherence-conditioned alpha: correct more when drifting, less when on-track.
            // effective_alpha = base * (1 - coherence). At coherence=1.0, alpha=0 (no correction).
            // At coherence=0.0, alpha=base (full correction).
            let alpha = if self.config.adaptive_compositional_alpha {
                let c = self.coherence_for_dt.clamp(0.0, 1.0);
                base_alpha * (1.0 - c)
            } else {
                base_alpha
            };
            if alpha > 1e-6 {
                let correction = output_hv.bind(thought_hv).normalize();
                ContinuousHV::weighted_bundle(&[&output_hv, &correction], &[1.0 - alpha, alpha])
            } else {
                output_hv
            }
        } else {
            output_hv
        };

        // 4. Weight-tied logits
        let mut logits = self.compute_logits(&logit_hv);
        let thought_residual = self.config.thought_logit_residual_weight.clamp(0.0, 1.0);
        if thought_residual > 1e-6 {
            let thought_query = thought_hv.bind(&pos_emb);
            let thought_logits = self.compute_logits(&thought_query);
            for (logit, thought_logit) in logits.iter_mut().zip(thought_logits.iter()) {
                *logit = (1.0 - thought_residual) * *logit + thought_residual * *thought_logit;
            }
        }

        // Apply temperature scaling
        if (self.config.logit_temperature - 1.0).abs() > 1e-6 {
            let inv_temp = 1.0 / self.config.logit_temperature;
            for l in &mut logits {
                *l *= inv_temp;
            }
        }

        // Ensure all logits are finite (replace NaN/Inf with 0.0)
        for l in &mut logits {
            if !l.is_finite() {
                *l = 0.0;
            }
        }

        self.current_pos = pos + 1;
        logits
    }

    /// Get the current network output HV (for coherence monitoring).
    pub fn output_hv(&self) -> ContinuousHV {
        self.network.output().normalize()
    }

    /// Get the controller configuration.
    pub fn config(&self) -> &LanguageControllerConfig {
        &self.config
    }

    /// Get GPU embedding cache for accelerated logit computation.
    /// Returns `None` when GPU features are disabled or cache not initialized.
    #[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
    pub fn gpu_embedding_cache(&self) -> Option<&GpuEmbeddingCache> {
        self.gpu_embeddings.as_ref()
    }

    /// Stub for when GPU features are disabled.
    #[cfg(not(any(feature = "mamba-cpu", feature = "gpu-logits")))]
    pub fn gpu_embedding_cache(&self) -> Option<&()> {
        None
    }

    /// Get mutable reference to the controller configuration.
    /// Used by training-time fusion to enable/disable flags per epoch.
    pub fn config_mut(&mut self) -> &mut LanguageControllerConfig {
        &mut self.config
    }

    /// Get token embedding, returning a zero vector for out-of-range IDs.
    fn get_token_embedding(&self, token_id: u32) -> ContinuousHV {
        self.token_embeddings
            .get(token_id as usize)
            .cloned()
            .unwrap_or_else(|| ContinuousHV::zero(HDC_DIMENSION))
    }

    /// Reset the network state (between generations).
    pub fn reset(&mut self) {
        self.network.reset();
        self.current_pos = 0;
        self.coherence_for_dt = 1.0;
        self.last_effective_dt = self.config.dt_per_token;
        self.last_logit_hv = None;
    }

    /// Seed CfC neuron states from the thought HV.
    ///
    /// Projects the thought HV into each neuron's state with per-layer binding
    /// to create distinct but thought-correlated initial states. This makes
    /// early generation tokens depend on the input thought (not just zeros),
    /// giving coherence monitoring a non-trivial baseline.
    pub fn seed_from_thought(&mut self, thought_hv: &ContinuousHV) {
        if !self.config.enable_thought_seeding {
            return;
        }
        let scale = self.config.thought_seed_scale;
        if scale <= 0.0 {
            return;
        }
        for layer_idx in 0..self.network.n_layers() {
            // Bind thought with layer index to create distinct seed per layer
            let layer_seed = thought_hv.permute(layer_idx + 1);
            let scaled = layer_seed.scale(scale);
            if let Some(layer) = self.network.layer_mut(layer_idx) {
                for (neuron_idx, neuron) in layer.iter_mut().enumerate() {
                    // Further differentiate per neuron via additional permutation
                    let neuron_seed = scaled.permute(neuron_idx + 1);
                    neuron.set_state(neuron_seed);
                }
            }
        }
    }

    /// Get the effective dt used in the last forward_step (for BPTT consistency).
    pub fn last_effective_dt(&self) -> f32 {
        self.last_effective_dt
    }

    /// Position base vector reference (used by GPU CfC trainer).
    pub fn position_base_ref(&self) -> &ContinuousHV {
        &self.position_base
    }

    /// Get the refined output HV from the last forward step.
    /// When compositional logits are enabled, returns the thought-conditioned output.
    pub fn refined_output_hv(&self) -> ContinuousHV {
        self.last_logit_hv
            .clone()
            .unwrap_or_else(|| self.network.output().normalize())
    }

    /// Set coherence feedback for adaptive dt (Phase 3).
    /// Called by the generator after each coherence update.
    pub fn set_coherence_feedback(&mut self, coherence: f32) {
        self.coherence_for_dt = if coherence.is_finite() {
            coherence
        } else {
            1.0
        };
    }

    /// Apply dropout to CfC hidden states (training-time regularization).
    /// Randomly zeros a fraction of hidden state dimensions, scaled by `1/(1-rate)`
    /// to maintain expected magnitude (inverted dropout).
    /// Uses deterministic masking based on `step` for reproducibility.
    pub fn apply_hidden_dropout(&mut self, rate: f32, step: usize) {
        if rate <= 0.0 || rate >= 1.0 {
            return;
        }
        let scale = 1.0 / (1.0 - rate);
        for layer_idx in 0..self.network.n_layers() {
            if let Some(layer) = self.network.layer_mut(layer_idx) {
                for (neuron_idx, neuron) in layer.iter_mut().enumerate() {
                    let state = neuron.state();
                    let src = state.as_slice();
                    let mut values: Vec<f32> = src.to_vec();
                    // Deterministic mask based on step + layer + neuron
                    let seed = step
                        .wrapping_mul(1000003)
                        .wrapping_add(layer_idx * 10007)
                        .wrapping_add(neuron_idx * 997);
                    for (j, v) in values.iter_mut().enumerate() {
                        // LCG-based deterministic coin flip
                        let hash = seed.wrapping_mul(j.wrapping_add(1)).wrapping_add(7) % 1000;
                        if (hash as f32) < rate * 1000.0 {
                            *v = 0.0;
                        } else {
                            *v *= scale;
                        }
                    }
                    neuron.set_state(ContinuousHV::from_values(values));
                }
            }
        }
    }

    /// Save a snapshot of the current CfC network state (Phase 4: soft veto).
    /// The snapshot captures per-neuron states for partial restoration.
    pub fn save_state(&self) -> NetworkSnapshot {
        let mut layer_states = Vec::with_capacity(self.network.n_layers());
        for l in 0..self.network.n_layers() {
            if let Some(layer) = self.network.layer(l) {
                layer_states.push(layer.iter().map(|n| n.state().clone()).collect());
            } else {
                layer_states.push(Vec::new());
            }
        }
        NetworkSnapshot { layer_states }
    }

    /// Partially restore CfC state from a snapshot (Phase 4: soft veto).
    /// `alpha = 1.0` fully restores to snapshot (equivalent to reset-to-snapshot).
    /// `alpha = 0.0` is a no-op (keeps current state).
    /// `alpha = 0.5` interpolates halfway between snapshot and current state.
    pub fn restore_partial(&mut self, snapshot: &NetworkSnapshot, alpha: f32) {
        let alpha = alpha.clamp(0.0, 1.0);
        if alpha < 1e-6 {
            return; // No-op
        }
        let one_minus_alpha = 1.0 - alpha;
        for (l, snap_layer) in snapshot.layer_states.iter().enumerate() {
            if let Some(layer) = self.network.layer_mut(l) {
                for (neuron, snap_state) in layer.iter_mut().zip(snap_layer.iter()) {
                    // Interpolate: new_state = alpha * snapshot + (1-alpha) * current
                    let current = neuron.state().clone();
                    let interpolated = ContinuousHV::weighted_bundle(
                        &[snap_state, &current],
                        &[alpha, one_minus_alpha],
                    );
                    *neuron.state_mut() = interpolated;
                }
            }
        }
    }

    /// Reset momentum on all CfC neurons. Call between BPTT training epochs
    /// to prevent accumulated directional bias from 67K+ gradient steps.
    pub fn reset_network_momentum(&mut self) {
        self.network.reset_momentum();
    }

    /// Append a new token embedding (for swarm vocabulary extension).
    /// The embedding is algebraically composed from context if `context_hvs` is provided.
    pub fn append_token(
        &mut self,
        genesis: &GenesisSeed,
        name: &str,
        context_hvs: Option<&[&ContinuousHV]>,
    ) {
        let emb = if let Some(ctx) = context_hvs {
            // Algebraically compose from surrounding semantics
            if ctx.is_empty() {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("broca::token_emb::{name}"),
                    HDC_DIMENSION,
                )
            } else {
                ContinuousHV::bundle(ctx)
            }
        } else {
            ContinuousHV::from_genesis(genesis, &format!("broca::token_emb::{name}"), HDC_DIMENSION)
        };
        self.token_embeddings.push(emb);
        // Invalidate GPU cache — will be rebuilt on next compute_logits if needed
        #[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
        {
            self.gpu_embeddings = Self::build_gpu_cache(&self.token_embeddings);
        }
    }

    /// Current vocabulary size (may grow via append_token).
    pub fn vocab_size(&self) -> usize {
        self.token_embeddings.len()
    }

    /// Forward step with sampled softmax (for training efficiency).
    ///
    /// Only computes cosine similarity for the specified `active_indices`,
    /// setting all other logits to `NEG_INFINITY`. This gives ~60× speedup
    /// for 4K vocabularies when using 64 negative samples.
    ///
    /// The CfC network evolution is identical to `forward_step`.
    pub fn forward_step_sampled(
        &mut self,
        thought_hv: &ContinuousHV,
        prev_token_id: u32,
        pos: usize,
        active_indices: &[usize],
    ) -> Vec<f32> {
        // 1. Compose input (consistent with forward_step_with_dt)
        let token_emb = self.get_token_embedding(prev_token_id);
        let pos_emb = if let Some(ref positions) = self.orthogonal_positions {
            positions[pos % positions.len()].clone()
        } else {
            self.position_base.permute(pos)
        };
        let input_hv = thought_hv.bind(&token_emb).bind(&pos_emb);

        let dt = if self.config.dt_per_token.is_finite() && self.config.dt_per_token > 0.0 {
            self.config.dt_per_token
        } else {
            0.02
        };

        // 2. Evolve network dynamics
        self.network.evolve_closed_form(dt, &input_hv);

        // 3. Get normalized output + compositional refinement (consistent with forward_step)
        let output_hv = self.network.output().normalize();
        let logit_hv = if self.config.enable_compositional_logits {
            let base_alpha = self.config.compositional_alpha.clamp(0.0, 1.0);
            let alpha = if self.config.adaptive_compositional_alpha {
                base_alpha * (1.0 - self.coherence_for_dt.clamp(0.0, 1.0))
            } else {
                base_alpha
            };
            if alpha > 1e-6 {
                let correction = output_hv.bind(thought_hv).normalize();
                ContinuousHV::weighted_bundle(&[&output_hv, &correction], &[1.0 - alpha, alpha])
            } else {
                output_hv
            }
        } else {
            output_hv
        };

        // 4. Sparse logits: only compute similarity for active indices
        let mut logits = vec![f32::NEG_INFINITY; self.token_embeddings.len()];
        let inv_temp = 1.0 / self.config.logit_temperature.max(1e-6);

        let scale = self.config.logit_scale;
        for &i in active_indices {
            if i < self.token_embeddings.len() {
                let mut s = scale * logit_hv.similarity(&self.token_embeddings[i]) * inv_temp;
                if !s.is_finite() {
                    s = 0.0;
                }
                logits[i] = s;
            }
        }

        self.current_pos = pos + 1;
        logits
    }

    /// Get current sequence position.
    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    /// Get learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Set learning rate.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    /// Set logit temperature.
    pub fn set_temperature(&mut self, temp: f32) {
        self.config.logit_temperature = temp;
    }

    /// Get dt per token.
    pub fn dt_per_token(&self) -> f32 {
        self.config.dt_per_token
    }

    /// Set dt per token (for thermodynamic subjective time).
    pub fn set_dt_per_token(&mut self, dt: f32) {
        self.config.dt_per_token = dt;
    }

    /// Get reference to the underlying network (for BPTT training).
    pub fn network(&self) -> &HdcLtcUnifiedNetwork {
        &self.network
    }

    /// Get mutable reference to the underlying network (for BPTT training).
    pub fn network_mut(&mut self) -> &mut HdcLtcUnifiedNetwork {
        &mut self.network
    }

    /// Get reference to token embeddings.
    pub fn token_embeddings(&self) -> &[ContinuousHV] {
        &self.token_embeddings
    }

    /// Get mutable reference to token embeddings (for training).
    pub fn token_embeddings_mut(&mut self) -> &mut Vec<ContinuousHV> {
        #[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
        {
            // The candle cache stores a pre-normalized snapshot of the embeddings.
            // Mutable access can change any row, so fail closed to the direct CPU
            // path until the caller explicitly rebuilds the cache.
            self.gpu_embeddings = None;
        }
        &mut self.token_embeddings
    }

    /// Rebuild the GPU embedding cache after training updates.
    /// Call after modifying embeddings (e.g., after gradient updates or normalization).
    #[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
    pub fn rebuild_gpu_cache(&mut self) {
        self.gpu_embeddings = Self::build_gpu_cache(&self.token_embeddings);
    }

    /// Normalize all token embeddings to a target L2 norm.
    ///
    /// Prevents embedding norm explosion during training (observed norms >3000
    /// without regularization). Called at the end of each training epoch.
    pub fn normalize_embeddings(&mut self, target_norm: f32) {
        for emb in &mut self.token_embeddings {
            let norm = emb.norm();
            if norm > target_norm * 1.1 {
                // Only normalize if significantly above target (avoid jitter)
                let scale = target_norm / norm;
                for v in emb.values.iter_mut() {
                    *v *= scale;
                }
            }
        }
        // Rebuild GPU cache with updated norms
        #[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
        self.rebuild_gpu_cache();
    }

    /// Backpropagate a loss gradient through the CfC network.
    ///
    /// Given the gradient of the loss w.r.t. the **normalized** network output HV (`d_output`),
    /// this projects the gradient onto the tangent plane of the unit sphere (to account for
    /// the normalize() in the forward pass), constructs a synthetic target for each neuron's
    /// backward() method, and applies gradients to CfC weights.
    pub fn backward_step(
        &mut self,
        d_output: &ContinuousHV,
        thought_hv: &ContinuousHV,
        prev_token_id: u32,
        pos: usize,
        dt: f32,
        network_lr: f32,
    ) {
        // Reconstruct the composed input from the forward step
        let token_emb = self.get_token_embedding(prev_token_id);
        let pos_emb = self.position_base.permute(pos);
        let input_hv = thought_hv.bind(&token_emb).bind(&pos_emb);

        let dt = if dt.is_finite() && dt > 0.0 {
            dt
        } else {
            self.config.dt_per_token
        };

        // --- Backprop through normalize() ---
        // Forward: output_hat = output_raw / ||output_raw||
        // Jacobian: d_hat/d_raw = (I - hat * hat^T) / ||raw||
        // So: d_raw = (d_output - dot(d_output, hat) * hat) / ||raw||
        let raw_output = self.network.output();
        let raw_norm = raw_output.norm().max(1e-8);
        let output_hat = raw_output.normalize();

        // Project d_output onto tangent plane of unit sphere
        let dot_d_hat: f32 = d_output
            .as_slice()
            .iter()
            .zip(output_hat.as_slice().iter())
            .map(|(d, h)| d * h)
            .sum();
        let d_raw_values: Vec<f32> = d_output
            .as_slice()
            .iter()
            .zip(output_hat.as_slice().iter())
            .map(|(d, h)| (d - dot_d_hat * h) / raw_norm)
            .collect();
        let d_raw = ContinuousHV::from_slice(&d_raw_values);

        // Backprop through the last layer: construct target = state - d_raw
        let last_layer_idx = self.network.n_layers() - 1;

        // Each neuron gets the same gradient signal (since output = avg(states))
        let inv_n = 1.0
            / self
                .network
                .layer(last_layer_idx)
                .map(|l| l.len())
                .unwrap_or(1) as f32;

        // Scale d_raw by 1/n (since output = mean of neuron states)
        let d_per_neuron = d_raw.scale(inv_n);

        // Get the input that was fed to each layer
        let layer_input = self.network.layer_input(last_layer_idx, &input_hv);

        // 1. Apply gradients to last layer and gather d_input for backprop
        let mut d_prev = ContinuousHV::zero(HDC_DIMENSION);
        if let Some(layer) = self.network.layer_mut(last_layer_idx) {
            for neuron in layer.iter_mut() {
                // Target = state - d_output (gradient descent direction)
                let target = neuron.state().subtract(&d_per_neuron);
                let grads = neuron.backward(&layer_input, &target, dt);

                // Accumulate d_input for next layer backward
                for (dp, &di) in d_prev.values.iter_mut().zip(grads.d_input.values.iter()) {
                    *dp += di;
                }

                // Use no_decay variant: per-step weight decay of 0.0001 applied
                // 67K times/epoch decays weights to ~0.1%, destroying the network.
                neuron.apply_gradients_no_decay(&grads, network_lr);
            }

            // Average the accumulated d_input
            if !layer.is_empty() {
                let inv = 1.0 / layer.len() as f32;
                for v in d_prev.values.iter_mut() {
                    *v *= inv;
                }
            }
        }

        // 2. Backprop through all earlier layers (intermediate + input layer 0)
        let mut d_prev_accum = d_prev;
        for layer_idx in (0..last_layer_idx).rev() {
            // Apply attenuation for this hop backward
            d_prev_accum = d_prev_accum.scale(self.config.gradient_attenuation);

            let layer_input = self.network.layer_input(layer_idx, &input_hv);
            let mut d_next_accum = ContinuousHV::zero(HDC_DIMENSION);

            if let Some(layer) = self.network.layer_mut(layer_idx) {
                let inv_n = 1.0 / layer.len().max(1) as f32;
                let d_per_neuron = d_prev_accum.scale(inv_n);

                for neuron in layer.iter_mut() {
                    let target = neuron.state().subtract(&d_per_neuron);
                    let grads = neuron.backward(&layer_input, &target, dt);

                    // Accumulate d_input for next layer backward
                    for (dp, &di) in d_next_accum
                        .values
                        .iter_mut()
                        .zip(grads.d_input.values.iter())
                    {
                        *dp += di;
                    }

                    // Apply gradients to this layer
                    neuron.apply_gradients_no_decay(
                        &grads,
                        network_lr * self.config.gradient_attenuation.powi((last_layer_idx - layer_idx) as i32),
                    );
                }

                // Average the accumulated d_input
                if !layer.is_empty() {
                    let inv = 1.0 / layer.len() as f32;
                    for v in d_next_accum.values.iter_mut() {
                        *v *= inv;
                    }
                }
            }
            d_prev_accum = d_next_accum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-broca-controller")
    }

    fn test_config() -> LanguageControllerConfig {
        LanguageControllerConfig {
            network_layers: 2,
            neurons_per_layer: 4,
            vocab_size: 32,
            max_seq_len: 16,
            ..Default::default()
        }
    }

    #[test]
    fn test_controller_creation() {
        let genesis = test_genesis();
        let config = test_config();
        let ctrl = LanguageController::new(&genesis, &config);
        assert_eq!(ctrl.vocab_size(), 32);
        assert_eq!(ctrl.current_pos(), 0);
    }

    #[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
    #[test]
    fn test_token_embedding_mutation_invalidates_candle_cache() {
        let genesis = test_genesis();
        let config = test_config();
        let mut ctrl = LanguageController::new(&genesis, &config);

        ctrl.rebuild_gpu_cache();
        assert!(
            ctrl.gpu_embedding_cache().is_some(),
            "cache should be available after explicit rebuild on CPU or CUDA candle device"
        );

        ctrl.token_embeddings_mut()[0].values[0] += 0.25;
        assert!(
            ctrl.gpu_embedding_cache().is_none(),
            "mutable embedding access must invalidate stale candle logits cache"
        );

        ctrl.rebuild_gpu_cache();
        assert!(
            ctrl.gpu_embedding_cache().is_some(),
            "explicit rebuild should restore cache after mutation"
        );
    }

    #[test]
    fn test_forward_step_produces_logits() {
        let genesis = test_genesis();
        let config = test_config();
        let mut ctrl = LanguageController::new(&genesis, &config);

        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);
        let logits = ctrl.forward_step(&thought_hv, 0, 0);

        assert_eq!(logits.len(), 32);
        // Logits should be finite
        assert!(
            logits.iter().all(|l| l.is_finite()),
            "all logits should be finite"
        );
        // At least some logits should be non-zero (network produces non-trivial output)
        assert!(
            logits.iter().any(|l| l.abs() > 1e-10),
            "should have non-zero logits"
        );
    }

    #[test]
    fn test_forward_determinism() {
        let genesis = test_genesis();
        let config = test_config();

        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);

        let mut ctrl1 = LanguageController::new(&genesis, &config);
        let logits1 = ctrl1.forward_step(&thought_hv, 0, 0);

        let mut ctrl2 = LanguageController::new(&genesis, &config);
        let logits2 = ctrl2.forward_step(&thought_hv, 0, 0);

        for (a, b) in logits1.iter().zip(logits2.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Same genesis should produce identical logits: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_different_tokens_different_logits() {
        let genesis = test_genesis();
        let config = test_config();
        let mut ctrl = LanguageController::new(&genesis, &config);

        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);

        let logits_a = ctrl.forward_step(&thought_hv, 5, 0);
        ctrl.reset();
        let logits_b = ctrl.forward_step(&thought_hv, 10, 0);

        let diff: f32 = logits_a
            .iter()
            .zip(logits_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "Different prev tokens should yield different logits: diff={diff}"
        );
    }

    #[test]
    fn test_different_positions_different_logits() {
        let genesis = test_genesis();
        let config = test_config();
        let mut ctrl = LanguageController::new(&genesis, &config);

        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);

        let logits_a = ctrl.forward_step(&thought_hv, 5, 0);
        ctrl.reset();
        let logits_b = ctrl.forward_step(&thought_hv, 5, 3);

        let diff: f32 = logits_a
            .iter()
            .zip(logits_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "Different positions should yield different logits: diff={diff}"
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let genesis = test_genesis();
        let config = test_config();
        let mut ctrl = LanguageController::new(&genesis, &config);

        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);
        let _ = ctrl.forward_step(&thought_hv, 5, 0);
        assert_eq!(ctrl.current_pos(), 1);

        ctrl.reset();
        assert_eq!(ctrl.current_pos(), 0);
    }

    #[test]
    fn test_append_token() {
        let genesis = test_genesis();
        let config = test_config();
        let mut ctrl = LanguageController::new(&genesis, &config);
        assert_eq!(ctrl.vocab_size(), 32);

        ctrl.append_token(&genesis, "new_concept", None);
        assert_eq!(ctrl.vocab_size(), 33);
    }

    #[test]
    fn test_logits_always_finite() {
        let genesis = test_genesis();
        let config = test_config();
        let mut ctrl = LanguageController::new(&genesis, &config);

        // Even with out-of-range token ID, logits should be finite
        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);
        let logits = ctrl.forward_step(&thought_hv, 99999, 0); // way beyond vocab
        assert!(
            logits.iter().all(|l| l.is_finite()),
            "Out-of-range token ID should still produce finite logits"
        );
    }

    #[test]
    fn test_nan_dt_falls_back_to_default() {
        let genesis = test_genesis();
        let config = test_config();

        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);

        // Normal dt
        let mut ctrl1 = LanguageController::new(&genesis, &config);
        let logits_normal = ctrl1.forward_step(&thought_hv, 0, 0);

        // NaN dt should fall back to default
        let mut ctrl2 = LanguageController::new(&genesis, &config);
        let logits_nan = ctrl2.forward_step_with_dt(&thought_hv, 0, 0, f32::NAN);

        // Should produce same result as default dt
        for (a, b) in logits_normal.iter().zip(logits_nan.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "NaN dt should fall back to default: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_temperature_scaling() {
        let genesis = test_genesis();
        let config = test_config();

        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);

        let mut ctrl1 = LanguageController::new(&genesis, &config);
        ctrl1.set_temperature(1.0);
        let logits_t1 = ctrl1.forward_step(&thought_hv, 0, 0);

        let mut ctrl2 = LanguageController::new(&genesis, &config);
        ctrl2.set_temperature(0.5);
        let logits_t05 = ctrl2.forward_step(&thought_hv, 0, 0);

        // With temp=0.5, logits should be 2x the temp=1.0 values
        for (a, b) in logits_t1.iter().zip(logits_t05.iter()) {
            if a.abs() > 1e-6 {
                assert!(
                    (b / a - 2.0).abs() < 0.01,
                    "temp=0.5 should double logits: {a} -> {b}"
                );
            }
        }
    }

    #[test]
    fn test_forward_step_sampled() {
        let genesis = test_genesis();
        let config = test_config();

        let thought_hv = ContinuousHV::from_genesis(&genesis, "test_thought", HDC_DIMENSION);

        // Full forward step
        let mut ctrl1 = LanguageController::new(&genesis, &config);
        let full_logits = ctrl1.forward_step(&thought_hv, 0, 0);

        // Sampled forward step with all indices = should match full
        let all_indices: Vec<usize> = (0..config.vocab_size).collect();
        let mut ctrl2 = LanguageController::new(&genesis, &config);
        let sampled_logits = ctrl2.forward_step_sampled(&thought_hv, 0, 0, &all_indices);

        assert_eq!(full_logits.len(), sampled_logits.len());
        for (a, b) in full_logits.iter().zip(sampled_logits.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Sampled (all indices) should match full: {a} vs {b}"
            );
        }

        // Sampled with subset: non-sampled indices should be -inf
        let subset = vec![0, 5, 10];
        let mut ctrl3 = LanguageController::new(&genesis, &config);
        let sparse_logits = ctrl3.forward_step_sampled(&thought_hv, 0, 0, &subset);
        assert!(sparse_logits[0].is_finite(), "Index 0 should be computed");
        assert!(sparse_logits[5].is_finite(), "Index 5 should be computed");
        assert_eq!(
            sparse_logits[1],
            f32::NEG_INFINITY,
            "Index 1 should be -inf"
        );
        assert_eq!(
            sparse_logits[3],
            f32::NEG_INFINITY,
            "Index 3 should be -inf"
        );
    }
}
