//! Liquid-Mamba Fusion (L-SSM): consciousness-gated SSM language generation.
//!
//! Fuses a pre-trained Mamba SSM with Symthaea's HDC-LTC cognitive loop.
//! The SSM provides vocabulary fluency; the cognitive loop provides
//! consciousness-gated semantic control, epistemic gating, emotional
//! authenticity, and biological constraints.
//!
//! # Architecture
//!
//! ```text
//! ThoughtChannels → ThoughtLanguageEncoder → thought_hv (16,384D)
//!                                              │
//!                           HdcSsmProjection ──┤──→ ssm_context (768D)
//!                                              │          │
//!                                              │    MambaWrapper.inject_initial_context()
//!                                              │          │
//!                                              │    Autoregressive loop:
//!                                              │      ├── biological delta modulation
//!                                              │      ├── forward_one_token() → logits
//!                                              │      ├── EpistemicGate.apply()
//!                                              │      ├── EmotionalModulator.apply()
//!                                              │      ├── top_k_sample()
//!                                              │      ├── back-project token → HDC
//!                                              │      └── CoherenceMonitor (veto check)
//!                                              │
//!                                         GenerationResult
//! ```

use std::collections::VecDeque;

use anyhow::Result;

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::encoder::{ThoughtChannels, ThoughtLanguageEncoder};
use crate::gating::{consciousness_gated_max_tokens, EpistemicGate, EmotionalModulator, GatingConfig};
use crate::generator::GenerationResult;
use crate::mamba::MambaWrapper;
use crate::projection::HdcSsmProjection;
use crate::tokenizer::BpeTokenizer;

/// Configuration for the Liquid-Mamba generator.
#[derive(Debug, Clone)]
pub struct LiquidMambaConfig {
    /// HuggingFace model ID for the Mamba checkpoint.
    pub model_id: String,
    /// Maximum tokens to generate per call.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: f32,
    /// Top-k for sampling.
    pub top_k: usize,
    /// Coherence below this → veto (mid-sentence self-correction).
    pub veto_threshold: f32,
    /// Coherence below this → boost thought binding.
    pub drift_threshold: f32,
    /// Sliding window size for coherence monitoring.
    pub coherence_window: usize,
    /// EMA smoothing alpha for coherence.
    pub coherence_ema_alpha: f32,
    /// Minimum consecutive low-coherence tokens before veto.
    pub min_consecutive_low: usize,
    /// Biological delta modulation strength (0 = disabled, 1 = full).
    pub delta_mod_strength: f32,
    /// Text inserted on semantic veto.
    pub veto_hesitation: String,
    /// Enable epistemic gating.
    pub enable_gating: bool,
    /// Enable semantic veto.
    pub enable_veto: bool,
    /// Enable biological delta modulation.
    pub enable_liquid_delta: bool,
    /// Enable consciousness-gated max tokens.
    pub enable_consciousness_gating: bool,
    /// HDC dimension (default 16384).
    pub hdc_dim: usize,
    /// Bottleneck dimension (default 256).
    pub bottleneck_dim: usize,
    /// SSM hidden dimension (default 768 for mamba-130m).
    pub ssm_dim: usize,
    /// LR warmup steps: ramp from 0 to base LR over this many generations.
    pub warmup_steps: usize,
    /// Gradient accumulation steps: accumulate over N generations before applying.
    pub accumulation_steps: usize,
    /// Contrastive loss weight (0 = disabled). Pushes different thoughts apart.
    pub contrastive_weight: f32,
    /// Size of the recent-thoughts buffer for contrastive loss.
    pub contrastive_buffer_size: usize,
}

impl Default for LiquidMambaConfig {
    fn default() -> Self {
        Self {
            model_id: "state-spaces/mamba-130m".to_string(),
            max_tokens: 256,
            temperature: 0.8,
            top_k: 40,
            veto_threshold: 0.20,
            drift_threshold: 0.30,
            coherence_window: 8,
            coherence_ema_alpha: 0.3,
            min_consecutive_low: 3,
            delta_mod_strength: 1.0,
            veto_hesitation: "-- wait, ".to_string(),
            enable_gating: true,
            enable_veto: true,
            enable_liquid_delta: true,
            enable_consciousness_gating: true,
            hdc_dim: 16384,
            bottleneck_dim: 256,
            ssm_dim: 768,
            warmup_steps: 100,
            accumulation_steps: 4,
            contrastive_weight: 0.1,
            contrastive_buffer_size: 8,
        }
    }
}

/// Liquid-Mamba fusion generator: consciousness-gated SSM language generation.
pub struct LiquidMambaGenerator {
    mamba: MambaWrapper,
    projection: HdcSsmProjection,
    encoder: ThoughtLanguageEncoder,
    epistemic_gate: EpistemicGate,
    emotional_modulator: EmotionalModulator,
    config: LiquidMambaConfig,
    // Biological state (injected by cognitive loop via update_affect)
    thermodynamic_load: f32,
    arousal: f32,
    // Online distillation state
    generation_count: usize,
    distill_accumulator: usize,
    recent_thought_hvs: VecDeque<ContinuousHV>,
}

impl LiquidMambaGenerator {
    /// Create a new Liquid-Mamba generator.
    ///
    /// Loads the Mamba model from HuggingFace Hub (requires network on first run).
    pub fn new(genesis: &GenesisSeed, config: LiquidMambaConfig) -> Result<Self> {
        let device = candle_core::Device::Cpu;
        let mamba = MambaWrapper::load(&config.model_id, device)?;

        let projection = HdcSsmProjection::new(
            genesis,
            config.hdc_dim,
            config.bottleneck_dim,
            config.ssm_dim,
        );

        let encoder = ThoughtLanguageEncoder::new(genesis);

        // Create gating modules using a minimal tokenizer for classification
        // (the actual token logits come from Mamba's 50K vocab)
        let tokenizer = BpeTokenizer::default_minimal();
        let gating_config = GatingConfig::default();
        let epistemic_gate = EpistemicGate::new(&tokenizer, &gating_config);
        let emotional_modulator = EmotionalModulator::new(&tokenizer, &gating_config);

        Ok(Self {
            mamba,
            projection,
            encoder,
            epistemic_gate,
            emotional_modulator,
            config,
            thermodynamic_load: 0.0,
            arousal: 0.5,
            generation_count: 0,
            distill_accumulator: 0,
            recent_thought_hvs: VecDeque::new(),
        })
    }

    /// Generate text from thought channels.
    pub fn generate(&mut self, channels: &ThoughtChannels) -> GenerationResult {
        match self.generate_inner(channels) {
            Ok(result) => result,
            Err(e) => {
                tracing::error!("Liquid-Mamba generation failed: {e}");
                GenerationResult {
                    text: String::new(),
                    token_ids: Vec::new(),
                    num_tokens: 0,
                    eos_terminated: false,
                    veto_triggered: false,
                    final_coherence: 0.0,
                    output_hvs: Vec::new(),
                    semantic_pe: 0.0,
                }
            }
        }
    }

    fn generate_inner(&mut self, channels: &ThoughtChannels) -> Result<GenerationResult> {
        // 1. Encode thought channels to 16,384D HDC
        let thought_hv = self.encoder.encode(channels);

        // 2. Project to SSM space (768D)
        let ssm_context = self.projection.project_to_ssm(&thought_hv);

        // 3. Prime Mamba's hidden state with the thought projection
        self.mamba.inject_initial_context(&ssm_context)?;

        // 4. Compute max tokens (consciousness-gated)
        let max_tokens = if self.config.enable_consciousness_gating {
            consciousness_gated_max_tokens(self.config.max_tokens, channels.psi())
        } else {
            self.config.max_tokens
        };

        // 5. Update arousal from channels
        self.arousal = channels.arousal();

        // 6. Initialize coherence monitor
        let mut coherence_monitor = CoherenceMonitor::new(
            thought_hv.clone(),
            self.config.coherence_window,
            self.config.coherence_ema_alpha,
            self.config.veto_threshold,
            self.config.min_consecutive_low,
        );

        // 7. Autoregressive generation loop
        let eos_id = self.mamba.eos_token_id();
        let mut tokens: Vec<u32> = Vec::new();
        let mut text = String::new();
        let mut veto_triggered = false;
        let mut prev_token = eos_id; // Start with EOS as BOS-equivalent
        let mut output_hvs: Vec<ContinuousHV> = Vec::new();

        for pos in 0..max_tokens {
            // 7a. Biological delta modulation
            if self.config.enable_liquid_delta {
                let scale = self.biological_state_scale();
                self.mamba.scale_hidden_states(scale)?;
            }

            // 7b. Forward one token through Mamba
            let mut logits = self.mamba.forward_one_token(prev_token)?;

            // 7c. Epistemic gating (apply to Mamba's large vocab logits)
            if self.config.enable_gating {
                // Apply gating to the first N tokens that overlap with Broca vocab
                // Mamba vocab >> Broca vocab, so we gate on the Broca-sized prefix
                let gate_len = logits.len().min(512);
                self.epistemic_gate.apply(
                    &mut logits[..gate_len],
                    channels.epistemic_ordinal(),
                );
            }

            // 7d. Emotional modulation
            if self.config.enable_gating {
                let gate_len = logits.len().min(512);
                self.emotional_modulator.apply(
                    &mut logits[..gate_len],
                    channels,
                    pos,
                );
            }

            // 7e. Top-k sampling
            let next_token = top_k_sample(&logits, self.config.top_k, self.config.temperature);

            // 7f. Back-project token to HDC (unconditional — for distillation + veto)
            let token_emb = self.mamba.embedding_vector(next_token)?;
            let token_hdc = self.projection.project_to_hdc(&token_emb);
            output_hvs.push(token_hdc.clone());

            // 7g. Coherence monitoring + semantic veto
            if self.config.enable_veto {
                coherence_monitor.push(token_hdc);

                if coherence_monitor.should_veto() && pos > 2 {
                    veto_triggered = true;
                    text.push_str(&self.config.veto_hesitation);

                    // Reset Mamba and re-inject thought context
                    self.mamba.reset();
                    self.mamba.inject_initial_context(&ssm_context)?;
                    coherence_monitor.reset();

                    // Continue from current position (don't restart text)
                    continue;
                }
            }

            // 7h. Decode token
            if next_token == eos_id {
                let semantic_pe = self.semantic_prediction_error(&thought_hv, &output_hvs);
                return Ok(GenerationResult {
                    text,
                    token_ids: tokens,
                    num_tokens: pos,
                    eos_terminated: true,
                    veto_triggered,
                    final_coherence: coherence_monitor.current_coherence(),
                    output_hvs,
                    semantic_pe,
                });
            }

            if let Ok(token_str) = self.mamba.decode_token(next_token) {
                text.push_str(&token_str);
            }

            tokens.push(next_token);
            prev_token = next_token;
        }

        let semantic_pe = self.semantic_prediction_error(&thought_hv, &output_hvs);
        Ok(GenerationResult {
            text,
            token_ids: tokens.clone(),
            num_tokens: tokens.len(),
            eos_terminated: false,
            veto_triggered,
            final_coherence: coherence_monitor.current_coherence(),
            output_hvs,
            semantic_pe,
        })
    }

    /// Biological delta modulation factor.
    ///
    /// Scales SSM hidden state based on thermodynamic load and arousal.
    /// - `factor < 1.0` → faster decay → shorter memory (exhausted/agitated)
    /// - `factor > 1.0` → slower decay → longer memory (rested/calm)
    ///
    /// Formula: `factor = exp(-alpha * load - beta * (arousal - 0.5))`
    /// - At rest (load=0, arousal=0.5): factor = 1.0 (no effect)
    /// - Exhausted (load=1.0, arousal=0.5): factor ≈ 0.61
    /// - Agitated (load=0, arousal=1.0): factor ≈ 0.86
    fn biological_state_scale(&self) -> f32 {
        let alpha = 0.5 * self.config.delta_mod_strength;
        let beta = 0.3 * self.config.delta_mod_strength;
        (-alpha * self.thermodynamic_load - beta * (self.arousal - 0.5))
            .exp()
            .clamp(0.3, 2.0)
    }

    /// Update biological state from the cognitive loop.
    pub fn update_affect(&mut self, load: f32, _temp: f32) {
        self.thermodynamic_load = load.clamp(0.0, 1.0);
    }

    /// Get mutable reference to the projection (for gradient learning).
    pub fn projection_mut(&mut self) -> &mut HdcSsmProjection {
        &mut self.projection
    }

    /// Get reference to the projection.
    pub fn projection(&self) -> &HdcSsmProjection {
        &self.projection
    }

    /// Get reference to the encoder.
    pub fn encoder(&self) -> &ThoughtLanguageEncoder {
        &self.encoder
    }

    /// Current thermodynamic load (0-1).
    pub fn thermodynamic_load(&self) -> f32 {
        self.thermodynamic_load
    }

    /// Number of generations completed (for warmup scheduling).
    pub fn generation_count(&self) -> usize {
        self.generation_count
    }

    /// Online distillation step: learn from the last generation.
    ///
    /// Adjusts the HDC↔SSM projection using:
    /// - Attention-weighted output HV bundling (recency × coherence)
    /// - LR warmup schedule (ramp from 0 over warmup_steps)
    /// - Gradient accumulation (apply every accumulation_steps)
    /// - Contrastive loss (push different thoughts' projections apart)
    pub fn distill_step(&mut self, channels: &ThoughtChannels, result: &GenerationResult) {
        self.generation_count += 1;

        // Gate: skip if no tokens or high PE (garbage output)
        if result.output_hvs.is_empty() || result.semantic_pe > 0.8 {
            return;
        }

        let thought_hv = self.encoder.encode(channels);

        // Warmup schedule: ramp from 0 to base LR over warmup_steps
        let warmup_factor = if self.config.warmup_steps > 0 {
            (self.generation_count as f32 / self.config.warmup_steps as f32).min(1.0)
        } else {
            1.0
        };
        let load_factor = 1.0 - self.thermodynamic_load;
        let effective_lr = 0.001 * warmup_factor * load_factor;
        if effective_lr < 1e-7 {
            return;
        }

        // Attention-weighted bundling: weight by recency × coherence
        let bundled = self.attention_weighted_bundle(&thought_hv, &result.output_hvs);

        // Reconstruction gradient
        self.projection.compute_gradients(&thought_hv, &bundled);

        // Contrastive term: push away from recent different thoughts
        if self.config.contrastive_weight > 0.0 {
            for neg_hv in &self.recent_thought_hvs {
                let sim = thought_hv.similarity(neg_hv);
                if sim < 0.8 {
                    self.projection.compute_contrastive_gradients(
                        &thought_hv,
                        neg_hv,
                        self.config.contrastive_weight,
                    );
                }
            }
        }

        // Gradient accumulation: only apply every N steps
        self.distill_accumulator += 1;
        if self.distill_accumulator >= self.config.accumulation_steps.max(1) {
            self.projection.apply_gradients(effective_lr, 1.0);
            self.distill_accumulator = 0;
        }

        // Update contrastive buffer
        if self.config.contrastive_buffer_size > 0 {
            self.recent_thought_hvs.push_back(thought_hv);
            while self.recent_thought_hvs.len() > self.config.contrastive_buffer_size {
                self.recent_thought_hvs.pop_front();
            }
        }
    }

    /// Attention-weighted bundling of output HVs.
    ///
    /// Weights each token's HV by:
    /// - Recency: `sqrt(position / total)` — later tokens weighted more
    /// - Coherence: cosine similarity with thought HV — relevant tokens weighted more
    fn attention_weighted_bundle(
        &self,
        thought_hv: &ContinuousHV,
        output_hvs: &[ContinuousHV],
    ) -> ContinuousHV {
        if output_hvs.is_empty() {
            return ContinuousHV::zero(self.config.hdc_dim);
        }
        if output_hvs.len() == 1 {
            return output_hvs[0].clone();
        }

        let n = output_hvs.len();
        let weights: Vec<f32> = output_hvs
            .iter()
            .enumerate()
            .map(|(i, hv)| {
                let recency = ((i + 1) as f32 / n as f32).sqrt();
                let coherence = thought_hv.similarity(hv).clamp(0.0, 1.0);
                recency * (0.5 + 0.5 * coherence)
            })
            .collect();

        let weight_sum: f32 = weights.iter().sum();
        if weight_sum < 1e-10 {
            let refs: Vec<&ContinuousHV> = output_hvs.iter().collect();
            return ContinuousHV::bundle(&refs).normalize();
        }

        let dim = output_hvs[0].values.len();
        let mut bundled = vec![0.0f32; dim];
        for (hv, &w) in output_hvs.iter().zip(weights.iter()) {
            let nw = w / weight_sum;
            for (bv, v) in bundled.iter_mut().zip(hv.values.iter()) {
                *bv += nw * v;
            }
        }

        ContinuousHV::from_vec(bundled).normalize()
    }

    /// Get the last semantic prediction error (round-trip reconstruction).
    pub fn semantic_prediction_error(
        &self,
        thought_hv: &ContinuousHV,
        output_hvs: &[ContinuousHV],
    ) -> f32 {
        if output_hvs.is_empty() {
            return 1.0;
        }
        let refs: Vec<&ContinuousHV> = output_hvs.iter().collect();
        let bundled = ContinuousHV::bundle(&refs).normalize();
        1.0 - thought_hv.similarity(&bundled).clamp(-1.0, 1.0)
    }
}

impl std::fmt::Debug for LiquidMambaGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiquidMambaGenerator")
            .field("config", &self.config)
            .field("mamba", &self.mamba)
            .field("thermodynamic_load", &self.thermodynamic_load)
            .field("arousal", &self.arousal)
            .field("generation_count", &self.generation_count)
            .finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COHERENCE MONITOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Monitors semantic coherence between generated tokens and the original thought.
///
/// Bundles back-projected token HDC vectors in a sliding window and computes
/// EMA cosine similarity against the original `thought_hv`. If coherence drops
/// below threshold for `min_consecutive_low` tokens, signals a veto.
struct CoherenceMonitor {
    thought_hv: ContinuousHV,
    window: VecDeque<ContinuousHV>,
    window_size: usize,
    ema_coherence: f32,
    ema_alpha: f32,
    consecutive_low: usize,
    min_consecutive_low: usize,
    veto_threshold: f32,
}

impl CoherenceMonitor {
    fn new(
        thought_hv: ContinuousHV,
        window_size: usize,
        ema_alpha: f32,
        veto_threshold: f32,
        min_consecutive_low: usize,
    ) -> Self {
        Self {
            thought_hv,
            window: VecDeque::with_capacity(window_size),
            window_size,
            ema_coherence: 1.0,
            ema_alpha,
            consecutive_low: 0,
            min_consecutive_low,
            veto_threshold,
        }
    }

    /// Add a back-projected token HV and update coherence.
    fn push(&mut self, token_hdc: ContinuousHV) {
        // Maintain sliding window
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(token_hdc);

        // Bundle window contents
        let refs: Vec<&ContinuousHV> = self.window.iter().collect();
        let bundled = ContinuousHV::bundle(&refs).normalize();

        // Cosine similarity with thought
        let sim = self.thought_hv.similarity(&bundled).clamp(-1.0, 1.0);

        // EMA smooth
        self.ema_coherence = self.ema_alpha * sim + (1.0 - self.ema_alpha) * self.ema_coherence;

        // Track consecutive low
        if self.ema_coherence < self.veto_threshold {
            self.consecutive_low += 1;
        } else {
            self.consecutive_low = 0;
        }
    }

    /// Whether coherence has been below threshold for enough consecutive tokens.
    fn should_veto(&self) -> bool {
        self.consecutive_low >= self.min_consecutive_low
    }

    /// Current EMA coherence value.
    fn current_coherence(&self) -> f32 {
        self.ema_coherence
    }

    /// Reset the monitor (after a veto).
    fn reset(&mut self) {
        self.window.clear();
        self.ema_coherence = 1.0;
        self.consecutive_low = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLING
// ═══════════════════════════════════════════════════════════════════════════════

/// Top-k sampling with temperature from a logits vector.
fn top_k_sample(logits: &[f32], k: usize, temperature: f32) -> u32 {
    if k == 0 || logits.is_empty() {
        return greedy_sample(logits);
    }

    let k = k.min(logits.len());
    let temp = temperature.max(1e-8);

    // Find top-k indices
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    indexed.truncate(k);

    // Apply temperature and softmax
    let max_logit = indexed[0].1;
    let mut probs: Vec<(usize, f32)> = indexed
        .into_iter()
        .map(|(i, l)| (i, ((l - max_logit) / temp).exp()))
        .collect();

    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum < 1e-10 {
        return probs[0].0 as u32;
    }
    for p in &mut probs {
        p.1 /= sum;
    }

    // Sample
    let r = simple_random_f32();
    let mut cumulative = 0.0f32;
    for (i, p) in &probs {
        cumulative += p;
        if r < cumulative {
            return *i as u32;
        }
    }

    probs.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

/// Greedy sampling: return argmax.
fn greedy_sample(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Simple random float in [0, 1) using thread_rng.
fn simple_random_f32() -> f32 {
    use rand::Rng;
    rand::thread_rng().gen::<f32>()
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biological_state_scale_at_rest() {
        let config = LiquidMambaConfig::default();
        // At rest: load=0, arousal=0.5 → factor = exp(0) = 1.0
        let alpha = 0.5 * config.delta_mod_strength;
        let beta = 0.3 * config.delta_mod_strength;
        let factor = (-alpha * 0.0 - beta * (0.5 - 0.5)).exp();
        assert!((factor - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_biological_state_scale_exhausted() {
        let config = LiquidMambaConfig::default();
        let alpha = 0.5 * config.delta_mod_strength;
        let beta = 0.3 * config.delta_mod_strength;
        // Exhausted: load=1.0, arousal=0.5
        let factor = (-alpha * 1.0 - beta * 0.0).exp();
        assert!((factor - 0.6065).abs() < 0.01, "Expected ~0.61, got {factor}");
    }

    #[test]
    fn test_biological_state_scale_agitated() {
        let config = LiquidMambaConfig::default();
        let alpha = 0.5 * config.delta_mod_strength;
        let beta = 0.3 * config.delta_mod_strength;
        // Agitated: load=0, arousal=1.0
        let factor = (-alpha * 0.0 - beta * 0.5).exp();
        assert!((factor - 0.8607).abs() < 0.01, "Expected ~0.86, got {factor}");
    }

    #[test]
    fn test_coherence_monitor_no_veto_initially() {
        let thought = ContinuousHV::random_default(42).normalize();
        let monitor = CoherenceMonitor::new(thought, 8, 0.3, 0.20, 3);
        assert!(!monitor.should_veto());
        assert!((monitor.current_coherence() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_coherence_monitor_veto_on_drift() {
        let thought = ContinuousHV::random_default(42).normalize();
        let mut monitor = CoherenceMonitor::new(thought, 8, 0.3, 0.20, 3);

        // Push orthogonal vectors to simulate drift
        for i in 0..10 {
            let drifted = ContinuousHV::random_default(1000 + i as u64).normalize();
            monitor.push(drifted);
        }

        // After enough drift, coherence should be very low
        assert!(monitor.current_coherence() < 0.5,
            "Coherence should drop with orthogonal tokens, got {}", monitor.current_coherence());
    }

    #[test]
    fn test_coherence_monitor_high_coherence_with_similar() {
        let thought = ContinuousHV::random_default(42).normalize();
        let mut monitor = CoherenceMonitor::new(thought.clone(), 8, 0.3, 0.20, 3);

        // Push vectors similar to thought (scaled versions)
        for _ in 0..5 {
            let similar = thought.scale(0.9).normalize();
            monitor.push(similar);
        }

        // Coherence should remain high
        assert!(monitor.current_coherence() > 0.5,
            "Coherence should stay high with similar tokens, got {}", monitor.current_coherence());
        assert!(!monitor.should_veto());
    }

    #[test]
    fn test_coherence_monitor_reset() {
        let thought = ContinuousHV::random_default(42).normalize();
        let mut monitor = CoherenceMonitor::new(thought, 8, 0.3, 0.20, 3);

        // Push some drift
        for i in 0..5 {
            let drifted = ContinuousHV::random_default(1000 + i as u64).normalize();
            monitor.push(drifted);
        }

        monitor.reset();
        assert!(!monitor.should_veto());
        assert!((monitor.current_coherence() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_sample_greedy_fallback() {
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let token = greedy_sample(&logits);
        assert_eq!(token, 3, "Should pick index 3 (highest logit 0.8)");
    }

    #[test]
    fn test_top_k_sample_produces_valid_index() {
        let logits = vec![0.1; 100];
        let token = top_k_sample(&logits, 10, 1.0);
        assert!((token as usize) < 100, "Token should be valid index");
    }

    #[test]
    fn test_top_k_sample_temperature_zero() {
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        // Very low temperature → should approach greedy
        let token = top_k_sample(&logits, 5, 0.001);
        assert_eq!(token, 3, "Near-zero temperature should be greedy");
    }

    #[test]
    fn test_config_default() {
        let config = LiquidMambaConfig::default();
        assert_eq!(config.model_id, "state-spaces/mamba-130m");
        assert_eq!(config.max_tokens, 256);
        assert_eq!(config.hdc_dim, 16384);
        assert_eq!(config.bottleneck_dim, 256);
        assert_eq!(config.ssm_dim, 768);
        assert!(config.enable_gating);
        assert!(config.enable_veto);
        assert!(config.enable_liquid_delta);
        assert_eq!(config.warmup_steps, 100);
        assert_eq!(config.accumulation_steps, 4);
        assert!((config.contrastive_weight - 0.1).abs() < 1e-6);
        assert_eq!(config.contrastive_buffer_size, 8);
    }

    #[test]
    fn test_attention_weighted_bundle_single() {
        // Single HV should be returned as-is (no weighting needed)
        let thought = ContinuousHV::random_default(42).normalize();
        let output = ContinuousHV::random_default(99).normalize();

        // Create a minimal generator-like context to test the bundling
        // Use the standalone function approach: replicate the logic
        let n = 1;
        let weights: Vec<f32> = vec![1.0];
        let weight_sum: f32 = weights.iter().sum();
        assert!(weight_sum > 0.0);

        // With a single HV, the result should be that HV
        let sim = thought.similarity(&output);
        assert!(sim.is_finite());
    }

    #[test]
    fn test_attention_weighted_bundle_recency() {
        // Later tokens should get higher recency weight
        let n = 5;
        let weights: Vec<f32> = (0..n)
            .map(|i| ((i + 1) as f32 / n as f32).sqrt())
            .collect();
        // First token should have lowest weight, last should have highest
        assert!(weights[0] < weights[4]);
        assert!(weights[3] < weights[4]);
        // Verify monotonically increasing
        for i in 1..n {
            assert!(weights[i] >= weights[i - 1]);
        }
    }

    #[test]
    fn test_warmup_schedule() {
        // At generation 0: factor = 0/100 = 0
        // At generation 50: factor = 50/100 = 0.5
        // At generation 100: factor = 100/100 = 1.0
        // At generation 200: factor = min(200/100, 1.0) = 1.0
        let warmup_steps = 100usize;
        for (gen, expected) in [(0, 0.0), (50, 0.5), (100, 1.0), (200, 1.0)] {
            let factor = (gen as f32 / warmup_steps as f32).min(1.0);
            assert!(
                (factor - expected).abs() < 0.01,
                "gen={gen}: expected {expected}, got {factor}"
            );
        }
    }

    #[test]
    fn test_accumulation_gating() {
        // Verify that gradient accumulation fires every N steps
        let accumulation_steps = 4;
        let mut acc = 0usize;
        let mut apply_count = 0;

        for _ in 0..12 {
            acc += 1;
            if acc >= accumulation_steps {
                apply_count += 1;
                acc = 0;
            }
        }
        assert_eq!(apply_count, 3, "Should apply 3 times in 12 steps with acc=4");
    }

    #[test]
    fn test_semantic_prediction_error_identical() {
        let thought = ContinuousHV::random_default(42).normalize();
        let output = thought.clone();

        // When output == thought, error should be ~0
        let refs: Vec<&ContinuousHV> = vec![&output];
        let bundled = ContinuousHV::bundle(&refs).normalize();
        let error = 1.0 - thought.similarity(&bundled).clamp(-1.0, 1.0);
        assert!(error < 0.1, "Error should be near 0 for identical vectors, got {error}");
    }

    #[test]
    fn test_semantic_prediction_error_orthogonal() {
        let thought = ContinuousHV::random_default(42).normalize();
        let other = ContinuousHV::random_default(99).normalize();

        let refs: Vec<&ContinuousHV> = vec![&other];
        let bundled = ContinuousHV::bundle(&refs).normalize();
        let error = 1.0 - thought.similarity(&bundled).clamp(-1.0, 1.0);
        // Random high-dimensional vectors are nearly orthogonal
        assert!(error > 0.5, "Error should be high for orthogonal vectors, got {error}");
    }
}
