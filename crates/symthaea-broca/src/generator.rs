//! BrocaGenerator: autoregressive text generation from thought channels.
//!
//! Orchestrates the full pipeline:
//! 1. Encode thought channels → ContinuousHV
//! 2. Autoregressive loop: forward step → gate → sample → decode
//! 3. Semantic veto: mid-sentence self-correction when coherence drops
//! 4. Thermodynamic subjective time: dt varies with system load

use serde::{Deserialize, Serialize};

use crate::controller::{LanguageController, LanguageControllerConfig};
use crate::encoder::{ThoughtChannels, ThoughtLanguageEncoder};
use crate::gating::{
    consciousness_gated_max_tokens, CoherenceFeedback, EmotionalModulator, EpistemicGate,
    GatingConfig,
};
use crate::tokenizer::BpeTokenizer;

use symthaea_core::genesis::GenesisSeed;

/// Sampling strategy for token selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Always pick the highest-logit token.
    Greedy,
    /// Sample from top-k tokens with temperature.
    TopK { k: usize, temperature: f32 },
    /// Sample from top-p (nucleus) tokens with temperature.
    TopP { p: f32, temperature: f32 },
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::Greedy
    }
}

/// Configuration for the Broca generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrocaConfig {
    /// Controller configuration.
    pub controller: LanguageControllerConfig,
    /// Gating configuration.
    pub gating: GatingConfig,
    /// Default sampling strategy.
    pub sampling: SamplingStrategy,
    /// Enable epistemic gating.
    pub enable_epistemic_gate: bool,
    /// Enable emotional modulation.
    pub enable_emotional_modulation: bool,
    /// Enable coherence feedback.
    pub enable_coherence_feedback: bool,
    /// Enable consciousness-gated verbosity.
    pub enable_consciousness_gating: bool,
    /// Enable semantic veto (mid-sentence self-correction).
    pub enable_semantic_veto: bool,
    /// Hesitation token for semantic veto.
    pub veto_hesitation: String,
}

impl Default for BrocaConfig {
    fn default() -> Self {
        Self {
            controller: LanguageControllerConfig::default(),
            gating: GatingConfig::default(),
            sampling: SamplingStrategy::Greedy,
            enable_epistemic_gate: true,
            enable_emotional_modulation: true,
            enable_coherence_feedback: true,
            enable_consciousness_gating: true,
            enable_semantic_veto: true,
            veto_hesitation: "-- wait, ".to_string(),
        }
    }
}

/// Result of a single generation.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Generated text.
    pub text: String,
    /// Generated token IDs (excluding BOS/EOS).
    pub token_ids: Vec<u32>,
    /// Number of tokens generated.
    pub num_tokens: usize,
    /// Whether generation was terminated by EOS.
    pub eos_terminated: bool,
    /// Whether a semantic veto was triggered.
    pub veto_triggered: bool,
    /// Final short-window coherence score.
    pub final_coherence: f32,
    /// Final long-window coherence score (Liquid-Mamba only, 0.0 for CfC-HDC).
    pub long_coherence: f32,
    /// Back-projected HDC vectors for each generated token (Liquid-Mamba only).
    #[cfg(feature = "mamba")]
    pub output_hvs: Vec<symthaea_core::hdc::ContinuousHV>,
    /// Semantic prediction error: round-trip reconstruction loss (Liquid-Mamba only).
    #[cfg(feature = "mamba")]
    pub semantic_pe: f32,
}

/// Broca generator: autoregressive thought-to-text.
pub struct BrocaGenerator {
    controller: LanguageController,
    tokenizer: BpeTokenizer,
    encoder: ThoughtLanguageEncoder,
    epistemic_gate: EpistemicGate,
    emotional_modulator: EmotionalModulator,
    coherence_feedback: CoherenceFeedback,
    config: BrocaConfig,
}

impl BrocaGenerator {
    /// Create a new generator from genesis seed and config.
    pub fn new(genesis: &GenesisSeed, config: BrocaConfig) -> Self {
        let tokenizer = BpeTokenizer::default_minimal();

        // Ensure controller vocab_size matches tokenizer
        let mut ctrl_config = config.controller.clone();
        ctrl_config.vocab_size = tokenizer.vocab_size();

        let controller = LanguageController::new(genesis, &ctrl_config);
        let encoder = ThoughtLanguageEncoder::new(genesis);
        let epistemic_gate = EpistemicGate::new(&tokenizer, &config.gating);
        let emotional_modulator = EmotionalModulator::new(&tokenizer, &config.gating);
        let coherence_feedback = CoherenceFeedback::new(config.gating.coherence_drift_threshold);

        Self {
            controller,
            tokenizer,
            encoder,
            epistemic_gate,
            emotional_modulator,
            coherence_feedback,
            config,
        }
    }

    /// Create a generator with a custom tokenizer.
    pub fn with_tokenizer(
        genesis: &GenesisSeed,
        config: BrocaConfig,
        tokenizer: BpeTokenizer,
    ) -> Self {
        let mut ctrl_config = config.controller.clone();
        ctrl_config.vocab_size = tokenizer.vocab_size();

        let controller = LanguageController::new(genesis, &ctrl_config);
        let encoder = ThoughtLanguageEncoder::new(genesis);
        let epistemic_gate = EpistemicGate::new(&tokenizer, &config.gating);
        let emotional_modulator = EmotionalModulator::new(&tokenizer, &config.gating);
        let coherence_feedback = CoherenceFeedback::new(config.gating.coherence_drift_threshold);

        Self {
            controller,
            tokenizer,
            encoder,
            epistemic_gate,
            emotional_modulator,
            coherence_feedback,
            config,
        }
    }

    /// Generate text from thought channels.
    pub fn generate(&mut self, channels: &ThoughtChannels) -> GenerationResult {
        self.generate_with_callback(channels, &mut |_| {})
    }

    /// Generate text with a per-token streaming callback.
    pub fn generate_with_callback(
        &mut self,
        channels: &ThoughtChannels,
        on_token: &mut dyn FnMut(&str),
    ) -> GenerationResult {
        // 1. Encode thought channels once
        let thought_hv = self.encoder.encode(channels);

        // 2. Compute max tokens (consciousness-gated)
        let max_tokens = if self.config.enable_consciousness_gating {
            consciousness_gated_max_tokens(self.config.gating.base_max_tokens, channels.psi())
        } else {
            self.config.gating.base_max_tokens
        };

        // 3. Reset controller state
        self.controller.reset();

        // 4. Autoregressive generation loop
        let mut tokens = Vec::new();
        let mut prev_token = self.tokenizer.thought_id; // Start with <thought> token
        let mut eos_terminated = false;
        let mut veto_triggered = false;
        let mut text = String::new();

        for pos in 0..max_tokens {
            // Forward step
            let mut logits = self.controller.forward_step(&thought_hv, prev_token, pos);

            // Apply gating
            if self.config.enable_epistemic_gate {
                self.epistemic_gate
                    .apply(&mut logits, channels.epistemic_ordinal());
            }

            if self.config.enable_emotional_modulation {
                self.emotional_modulator.apply(&mut logits, channels, pos);
            }

            // Coherence feedback
            if self.config.enable_coherence_feedback {
                let output_hv = self.controller.output_hv();
                let _weight = self.coherence_feedback.update(&output_hv, &thought_hv);

                // Semantic veto: mid-sentence self-correction
                if self.config.enable_semantic_veto
                    && self.coherence_feedback.should_veto()
                    && pos > 2
                {
                    veto_triggered = true;
                    text.push_str(&self.config.veto_hesitation);
                    on_token(&self.config.veto_hesitation);
                    // Reset network state and re-inject thought
                    self.controller.reset();
                    // Continue from current position (don't reset pos)
                }
            }

            // Sample next token
            let next_token = self.sample(&logits);

            // Check EOS
            if next_token == self.tokenizer.eos_id {
                eos_terminated = true;
                break;
            }

            // Skip special tokens in output
            if !self.tokenizer.is_special(next_token) {
                let token_str = self.tokenizer.token_str(next_token);
                text.push_str(token_str);
                on_token(token_str);
            }

            tokens.push(next_token);
            prev_token = next_token;
        }

        let final_coherence = self.coherence_feedback.coherence();

        GenerationResult {
            text,
            token_ids: tokens.clone(),
            num_tokens: tokens.len(),
            eos_terminated,
            veto_triggered,
            final_coherence,
            long_coherence: 0.0, // CfC-HDC backend doesn't use long window
            #[cfg(feature = "mamba")]
            output_hvs: Vec::new(),
            #[cfg(feature = "mamba")]
            semantic_pe: 0.0,
        }
    }

    /// Sample a token from logits using the configured strategy.
    fn sample(&self, logits: &[f32]) -> u32 {
        match &self.config.sampling {
            SamplingStrategy::Greedy => greedy_sample(logits),
            SamplingStrategy::TopK { k, temperature } => top_k_sample(logits, *k, *temperature),
            SamplingStrategy::TopP { p, temperature } => top_p_sample(logits, *p, *temperature),
        }
    }

    /// Get reference to the tokenizer.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }

    /// Get mutable reference to the controller (for training).
    pub fn controller_mut(&mut self) -> &mut LanguageController {
        &mut self.controller
    }

    /// Get reference to the encoder.
    pub fn encoder(&self) -> &ThoughtLanguageEncoder {
        &self.encoder
    }

    /// Get reference to the controller.
    pub fn controller(&self) -> &LanguageController {
        &self.controller
    }

    /// Get reference to the config.
    pub fn config(&self) -> &BrocaConfig {
        &self.config
    }
}

/// Greedy sampling: return the argmax token.
fn greedy_sample(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Top-k sampling with temperature.
fn top_k_sample(logits: &[f32], k: usize, temperature: f32) -> u32 {
    let k = k.min(logits.len());
    if k == 0 {
        return greedy_sample(logits);
    }

    // Find top-k indices
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
    indexed.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    indexed.truncate(k);

    // Apply temperature and softmax
    let temp = temperature.max(1e-6);
    let max_logit = indexed[0].1;
    let probs: Vec<(usize, f32)> = indexed
        .iter()
        .map(|&(i, l)| {
            let exp = ((l - max_logit) / temp).exp();
            (i, exp)
        })
        .collect();

    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum < 1e-10 {
        return indexed[0].0 as u32;
    }

    // Sample from distribution using simple LCG
    let r = simple_random_f32();
    let mut cumulative = 0.0;
    for (i, p) in &probs {
        cumulative += p / sum;
        if r < cumulative {
            return *i as u32;
        }
    }

    probs.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

/// Top-p (nucleus) sampling with temperature.
fn top_p_sample(logits: &[f32], p: f32, temperature: f32) -> u32 {
    // Sort by logit descending
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
    indexed.sort_by(|(_, a), (_, b)| b.total_cmp(a));

    // Apply temperature and softmax
    let temp = temperature.max(1e-6);
    let max_logit = indexed[0].1;
    let probs: Vec<(usize, f32)> = indexed
        .iter()
        .map(|&(i, l)| {
            let exp = ((l - max_logit) / temp).exp();
            (i, exp)
        })
        .collect();

    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum < 1e-10 {
        return indexed[0].0 as u32;
    }

    // Find nucleus: smallest set with cumulative prob >= p
    let mut cumulative = 0.0;
    let mut nucleus = Vec::new();
    for (i, prob) in &probs {
        cumulative += prob / sum;
        nucleus.push((*i, *prob / sum));
        if cumulative >= p {
            break;
        }
    }

    // Sample from nucleus
    let r = simple_random_f32();
    let nucleus_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
    let mut c = 0.0;
    for (i, prob) in &nucleus {
        c += prob / nucleus_sum;
        if r < c {
            return *i as u32;
        }
    }

    nucleus.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

/// Simple thread-local random f32 in [0, 1) for sampling.
/// Uses std random for simplicity (not genesis-seeded — sampling is intentionally stochastic).
fn simple_random_f32() -> f32 {
    use rand::Rng;
    rand::thread_rng().gen::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-broca-generator")
    }

    fn test_config() -> BrocaConfig {
        BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 2,
                neurons_per_layer: 4,
                vocab_size: 32, // Will be overridden by tokenizer
                max_seq_len: 16,
                ..Default::default()
            },
            gating: GatingConfig {
                base_max_tokens: 20, // Short for testing
                ..Default::default()
            },
            sampling: SamplingStrategy::Greedy,
            enable_coherence_feedback: false, // Disable for deterministic tests
            enable_semantic_veto: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_generator_creation() {
        let genesis = test_genesis();
        let config = test_config();
        let gen = BrocaGenerator::new(&genesis, config);
        assert!(gen.tokenizer().vocab_size() > 100);
    }

    #[test]
    fn test_greedy_determinism() {
        let genesis = test_genesis();
        let config = test_config();

        let mut gen1 = BrocaGenerator::new(&genesis, config.clone());
        let mut gen2 = BrocaGenerator::new(&genesis, config);

        let channels = ThoughtChannels::default();
        let result1 = gen1.generate(&channels);
        let result2 = gen2.generate(&channels);

        assert_eq!(
            result1.token_ids, result2.token_ids,
            "Greedy generation should be deterministic"
        );
        assert_eq!(result1.text, result2.text);
    }

    #[test]
    fn test_generation_produces_tokens() {
        let genesis = test_genesis();
        let config = test_config();
        let mut gen = BrocaGenerator::new(&genesis, config);

        let channels = ThoughtChannels::default();
        let result = gen.generate(&channels);

        assert!(result.num_tokens > 0, "Should generate at least 1 token");
    }

    #[test]
    fn test_max_tokens_limit() {
        let genesis = test_genesis();
        let mut config = test_config();
        config.gating.base_max_tokens = 5;
        config.enable_consciousness_gating = false;

        let mut gen = BrocaGenerator::new(&genesis, config);
        let channels = ThoughtChannels::default();
        let result = gen.generate(&channels);

        assert!(
            result.num_tokens <= 5,
            "Should respect max_tokens limit: got {}",
            result.num_tokens
        );
    }

    #[test]
    fn test_streaming_callback() {
        let genesis = test_genesis();
        let config = test_config();
        let mut gen = BrocaGenerator::new(&genesis, config);

        let channels = ThoughtChannels::default();
        let mut streamed = String::new();
        let result = gen.generate_with_callback(&channels, &mut |token| {
            streamed.push_str(token);
        });

        assert_eq!(
            streamed, result.text,
            "Streamed tokens should match final text"
        );
    }

    #[test]
    fn test_different_intents_different_output() {
        let genesis = test_genesis();
        let config = test_config();

        let answer_channels = ThoughtChannels::with_intent(1); // Answer
        let clarify_channels = ThoughtChannels::with_intent(2); // Clarify

        let mut gen1 = BrocaGenerator::new(&genesis, config.clone());
        let result1 = gen1.generate(&answer_channels);

        let mut gen2 = BrocaGenerator::new(&genesis, config);
        let result2 = gen2.generate(&clarify_channels);

        // Different intents should produce different token sequences
        assert_ne!(
            result1.token_ids, result2.token_ids,
            "Different intents should generate different outputs"
        );
    }

    #[test]
    fn test_greedy_sample_fn() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(greedy_sample(&logits), 3);
    }

    #[test]
    fn test_consciousness_gated_max_tokens() {
        let genesis = test_genesis();
        let mut config = test_config();
        config.enable_consciousness_gating = true;
        config.gating.base_max_tokens = 100;

        let mut gen = BrocaGenerator::new(&genesis, config);

        // Low psi → shorter
        let mut low_psi = ThoughtChannels::default();
        low_psi.set_consciousness(0.1, 0.5, 0.5);
        let result_low = gen.generate(&low_psi);

        let mut gen2 = BrocaGenerator::new(
            &test_genesis(),
            BrocaConfig {
                controller: LanguageControllerConfig {
                    network_layers: 2,
                    neurons_per_layer: 4,
                    vocab_size: 32,
                    max_seq_len: 16,
                    ..Default::default()
                },
                gating: GatingConfig {
                    base_max_tokens: 100,
                    ..Default::default()
                },
                enable_consciousness_gating: true,
                enable_coherence_feedback: false,
                enable_semantic_veto: false,
                sampling: SamplingStrategy::Greedy,
                ..Default::default()
            },
        );

        // High psi → longer (but may hit EOS or max_tokens)
        let mut high_psi = ThoughtChannels::default();
        high_psi.set_consciousness(0.9, 0.5, 0.5);
        let result_high = gen2.generate(&high_psi);

        // Just verify both produce output — the effective max differs
        assert!(result_low.num_tokens > 0);
        assert!(result_high.num_tokens > 0);
    }
}
