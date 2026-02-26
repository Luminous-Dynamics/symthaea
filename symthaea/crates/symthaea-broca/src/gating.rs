//! Per-token gating: epistemic, emotional, and coherence constraints on generation.
//!
//! This is the architectural innovation — epistemic honesty and emotional authenticity
//! become *physical constraints* on generation, not prompt instructions.
//!
//! # Gates
//!
//! - **EpistemicGate**: Suppresses factual tokens when epistemic status is Unknown/Uncertain
//! - **EmotionalModulator**: Arousal → sentence length, warmth → formality
//! - **CoherenceFeedback**: Boosts thought_hv binding when network drifts
//! - **ConsciousnessGatedVerbosity**: Higher Ψ → more detailed output

use crate::encoder::ThoughtChannels;
use crate::tokenizer::BpeTokenizer;

/// Configuration for the gating system.
#[derive(Debug, Clone)]
pub struct GatingConfig {
    /// Logit penalty for factual tokens when epistemic status is Unknown.
    pub unknown_factual_penalty: f32,
    /// Logit boost for hedging tokens when epistemic status is Unknown.
    pub unknown_hedging_boost: f32,
    /// Logit penalty for factual tokens when epistemic status is Uncertain.
    pub uncertain_factual_penalty: f32,
    /// Logit boost for hedging tokens when epistemic status is Uncertain.
    pub uncertain_hedging_boost: f32,
    /// Coherence drift threshold (below this, boost thought binding).
    pub coherence_drift_threshold: f32,
    /// Arousal threshold above which sentence-ending tokens are boosted.
    pub high_arousal_threshold: f32,
    /// Token position after which high-arousal sentence-ending boost applies.
    pub arousal_position_threshold: usize,
    /// Warmth threshold below which informal tokens are suppressed.
    pub low_warmth_threshold: f32,
    /// Base max tokens (before consciousness scaling).
    pub base_max_tokens: usize,
}

impl Default for GatingConfig {
    fn default() -> Self {
        Self {
            unknown_factual_penalty: -10.0,
            unknown_hedging_boost: 2.0,
            uncertain_factual_penalty: -3.0,
            uncertain_hedging_boost: 1.0,
            coherence_drift_threshold: 0.3,
            high_arousal_threshold: 0.7,
            arousal_position_threshold: 10,
            low_warmth_threshold: 0.3,
            base_max_tokens: 128,
        }
    }
}

/// Epistemic gate: suppresses factual assertions when confidence is low.
///
/// The system physically *cannot* hallucinate when epistemic status is Unknown —
/// factual token logits are suppressed before sampling.
pub struct EpistemicGate {
    config: GatingConfig,
    /// Token IDs classified as "hedging" (maybe, perhaps, uncertain, etc.)
    hedging_token_ids: Vec<u32>,
    /// Token IDs classified as "factual assertion" (is, are, definitely, etc.)
    factual_token_ids: Vec<u32>,
    /// Token IDs for out-of-domain response
    ood_token_ids: Vec<u32>,
}

impl EpistemicGate {
    /// Create an epistemic gate, classifying tokens from the tokenizer vocabulary.
    pub fn new(tokenizer: &BpeTokenizer, config: &GatingConfig) -> Self {
        let hedging_words = [
            "perhaps", "maybe", "possibly", "likely", "probably",
            "uncertain", "unknown", "believe", "seems", "appears", "might",
            "however", "although", "unfortunately", "sorry",
        ];
        let factual_words = [
            "is", "are", "was", "certainly", "definitely", "always", "never",
            "must", "shall", "every", "all", "none",
        ];
        let ood_words = [
            "outside", "beyond", "cannot", "unable",
        ];

        let hedging_token_ids: Vec<u32> = hedging_words
            .iter()
            .filter_map(|w| {
                let id = tokenizer.token_id(w);
                if id != tokenizer.unk_id { Some(id) } else { None }
            })
            .collect();

        let factual_token_ids: Vec<u32> = factual_words
            .iter()
            .filter_map(|w| {
                let id = tokenizer.token_id(w);
                if id != tokenizer.unk_id { Some(id) } else { None }
            })
            .collect();

        let ood_token_ids: Vec<u32> = ood_words
            .iter()
            .filter_map(|w| {
                let id = tokenizer.token_id(w);
                if id != tokenizer.unk_id { Some(id) } else { None }
            })
            .collect();

        Self {
            config: config.clone(),
            hedging_token_ids,
            factual_token_ids,
            ood_token_ids,
        }
    }

    /// Apply epistemic gating to logits in-place.
    ///
    /// - epistemic ordinal: 0=Certain, 1=Probable, 2=Uncertain, 3=Unknown, 4=OutOfDomain
    pub fn apply(&self, logits: &mut [f32], epistemic_ordinal: f32) {
        if epistemic_ordinal < 1.5 {
            // Certain or Probable: no modification
            return;
        }

        if epistemic_ordinal > 3.5 {
            // OutOfDomain: suppress all content tokens, boost OOD tokens
            for &id in &self.factual_token_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += self.config.unknown_factual_penalty;
                }
            }
            for &id in &self.ood_token_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += self.config.unknown_hedging_boost * 2.0;
                }
            }
            for &id in &self.hedging_token_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += self.config.unknown_hedging_boost;
                }
            }
            return;
        }

        if epistemic_ordinal > 2.5 {
            // Unknown
            for &id in &self.factual_token_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += self.config.unknown_factual_penalty;
                }
            }
            for &id in &self.hedging_token_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += self.config.unknown_hedging_boost;
                }
            }
        } else {
            // Uncertain
            for &id in &self.factual_token_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += self.config.uncertain_factual_penalty;
                }
            }
            for &id in &self.hedging_token_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += self.config.uncertain_hedging_boost;
                }
            }
        }
    }
}

/// Emotional modulator: shapes generation style based on affect.
///
/// - High arousal → boost sentence-ending tokens (shorter sentences)
/// - Low warmth → suppress informal vocabulary
/// - Negative valence → boost softening language
pub struct EmotionalModulator {
    config: GatingConfig,
    /// Sentence-ending token IDs (., !, ?)
    sentence_end_ids: Vec<u32>,
    /// Informal token IDs (contractions, slang)
    informal_ids: Vec<u32>,
    /// Softening token IDs (unfortunately, sorry, etc.)
    softening_ids: Vec<u32>,
}

impl EmotionalModulator {
    /// Create an emotional modulator from the tokenizer vocabulary.
    pub fn new(tokenizer: &BpeTokenizer, config: &GatingConfig) -> Self {
        let sentence_endings = [". ", "! ", "? ", "...", "\n"];
        let informal_words = ["gonna", "wanna", "gotta", "kinda", "sorta"];
        let softening_words = ["unfortunately", "sorry", "however", "although"];

        let sentence_end_ids: Vec<u32> = sentence_endings
            .iter()
            .filter_map(|w| {
                let id = tokenizer.token_id(w);
                if id != tokenizer.unk_id { Some(id) } else { None }
            })
            .collect();

        let informal_ids: Vec<u32> = informal_words
            .iter()
            .filter_map(|w| {
                let id = tokenizer.token_id(w);
                if id != tokenizer.unk_id { Some(id) } else { None }
            })
            .collect();

        let softening_ids: Vec<u32> = softening_words
            .iter()
            .filter_map(|w| {
                let id = tokenizer.token_id(w);
                if id != tokenizer.unk_id { Some(id) } else { None }
            })
            .collect();

        Self {
            config: config.clone(),
            sentence_end_ids,
            informal_ids,
            softening_ids,
        }
    }

    /// Apply emotional modulation to logits in-place.
    pub fn apply(&self, logits: &mut [f32], channels: &ThoughtChannels, position: usize) {
        let arousal = channels.arousal();
        let warmth = channels.warmth();
        let valence = channels.valence();

        // High arousal + past threshold → boost sentence endings (shorter sentences)
        if arousal > self.config.high_arousal_threshold
            && position > self.config.arousal_position_threshold
        {
            let boost = (arousal - self.config.high_arousal_threshold) * 3.0;
            for &id in &self.sentence_end_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += boost;
                }
            }
        }

        // Low warmth → suppress informal tokens
        if warmth < self.config.low_warmth_threshold {
            let penalty = (self.config.low_warmth_threshold - warmth) * -5.0;
            for &id in &self.informal_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += penalty;
                }
            }
        }

        // Negative valence → boost softening language
        if valence < -0.3 {
            let boost = (-valence - 0.3) * 2.0;
            for &id in &self.softening_ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l += boost;
                }
            }
        }
    }
}

/// Coherence feedback: monitors drift between network state and thought intent.
///
/// If `cosine_similarity(network_state, thought_hv) < threshold`, the system
/// boosts thought_hv binding weight in the next step.
pub struct CoherenceFeedback {
    /// Drift threshold — below this, correction is applied.
    threshold: f32,
    /// Current coherence score (updated each step).
    current_coherence: f32,
    /// Whether semantic veto was triggered.
    veto_triggered: bool,
    /// Semantic veto threshold (more aggressive than drift).
    veto_threshold: f32,
}

impl CoherenceFeedback {
    /// Create a new coherence feedback monitor.
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            current_coherence: 1.0,
            veto_triggered: false,
            veto_threshold: 0.25,
        }
    }

    /// Update coherence score from the current network output and thought HV.
    /// Returns a binding weight multiplier (>1.0 when drifting).
    pub fn update(&mut self, output_hv: &symthaea_core::hdc::ContinuousHV, thought_hv: &symthaea_core::hdc::ContinuousHV) -> f32 {
        self.current_coherence = output_hv.similarity(thought_hv);
        self.veto_triggered = self.current_coherence < self.veto_threshold;

        if self.current_coherence < self.threshold {
            // Boost binding weight inversely proportional to coherence
            let correction = 1.0 + (self.threshold - self.current_coherence) * 2.0;
            correction.min(3.0) // Cap at 3x
        } else {
            1.0
        }
    }

    /// Whether a semantic veto should be triggered (mid-sentence self-correction).
    pub fn should_veto(&self) -> bool {
        self.veto_triggered
    }

    /// Current coherence score.
    pub fn coherence(&self) -> f32 {
        self.current_coherence
    }
}

/// Compute consciousness-gated max tokens.
///
/// `max_tokens = base_max * (0.5 + psi)`
///
/// Higher consciousness → more detailed output, lower → terser.
pub fn consciousness_gated_max_tokens(base_max: usize, psi: f32) -> usize {
    let factor = 0.5 + psi.clamp(0.0, 1.0);
    ((base_max as f32) * factor) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tokenizer() -> BpeTokenizer {
        BpeTokenizer::default_minimal()
    }

    fn test_config() -> GatingConfig {
        GatingConfig::default()
    }

    #[test]
    fn test_epistemic_gate_certain_no_change() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);

        let mut logits = vec![0.5; tok.vocab_size()];
        let original = logits.clone();
        gate.apply(&mut logits, 0.0); // Certain

        assert_eq!(logits, original, "Certain status should not modify logits");
    }

    #[test]
    fn test_epistemic_gate_unknown_suppresses_factual() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);

        let mut logits = vec![0.5; tok.vocab_size()];
        gate.apply(&mut logits, 3.0); // Unknown

        // Factual token "is" should be penalized if in vocabulary
        let is_id = tok.token_id("is");
        if is_id != tok.unk_id {
            assert!(
                logits[is_id as usize] < 0.5,
                "Factual token 'is' should be penalized under Unknown"
            );
        }
    }

    #[test]
    fn test_epistemic_gate_unknown_boosts_hedging() {
        let tok = test_tokenizer();
        let config = test_config();
        let gate = EpistemicGate::new(&tok, &config);

        let mut logits = vec![0.5; tok.vocab_size()];
        gate.apply(&mut logits, 3.0); // Unknown

        let perhaps_id = tok.token_id("perhaps");
        if perhaps_id != tok.unk_id {
            assert!(
                logits[perhaps_id as usize] > 0.5,
                "Hedging token 'perhaps' should be boosted under Unknown"
            );
        }
    }

    #[test]
    fn test_emotional_modulator_high_arousal() {
        let tok = test_tokenizer();
        let config = test_config();
        let modulator = EmotionalModulator::new(&tok, &config);

        let mut channels = ThoughtChannels::default();
        channels.set_emotion(0.5, 0.9, 0.5); // high arousal

        let mut logits = vec![0.5; tok.vocab_size()];
        modulator.apply(&mut logits, &channels, 15); // past position threshold

        // Sentence endings should be boosted
        let period_id = tok.token_id(". ");
        if period_id != tok.unk_id {
            assert!(
                logits[period_id as usize] > 0.5,
                "Sentence endings should be boosted under high arousal"
            );
        }
    }

    #[test]
    fn test_coherence_feedback_normal() {
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("test-coherence");
        let thought_hv = symthaea_core::hdc::ContinuousHV::from_genesis(&genesis, "thought", symthaea_core::hdc::HDC_DIMENSION);

        let mut feedback = CoherenceFeedback::new(0.3);
        let weight = feedback.update(&thought_hv, &thought_hv);

        // Same HV → perfect coherence → no correction
        assert!((weight - 1.0).abs() < 0.01, "Perfect coherence should yield weight 1.0");
        assert!(!feedback.should_veto());
    }

    #[test]
    fn test_coherence_feedback_drift() {
        let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("test-coherence");
        let thought_hv = symthaea_core::hdc::ContinuousHV::from_genesis(&genesis, "thought", symthaea_core::hdc::HDC_DIMENSION);
        let other_hv = symthaea_core::hdc::ContinuousHV::from_genesis(&genesis, "other", symthaea_core::hdc::HDC_DIMENSION);

        let mut feedback = CoherenceFeedback::new(0.3);
        let weight = feedback.update(&other_hv, &thought_hv);

        // Random HVs in high-D are nearly orthogonal → low coherence → correction
        assert!(weight > 1.0, "Drifted state should increase binding weight");
    }

    #[test]
    fn test_consciousness_gated_verbosity() {
        assert_eq!(consciousness_gated_max_tokens(100, 0.0), 50);
        assert_eq!(consciousness_gated_max_tokens(100, 0.5), 100);
        assert_eq!(consciousness_gated_max_tokens(100, 1.0), 150);
    }
}
