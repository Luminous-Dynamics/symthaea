// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BrocaLite bridge: lightweight always-on language generation fallback.
//!
//! When the full `ssm_language` feature is not enabled (or its checkpoint
//! is unavailable), BrocaLite provides consciousness-coupled text generation
//! using a 512-token vocabulary and 1024D element-wise gated recurrence.
//!
//! Memory cost: ~2MB. No matrix multiplies. No checkpoint file required.
//! Deterministic via xorshift — same seed produces identical output.

use super::broca_bridge::BrocaConsciousnessSignals;

/// Result of a BrocaLite generation call, compatible with the cognitive loop.
#[derive(Debug, Clone)]
pub struct LiteGenerationResult {
    /// Generated text.
    pub text: String,
    /// Number of tokens generated.
    pub num_tokens: usize,
    /// Whether generation stopped due to EOS.
    pub eos_terminated: bool,
    /// Final coherence estimate (always 1.0 for structured, cosine_sim for autoregressive).
    pub coherence: f32,
}

/// Manager wrapping symthaea-spore's BrocaLite with consciousness gating.
#[cfg(feature = "broca_lite")]
pub struct BrocaLiteManager {
    generator: symthaea_spore::broca::BrocaLite,
    /// Minimum consciousness level to generate (below this, silence).
    consciousness_threshold: f32,
    /// Maximum tokens per generation.
    max_tokens: usize,
}

#[cfg(feature = "broca_lite")]
impl BrocaLiteManager {
    /// Create a new BrocaLite manager with deterministic initialization.
    pub fn new(seed: u64) -> Self {
        Self {
            generator: symthaea_spore::broca::BrocaLite::new(seed),
            consciousness_threshold: 0.05,
            max_tokens: 32,
        }
    }

    /// Generate text from consciousness signals (fallback path).
    ///
    /// When `input_text` is provided, intent channels are enriched with
    /// keyword scanning (curiosity from questions, valence from sentiment,
    /// abstraction from philosophical terms, self-reference from "you/your").
    ///
    /// Returns `None` if consciousness is too low or ethics blocks generation.
    pub fn generate_from_signals(
        &mut self,
        signals: &BrocaConsciousnessSignals,
    ) -> Option<LiteGenerationResult> {
        self.generate_from_signals_with_input(signals, None)
    }

    /// Generate text with input-awareness.
    pub fn generate_from_signals_with_input(
        &mut self,
        signals: &BrocaConsciousnessSignals,
        input_text: Option<&str>,
    ) -> Option<LiteGenerationResult> {
        // Ethics gate
        if signals.ethics_blocked {
            return None;
        }

        // Consciousness gate
        if signals.consciousness_level < self.consciousness_threshold {
            return None;
        }

        // Map BrocaConsciousnessSignals → Spore ThoughtChannels (12D)
        let mut channels = signals_to_lite_channels(signals);

        // Enrich intent channels from user input text (makes output responsive)
        if let Some(text) = input_text {
            channels.inject_intent(text);
        }

        // T1.6: Consciousness-scaled max_tokens
        self.max_tokens = match signals.consciousness_level {
            c if c >= 0.7 => 64,
            c if c >= 0.5 => 48,
            c if c >= 0.2 => 32,
            _ => 16,
        };

        let result = self.generator.generate(&channels, self.max_tokens);

        if result.text.is_empty() {
            return None;
        }

        // T1.4: Honest coherence from type-token ratio
        let ttr = {
            let words: Vec<&str> = result.text.split_whitespace().collect();
            if words.is_empty() {
                1.0
            } else {
                let unique: std::collections::HashSet<_> = words.iter().collect();
                (unique.len() as f32 / words.len() as f32).clamp(0.1, 1.0)
            }
        };

        Some(LiteGenerationResult {
            text: result.text,
            num_tokens: result.num_tokens,
            eos_terminated: result.eos_terminated,
            coherence: ttr,
        })
    }
}

/// Map the full 20+ field BrocaConsciousnessSignals to Spore's 12-channel ThoughtChannels.
///
/// Channel layout (Spore):
/// - 0..3: intent (curiosity, valence, abstraction, self-reference)
/// - 4: epistemic_status (0=certain .. 1=uncertain)
/// - 5: valence
/// - 6: arousal
/// - 7: consciousness_level
/// - 8: prediction_error (mapped from 1 - epistemic_confidence)
/// - 9: harmony (mapped from coherence)
/// - 10: dopamine (mapped from emotional_warmth)
/// - 11: serotonin (mapped from (1 + valence) / 2)
#[cfg(feature = "broca_lite")]
fn signals_to_lite_channels(
    signals: &BrocaConsciousnessSignals,
) -> symthaea_spore::broca::ThoughtChannels {
    let mut channels = symthaea_spore::broca::ThoughtChannels {
        channels: [0.0; 12],
    };

    // Intent channels (set neutral defaults — could be enriched from detected_primitives)
    channels.channels[0] = (1.0 - signals.epistemic_confidence).clamp(0.1, 0.85); // curiosity from epistemic uncertainty
    channels.channels[1] = ((signals.emotional_valence + 1.0) / 2.0).clamp(0.0, 1.0); // valence [-1,1] → [0,1]
    channels.channels[2] = if signals.knowledge_grounding > 0.5 {
        0.5
    } else {
        0.3
    }; // abstraction
    channels.channels[3] = 0.2; // self-reference (moderate default)

    // Epistemic status: invert confidence → uncertainty
    channels.channels[4] = (1.0 - signals.epistemic_confidence).clamp(0.0, 1.0);

    // Emotional dimensions
    channels.channels[5] = ((signals.emotional_valence + 1.0) / 2.0).clamp(0.0, 1.0);
    channels.channels[6] = signals.emotional_arousal.clamp(0.0, 1.0);

    // Consciousness metrics
    channels.channels[7] = signals.consciousness_level.clamp(0.0, 1.0);
    channels.channels[8] = (1.0 - signals.epistemic_confidence).clamp(0.0, 1.0); // prediction error proxy
    channels.channels[9] = signals.coherence.clamp(0.0, 1.0); // harmony

    // Neuromodulator proxies
    channels.channels[10] = signals.emotional_warmth.clamp(0.0, 1.0); // dopamine proxy
    channels.channels[11] = ((signals.emotional_valence + 1.0) / 2.0).clamp(0.0, 1.0); // serotonin proxy

    // Enrich intent from detected primitives if available
    if !signals.detected_primitives.is_empty() {
        let lower_primes: Vec<String> = signals
            .detected_primitives
            .iter()
            .map(|p| p.to_lowercase())
            .collect();

        // Curiosity boost from question-like primitives
        if lower_primes.iter().any(|p| p == "want" || p == "think") {
            channels.channels[0] = (channels.channels[0] + 0.3).min(1.0);
        }

        // Abstraction boost from meta-cognitive primitives
        if lower_primes.iter().any(|p| p == "know" || p == "cause") {
            channels.channels[2] = (channels.channels[2] + 0.2).min(1.0);
        }
    }

    channels
}

// ── Stub when feature is disabled ────────────────────────────────────────

/// Stub manager when broca_lite feature is not enabled.
/// Always returns None from generate_from_signals.
#[cfg(not(feature = "broca_lite"))]
pub struct BrocaLiteManager;

#[cfg(not(feature = "broca_lite"))]
impl BrocaLiteManager {
    pub fn new(_seed: u64) -> Self {
        Self
    }

    pub fn generate_from_signals(
        &mut self,
        _signals: &BrocaConsciousnessSignals,
    ) -> Option<LiteGenerationResult> {
        None
    }

    pub fn generate_from_signals_with_input(
        &mut self,
        _signals: &BrocaConsciousnessSignals,
        _input_text: Option<&str>,
    ) -> Option<LiteGenerationResult> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signals_mapping_defaults() {
        let signals = BrocaConsciousnessSignals::default();
        // Default signals should not crash the mapping
        #[cfg(feature = "broca_lite")]
        {
            let channels = signals_to_lite_channels(&signals);
            assert_eq!(channels.channels.len(), 12);
            // All values should be in [0, 1]
            for &v in &channels.channels {
                assert!(v >= 0.0 && v <= 1.0, "channel out of range: {}", v);
            }
        }
    }

    #[test]
    fn test_lite_manager_ethics_blocked() {
        let mut manager = BrocaLiteManager::new(42);
        let mut signals = BrocaConsciousnessSignals::default();
        signals.ethics_blocked = true;
        signals.consciousness_level = 0.5;
        let result = manager.generate_from_signals(&signals);
        assert!(result.is_none(), "ethics blocked should prevent generation");
    }

    #[test]
    fn test_lite_manager_low_consciousness() {
        let mut manager = BrocaLiteManager::new(42);
        let mut signals = BrocaConsciousnessSignals::default();
        signals.consciousness_level = 0.01; // Below threshold
        let result = manager.generate_from_signals(&signals);
        assert!(
            result.is_none(),
            "low consciousness should prevent generation"
        );
    }

    #[cfg(feature = "broca_lite")]
    #[test]
    fn test_lite_manager_generates_text() {
        let mut manager = BrocaLiteManager::new(42);
        let mut signals = BrocaConsciousnessSignals::default();
        signals.consciousness_level = 0.5;
        signals.epistemic_confidence = 0.7;
        signals.coherence = 0.6;
        signals.emotional_valence = 0.3;
        signals.emotional_arousal = 0.4;
        signals.emotional_warmth = 0.5;
        let result = manager.generate_from_signals(&signals);
        assert!(
            result.is_some(),
            "should generate text at moderate consciousness"
        );
        let text = result.unwrap().text;
        assert!(!text.is_empty(), "generated text should not be empty");
    }

    #[cfg(feature = "broca_lite")]
    #[test]
    fn test_lite_deterministic() {
        let mut signals = BrocaConsciousnessSignals::default();
        signals.consciousness_level = 0.4;
        signals.epistemic_confidence = 0.6;
        signals.coherence = 0.5;

        let mut m1 = BrocaLiteManager::new(123);
        let r1 = m1.generate_from_signals(&signals);

        let mut m2 = BrocaLiteManager::new(123);
        let r2 = m2.generate_from_signals(&signals);

        assert_eq!(
            r1.as_ref().map(|r| &r.text),
            r2.as_ref().map(|r| &r.text),
            "same seed should produce same output"
        );
    }

    #[cfg(feature = "broca_lite")]
    #[test]
    fn test_lite_output_quality_demo() {
        let mut manager = BrocaLiteManager::new(42);

        // Scenario 1: Calm, coherent, moderate consciousness
        let mut signals = BrocaConsciousnessSignals::default();
        signals.consciousness_level = 0.4;
        signals.epistemic_confidence = 0.8;
        signals.coherence = 0.7;
        signals.emotional_valence = 0.5;
        signals.emotional_arousal = 0.3;
        signals.emotional_warmth = 0.6;
        let r1 = manager.generate_from_signals(&signals).unwrap();
        eprintln!("[Calm/Coherent] {}", r1.text);

        // Scenario 2: High uncertainty, curious
        signals.consciousness_level = 0.3;
        signals.epistemic_confidence = 0.2;
        signals.coherence = 0.4;
        signals.emotional_valence = 0.0;
        signals.emotional_arousal = 0.6;
        signals.detected_primitives = vec!["THINK".into(), "WANT".into()];
        let r2 = manager.generate_from_signals(&signals).unwrap();
        eprintln!("[Uncertain/Curious] {}", r2.text);

        // Scenario 3: High consciousness, positive affect
        signals.consciousness_level = 0.7;
        signals.epistemic_confidence = 0.9;
        signals.coherence = 0.8;
        signals.emotional_valence = 0.8;
        signals.emotional_arousal = 0.5;
        signals.emotional_warmth = 0.9;
        signals.detected_primitives = vec!["KNOW".into(), "CAUSE".into()];
        let r3 = manager.generate_from_signals(&signals).unwrap();
        eprintln!("[High-Psi/Positive] {}", r3.text);

        // Scenario 4: Low consciousness, fragmented
        signals.consciousness_level = 0.1;
        signals.epistemic_confidence = 0.3;
        signals.coherence = 0.2;
        signals.emotional_valence = -0.5;
        signals.emotional_arousal = 0.2;
        let r4 = manager.generate_from_signals(&signals).unwrap();
        eprintln!("[Low-Psi/Fragmented] {}", r4.text);

        // All should produce non-empty text
        assert!(!r1.text.is_empty());
        assert!(!r2.text.is_empty());
        assert!(!r3.text.is_empty());
        assert!(!r4.text.is_empty());
    }

    #[cfg(feature = "broca_lite")]
    #[test]
    fn test_lite_input_awareness() {
        let mut signals = BrocaConsciousnessSignals::default();
        signals.consciousness_level = 0.35;
        signals.epistemic_confidence = 0.6;
        signals.coherence = 0.5;
        signals.emotional_valence = 0.3;

        // Without input — neutral curiosity
        let mut m1 = BrocaLiteManager::new(42);
        let r_neutral = m1.generate_from_signals(&signals).unwrap();
        eprintln!("[No input] {}", r_neutral.text);

        // With question input — should boost curiosity channel
        let mut m2 = BrocaLiteManager::new(42);
        let r_question = m2
            .generate_from_signals_with_input(&signals, Some("why does consciousness emerge?"))
            .unwrap();
        eprintln!("[Question] {}", r_question.text);

        // With positive emotional input
        let mut m3 = BrocaLiteManager::new(42);
        let r_positive = m3
            .generate_from_signals_with_input(&signals, Some("I feel love and hope"))
            .unwrap();
        eprintln!("[Positive] {}", r_positive.text);

        // With self-referential input
        let mut m4 = BrocaLiteManager::new(42);
        let r_self = m4
            .generate_from_signals_with_input(&signals, Some("what are you thinking about?"))
            .unwrap();
        eprintln!("[Self-ref] {}", r_self.text);

        // Input should change the output (different intent channels → different patterns)
        let unique_count = [
            &r_neutral.text,
            &r_question.text,
            &r_positive.text,
            &r_self.text,
        ]
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
        assert!(
            unique_count >= 2,
            "different inputs should produce different outputs, got {} unique from 4",
            unique_count,
        );
    }
}
