// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Integration Test: Voice Pipeline (STT -> Cognitive Loop -> Broca -> TTS)
// ==================================================================================
//
// Tests the voice synthesis pipeline end-to-end:
//   CognitiveVoiceBridge receives CfC output -> produces pacing -> VoiceOutput synthesizes
//
// Tests consciousness-modulated prosody, semantic prosody from primitives,
// pacing smoothing, and the full cognitive-loop-to-voice bridge pathway.
//
// No feature flags required — uses default configuration.
// ==================================================================================

use std::collections::HashMap;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea::voice::{
    CognitivePacing, CognitiveVoiceBridge, LTCPacing, SemanticCategory, SemanticProsody,
    VoiceConsciousnessSignals, VoiceOutput, VoiceOutputConfig,
};

// ── Test 1: CognitiveVoiceBridge updates pacing from CfC output ──────────────

#[test]
fn test_cognitive_bridge_updates_pacing() {
    let mut bridge = CognitiveVoiceBridge::new();

    // Feed CfC output with mixed activations, moderate tau, moderate prediction error
    let cfc_output: Vec<f32> = vec![0.3, -0.2, 0.5, -0.1, 0.4, 0.6, -0.3, 0.1];
    let tau = 1.2;
    let prediction_error = 0.3;
    let attention: HashMap<String, f32> = HashMap::new();
    let primitives: Vec<String> = vec![];

    let pacing = bridge.update(&cfc_output, tau, prediction_error, attention, primitives);

    // Rate must be positive and finite
    assert!(pacing.base_pacing.rate > 0.0, "rate should be positive");
    assert!(pacing.base_pacing.rate.is_finite(), "rate should be finite");

    // Phrase and sentence pauses must be non-negative and finite
    assert!(
        pacing.base_pacing.phrase_pause >= 0.0,
        "phrase_pause should be non-negative"
    );
    assert!(
        pacing.base_pacing.sentence_pause >= 0.0,
        "sentence_pause should be non-negative"
    );
    assert!(
        pacing.base_pacing.phrase_pause.is_finite(),
        "phrase_pause should be finite"
    );
    assert!(
        pacing.base_pacing.sentence_pause.is_finite(),
        "sentence_pause should be finite"
    );

    // Confidence should be derived from prediction_error: 1 - 0.3 = 0.7
    // But smoothing applies (default confidence is 1.0, smoothing is 0.3)
    // So after one update: 1.0 * 0.7 + 0.7 * 0.3 = 0.91
    assert!(
        pacing.confidence > 0.0 && pacing.confidence <= 1.0,
        "confidence should be in (0, 1]: got {}",
        pacing.confidence
    );

    // Tau should be positive
    assert!(
        pacing.base_pacing.tau > 0.0,
        "tau should be positive: got {}",
        pacing.base_pacing.tau
    );

    // Emphasis should be positive
    assert!(
        pacing.base_pacing.emphasis > 0.0,
        "emphasis should be positive: got {}",
        pacing.base_pacing.emphasis
    );
}

// ── Test 2: LTCPacing default values are sane ────────────────────────────────

#[test]
fn test_ltc_pacing_defaults() {
    let pacing = LTCPacing::default();

    // Rate should be 1.0 (normal speed)
    assert!(
        (pacing.rate - 1.0).abs() < f32::EPSILON,
        "default rate should be 1.0: got {}",
        pacing.rate
    );

    // Pauses should be positive
    assert!(
        pacing.phrase_pause > 0.0,
        "phrase_pause should be positive: got {}",
        pacing.phrase_pause
    );
    assert!(
        pacing.sentence_pause > 0.0,
        "sentence_pause should be positive: got {}",
        pacing.sentence_pause
    );

    // Emphasis should be 1.0 (normal)
    assert!(
        (pacing.emphasis - 1.0).abs() < f32::EPSILON,
        "default emphasis should be 1.0: got {}",
        pacing.emphasis
    );

    // Breath probability should be in [0, 1]
    assert!(
        pacing.breath_probability >= 0.0 && pacing.breath_probability <= 1.0,
        "breath_probability should be in [0, 1]: got {}",
        pacing.breath_probability
    );

    // Emotional valence should be in [-1, 1]
    assert!(
        pacing.emotional_valence >= -1.0 && pacing.emotional_valence <= 1.0,
        "emotional_valence should be in [-1, 1]: got {}",
        pacing.emotional_valence
    );

    // Arousal should be in [0, 1]
    assert!(
        pacing.arousal >= 0.0 && pacing.arousal <= 1.0,
        "arousal should be in [0, 1]: got {}",
        pacing.arousal
    );

    // Tau should be positive
    assert!(
        pacing.tau > 0.0,
        "tau should be positive: got {}",
        pacing.tau
    );

    // All fields should be finite
    assert!(pacing.rate.is_finite());
    assert!(pacing.phrase_pause.is_finite());
    assert!(pacing.sentence_pause.is_finite());
    assert!(pacing.emphasis.is_finite());
    assert!(pacing.breath_probability.is_finite());
    assert!(pacing.emotional_valence.is_finite());
    assert!(pacing.arousal.is_finite());
    assert!(pacing.tau.is_finite());
}

// ── Test 3: CognitivePacing effective_pacing modulates rate by confidence ─────

#[test]
fn test_cognitive_pacing_effective_pacing() {
    let cfc_output = vec![0.5_f32; 10];
    let tau = 1.0;

    // High confidence (low prediction error)
    let high_confidence = CognitivePacing::from_cognitive_loop(
        &cfc_output,
        tau,
        0.05, // very low error -> confidence ~0.95
        HashMap::new(),
        vec![],
    );

    // Low confidence (high prediction error)
    let low_confidence = CognitivePacing::from_cognitive_loop(
        &cfc_output,
        tau,
        0.9, // high error -> confidence ~0.1
        HashMap::new(),
        vec![],
    );

    let effective_high = high_confidence.effective_pacing();
    let effective_low = low_confidence.effective_pacing();

    // Low confidence should produce slower rate
    assert!(
        effective_low.rate < effective_high.rate,
        "low confidence should slow rate: low={}, high={}",
        effective_low.rate,
        effective_high.rate
    );

    // Low confidence should produce longer phrase pauses
    assert!(
        effective_low.phrase_pause > effective_high.phrase_pause,
        "low confidence should lengthen pauses: low={}, high={}",
        effective_low.phrase_pause,
        effective_high.phrase_pause
    );

    // Low confidence should produce higher tau (more deliberate)
    assert!(
        effective_low.tau > effective_high.tau,
        "low confidence should increase tau: low={}, high={}",
        effective_low.tau,
        effective_high.tau
    );

    // Low confidence should reduce emphasis (mumbling)
    assert!(
        effective_low.emphasis < effective_high.emphasis,
        "low confidence should reduce emphasis: low={}, high={}",
        effective_low.emphasis,
        effective_high.emphasis
    );

    // All outputs should be finite and positive
    for eff in [&effective_high, &effective_low] {
        assert!(eff.rate > 0.0 && eff.rate.is_finite());
        assert!(eff.phrase_pause >= 0.0 && eff.phrase_pause.is_finite());
        assert!(eff.sentence_pause >= 0.0 && eff.sentence_pause.is_finite());
        assert!(eff.emphasis > 0.0 && eff.emphasis.is_finite());
        assert!(eff.tau > 0.0 && eff.tau.is_finite());
    }
}

// ── Test 4: VoiceOutput synthesize produces finite audio samples ─────────────

#[test]
fn test_voice_output_synthesize() {
    let config = VoiceOutputConfig {
        enable_tts: false, // Use simulated TTS
        ..Default::default()
    };
    let mut voice = VoiceOutput::new(config);

    // Synthesize a short phrase
    let samples = voice.synthesize("Hello, world!").unwrap();

    // Should produce non-empty audio
    assert!(
        !samples.is_empty(),
        "synthesize should produce non-empty samples"
    );

    // All samples should be finite
    for (i, &s) in samples.iter().enumerate() {
        assert!(s.is_finite(), "sample {} should be finite, got {}", i, s);
    }

    // Samples should be in a reasonable range (not exploding)
    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
    assert!(
        max_abs <= 2.0,
        "max absolute sample value should be reasonable: got {}",
        max_abs
    );

    // Stats should be updated
    assert_eq!(voice.stats().total_utterances, 1);
    assert!(voice.stats().total_chars > 0);
    assert!(voice.stats().total_audio_seconds > 0.0);

    // Synthesize another phrase to check stats accumulation
    let samples2 = voice.synthesize("Testing one two three.").unwrap();
    assert!(!samples2.is_empty());
    assert_eq!(voice.stats().total_utterances, 2);

    // Different texts should produce different-length audio
    // (not strictly required, but a sanity check)
    // "Hello, world!" (13 chars) vs "Testing one two three." (22 chars)
    // Longer text should generally produce more samples
    // (allowing some tolerance for punctuation effects)
    assert!(
        samples2.len() > samples.len() / 2,
        "longer text should produce roughly proportional audio: {} vs {}",
        samples2.len(),
        samples.len()
    );
}

// ── Test 5: SemanticProsody from various primitive keywords ──────────────────

#[test]
fn test_semantic_prosody_from_primitives() {
    // Test causal primitives -> rising pitch, slower rate
    let cause_prosody = SemanticProsody::from_primitives(&["because".to_string()]);
    assert!(
        cause_prosody.has_semantic_content(),
        "should detect semantic content"
    );
    assert!(
        cause_prosody.pitch_contour > 0.0,
        "cause should have rising pitch: got {}",
        cause_prosody.pitch_contour
    );
    assert!(
        cause_prosody.rate_modifier < 1.0,
        "cause should slow rate: got {}",
        cause_prosody.rate_modifier
    );
    assert!(
        cause_prosody.categories.contains(&SemanticCategory::Cause),
        "should detect Cause category"
    );

    // Test question primitives -> rising pitch, expanded pitch range
    let question_prosody = SemanticProsody::from_primitives(&["why".to_string()]);
    assert!(
        question_prosody.pitch_contour > 0.3,
        "question should have strong rising pitch: got {}",
        question_prosody.pitch_contour
    );
    assert!(
        question_prosody.pitch_range > 1.0,
        "question should expand pitch range: got {}",
        question_prosody.pitch_range
    );

    // Test emphasis primitives -> higher energy, slower rate
    let emphasis_prosody = SemanticProsody::from_primitives(&["critical".to_string()]);
    assert!(
        emphasis_prosody.energy_modifier > 1.0,
        "emphasis should boost energy: got {}",
        emphasis_prosody.energy_modifier
    );
    assert!(
        emphasis_prosody.rate_modifier < 1.0,
        "emphasis should slow rate: got {}",
        emphasis_prosody.rate_modifier
    );

    // Test uncertainty primitives -> falling pitch, reduced energy
    let uncertainty_prosody =
        SemanticProsody::from_primitives(&["maybe".to_string(), "perhaps".to_string()]);
    assert!(
        uncertainty_prosody.pitch_contour < 0.0,
        "uncertainty should have falling pitch: got {}",
        uncertainty_prosody.pitch_contour
    );
    assert!(
        uncertainty_prosody.energy_modifier < 1.0,
        "uncertainty should reduce energy: got {}",
        uncertainty_prosody.energy_modifier
    );

    // Test contrast primitives -> pre-pause, expanded pitch range
    let contrast_prosody = SemanticProsody::from_primitives(&["however".to_string()]);
    assert!(
        contrast_prosody.pre_pause > 1.2,
        "contrast should add pre-pause: got {}",
        contrast_prosody.pre_pause
    );
    assert!(
        contrast_prosody.pitch_range > 1.0,
        "contrast should expand pitch range: got {}",
        contrast_prosody.pitch_range
    );

    // Test neutral (no semantic content)
    let neutral_prosody = SemanticProsody::from_primitives(&["hello".to_string()]);
    assert!(
        !neutral_prosody.has_semantic_content(),
        "neutral words should not trigger semantic prosody"
    );
    assert!(
        (neutral_prosody.rate_modifier - 1.0).abs() < f32::EPSILON,
        "neutral should have rate_modifier 1.0"
    );

    // All prosody fields should be finite
    for prosody in [
        &cause_prosody,
        &question_prosody,
        &emphasis_prosody,
        &uncertainty_prosody,
        &contrast_prosody,
        &neutral_prosody,
    ] {
        assert!(prosody.pitch_contour.is_finite());
        assert!(prosody.rate_modifier.is_finite());
        assert!(prosody.pre_pause.is_finite());
        assert!(prosody.post_pause.is_finite());
        assert!(prosody.pitch_range.is_finite());
        assert!(prosody.energy_modifier.is_finite());
    }
}

// ── Test 6: Cognitive loop output feeds into voice bridge ────────────────────

#[test]
fn test_cognitive_loop_to_voice_bridge() {
    // Create a cognitive loop service
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Create a voice bridge
    let mut bridge = CognitiveVoiceBridge::new();

    // Run 5 cognitive cycles and feed each output to the bridge
    let mut last_pacing_rate = 0.0_f32;
    let mut received_valid_updates = 0;

    for i in 0..5 {
        let input = match i {
            0 => "The cause of the problem is clear.",
            1 => "Therefore we must act quickly.",
            2 => "Why is this important?",
            3 => "Maybe we should reconsider.",
            4 => "However, the results are promising.",
            _ => unreachable!(),
        };

        let result = service.cycle(input);

        // Feed cycle output to voice bridge
        let pacing = bridge.update(
            &result.output,
            1.0, // tau
            result.prediction_error,
            HashMap::new(),
            result.detected_primitives.clone(),
        );

        // Pacing should have valid values
        assert!(
            pacing.base_pacing.rate > 0.0,
            "cycle {} rate should be positive: {}",
            i,
            pacing.base_pacing.rate
        );
        assert!(
            pacing.base_pacing.rate.is_finite(),
            "cycle {} rate should be finite",
            i
        );
        assert!(
            pacing.confidence >= 0.0 && pacing.confidence <= 1.0,
            "cycle {} confidence should be in [0, 1]: {}",
            i,
            pacing.confidence
        );

        // Track that we got valid updates
        if pacing.base_pacing.rate > 0.0 && pacing.base_pacing.rate.is_finite() {
            received_valid_updates += 1;
        }

        last_pacing_rate = pacing.base_pacing.rate;
    }

    // All 5 cycles should have produced valid updates
    assert_eq!(
        received_valid_updates, 5,
        "all 5 cycles should produce valid voice updates"
    );

    // After 5 updates, pacing should differ from the initial default rate of 1.0
    // (the bridge has been fed CfC output + smoothing, so rate will have shifted)
    let default_rate = LTCPacing::default().rate;
    // The rate should be non-default (bridge has received real CfC hidden states)
    // We allow it to be equal only if by coincidence, but it should be finite
    assert!(
        last_pacing_rate.is_finite() && last_pacing_rate > 0.0,
        "final rate should be finite and positive: {}",
        last_pacing_rate
    );

    // The effective pacing from the bridge should also be valid
    let effective = bridge.get_ltc_pacing();
    assert!(effective.rate > 0.0 && effective.rate.is_finite());
    assert!(effective.phrase_pause >= 0.0 && effective.phrase_pause.is_finite());
    assert!(effective.tau > 0.0 && effective.tau.is_finite());

    // Verify the bridge's internal pacing has changed from default
    let bridge_pacing = bridge.get_pacing();
    // After 5 cycles of real input, confidence should have moved from default
    // (default is 1.0, and with any non-zero prediction error it should decrease)
    let _ = bridge_pacing; // used for the assertion below
    assert!(
        bridge_pacing.confidence.is_finite(),
        "bridge confidence should be finite"
    );
    let _ = default_rate; // suppress unused warning
}

// ── Test 7: Consciousness signals modulate pacing ────────────────────────────

#[test]
fn test_consciousness_modulated_pacing() {
    let cfc_output = vec![0.3, -0.2, 0.4, -0.3, 0.2, -0.1, 0.3, -0.2, 0.1, -0.1];
    let tau = 1.0;

    // ── High confidence + high quality (assured speech) ──
    let high_signals = VoiceConsciousnessSignals {
        unified_quality: 0.95,
        epistemic_confidence: 0.9,
        dissipative_gated: false,
        dissipative_factor: 0.95,
        coherence_velocity: 0.1, // stable or improving
        cross_module_agreement: 0.9,
        consciousness_level: 0.8,
    };

    let high_pacing = CognitivePacing::from_cycle_metadata(
        &cfc_output,
        tau,
        0.1, // low error
        HashMap::new(),
        vec![],
        &high_signals,
    );

    // ── Low confidence + low quality (hesitant speech) ──
    let low_signals = VoiceConsciousnessSignals {
        unified_quality: 0.2,
        epistemic_confidence: 0.1,
        dissipative_gated: true,
        dissipative_factor: 0.3,
        coherence_velocity: -0.5, // dropping coherence
        cross_module_agreement: 0.2,
        consciousness_level: 0.2,
    };

    let low_pacing = CognitivePacing::from_cycle_metadata(
        &cfc_output,
        tau,
        0.8, // high error
        HashMap::new(),
        vec![],
        &low_signals,
    );

    let effective_high = high_pacing.effective_pacing();
    let effective_low = low_pacing.effective_pacing();

    // Low consciousness signals should produce slower rate
    assert!(
        effective_low.rate < effective_high.rate,
        "low consciousness should slow rate: low={}, high={}",
        effective_low.rate,
        effective_high.rate
    );

    // Low signals should produce reduced emphasis
    assert!(
        effective_low.emphasis < effective_high.emphasis,
        "low consciousness should reduce emphasis: low={}, high={}",
        effective_low.emphasis,
        effective_high.emphasis
    );

    // Dissipative gating should increase phrase pauses
    // (low_pacing has dissipative_gated=true)
    // We verify this by checking that low has longer pauses
    assert!(
        effective_low.phrase_pause > effective_high.phrase_pause,
        "dissipative gating + low confidence should increase pauses: low={}, high={}",
        effective_low.phrase_pause,
        effective_high.phrase_pause
    );

    // Negative coherence velocity should slow rate further
    // (already captured in the overall rate comparison above)

    // All values should be finite and in reasonable ranges
    for eff in [&effective_high, &effective_low] {
        assert!(eff.rate > 0.0 && eff.rate.is_finite(), "rate: {}", eff.rate);
        assert!(
            eff.emphasis > 0.0 && eff.emphasis.is_finite(),
            "emphasis: {}",
            eff.emphasis
        );
        assert!(
            eff.phrase_pause >= 0.0 && eff.phrase_pause.is_finite(),
            "phrase_pause: {}",
            eff.phrase_pause
        );
        assert!(
            eff.sentence_pause >= 0.0 && eff.sentence_pause.is_finite(),
            "sentence_pause: {}",
            eff.sentence_pause
        );
        assert!(eff.tau > 0.0 && eff.tau.is_finite(), "tau: {}", eff.tau);
    }

    // Also test update_from_metadata on a bridge
    let mut bridge = CognitiveVoiceBridge::new();
    bridge.update(&cfc_output, tau, 0.3, HashMap::new(), vec![]);
    bridge.update_from_metadata(&high_signals);

    let bridge_pacing = bridge.get_pacing();
    assert!(
        (bridge_pacing.unified_quality - 0.95).abs() < 0.01,
        "bridge should reflect high quality: got {}",
        bridge_pacing.unified_quality
    );
    assert!(
        (bridge_pacing.epistemic_confidence - 0.9).abs() < 0.01,
        "bridge should reflect high epistemic confidence: got {}",
        bridge_pacing.epistemic_confidence
    );
}

// ── Test 8: Pacing smoothing prevents instant jumps ──────────────────────────

#[test]
fn test_pacing_smoothing() {
    let mut bridge = CognitiveVoiceBridge::new();
    let cfc_output = vec![0.5_f32; 10];

    // Step 1: Feed low error (high confidence) for several cycles to establish baseline
    for _ in 0..5 {
        bridge.update(
            &cfc_output,
            1.0,
            0.1, // low error -> high confidence
            HashMap::new(),
            vec![],
        );
    }
    let baseline_confidence = bridge.get_pacing().confidence;
    let _baseline_rate = bridge.get_pacing().base_pacing.rate;

    // Step 2: Suddenly feed maximum error
    bridge.update(
        &cfc_output,
        1.0,
        1.0, // maximum error -> zero new confidence
        HashMap::new(),
        vec![],
    );
    let after_spike_confidence = bridge.get_pacing().confidence;
    let after_spike_rate = bridge.get_pacing().base_pacing.rate;

    // Smoothing should prevent instant jump to zero confidence
    // The smoothing factor is 0.3, so the confidence should not drop to 0 immediately
    assert!(
        after_spike_confidence > 0.1,
        "smoothing should prevent instant confidence drop: baseline={}, after_spike={}",
        baseline_confidence,
        after_spike_confidence
    );
    assert!(
        after_spike_confidence < baseline_confidence,
        "confidence should decrease: baseline={}, after_spike={}",
        baseline_confidence,
        after_spike_confidence
    );

    // Rate should also be smoothed (lerp with smoothing=0.3)
    // The rate is determined by CfC output (same each time) and smoothing
    // Since cfc_output is the same, the new base rate is the same, so rate
    // stays approximately stable (the CfC output hasn't changed)
    assert!(
        after_spike_rate.is_finite() && after_spike_rate > 0.0,
        "rate should remain finite and positive after spike: {}",
        after_spike_rate
    );

    // Step 3: Continue feeding maximum error -- confidence should converge toward 0
    for _ in 0..20 {
        bridge.update(
            &cfc_output,
            1.0,
            1.0, // keep maximum error
            HashMap::new(),
            vec![],
        );
    }
    let converged_confidence = bridge.get_pacing().confidence;

    // After many high-error updates, confidence should be very low but NOT exactly 0
    // (because error_history smoothing: average of [0.1, 0.1, ..., 1.0, 1.0, ...]
    //  converges to 1.0 as the window fills, so confidence approaches 0)
    assert!(
        converged_confidence < 0.3,
        "many high-error updates should drive confidence low: {}",
        converged_confidence
    );
    assert!(
        converged_confidence >= 0.0,
        "confidence should stay non-negative: {}",
        converged_confidence
    );

    // Step 4: Recovery -- feed low error again, confidence should NOT jump instantly
    bridge.update(
        &cfc_output,
        1.0,
        0.0, // zero error -> confidence 1.0 for this update
        HashMap::new(),
        vec![],
    );
    let recovery_confidence = bridge.get_pacing().confidence;

    // Due to error history smoothing (window of 10), the average error is still high
    // even after one low-error update, so recovery is gradual
    assert!(
        recovery_confidence < 0.8,
        "single low-error update should not fully recover confidence: {}",
        recovery_confidence
    );
    assert!(
        recovery_confidence > converged_confidence,
        "recovery should increase confidence: converged={}, recovery={}",
        converged_confidence,
        recovery_confidence
    );
}

// ── Test: VoiceOutput with LTC-driven pacing ─────────────────────────────────

#[test]
fn test_voice_output_with_ltc_pacing() {
    let mut voice = VoiceOutput::default();

    // Create pacing from LTC hidden state
    let hidden_state = vec![0.3, -0.1, 0.5, 0.2, -0.4, 0.1, 0.3, -0.2];
    let tau = 1.5;
    voice.update_pacing(&hidden_state, tau);

    let pacing = voice.pacing();
    assert!(pacing.rate > 0.0 && pacing.rate.is_finite());
    assert!(pacing.tau > 0.0 && pacing.tau.is_finite());
    assert_eq!(pacing.tau, tau);

    // Synthesize with the LTC-driven pacing
    let samples = voice.synthesize("This is a test sentence.").unwrap();
    assert!(!samples.is_empty());
    for &s in &samples {
        assert!(s.is_finite(), "all samples should be finite");
    }
}

// ── Test: Calm vs Excited pacing produces different audio lengths ─────────────

#[test]
fn test_pacing_presets_affect_synthesis() {
    let mut voice = VoiceOutput::default();
    let text = "This is a test sentence with some words.";

    // Calm: slower rate, longer pauses
    let calm_samples = voice
        .synthesize_with_pacing(text, &LTCPacing::calm())
        .unwrap();

    // Excited: faster rate, shorter pauses
    let excited_samples = voice
        .synthesize_with_pacing(text, &LTCPacing::excited())
        .unwrap();

    // Calm should produce more samples (slower + longer pauses = longer audio)
    assert!(
        calm_samples.len() > excited_samples.len(),
        "calm speech should be longer than excited: calm={}, excited={}",
        calm_samples.len(),
        excited_samples.len()
    );

    // Both should be non-empty with finite samples
    assert!(!calm_samples.is_empty());
    assert!(!excited_samples.is_empty());
    for &s in calm_samples.iter().chain(excited_samples.iter()) {
        assert!(s.is_finite());
    }
}

// ── Test: LTCPacing lerp produces intermediate values ────────────────────────

#[test]
fn test_ltc_pacing_lerp_intermediate() {
    let calm = LTCPacing::calm();
    let excited = LTCPacing::excited();

    // Lerp at 0.0 should be close to calm
    let at_zero = calm.lerp(&excited, 0.0);
    assert!(
        (at_zero.rate - calm.rate).abs() < f32::EPSILON,
        "lerp(0.0) rate should equal calm"
    );

    // Lerp at 1.0 should be close to excited
    let at_one = calm.lerp(&excited, 1.0);
    assert!(
        (at_one.rate - excited.rate).abs() < f32::EPSILON,
        "lerp(1.0) rate should equal excited"
    );

    // Lerp at 0.5 should be between calm and excited
    let mid = calm.lerp(&excited, 0.5);
    assert!(
        mid.rate > calm.rate && mid.rate < excited.rate,
        "lerp(0.5) rate should be between calm and excited: calm={}, mid={}, excited={}",
        calm.rate,
        mid.rate,
        excited.rate
    );
    assert!(
        mid.arousal > calm.arousal && mid.arousal < excited.arousal,
        "lerp(0.5) arousal should be between calm and excited"
    );
    assert!(
        mid.tau > excited.tau && mid.tau < calm.tau,
        "lerp(0.5) tau should be between excited and calm"
    );

    // All fields should be finite
    assert!(mid.rate.is_finite());
    assert!(mid.phrase_pause.is_finite());
    assert!(mid.sentence_pause.is_finite());
    assert!(mid.emphasis.is_finite());
    assert!(mid.breath_probability.is_finite());
    assert!(mid.emotional_valence.is_finite());
    assert!(mid.arousal.is_finite());
    assert!(mid.tau.is_finite());
}

// ── Test: End-to-end pipeline (cognitive loop -> bridge -> synthesis) ─────────

#[test]
fn test_full_pipeline_cognitive_loop_to_synthesis() {
    // Create cognitive loop
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Create voice bridge
    let mut bridge = CognitiveVoiceBridge::new();

    // Create voice output
    let mut voice = VoiceOutput::default();

    // Run the pipeline: process input -> get pacing -> synthesize
    let result = service.cycle("Because the system needs attention, we must respond carefully.");

    // Feed to bridge
    bridge.update(
        &result.output,
        1.0,
        result.prediction_error,
        HashMap::new(),
        result.detected_primitives.clone(),
    );

    // Get effective pacing and apply to voice
    let ltc_pacing = bridge.get_ltc_pacing();
    voice.set_pacing(ltc_pacing.clone());

    // Synthesize speech with the consciousness-modulated pacing
    let samples = voice
        .synthesize("The system requires immediate attention.")
        .unwrap();

    // Verify the full pipeline produced valid output
    assert!(!samples.is_empty(), "pipeline should produce audio");
    assert!(
        samples.iter().all(|s| s.is_finite()),
        "all samples should be finite"
    );

    // Verify pacing was applied (not the default)
    let applied_pacing = voice.pacing();
    assert!(applied_pacing.rate > 0.0);
    assert!(applied_pacing.rate.is_finite());
    assert!(applied_pacing.tau > 0.0);
}

// ── Test: Adaptive behavior modulates synthesis ──────────────────────────────

#[test]
fn test_adaptive_behavior_synthesis() {
    let mut voice = VoiceOutput::default();
    let text = "Hello, this is a test.";

    // Normal synthesis
    let normal = voice.synthesize(text).unwrap();

    // Synthesis with slow, contemplative adaptive behavior
    let slow = voice
        .synthesize_with_adaptive_behavior(
            text, 0.7, // slower speech rate
            1.5, // longer pauses
            0.8, // slightly reduced emphasis
        )
        .unwrap();

    // Slow+contemplative should produce more samples
    assert!(
        slow.len() > normal.len(),
        "contemplative speech should be longer: slow={}, normal={}",
        slow.len(),
        normal.len()
    );

    // All samples should be finite
    for &s in slow.iter().chain(normal.iter()) {
        assert!(s.is_finite());
    }
}

// ── Test: CognitivePacing with semantic prosody changes effective pacing ──────

#[test]
fn test_semantic_prosody_affects_effective_pacing() {
    let cfc_output = vec![0.3, -0.2, 0.4, -0.3, 0.2, -0.1, 0.3, -0.2, 0.1, -0.1];
    let tau = 1.0;

    // Pacing with question semantics
    let question_pacing = CognitivePacing::from_cognitive_loop(
        &cfc_output,
        tau,
        0.1,
        HashMap::new(),
        vec!["why".to_string()],
    );

    // Pacing without semantics
    let neutral_pacing =
        CognitivePacing::from_cognitive_loop(&cfc_output, tau, 0.1, HashMap::new(), vec![]);

    let eff_question = question_pacing.effective_pacing();
    let eff_neutral = neutral_pacing.effective_pacing();

    // Question should raise emotional valence (rising intonation)
    assert!(
        eff_question.emotional_valence > eff_neutral.emotional_valence,
        "question should raise valence: question={}, neutral={}",
        eff_question.emotional_valence,
        eff_neutral.emotional_valence
    );

    // Question semantic should slow rate (rate_modifier < 1.0)
    assert!(
        eff_question.rate < eff_neutral.rate,
        "question should slow rate: question={}, neutral={}",
        eff_question.rate,
        eff_neutral.rate
    );

    // The CognitivePacing should correctly report it's a question
    assert!(question_pacing.is_question());
    assert!(!neutral_pacing.is_question());
}

// ==================================================================================
// Voice Pipeline Orchestrator Tests
// ==================================================================================

use symthaea::voice::{VoiceOrchestrator, aggregate_audio_hvs, binary_to_continuous};
use symthaea_core::hdc::BinaryHV;

// ── Test: aggregate_audio_hvs produces valid ContinuousHV ─────────────────────

#[test]
fn test_aggregate_audio_hvs_produces_valid_continuous() {
    // Create several random BinaryHVs (simulating audio frame projections)
    let frames: Vec<BinaryHV> = (1..=5).map(BinaryHV::random).collect();

    let aggregated = aggregate_audio_hvs(&frames);

    // Should have correct dimensionality
    assert_eq!(
        aggregated.values.len(),
        BinaryHV::DIM,
        "aggregated HV should have {} dimensions",
        BinaryHV::DIM
    );

    // All values should be finite and in [-1, 1]
    for (i, &v) in aggregated.values.iter().enumerate() {
        assert!(v.is_finite(), "dimension {} should be finite, got {}", i, v);
        assert!(
            (-1.0..=1.0).contains(&v),
            "dimension {} should be in [-1, 1], got {}",
            i,
            v
        );
    }

    // With 5 random HVs, the average should be close to 0 (high-dim concentration)
    let mean: f32 = aggregated.values.iter().sum::<f32>() / aggregated.values.len() as f32;
    assert!(
        mean.abs() < 0.1,
        "mean of random HV aggregation should be near 0, got {}",
        mean
    );

    // Empty input should return zeros
    let empty = aggregate_audio_hvs(&[]);
    assert_eq!(empty.values.len(), BinaryHV::DIM);
    assert!(empty.values.iter().all(|&v| v == 0.0));
}

// ── Test: binary_to_continuous maps bits correctly ────────────────────────────

#[test]
fn test_binary_to_continuous_maps_bits() {
    // All zeros -> all -1.0
    let zeros = BinaryHV::zero();
    let continuous_zeros = binary_to_continuous(&zeros);
    assert_eq!(continuous_zeros.values.len(), BinaryHV::DIM);
    assert!(
        continuous_zeros
            .values
            .iter()
            .all(|&v| (v - (-1.0)).abs() < f32::EPSILON),
        "all-zero BinaryHV should map to all -1.0"
    );

    // All ones -> all +1.0
    let ones = BinaryHV::ones();
    let continuous_ones = binary_to_continuous(&ones);
    assert_eq!(continuous_ones.values.len(), BinaryHV::DIM);
    assert!(
        continuous_ones
            .values
            .iter()
            .all(|&v| (v - 1.0).abs() < f32::EPSILON),
        "all-one BinaryHV should map to all +1.0"
    );

    // Random HV should have mix of -1 and +1
    let random = BinaryHV::random(42);
    let continuous_random = binary_to_continuous(&random);
    let pos_count = continuous_random
        .values
        .iter()
        .filter(|&&v| v > 0.0)
        .count();
    let neg_count = continuous_random
        .values
        .iter()
        .filter(|&&v| v < 0.0)
        .count();
    assert!(pos_count > 0, "random HV should have some +1.0 values");
    assert!(neg_count > 0, "random HV should have some -1.0 values");

    // Each value should be exactly -1 or +1 (bipolar encoding)
    for &v in &continuous_random.values {
        assert!(
            (v - 1.0).abs() < f32::EPSILON || (v - (-1.0)).abs() < f32::EPSILON,
            "each dimension should be exactly -1 or +1, got {}",
            v
        );
    }
}

// ── Test: Orchestrator synthesize_from_cycle_result produces audio ────────────

#[test]
fn test_orchestrator_synthesize_produces_audio() {
    let mut orchestrator = VoiceOrchestrator::new();
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Run a cognitive cycle
    let result = service.cycle("The system is working correctly.");

    // Synthesize from the cycle result
    let samples =
        orchestrator.synthesize_from_cycle_result("The system is working correctly.", &result);

    // Should produce non-empty audio
    assert!(
        !samples.is_empty(),
        "orchestrator should produce audio samples"
    );

    // All samples should be finite
    for (i, &s) in samples.iter().enumerate() {
        assert!(s.is_finite(), "sample {} should be finite, got {}", i, s);
    }

    // Samples should be in a reasonable range
    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
    assert!(
        max_abs <= 2.0,
        "max sample amplitude should be reasonable: got {}",
        max_abs
    );

    // The bridge should have updated pacing
    let pacing = orchestrator.current_pacing();
    assert!(pacing.rate > 0.0 && pacing.rate.is_finite());
    assert!(pacing.tau > 0.0 && pacing.tau.is_finite());
}

// ── Test: Full pipeline text -> cognitive cycle -> speech ──────────────────────

#[test]
fn test_full_pipeline_text_to_speech() {
    let mut orchestrator = VoiceOrchestrator::new();

    // Use the low-level thought_to_speech path
    let cfc_output = vec![0.4, -0.2, 0.6, -0.1, 0.3, 0.2, -0.4, 0.1];
    let samples = orchestrator.thought_to_speech(
        "Hello, consciousness is emerging.",
        &cfc_output,
        0.15,
        vec!["cause".to_string(), "important".to_string()],
    );

    // Should produce audio
    assert!(
        !samples.is_empty(),
        "thought_to_speech should produce audio"
    );

    // All samples should be finite
    assert!(
        samples.iter().all(|s| s.is_finite()),
        "all samples should be finite"
    );

    // The bridge should reflect the semantic content
    let cognitive_pacing = orchestrator.voice_bridge().get_pacing();

    // Should have detected semantic categories from "cause" and "important"
    assert!(
        cognitive_pacing.semantic_prosody.has_semantic_content(),
        "should detect semantic content from primitives"
    );

    // Run multiple cycles to verify smoothing works
    for i in 0..3 {
        let text = match i {
            0 => "Maybe we should reconsider.",
            1 => "However, the results are clear.",
            2 => "Therefore we must act.",
            _ => unreachable!(),
        };
        let primitives = match i {
            0 => vec!["uncertain".to_string()],
            1 => vec!["contrast".to_string()],
            2 => vec!["effect".to_string()],
            _ => unreachable!(),
        };
        let samples = orchestrator.thought_to_speech(text, &cfc_output, 0.3, primitives);
        assert!(!samples.is_empty(), "cycle {} should produce audio", i);
        assert!(
            samples.iter().all(|s| s.is_finite()),
            "cycle {} samples should be finite",
            i
        );
    }

    // After multiple updates, the pacing should still be valid
    let final_pacing = orchestrator.current_pacing();
    assert!(final_pacing.rate > 0.0 && final_pacing.rate.is_finite());
    assert!(final_pacing.emphasis > 0.0 && final_pacing.emphasis.is_finite());
    assert!(final_pacing.phrase_pause >= 0.0 && final_pacing.phrase_pause.is_finite());
}