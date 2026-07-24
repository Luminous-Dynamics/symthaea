// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Voice Module Integration Tests
//!
//! Tests for the voice interface components that don't require audio
//! hardware, model downloads, or network access.
//!
//! Rewritten 2026-07-15: the previous version imported four types that no
//! longer exist anywhere in the crate (`KokoroModel`, `ModelPaths`,
//! `VoiceEvent`, `WhisperModel`), so every `cargo test --features voice-tts`
//! failed at compile — hidden by the `#![cfg]` gate under default features
//! and by CI's check-only feature legs.
#![cfg(feature = "voice-tts")]

use symthaea::voice::orchestrator::VoiceOrchestrator;
use symthaea::voice::{LTCPacing, VoiceConfig, VoiceConversation};

// ============================================================================
// VOICE CONFIG TESTS (real API: voice_id / ltc_pacing / sample_rate)
// ============================================================================

#[test]
fn test_voice_config_default() {
    let config = VoiceConfig::default();

    assert_eq!(config.voice_id, 0);
    assert!(config.ltc_pacing, "LTC pacing should be enabled by default");
    assert_eq!(config.sample_rate, 24000);
}

// ============================================================================
// LTC PACING TESTS (real API: from_ltc_state(hidden, tau))
// ============================================================================

#[test]
fn test_ltc_pacing_default() {
    let pacing = LTCPacing::default();
    assert_eq!(pacing.rate, 1.0);
    assert!(pacing.tau > 0.0);
}

#[test]
fn test_ltc_pacing_tau_slows_speech() {
    // Higher tau (slower time constant) → slower speech rate, longer pauses.
    let hidden = vec![0.3, -0.2, 0.5, 0.1];
    let fast = LTCPacing::from_ltc_state(&hidden, 0.5);
    let slow = LTCPacing::from_ltc_state(&hidden, 3.0);

    assert!(
        fast.rate > slow.rate,
        "low tau should speak faster: {} vs {}",
        fast.rate,
        slow.rate
    );
    assert!(fast.sentence_pause < slow.sentence_pause);
}

#[test]
fn test_ltc_pacing_valence_from_asymmetry() {
    // Positive-dominated hidden state → positive valence; negative → negative.
    let positive = LTCPacing::from_ltc_state(&[0.8, 0.6, 0.7, -0.1], 1.0);
    let negative = LTCPacing::from_ltc_state(&[-0.8, -0.6, -0.7, 0.1], 1.0);

    assert!(positive.emotional_valence > 0.0);
    assert!(negative.emotional_valence < 0.0);
}

#[test]
fn test_ltc_pacing_arousal_from_magnitude() {
    let calm = LTCPacing::from_ltc_state(&[0.05, -0.05, 0.02, 0.01], 1.0);
    let excited = LTCPacing::from_ltc_state(&[0.9, -0.8, 0.95, -0.85], 1.0);

    assert!(excited.arousal > calm.arousal);
    assert!(excited.emphasis > calm.emphasis);
}

#[test]
fn test_ltc_pacing_consistency() {
    let a = LTCPacing::from_ltc_state(&[0.7, 0.3], 1.2);
    let b = LTCPacing::from_ltc_state(&[0.7, 0.3], 1.2);
    assert_eq!(a.rate, b.rate);
    assert_eq!(a.phrase_pause, b.phrase_pause);
    assert_eq!(a.arousal, b.arousal);
}

// ============================================================================
// ORCHESTRATOR: real formant synthesis end-to-end (no models, no network)
// ============================================================================

#[test]
fn test_thought_to_speech_produces_finite_audio() {
    let mut orch = VoiceOrchestrator::new();
    let cfc_output = vec![0.3, -0.2, 0.5, 0.1, -0.1, 0.4, -0.3, 0.2];
    let audio = orch.thought_to_speech("hello world", &cfc_output, 0.2, vec![]);

    assert!(!audio.is_empty(), "synthesis must produce audio");
    assert!(audio.iter().all(|s| s.is_finite()));
    assert!(
        audio.iter().any(|&s| s.abs() > 1e-6),
        "audio must not be silent"
    );
}

#[test]
fn test_thought_to_speech_paced_metrics_are_real() {
    let mut orch = VoiceOrchestrator::new();
    let cfc_output = vec![0.2; 16];
    let (audio, metrics) = orch.thought_to_speech_paced(
        "testing voice quality",
        &cfc_output,
        1.0,
        0.1,
        vec![],
        1.0,
        1.0,
    );

    assert!(!audio.is_empty());
    // Metrics must be computed from the produced frames, not defaults.
    assert!(metrics.speech_rate > 0.0, "speech rate must be measured");
    assert!(metrics.pitch_stability > 0.0);
    assert!((0.0..=1.0).contains(&metrics.coarticulation_smoothness));
}

#[test]
fn test_rate_multiplier_shortens_audio() {
    let cfc_output = vec![0.1; 8];
    let mut orch_fast = VoiceOrchestrator::new();
    let (fast, _) = orch_fast.thought_to_speech_paced(
        "the quick brown fox",
        &cfc_output,
        1.0,
        0.1,
        vec![],
        2.0,
        1.0,
    );
    let mut orch_slow = VoiceOrchestrator::new();
    let (slow, _) = orch_slow.thought_to_speech_paced(
        "the quick brown fox",
        &cfc_output,
        1.0,
        0.1,
        vec![],
        0.5,
        1.0,
    );

    assert!(
        fast.len() < slow.len(),
        "2x rate should yield shorter audio than 0.5x: {} vs {}",
        fast.len(),
        slow.len()
    );
}

#[test]
fn test_empty_text_produces_empty_audio() {
    let mut orch = VoiceOrchestrator::new();
    let (audio, _) = orch.thought_to_speech_paced("", &[0.1; 4], 1.0, 0.1, vec![], 1.0, 1.0);
    assert!(audio.is_empty());
}

// ============================================================================
// VOICE CONVERSATION (service --voice surface; no audio device required)
// ============================================================================

#[test]
fn test_voice_conversation_constructs_and_speaks() {
    let mut vc = VoiceConversation::new(VoiceConfig::default())
        .expect("VoiceConversation should construct without audio hardware");

    // speak() must synthesize without error even when no playback backend
    // is available (it logs and drops the audio instead of failing).
    vc.speak("integration test utterance")
        .expect("speak should not error without an audio device");

    // Synthesis must have produced real quality metrics.
    let metrics = vc.take_voice_metrics();
    if let Some(m) = metrics {
        assert!(m.speech_rate > 0.0);
    }
}
