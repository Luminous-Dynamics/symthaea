// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Voice Module Integration Tests
//!
//! Tests for the voice interface components that don't require
//! actual audio hardware or TTS/STT features enabled.
#![cfg(feature = "voice-tts")]

use symthaea::voice::{KokoroModel, LTCPacing, ModelPaths, VoiceConfig, VoiceEvent, WhisperModel};

// ============================================================================
// VOICE CONFIG TESTS
// ============================================================================

#[test]
fn test_voice_config_default() {
    let config = VoiceConfig::default();

    assert_eq!(config.whisper_model, WhisperModel::Base);
    assert_eq!(config.kokoro_model, KokoroModel::V019);
    assert!(config.ltc_pacing, "LTC pacing should be enabled by default");
    assert_eq!(config.voice_name, "af_bella");
    assert!(config.language.is_some());
    assert_eq!(config.language.as_deref(), Some("en"));
}

#[test]
fn test_voice_config_custom() {
    let config = VoiceConfig {
        whisper_model: WhisperModel::Large,
        kokoro_model: KokoroModel::V019,
        language: Some("de".to_string()),
        ltc_pacing: false,
        voice_name: "bf_emma".to_string(),
        wake_word: Some("hey symthaea".to_string()),
        silence_threshold: 0.05,
        debug: true,
    };

    assert_eq!(config.whisper_model, WhisperModel::Large);
    assert!(!config.ltc_pacing);
    assert_eq!(config.voice_name, "bf_emma");
    assert_eq!(config.wake_word.as_deref(), Some("hey symthaea"));
    assert!(config.debug);
}

// ============================================================================
// VOICE EVENT TESTS
// ============================================================================

#[test]
fn test_voice_event_variants() {
    // Test all event variants can be constructed
    let events = vec![
        VoiceEvent::Ready,
        VoiceEvent::Listening,
        VoiceEvent::SpeechDetected,
        VoiceEvent::Transcribed {
            text: "Hello".to_string(),
            confidence: 0.95,
        },
        VoiceEvent::Processing,
        VoiceEvent::Response {
            text: "Hi there!".to_string(),
        },
        VoiceEvent::Speaking,
        VoiceEvent::Done,
        VoiceEvent::Error {
            message: "Test error".to_string(),
        },
        VoiceEvent::Stopped,
    ];

    assert_eq!(events.len(), 10, "Should have 10 event variants");
}

#[test]
fn test_voice_event_debug() {
    let event = VoiceEvent::Transcribed {
        text: "Test transcription".to_string(),
        confidence: 0.87,
    };

    let debug_str = format!("{:?}", event);
    assert!(debug_str.contains("Transcribed"));
    assert!(debug_str.contains("Test transcription"));
}

// ============================================================================
// LTC PACING TESTS
// ============================================================================

#[test]
fn test_ltc_pacing_default() {
    let pacing = LTCPacing::default();

    // Default should have neutral speech rate
    assert!(pacing.speech_rate >= 0.8 && pacing.speech_rate <= 1.2);
    assert!(!pacing.peak_flow);
}

#[test]
fn test_ltc_pacing_from_ltc() {
    // Test with high flow state
    let high_flow = LTCPacing::from_ltc(0.9, 0.5);
    assert!(
        high_flow.peak_flow,
        "High flow (0.9) should trigger peak_flow"
    );
    assert!(
        high_flow.speech_rate > 1.0,
        "High flow should increase speech rate"
    );

    // Test with low flow state
    let low_flow = LTCPacing::from_ltc(0.2, -0.3);
    assert!(!low_flow.peak_flow, "Low flow should not be peak");
    assert!(
        low_flow.speech_rate <= 1.0,
        "Low flow should not increase rate"
    );
}

#[test]
fn test_ltc_pacing_pause_duration() {
    // Rising Φ trend should have shorter pauses
    let rising = LTCPacing::from_ltc(0.5, 0.8);
    let falling = LTCPacing::from_ltc(0.5, -0.8);

    assert!(
        rising.pause_ms < falling.pause_ms,
        "Rising Φ ({}) should have shorter pause than falling ({})",
        rising.pause_ms,
        falling.pause_ms
    );
}

// ============================================================================
// MODEL PATHS AND MANAGER TESTS
// ============================================================================

#[test]
fn test_model_paths() {
    let paths = ModelPaths::default();

    // Should have sensible default paths
    assert!(!paths.base_dir.as_os_str().is_empty());
    assert!(!paths.whisper_dir.as_os_str().is_empty());
    assert!(!paths.kokoro_dir.as_os_str().is_empty());
}

#[test]
fn test_whisper_model_variants() {
    let models = [
        WhisperModel::Tiny,
        WhisperModel::Base,
        WhisperModel::Small,
        WhisperModel::Medium,
        WhisperModel::Large,
    ];

    // Each model should have a distinct filename
    let filenames: Vec<&str> = models.iter().map(|m| m.filename()).collect();
    let unique: std::collections::HashSet<&str> = filenames.iter().cloned().collect();
    assert_eq!(
        filenames.len(),
        unique.len(),
        "All model filenames should be unique"
    );
}

#[test]
fn test_whisper_model_sizes() {
    // Larger models should require more memory (using size_bytes)
    assert!(WhisperModel::Tiny.size_bytes() < WhisperModel::Base.size_bytes());
    assert!(WhisperModel::Base.size_bytes() < WhisperModel::Small.size_bytes());
    assert!(WhisperModel::Small.size_bytes() < WhisperModel::Medium.size_bytes());
    assert!(WhisperModel::Medium.size_bytes() < WhisperModel::Large.size_bytes());
}

#[test]
fn test_kokoro_model_variant() {
    let model = KokoroModel::V019;
    assert!(!model.filename().is_empty());

    // Test HuggingFace path
    let (repo, path) = model.hf_path();
    assert!(!repo.is_empty());
    assert!(!path.is_empty());
}

#[test]
fn test_model_paths_whisper_model() {
    let paths = ModelPaths::default();
    let model_path = paths.whisper_model(WhisperModel::Base);

    // Should end with the correct filename
    assert!(model_path.ends_with("ggml-base.bin"));
}

#[test]
fn test_model_paths_kokoro_model() {
    let paths = ModelPaths::default();
    let model_path = paths.kokoro_model(KokoroModel::V019);

    // Should end with the correct filename
    assert!(model_path.ends_with("kokoro-v0_19.onnx"));
}

// ============================================================================
// PROSODY / EXPRESSION TESTS
// ============================================================================

#[test]
fn test_ltc_pacing_speech_rate_bounds() {
    // Test various LTC states don't produce out-of-bounds rates
    let test_cases = [
        (0.0, 0.0),  // Minimum
        (1.0, 1.0),  // Maximum
        (0.5, 0.0),  // Neutral
        (0.8, 0.9),  // High flow, rising phi
        (0.1, -0.9), // Low flow, falling phi
    ];

    for (flow, trend) in test_cases {
        let pacing = LTCPacing::from_ltc(flow, trend);
        assert!(
            pacing.speech_rate >= 0.5 && pacing.speech_rate <= 2.0,
            "Speech rate {} out of bounds for flow={}, trend={}",
            pacing.speech_rate,
            flow,
            trend
        );
    }
}

#[test]
fn test_ltc_pacing_consistency() {
    // Same inputs should produce same outputs
    let pacing1 = LTCPacing::from_ltc(0.7, 0.3);
    let pacing2 = LTCPacing::from_ltc(0.7, 0.3);

    assert_eq!(pacing1.speech_rate, pacing2.speech_rate);
    assert_eq!(pacing1.pause_ms, pacing2.pause_ms);
    assert_eq!(pacing1.peak_flow, pacing2.peak_flow);
}