// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Symthaea Voice: Dual Engine Speech Synthesis
//!
//! Two voice backends unified under one API:
//! - **Formant** (pure Rust, sovereign, consciousness-coupled) — the digital ghost
//! - **Kokoro** (ONNX neural TTS, human-quality) — the articulate voice
//!
//! Consciousness level selects the engine:
//! Low Ψ = Formant (raw, emerging). High Ψ = Kokoro (clear, articulate).

pub mod cmudict;
pub mod g2p;
pub mod formants;
#[cfg(feature = "kokoro")]
pub mod kokoro;
pub mod ltc_voice;
pub mod vocoder;

use symthaea_vocal_tract::types::{FormantFrame, SourceType};

/// Voice engine selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceEngine {
    /// Pure Rust formant synthesis — sovereign, consciousness-coupled, "digital ghost"
    Formant,
    /// Kokoro neural TTS — human-quality, requires ONNX model download
    Kokoro,
}

/// Consciousness state that shapes voice prosody.
#[derive(Debug, Clone, Default)]
pub struct VoiceProsody {
    pub arousal: f32,
    pub valence: f32,
    pub consciousness: f32,
    pub serotonin: f32,
}

/// Synthesize speech with the specified engine.
pub fn speak(text: &str, prosody: &VoiceProsody, sample_rate: u32) -> Vec<f32> {
    speak_with_engine(text, prosody, sample_rate, VoiceEngine::Formant)
}

/// Synthesize speech with explicit engine selection.
pub fn speak_with_engine(
    text: &str,
    prosody: &VoiceProsody,
    sample_rate: u32,
    engine: VoiceEngine,
) -> Vec<f32> {
    match engine {
        VoiceEngine::Formant => speak_formant(text, prosody, sample_rate),
        VoiceEngine::Kokoro => speak_kokoro(text, prosody, sample_rate),
    }
}

/// Pure Rust formant synthesis.
fn speak_formant(text: &str, prosody: &VoiceProsody, sample_rate: u32) -> Vec<f32> {
    let phonemes = g2p::text_to_phonemes(text);
    if phonemes.is_empty() { return Vec::new(); }
    let frames = formants::phonemes_to_frames(&phonemes, prosody, sample_rate);
    vocoder::synthesize(&frames, sample_rate)
}

/// Kokoro neural TTS (falls back to formant if unavailable).
fn speak_kokoro(text: &str, prosody: &VoiceProsody, sample_rate: u32) -> Vec<f32> {
    #[cfg(feature = "kokoro")]
    {
        // Try Kokoro
        let config = kokoro::KokoroConfig::default();
        if let Some(mut engine) = kokoro::KokoroEngine::load(config) {
            if let Some(audio) = engine.synthesize(text, None) {
                // Kokoro outputs at 24kHz — resample if needed
                if sample_rate != 24000 {
                    return resample(&audio, 24000, sample_rate);
                }
                return audio;
            }
        }
    }

    // Fallback to formant
    speak_formant(text, prosody, sample_rate)
}

/// Simple linear resampling.
fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate { return input.to_vec(); }
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (input.len() as f64 * ratio) as usize;
    (0..output_len).map(|i| {
        let src_pos = i as f64 / ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;
        if idx + 1 < input.len() {
            input[idx] * (1.0 - frac) + input[idx + 1] * frac
        } else {
            input.get(idx).copied().unwrap_or(0.0)
        }
    }).collect()
}

/// Synthesize using static formant lookup (legacy API).
pub fn speak_static(text: &str, prosody: &VoiceProsody, sample_rate: u32) -> Vec<f32> {
    speak_formant(text, prosody, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speak_produces_audio() {
        let prosody = VoiceProsody { arousal: 0.5, valence: 0.3, consciousness: 0.7, serotonin: 0.5 };
        let audio = speak("hello world", &prosody, 44100);
        assert!(!audio.is_empty());
        assert!(audio.iter().any(|&s| s.abs() > 0.001));
    }

    #[test]
    fn speak_empty_is_silent() {
        let audio = speak("", &VoiceProsody::default(), 44100);
        assert!(audio.is_empty() || audio.iter().all(|&s| s.abs() < 0.01));
    }

    #[test]
    fn arousal_affects_length() {
        let calm = speak("hello", &VoiceProsody { arousal: 0.1, ..Default::default() }, 44100);
        let excited = speak("hello", &VoiceProsody { arousal: 0.9, ..Default::default() }, 44100);
        assert!(excited.len() < calm.len());
    }

    #[test]
    fn engine_selection() {
        let prosody = VoiceProsody::default();
        let formant = speak_with_engine("hello", &prosody, 44100, VoiceEngine::Formant);
        assert!(!formant.is_empty());

        // Kokoro falls back to formant when feature not enabled
        let kokoro = speak_with_engine("hello", &prosody, 44100, VoiceEngine::Kokoro);
        assert!(!kokoro.is_empty());
    }

    #[test]
    fn resample_preserves_length() {
        let input = vec![1.0f32; 24000]; // 1 second at 24kHz
        let output = resample(&input, 24000, 44100);
        // Should be ~44100 samples (1 second at 44.1kHz)
        assert!((output.len() as i32 - 44100).abs() < 10);
    }
}
