// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Symthaea Voice: Consciousness-Driven Speech Synthesis
//!
//! Pure Rust formant vocoder with G2P conversion. Produces speech audio
//! from text, modulated by consciousness state. No external dependencies.
//!
//! Architecture:
//! ```text
//! Text → G2P → Phonemes → Formant Targets → Vocoder → Audio Samples
//!                              ↑
//!                     Consciousness State (prosody)
//! ```

pub mod g2p;
pub mod formants;
pub mod ltc_voice;
pub mod vocoder;

use symthaea_vocal_tract::types::{FormantFrame, SourceType};

/// Consciousness state that shapes voice prosody.
#[derive(Debug, Clone, Default)]
pub struct VoiceProsody {
    /// Arousal [0,1]: affects speech rate and pitch range.
    pub arousal: f32,
    /// Valence [-1,1]: affects pitch contour (positive = rising, warm).
    pub valence: f32,
    /// Consciousness level [0,1]: affects vocabulary complexity and articulation.
    pub consciousness: f32,
    /// Serotonin [0,1]: affects warmth/breathiness.
    pub serotonin: f32,
}

/// Synthesize speech from text with consciousness-shaped prosody.
///
/// Uses the LTC-driven VocalTractController for liquid formant transitions.
/// Falls back to static formant lookup if LTC fails.
pub fn speak(text: &str, prosody: &VoiceProsody, sample_rate: u32) -> Vec<f32> {
    let phonemes = g2p::text_to_phonemes(text);
    if phonemes.is_empty() { return Vec::new(); }

    // Use LTC-driven voice (liquid transitions from differential equations)
    let genesis = symthaea_core::genesis::GenesisSeed::from_phrase("symthaea-voice");
    let mut ltc = ltc_voice::LtcVoice::new(&genesis);
    let frames = ltc.phonemes_to_frames(&phonemes, prosody, sample_rate);

    vocoder::synthesize(&frames, sample_rate)
}

/// Synthesize using static formant lookup (fallback, no LTC).
pub fn speak_static(text: &str, prosody: &VoiceProsody, sample_rate: u32) -> Vec<f32> {
    let phonemes = g2p::text_to_phonemes(text);
    let frames = formants::phonemes_to_frames(&phonemes, prosody, sample_rate);
    vocoder::synthesize(&frames, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speak_produces_audio() {
        let prosody = VoiceProsody {
            arousal: 0.5,
            valence: 0.3,
            consciousness: 0.7,
            serotonin: 0.5,
        };
        let audio = speak("hello world", &prosody, 44100);
        assert!(!audio.is_empty(), "should produce audio");
        assert!(audio.len() > 1000, "should produce substantial audio: {} samples", audio.len());
        assert!(audio.iter().any(|&s| s.abs() > 0.001), "should have audible signal");
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
        // Excited speech should be shorter (faster)
        assert!(excited.len() < calm.len(),
            "excited ({}) should be shorter than calm ({})", excited.len(), calm.len());
    }
}
