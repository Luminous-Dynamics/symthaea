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
pub mod formants;
pub mod g2p;
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
    if phonemes.is_empty() {
        return Vec::new();
    }
    let frames = formants::phonemes_to_frames(&phonemes, prosody, sample_rate);
    vocoder::synthesize(&frames, sample_rate)
}

/// Kokoro neural TTS (falls back to formant if unavailable).
/// Caches the engine across calls for performance.
fn speak_kokoro(text: &str, prosody: &VoiceProsody, sample_rate: u32) -> Vec<f32> {
    #[cfg(feature = "kokoro")]
    {
        use std::sync::Mutex;
        use std::sync::OnceLock;

        static KOKORO: OnceLock<Mutex<Option<kokoro::KokoroEngine>>> = OnceLock::new();

        let engine_lock = KOKORO.get_or_init(|| {
            let config = kokoro::KokoroConfig::default();
            Mutex::new(kokoro::KokoroEngine::load(config))
        });

        if let Ok(mut guard) = engine_lock.lock() {
            if let Some(ref mut engine) = *guard {
                // LIQUID KOKORO: consciousness → speech rate + voice character
                // Speed: confused/low Ψ = slow (0.75x), confident/high Ψ = faster (1.1x)
                let speed = 0.75 + prosody.consciousness * 0.35;
                engine.speed = Some(speed);
                // Voice blend: valence + arousal shift voice embedding
                engine.voice_blend = Some((prosody.valence, prosody.arousal));

                if let Some(mut audio) = engine.synthesize(text, None) {
                    // Vocal chain: normalize → presence EQ → compress
                    vocal_chain(&mut audio);

                    if sample_rate != 24000 {
                        return resample(&audio, 24000, sample_rate);
                    }
                    return audio;
                }
            }
        }
    }

    speak_formant(text, prosody, sample_rate)
}

/// Vocal processing chain: normalize → presence EQ → compress.
/// Makes the neural voice crisp, loud, and present.
fn vocal_chain(audio: &mut Vec<f32>) {
    if audio.is_empty() {
        return;
    }

    // 0. Fade-in first 10ms to eliminate boundary clicks
    let fade_in_samples = (24000.0 * 0.01) as usize; // 10ms at 24kHz
    for i in 0..fade_in_samples.min(audio.len()) {
        audio[i] *= i as f32 / fade_in_samples as f32;
    }

    // 1. Peak normalize to -1 dBFS
    let peak = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.001 {
        let target = 0.89; // -1 dBFS
        let gain = target / peak;
        for s in audio.iter_mut() {
            *s *= gain;
        }
    }

    // 2. Presence EQ: high-shelf boost above 4kHz (+4dB)
    // Simple one-pole high-shelf at 24kHz sample rate
    let mut hp_state = 0.0f32;
    let shelf_coeff = 0.7; // cutoff ~4kHz at 24kHz SR
    let shelf_gain = 1.6; // +4dB
    for s in audio.iter_mut() {
        hp_state += shelf_coeff * (*s - hp_state);
        let high = *s - hp_state; // high-frequency component
        *s += high * (shelf_gain - 1.0); // boost highs
    }

    // 3. Soft-knee compressor: 4:1 ratio above -12dBFS
    let threshold = 0.25; // -12 dBFS
    let ratio = 4.0;
    let mut env = 0.0f32;
    let attack = 0.001; // fast attack
    let release = 0.02; // moderate release
    for s in audio.iter_mut() {
        let level = s.abs();
        // Envelope follower
        if level > env {
            env += attack * (level - env);
        } else {
            env += release * (level - env);
        }

        // Gain reduction
        if env > threshold {
            let over = env - threshold;
            let compressed = threshold + over / ratio;
            let gain = compressed / env.max(0.0001);
            *s *= gain;
        }
    }

    // 4. Make-up gain
    let post_peak = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if post_peak > 0.001 && post_peak < 0.85 {
        let makeup = 0.85 / post_peak;
        for s in audio.iter_mut() {
            *s *= makeup;
        }
    }

    // 5. Tail padding: 300ms silence so words don't clip at the end
    let sr = 24000; // Kokoro's native rate
    let tail_samples = (sr as f32 * 0.3) as usize; // 300ms
    audio.extend(vec![0.0f32; tail_samples]);

    // 6. Room reverb: simple comb filter (10% wet) for spatial presence
    let delay_samples = (sr as f32 * 0.035) as usize; // 35ms early reflection
    let mut reverb_buf = vec![0.0f32; delay_samples.max(1)];
    let mut idx = 0usize;
    let feedback = 0.25;
    let wet = 0.12; // subtle room
    for s in audio.iter_mut() {
        let delayed = reverb_buf[idx % delay_samples];
        reverb_buf[idx % delay_samples] = *s + delayed * feedback;
        idx += 1;
        *s = *s * (1.0 - wet) + delayed * wet;
    }

    // 7. Fade out last 50ms to prevent any click at very end
    let fade_samples = (sr as f32 * 0.05) as usize;
    let start = audio.len().saturating_sub(fade_samples);
    for i in start..audio.len() {
        let t = (audio.len() - i) as f32 / fade_samples as f32;
        audio[i] *= t;
    }
}

/// Simple linear resampling.
fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return input.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (input.len() as f64 * ratio) as usize;
    (0..output_len)
        .map(|i| {
            let src_pos = i as f64 / ratio;
            let idx = src_pos as usize;
            let frac = (src_pos - idx as f64) as f32;
            if idx + 1 < input.len() {
                input[idx] * (1.0 - frac) + input[idx + 1] * frac
            } else {
                input.get(idx).copied().unwrap_or(0.0)
            }
        })
        .collect()
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
        let prosody = VoiceProsody {
            arousal: 0.5,
            valence: 0.3,
            consciousness: 0.7,
            serotonin: 0.5,
        };
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
        let calm = speak(
            "hello",
            &VoiceProsody {
                arousal: 0.1,
                ..Default::default()
            },
            44100,
        );
        let excited = speak(
            "hello",
            &VoiceProsody {
                arousal: 0.9,
                ..Default::default()
            },
            44100,
        );
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
