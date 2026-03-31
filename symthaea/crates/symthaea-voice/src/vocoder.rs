// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Formant vocoder: source-filter synthesis from formant frames.
//!
//! Converts FormantFrame sequences into audio samples using:
//! 1. Glottal pulse source (voiced) + noise source (unvoiced)
//! 2. Three resonant bandpass filters (F1, F2, F3)
//!
//! This produces the characteristic "Hawking-esque" formant synthesis
//! that is Symthaea's authentic voice.

use symthaea_vocal_tract::types::{FormantFrame, SourceType};

/// Synthesize audio from formant frames.
pub fn synthesize(frames: &[FormantFrame], sample_rate: u32) -> Vec<f32> {
    if frames.is_empty() { return Vec::new(); }

    let sr = sample_rate as f32;
    let frame_rate = 200.0; // frames per second
    let samples_per_frame = (sr / frame_rate) as usize;

    let mut output = Vec::with_capacity(frames.len() * samples_per_frame);
    let mut glottal_phase = 0.0f32;
    let mut noise_state = 42u32;

    // Resonator states (F1, F2, F3)
    let mut res = [[0.0f32; 2]; 3];

    for frame in frames {
        let f0 = frame.f0.max(50.0);
        let energy = frame.energy;
        let voicing = frame.voicing;

        let formants = [(frame.f1, frame.b1), (frame.f2, frame.b2), (frame.f3, frame.b3)];

        for _ in 0..samples_per_frame {
            // ── Source ──
            let mut source = 0.0f32;

            match frame.source_type {
                SourceType::Vowel | SourceType::Liquid | SourceType::Nasal => {
                    // Glottal pulse (simplified Rosenberg model)
                    glottal_phase += f0 / sr;
                    if glottal_phase >= 1.0 { glottal_phase -= 1.0; }

                    let pulse = if glottal_phase < 0.4 {
                        // Open phase: half-sine
                        (glottal_phase / 0.4 * std::f32::consts::PI).sin()
                    } else if glottal_phase < 0.5 {
                        // Closing phase: fast decay
                        let t = (glottal_phase - 0.4) / 0.1;
                        (1.0 - t) * (1.0 - t)
                    } else {
                        0.0 // closed phase
                    };

                    source = pulse * voicing * energy;

                    // Add aspiration noise during open phase
                    if glottal_phase < 0.4 {
                        noise_state = noise_state.wrapping_mul(1664525).wrapping_add(1013904223);
                        let noise = (noise_state >> 16) as f32 / 32768.0 - 1.0;
                        source += noise * 0.05 * energy; // subtle breathiness
                    }
                }
                SourceType::Fricative => {
                    // Shaped noise source
                    noise_state = noise_state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let noise = (noise_state >> 16) as f32 / 32768.0 - 1.0;
                    source = noise * energy * 0.5;

                    // Add voicing bar for voiced fricatives
                    if voicing > 0.3 {
                        glottal_phase += f0 / sr;
                        if glottal_phase >= 1.0 { glottal_phase -= 1.0; }
                        let pulse = (glottal_phase * std::f32::consts::TAU).sin();
                        source += pulse * voicing * energy * 0.3;
                    }
                }
                SourceType::Stop => {
                    // Silence (closure) with brief burst
                    source = 0.0;
                }
                SourceType::Affricate => {
                    // Brief stop + fricative
                    noise_state = noise_state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let noise = (noise_state >> 16) as f32 / 32768.0 - 1.0;
                    source = noise * energy * 0.3;
                }
                SourceType::Silent => {
                    source = 0.0;
                }
            }

            // ── Filter: cascade of 3 resonators ──
            let mut filtered = source;
            for (i, &(freq, bw)) in formants.iter().enumerate() {
                if freq < 10.0 { continue; } // skip inactive formants
                filtered = resonate(&mut res[i], filtered, freq, bw, sr);
            }

            output.push(filtered * 0.3); // master volume
        }
    }

    output
}

/// Second-order resonator (digital formant filter).
///
/// Implementation: two-pole IIR filter with center frequency and bandwidth.
fn resonate(state: &mut [f32; 2], input: f32, freq: f32, bandwidth: f32, sr: f32) -> f32 {
    let omega = std::f32::consts::TAU * freq / sr;
    let r = (-std::f32::consts::PI * bandwidth / sr).exp();
    let a1 = -2.0 * r * omega.cos();
    let a2 = r * r;

    let output = input - a1 * state[0] - a2 * state[1];
    state[1] = state[0];
    state[0] = output;

    output * (1.0 - r) // normalize gain
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_frame(f1: f32, f2: f32, f0: f32, energy: f32) -> FormantFrame {
        FormantFrame {
            f1, f2, f3: 2500.0, b1: 60.0, b2: 80.0, b3: 100.0,
            f0, energy, voicing: 0.9, time: 0.0,
            source_type: SourceType::Vowel,
            nasal_zero_freq: 0.0, nasal_zero_bw: 0.0,
        }
    }

    #[test]
    fn synthesize_produces_audio() {
        let frames = vec![test_frame(730.0, 1090.0, 120.0, 0.8); 100];
        let audio = synthesize(&frames, 44100);
        assert!(!audio.is_empty());
        assert!(audio.iter().any(|&s| s.abs() > 0.001), "should produce audible signal");
    }

    #[test]
    fn silence_is_silent() {
        let frames = vec![FormantFrame {
            f1: 0.0, f2: 0.0, f3: 0.0, b1: 60.0, b2: 80.0, b3: 100.0,
            f0: 0.0, energy: 0.0, voicing: 0.0, time: 0.0,
            source_type: SourceType::Silent,
            nasal_zero_freq: 0.0, nasal_zero_bw: 0.0,
        }; 50];
        let audio = synthesize(&frames, 44100);
        assert!(audio.iter().all(|&s| s.abs() < 0.01), "silence should be silent");
    }

    #[test]
    fn different_vowels_sound_different() {
        // /ɑ/ (as in "father") vs /iː/ (as in "see")
        let ah = synthesize(&vec![test_frame(730.0, 1090.0, 120.0, 0.8); 50], 44100);
        let ee = synthesize(&vec![test_frame(270.0, 2290.0, 120.0, 0.8); 50], 44100);

        // They should have different spectral content
        let ah_energy: f32 = ah.iter().map(|s| s * s).sum();
        let ee_energy: f32 = ee.iter().map(|s| s * s).sum();
        assert!((ah_energy - ee_energy).abs() > 0.001, "different vowels should differ");
    }
}
