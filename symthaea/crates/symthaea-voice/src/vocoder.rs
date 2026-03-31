// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Formant vocoder with coarticulation and natural glottal source.
//!
//! Fixes from feedback:
//! 1. Coarticulation: formants interpolate smoothly between phonemes
//! 2. Natural source: jitter + shimmer + aspiration noise on glottal pulse
//! 3. Click-free: sample-level parameter smoothing

use symthaea_vocal_tract::types::{FormantFrame, SourceType};

/// Synthesize audio from formant frames with smooth coarticulation.
pub fn synthesize(frames: &[FormantFrame], sample_rate: u32) -> Vec<f32> {
    if frames.is_empty() { return Vec::new(); }

    let sr = sample_rate as f32;
    let frame_rate = 200.0;
    let samples_per_frame = (sr / frame_rate) as usize;

    let mut output = Vec::with_capacity(frames.len() * samples_per_frame);
    let mut glottal_phase = 0.0f32;
    let mut noise_state = 42u32;

    // Resonator states
    let mut res = [[0.0f32; 2]; 3];

    // Smoothed parameters (for coarticulation — no clicks)
    let mut smooth_f1 = frames[0].f1;
    let mut smooth_f2 = frames[0].f2;
    let mut smooth_f3 = frames[0].f3;
    let mut smooth_f0 = frames[0].f0;
    let mut smooth_energy = frames[0].energy;
    let mut smooth_voicing = frames[0].voicing;

    // Smoothing coefficient: lower = smoother (more coarticulation)
    let coartic_alpha = 0.08; // ~12 frames to converge = 60ms glide

    for (frame_idx, frame) in frames.iter().enumerate() {
        let target_f0 = frame.f0.max(50.0);
        let target_energy = frame.energy;
        let target_voicing = frame.voicing;

        // Bandwidth widens during transitions (source type changes)
        let prev_source = if frame_idx > 0 { frames[frame_idx - 1].source_type } else { frame.source_type };
        let in_transition = prev_source != frame.source_type;
        let transition_bw_mult = if in_transition { 1.5 } else { 1.0 };

        for sample_idx in 0..samples_per_frame {
            // ── Coarticulation: smooth parameter interpolation ──
            smooth_f1 += coartic_alpha * (frame.f1 - smooth_f1);
            smooth_f2 += coartic_alpha * (frame.f2 - smooth_f2);
            smooth_f3 += coartic_alpha * (frame.f3 - smooth_f3);
            smooth_f0 += coartic_alpha * (target_f0 - smooth_f0);
            smooth_energy += coartic_alpha * (target_energy - smooth_energy);
            smooth_voicing += coartic_alpha * (target_voicing - smooth_voicing);

            // ── Source: glottal pulse with jitter + shimmer ──
            let mut source = 0.0f32;

            match frame.source_type {
                SourceType::Vowel | SourceType::Liquid | SourceType::Nasal => {
                    // Jitter: ±2% random F0 variation (biological vocal fold irregularity)
                    noise_state = noise_state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let jitter = 1.0 + ((noise_state >> 16) as f32 / 65536.0 - 0.5) * 0.04;
                    let f0_jittered = smooth_f0 * jitter;

                    glottal_phase += f0_jittered / sr;
                    if glottal_phase >= 1.0 { glottal_phase -= 1.0; }

                    // LF-like glottal pulse (more natural than simple half-sine)
                    let pulse = if glottal_phase < 0.35 {
                        // Open phase: rising with concave shape
                        let t = glottal_phase / 0.35;
                        t * t * (3.0 - 2.0 * t) // smoothstep
                    } else if glottal_phase < 0.45 {
                        // Return phase: rapid closure
                        let t = (glottal_phase - 0.35) / 0.10;
                        (1.0 - t) * (1.0 - t) * (1.0 - t) // cubic decay
                    } else {
                        0.0 // closed phase
                    };

                    // Shimmer: ±5% random amplitude variation
                    noise_state = noise_state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let shimmer = 1.0 + ((noise_state >> 20) as f32 / 4096.0 - 0.5) * 0.10;

                    source = pulse * smooth_voicing * smooth_energy * shimmer;

                    // Aspiration: breathy noise during open phase
                    if glottal_phase < 0.35 {
                        noise_state = noise_state.wrapping_mul(1664525).wrapping_add(1013904223);
                        let aspiration = (noise_state >> 16) as f32 / 32768.0 - 1.0;
                        source += aspiration * 0.08 * smooth_energy;
                    }
                }
                SourceType::Fricative => {
                    noise_state = noise_state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let noise = (noise_state >> 16) as f32 / 32768.0 - 1.0;
                    source = noise * smooth_energy * 0.6;

                    if smooth_voicing > 0.3 {
                        glottal_phase += smooth_f0 / sr;
                        if glottal_phase >= 1.0 { glottal_phase -= 1.0; }
                        let pulse = (glottal_phase * std::f32::consts::TAU).sin();
                        source += pulse * smooth_voicing * smooth_energy * 0.3;
                    }
                }
                SourceType::Stop => {
                    // Brief burst at release
                    if sample_idx < samples_per_frame / 4 {
                        noise_state = noise_state.wrapping_mul(1664525).wrapping_add(1013904223);
                        source = (noise_state >> 16) as f32 / 32768.0 - 1.0;
                        source *= smooth_energy * 0.4 * (1.0 - sample_idx as f32 / (samples_per_frame as f32 / 4.0));
                    }
                }
                SourceType::Affricate => {
                    noise_state = noise_state.wrapping_mul(1664525).wrapping_add(1013904223);
                    source = (noise_state >> 16) as f32 / 32768.0 - 1.0;
                    source *= smooth_energy * 0.4;
                }
                SourceType::Silent => {}
            }

            // ── Filter: cascade of 3 resonators with smoothed formants ──
            let formants = [
                (smooth_f1, frame.b1 * transition_bw_mult),
                (smooth_f2, frame.b2 * transition_bw_mult),
                (smooth_f3, frame.b3 * transition_bw_mult),
            ];

            let mut filtered = source;
            for (i, &(freq, bw)) in formants.iter().enumerate() {
                if freq < 10.0 { continue; }
                filtered = resonate(&mut res[i], filtered, freq, bw, sr);
            }

            // Soft clip to prevent harsh distortion
            let clipped = if filtered.abs() > 1.0 {
                filtered.signum() * (1.0 - (-filtered.abs() + 1.0).exp())
            } else {
                filtered
            };

            output.push(clipped * 8.0); // master gain (reduced from 12 to avoid clipping)
        }
    }

    output
}

/// Second-order resonator (digital formant filter).
fn resonate(state: &mut [f32; 2], input: f32, freq: f32, bandwidth: f32, sr: f32) -> f32 {
    let omega = std::f32::consts::TAU * freq / sr;
    let r = (-std::f32::consts::PI * bandwidth / sr).exp();
    let a1 = -2.0 * r * omega.cos();
    let a2 = r * r;

    let output = input - a1 * state[0] - a2 * state[1];
    state[1] = state[0];
    state[0] = output;

    output * (1.0 - r)
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
        assert!(audio.iter().any(|&s| s.abs() > 0.01), "should be audible");
    }

    #[test]
    fn coarticulation_smooths_transitions() {
        // Jump from /ɑ/ to /iː/ — should glide, not click
        let mut frames = vec![test_frame(730.0, 1090.0, 120.0, 0.8); 50]; // /ɑ/
        frames.extend(vec![test_frame(270.0, 2290.0, 120.0, 0.8); 50]); // /iː/
        let audio = synthesize(&frames, 44100);

        // Check that the transition region doesn't have huge jumps
        let transition_start: usize = 50 * (44100 / 200);
        let max_jump = audio.windows(2)
            .skip(transition_start.saturating_sub(100))
            .take(200)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_jump < 0.5, "transition should be smooth: max_jump={max_jump}");
    }

    #[test]
    fn silence_is_silent() {
        let frames = vec![FormantFrame {
            f1: 0.0, f2: 0.0, f3: 0.0, b1: 60.0, b2: 80.0, b3: 100.0,
            f0: 0.0, energy: 0.0, voicing: 0.0, time: 0.0,
            source_type: SourceType::Silent, nasal_zero_freq: 0.0, nasal_zero_bw: 0.0,
        }; 50];
        let audio = synthesize(&frames, 44100);
        assert!(audio.iter().all(|&s| s.abs() < 0.01));
    }
}
