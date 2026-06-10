// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Formant vocoder with warmth, space, and organic feel.
//!
//! Four acoustic improvements:
//! 1. Composite excitation: filtered pulse + pink noise (warmth)
//! 2. Dynamic bandwidths: wider during transitions (organic)
//! 3. Spectral tilt: -6dB/octave roll-off (removes digital edge)
//! 4. Simple reverb tail (spatial depth)

use symthaea_vocal_tract::types::{FormantFrame, SourceType};

/// Stable formant resonator with coefficient interpolation.
struct StableResonator {
    y1: f32,
    y2: f32,
    a1: f32,
    a2: f32,
    gain: f32,
    target_a1: f32,
    target_a2: f32,
    target_gain: f32,
}

impl StableResonator {
    fn new() -> Self {
        Self {
            y1: 0.0,
            y2: 0.0,
            a1: 0.0,
            a2: 0.0,
            gain: 0.01,
            target_a1: 0.0,
            target_a2: 0.0,
            target_gain: 0.01,
        }
    }

    fn set_target(&mut self, freq: f32, bandwidth: f32, sr: f32) {
        let bw = bandwidth.max(150.0); // wider minimum prevents metallic spikes
        let omega = std::f32::consts::TAU * freq / sr;
        let r = (-std::f32::consts::PI * bw / sr).exp().clamp(0.0, 0.99);
        self.target_a1 = -2.0 * r * omega.cos();
        self.target_a2 = r * r;
        self.target_gain = (1.0 - r).max(0.001);
    }

    fn tick(&mut self, input: f32) -> f32 {
        // Slower coefficient interpolation = smoother transitions = less stutter
        // 0.002 ≈ 150ms to converge (human coarticulation range)
        let alpha = 0.002;
        self.a1 += alpha * (self.target_a1 - self.a1);
        self.a2 += alpha * (self.target_a2 - self.a2);
        self.gain += alpha * (self.target_gain - self.gain);
        let output = (input - self.a1 * self.y1 - self.a2 * self.y2).clamp(-5.0, 5.0);
        self.y2 = self.y1;
        self.y1 = output;
        output * self.gain
    }
}

/// Synthesize audio with warmth and depth.
pub fn synthesize(frames: &[FormantFrame], sample_rate: u32) -> Vec<f32> {
    if frames.is_empty() {
        return Vec::new();
    }

    let sr = sample_rate as f32;
    let frame_rate = 200.0;
    let samples_per_frame = (sr / frame_rate) as usize;

    let mut output = Vec::with_capacity(frames.len() * samples_per_frame);
    let mut glottal_phase = 0.0f32;
    let mut noise_state = 42u32;
    let mut res = [
        StableResonator::new(),
        StableResonator::new(),
        StableResonator::new(),
    ];
    let mut smooth_f0 = frames[0].f0.max(80.0);
    let mut smooth_energy = 0.0f32;

    // Spectral tilt: aggressive LP to kill piercing highs
    let mut tilt_state = 0.0f32;
    let tilt_coeff = 0.95; // strong roll-off (was 0.85 — too gentle)

    // FIX 4: Simple reverb (comb filter delay line)
    let reverb_len = (sr * 0.03) as usize; // 30ms early reflection
    let mut reverb_buf = vec![0.0f32; reverb_len.max(1)];
    let mut reverb_idx = 0usize;
    let reverb_feedback = 0.3;
    let reverb_mix = 0.15; // subtle

    for (frame_idx, frame) in frames.iter().enumerate() {
        // FIX 2: Dynamic bandwidth — wider during transitions
        let prev_source = if frame_idx > 0 {
            frames[frame_idx - 1].source_type
        } else {
            frame.source_type
        };
        let bw_mult = if prev_source != frame.source_type {
            1.4
        } else {
            1.0
        };

        res[0].set_target(frame.f1, frame.b1 * bw_mult, sr);
        res[1].set_target(frame.f2, frame.b2 * bw_mult, sr);
        res[2].set_target(frame.f3, frame.b3 * bw_mult, sr);

        let target_f0 = frame.f0.max(80.0);
        let target_energy = frame.energy;

        for _ in 0..samples_per_frame {
            smooth_f0 += 0.01 * (target_f0 - smooth_f0);
            smooth_energy += 0.02 * (target_energy - smooth_energy);

            // ── FIX 1: Composite excitation (pulse + pink noise = warmth) ──
            let mut source = 0.0f32;

            match frame.source_type {
                SourceType::Vowel | SourceType::Liquid | SourceType::Nasal => {
                    glottal_phase += smooth_f0 / sr;
                    if glottal_phase >= 1.0 {
                        glottal_phase -= 1.0;
                    }

                    // Glottal pulse with softer shape
                    let pulse = if glottal_phase < 0.35 {
                        let t = glottal_phase / 0.35;
                        t * t * (3.0 - 2.0 * t)
                    } else if glottal_phase < 0.45 {
                        let t = (glottal_phase - 0.35) / 0.10;
                        (1.0 - t).powi(3)
                    } else {
                        0.0
                    };

                    // Composite: 85% pulse + 15% subtle noise (warmth without harshness)
                    noise_state = lcg(&mut noise_state);
                    let pink = noise_f32(noise_state) * 0.10; // reduced from 0.25
                    source = (pulse * 0.85 + pink) * smooth_energy;

                    // Gentle breathiness
                    if glottal_phase < 0.4 {
                        noise_state = lcg(&mut noise_state);
                        source += noise_f32(noise_state) * 0.05 * smooth_energy;
                        // reduced from 0.10
                    }
                }
                SourceType::Fricative => {
                    // Strong noise for consonants — must be clearly audible to break up vowels
                    noise_state = lcg(&mut noise_state);
                    source = noise_f32(noise_state) * smooth_energy * 1.5; // boosted from 0.8
                    if frame.voicing > 0.3 {
                        glottal_phase += smooth_f0 / sr;
                        if glottal_phase >= 1.0 {
                            glottal_phase -= 1.0;
                        }
                        source +=
                            (glottal_phase * std::f32::consts::TAU).sin() * smooth_energy * 0.3;
                    }
                }
                SourceType::Stop => {
                    // Brief noise burst for plosives (p, t, k, b, d, g)
                    noise_state = lcg(&mut noise_state);
                    source = noise_f32(noise_state) * smooth_energy * 1.2;
                }
                SourceType::Affricate => {
                    noise_state = lcg(&mut noise_state);
                    source = noise_f32(noise_state) * smooth_energy * 0.5;
                }
                SourceType::Silent => {
                    source = 0.0;
                    for r in &mut res {
                        r.y1 *= 0.99;
                        r.y2 *= 0.99;
                    }
                }
            }

            // Cascade resonators
            let mut filtered = source;
            for r in &mut res {
                filtered = r.tick(filtered);
            }

            // FIX 3: Spectral tilt (-6dB/oct removes digital edge)
            tilt_state += tilt_coeff * (filtered - tilt_state);
            filtered = tilt_state;

            // Soft clip
            let dry = if filtered.abs() > 1.0 {
                filtered.signum() * (1.0 - (-filtered.abs() + 1.0).exp())
            } else {
                filtered
            };

            // FIX 4: Simple reverb (early reflection for spatial depth)
            let delayed = reverb_buf[reverb_idx % reverb_len];
            reverb_buf[reverb_idx % reverb_len] = dry + delayed * reverb_feedback;
            reverb_idx += 1;
            let wet = dry * (1.0 - reverb_mix) + delayed * reverb_mix;

            output.push(wet * 25.0); // balanced: audible but not harsh
        }
    }

    output
}

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    *state
}

fn noise_f32(state: u32) -> f32 {
    (state >> 16) as f32 / 32768.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vowel_frame(f1: f32, f2: f32, f0: f32) -> FormantFrame {
        FormantFrame {
            f1,
            f2,
            f3: 2500.0,
            b1: 80.0,
            b2: 100.0,
            b3: 120.0,
            f0,
            energy: 0.8,
            voicing: 0.9,
            time: 0.0,
            source_type: SourceType::Vowel,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        }
    }

    #[test]
    fn produces_audible_output() {
        let frames = vec![vowel_frame(730.0, 1090.0, 120.0); 100];
        let audio = synthesize(&frames, 44100);
        assert!(audio.iter().any(|&s| s.abs() > 0.05));
    }

    #[test]
    fn smooth_transition_no_clicks() {
        let mut frames = vec![vowel_frame(730.0, 1090.0, 120.0); 60];
        frames.extend(vec![vowel_frame(270.0, 2290.0, 120.0); 60]);
        let audio = synthesize(&frames, 44100);
        let max_jump: f32 = audio
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0, f32::max);
        assert!(max_jump < 1.0, "smooth: {max_jump}");
    }

    #[test]
    fn silence_is_silent() {
        let frames = vec![
            FormantFrame {
                f1: 0.0,
                f2: 0.0,
                f3: 0.0,
                b1: 80.0,
                b2: 100.0,
                b3: 120.0,
                f0: 0.0,
                energy: 0.0,
                voicing: 0.0,
                time: 0.0,
                source_type: SourceType::Silent,
                nasal_zero_freq: 0.0,
                nasal_zero_bw: 0.0,
            };
            50
        ];
        let audio = synthesize(&frames, 44100);
        assert!(audio.iter().all(|&s| s.abs() < 0.5));
    }

    #[test]
    fn fricative_produces_noise() {
        let frames = vec![
            FormantFrame {
                f1: 300.0,
                f2: 1800.0,
                f3: 4500.0,
                b1: 80.0,
                b2: 100.0,
                b3: 120.0,
                f0: 0.0,
                energy: 0.8,
                voicing: 0.0,
                time: 0.0,
                source_type: SourceType::Fricative,
                nasal_zero_freq: 0.0,
                nasal_zero_bw: 0.0,
            };
            50
        ];
        let audio = synthesize(&frames, 44100);
        assert!(
            audio.iter().any(|&s| s.abs() > 0.01),
            "fricative should produce noise"
        );
    }
}
