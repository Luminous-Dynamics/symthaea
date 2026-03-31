// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Formant vocoder with stable coefficient interpolation.
//!
//! Key insight: IIR filter coefficients must be interpolated smoothly,
//! not just the target frequencies. Computing new coefficients every
//! sample from changing frequencies causes instability.

use symthaea_vocal_tract::types::{FormantFrame, SourceType};

/// Stable formant resonator with coefficient interpolation.
struct StableResonator {
    /// Current filter state
    y1: f32,
    y2: f32,
    /// Current coefficients (smoothed)
    a1: f32,
    a2: f32,
    gain: f32,
    /// Target coefficients
    target_a1: f32,
    target_a2: f32,
    target_gain: f32,
}

impl StableResonator {
    fn new() -> Self {
        Self {
            y1: 0.0, y2: 0.0,
            a1: 0.0, a2: 0.0, gain: 0.01,
            target_a1: 0.0, target_a2: 0.0, target_gain: 0.01,
        }
    }

    /// Set target formant (called once per frame, not per sample).
    fn set_target(&mut self, freq: f32, bandwidth: f32, sr: f32) {
        let bw = bandwidth.max(120.0); // wide minimum BW for stability
        let omega = std::f32::consts::TAU * freq / sr;
        let r = (-std::f32::consts::PI * bw / sr).exp().clamp(0.0, 0.99);
        self.target_a1 = -2.0 * r * omega.cos();
        self.target_a2 = r * r;
        self.target_gain = (1.0 - r).max(0.001);
    }

    /// Process one sample with coefficient interpolation.
    fn tick(&mut self, input: f32) -> f32 {
        // Slew-rate limit: coefficients glide toward target (~5ms time constant)
        let alpha = 0.005; // very slow = very smooth
        self.a1 += alpha * (self.target_a1 - self.a1);
        self.a2 += alpha * (self.target_a2 - self.a2);
        self.gain += alpha * (self.target_gain - self.gain);

        let output = input - self.a1 * self.y1 - self.a2 * self.y2;
        let output = output.clamp(-5.0, 5.0); // prevent runaway

        self.y2 = self.y1;
        self.y1 = output;

        output * self.gain
    }

    fn reset(&mut self) {
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Synthesize audio from formant frames.
pub fn synthesize(frames: &[FormantFrame], sample_rate: u32) -> Vec<f32> {
    if frames.is_empty() { return Vec::new(); }

    let sr = sample_rate as f32;
    let frame_rate = 200.0;
    let samples_per_frame = (sr / frame_rate) as usize;

    let mut output = Vec::with_capacity(frames.len() * samples_per_frame);
    let mut glottal_phase = 0.0f32;
    let mut noise_state = 42u32;

    // Three stable resonators with coefficient interpolation
    let mut res = [StableResonator::new(), StableResonator::new(), StableResonator::new()];

    // Smoothed control parameters
    let mut smooth_f0 = frames[0].f0.max(80.0);
    let mut smooth_energy = 0.0f32; // start silent, fade in

    for frame in frames {
        // Set resonator targets ONCE per frame (not per sample)
        res[0].set_target(frame.f1, frame.b1, sr);
        res[1].set_target(frame.f2, frame.b2, sr);
        res[2].set_target(frame.f3, frame.b3, sr);

        let target_f0 = frame.f0.max(80.0);
        let target_energy = frame.energy;

        for _ in 0..samples_per_frame {
            // Smooth control signals
            smooth_f0 += 0.01 * (target_f0 - smooth_f0);
            smooth_energy += 0.02 * (target_energy - smooth_energy);

            // ── Excitation source ──
            let mut source = 0.0f32;

            match frame.source_type {
                SourceType::Vowel | SourceType::Liquid | SourceType::Nasal => {
                    // Glottal pulse
                    glottal_phase += smooth_f0 / sr;
                    if glottal_phase >= 1.0 { glottal_phase -= 1.0; }

                    let pulse = if glottal_phase < 0.35 {
                        let t = glottal_phase / 0.35;
                        t * t * (3.0 - 2.0 * t) // smoothstep open
                    } else if glottal_phase < 0.45 {
                        let t = (glottal_phase - 0.35) / 0.10;
                        (1.0 - t).powi(3) // cubic close
                    } else {
                        0.0
                    };

                    source = pulse * smooth_energy;

                    // Aspiration noise (breathiness)
                    if glottal_phase < 0.4 {
                        noise_state = lcg(&mut noise_state);
                        source += noise_f32(noise_state) * 0.06 * smooth_energy;
                    }
                }
                SourceType::Fricative => {
                    // Broadband noise — MUST be strong for /s/, /f/, /ʃ/ to be audible
                    noise_state = lcg(&mut noise_state);
                    source = noise_f32(noise_state) * smooth_energy * 0.8;

                    // Voiced fricatives (/v/, /z/, /ð/) add glottal pulse
                    if frame.voicing > 0.3 {
                        glottal_phase += smooth_f0 / sr;
                        if glottal_phase >= 1.0 { glottal_phase -= 1.0; }
                        source += (glottal_phase * std::f32::consts::TAU).sin() * smooth_energy * 0.3;
                    }
                }
                SourceType::Stop => {
                    // Silence during closure, then brief noise burst
                    // (the burst happens at the start of the next phoneme)
                    source = 0.0;
                }
                SourceType::Affricate => {
                    noise_state = lcg(&mut noise_state);
                    source = noise_f32(noise_state) * smooth_energy * 0.5;
                }
                SourceType::Silent => {
                    source = 0.0;
                    // Fade resonator state during silence to prevent ringing
                    for r in &mut res { r.y1 *= 0.99; r.y2 *= 0.99; }
                }
            }

            // ── Filter: cascade of 3 resonators with interpolated coefficients ──
            let mut filtered = source;
            for r in &mut res {
                filtered = r.tick(filtered);
            }

            // Soft clip
            let out = if filtered.abs() > 1.0 {
                filtered.signum() * (1.0 - (-filtered.abs() + 1.0).exp())
            } else {
                filtered
            };

            output.push(out * 40.0); // master gain (needs high due to wide BW)
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
            f1, f2, f3: 2500.0, b1: 80.0, b2: 100.0, b3: 120.0,
            f0, energy: 0.8, voicing: 0.9, time: 0.0,
            source_type: SourceType::Vowel,
            nasal_zero_freq: 0.0, nasal_zero_bw: 0.0,
        }
    }

    #[test]
    fn produces_audible_output() {
        let frames = vec![vowel_frame(730.0, 1090.0, 120.0); 100]; // /ɑ/
        let audio = synthesize(&frames, 44100);
        assert!(audio.iter().any(|&s| s.abs() > 0.05), "should be audible");
    }

    #[test]
    fn smooth_transition_no_clicks() {
        let mut frames = vec![vowel_frame(730.0, 1090.0, 120.0); 60]; // /ɑ/
        frames.extend(vec![vowel_frame(270.0, 2290.0, 120.0); 60]);   // /iː/
        let audio = synthesize(&frames, 44100);
        let max_jump: f32 = audio.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0, f32::max);
        assert!(max_jump < 1.0, "transition should be smooth: {max_jump}");
    }

    #[test]
    fn silence_is_silent() {
        let frames = vec![FormantFrame {
            f1: 0.0, f2: 0.0, f3: 0.0, b1: 80.0, b2: 100.0, b3: 120.0,
            f0: 0.0, energy: 0.0, voicing: 0.0, time: 0.0,
            source_type: SourceType::Silent, nasal_zero_freq: 0.0, nasal_zero_bw: 0.0,
        }; 50];
        let audio = synthesize(&frames, 44100);
        assert!(audio.iter().all(|&s| s.abs() < 0.1));
    }

    #[test]
    fn fricative_produces_noise() {
        let frames = vec![FormantFrame {
            f1: 300.0, f2: 1800.0, f3: 4500.0, b1: 80.0, b2: 100.0, b3: 120.0,
            f0: 0.0, energy: 0.8, voicing: 0.0, time: 0.0,
            source_type: SourceType::Fricative, nasal_zero_freq: 0.0, nasal_zero_bw: 0.0,
        }; 50];
        let audio = synthesize(&frames, 44100);
        assert!(audio.iter().any(|&s| s.abs() > 0.05), "/s/ should be audible noise");
    }
}
