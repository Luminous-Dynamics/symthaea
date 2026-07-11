// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness-driven reverb with early reflections from the Eight Harmonies.
//!
//! Extends the base Freeverb with:
//! - Pre-delay (consciousness_level → room distance)
//! - 8 early reflection taps from harmony activations
//! - Air absorption LP filter (serotonin → warmth)
//! - Phi → room size mapping (closet → cathedral)

use crate::MusicalState;
use crate::synth::{Freeverb, flush_denormal};

/// Consciousness-driven reverb engine.
///
/// All per-channel state (pre-delay, air absorption) is duplicated L/R: a
/// previous version pushed both channels through one mono delay line and one
/// mono filter, which halved the effective pre-delay and cross-contaminated
/// the channels.
pub struct ConsciousnessReverb {
    pre_delay_l: DelayLine,
    pre_delay_r: DelayLine,
    early_reflections: EarlyReflections,
    late_reverb: Freeverb,
    air_absorption_l: OnePoleLP,
    air_absorption_r: OnePoleLP,
    sample_rate: f32,
}

/// Mono delay line with click-free delay changes via two-tap crossfade.
///
/// A previous version *glided* one read tap toward the new delay, which
/// swept it through the buffer at several samples per output sample — a
/// pitch-warp artifact, plus a hard step when the tap crossed from written
/// signal into unwritten (zero) buffer (measured 0.54 sample-to-sample jump
/// on a 0.5-amplitude sine). Crossfading between a stationary old tap and a
/// stationary new tap is the standard fix: both taps read coherent audio and
/// only their mix changes, at the crossfade rate.
struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
    current_delay: f32, // outgoing tap (fully active when xfade == 0)
    target_delay: f32,  // incoming tap (fully active when xfade == 1)
    xfade: f32,         // crossfade progress [0, 1]
    xfade_step: f32,    // per-sample progress (~45ms total at 44.1kHz)
}

impl DelayLine {
    fn new(max_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; max_samples.max(1)],
            write_pos: 0,
            current_delay: 0.0,
            target_delay: 0.0,
            xfade: 1.0,
            xfade_step: 0.0005,
        }
    }

    fn set_delay(&mut self, samples: usize) {
        let new_target = (samples as f32).min((self.buffer.len() - 1) as f32);
        if (new_target - self.target_delay).abs() < 0.5 {
            return; // no real change — don't restart the fade
        }
        // The tap we've faded furthest toward becomes the outgoing tap.
        if self.xfade >= 0.5 {
            self.current_delay = self.target_delay;
        }
        self.target_delay = new_target;
        self.xfade = 0.0;
    }

    /// Read a stationary fractional tap (linear interpolation).
    fn tap(&self, delay: f32) -> f32 {
        let delay_int = delay as usize;
        let frac = delay - delay_int as f32;
        let n = self.buffer.len();
        let p0 = (self.write_pos + n - delay_int) % n;
        let p1 = (self.write_pos + n - delay_int - 1) % n;
        self.buffer[p0] * (1.0 - frac) + self.buffer[p1] * frac
    }

    fn process(&mut self, input: f32) -> f32 {
        self.buffer[self.write_pos] = input;

        let output = if self.xfade >= 1.0 {
            self.tap(self.target_delay)
        } else {
            let a = self.tap(self.current_delay);
            let b = self.tap(self.target_delay);
            self.xfade = (self.xfade + self.xfade_step).min(1.0);
            a * (1.0 - self.xfade) + b * self.xfade
        };

        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        output
    }
}

/// Early reflections: 8 taps driven by the Eight Harmonies.
struct EarlyReflections {
    taps: [ReflectionTap; 8],
}

struct ReflectionTap {
    delay: DelayLine,
    gain: f32,
    pan: f32, // [-1, 1]
}

impl EarlyReflections {
    fn new(sample_rate: u32) -> Self {
        let sr = sample_rate as f32;
        // Base delays (ms) for each harmony's spatial position
        let delays_ms = [12.0, 18.0, 25.0, 33.0, 40.0, 48.0, 55.0, 65.0];
        // Pan positions: harmonies distributed across stereo field
        let pans = [0.0, 0.3, -0.4, 0.6, -0.6, 0.2, -0.3, 0.0];
        let max_delay = (80.0 * 0.001 * sr) as usize + 1;

        let mut taps: [ReflectionTap; 8] = std::array::from_fn(|i| ReflectionTap {
            delay: DelayLine::new(max_delay),
            gain: 0.0,
            pan: pans[i],
        });

        for (i, tap) in taps.iter_mut().enumerate() {
            tap.delay.set_delay((delays_ms[i] * 0.001 * sr) as usize);
        }

        Self { taps }
    }

    /// Update tap gains from harmony activations (capped total gain).
    fn update_harmonies(&mut self, harmonies: &[f32; 8]) {
        for (tap, &activation) in self.taps.iter_mut().zip(harmonies.iter()) {
            tap.gain = activation * 0.06; // max 0.06 per tap (0.48 total max)
        }
    }

    /// Process stereo input, adding early reflections.
    fn process(&mut self, input_l: f32, input_r: f32) -> (f32, f32) {
        let mono = (input_l + input_r) * 0.5;
        let (mut out_l, mut out_r) = (0.0f32, 0.0f32);

        for tap in &mut self.taps {
            let reflected = tap.delay.process(mono) * tap.gain;
            let theta = (tap.pan + 1.0) * std::f32::consts::FRAC_PI_4;
            out_l += reflected * theta.cos();
            out_r += reflected * theta.sin();
        }

        (input_l + out_l, input_r + out_r)
    }
}

/// Simple one-pole low-pass filter for air absorption.
struct OnePoleLP {
    state: f32,
    coeff: f32,
}

impl OnePoleLP {
    fn new(sample_rate: f32, cutoff_hz: f32) -> Self {
        let coeff = Self::compute_coeff(sample_rate, cutoff_hz);
        Self { state: 0.0, coeff }
    }

    fn compute_coeff(sample_rate: f32, cutoff_hz: f32) -> f32 {
        let rc = 1.0 / (std::f32::consts::TAU * cutoff_hz);
        let dt = 1.0 / sample_rate;
        dt / (rc + dt)
    }

    fn set_cutoff(&mut self, sample_rate: f32, cutoff_hz: f32) {
        self.coeff = Self::compute_coeff(sample_rate, cutoff_hz);
    }

    fn process(&mut self, input: f32) -> f32 {
        self.state = flush_denormal(self.state + self.coeff * (input - self.state));
        self.state
    }
}

impl ConsciousnessReverb {
    /// Create a new consciousness reverb.
    pub fn new(sample_rate: u32) -> Self {
        let sr = sample_rate as f32;
        let max_predelay = (100.0 * 0.001 * sr) as usize + 1; // 100ms max

        Self {
            pre_delay_l: DelayLine::new(max_predelay),
            pre_delay_r: DelayLine::new(max_predelay),
            early_reflections: EarlyReflections::new(sample_rate),
            late_reverb: Freeverb::new(sample_rate, 0.5, 0.5, 0.3),
            air_absorption_l: OnePoleLP::new(sr, 8000.0),
            air_absorption_r: OnePoleLP::new(sr, 8000.0),
            sample_rate: sr,
        }
    }

    /// Update reverb parameters from consciousness state.
    ///
    /// Called once per chunk (32ms). Maps:
    /// - consciousness_level → pre-delay (0-80ms)
    /// - consciousness_level → room size (Freeverb feedback)
    /// - harmony_activations → early reflection tap gains
    /// - serotonin → air absorption cutoff (warmth)
    /// - arousal → late reverb diffusion
    pub fn update_state(&mut self, state: &MusicalState) {
        let psi = state.consciousness_level;

        // Pre-delay: higher consciousness = larger room = more pre-delay
        let predelay_ms = psi * 80.0; // 0-80ms
        let predelay_samples = (predelay_ms * 0.001 * self.sample_rate) as usize;
        self.pre_delay_l.set_delay(predelay_samples);
        self.pre_delay_r.set_delay(predelay_samples);

        // Room size: consciousness maps to Freeverb feedback.
        // CRITICAL: update params in-place to preserve reverb tail.
        let room = 0.1 + psi * 0.8;
        let damping = 0.3 + state.serotonin * 0.4;
        let wet = 0.1 + psi * 0.3;
        self.late_reverb.set_params(room, damping, wet);

        // Early reflections from harmony activations
        self.early_reflections
            .update_harmonies(&state.harmony_activations);

        // Air absorption: serotonin → warmth (lower cutoff = more absorption)
        let cutoff = 4000.0 + (1.0 - state.serotonin) * 12000.0; // 4-16kHz
        self.air_absorption_l.set_cutoff(self.sample_rate, cutoff);
        self.air_absorption_r.set_cutoff(self.sample_rate, cutoff);
    }

    /// Process a stereo sample pair through the full reverb chain.
    ///
    /// Signal flow: pre-delay → early reflections → late reverb → air absorption
    pub fn process_stereo(&mut self, input_l: f32, input_r: f32) -> (f32, f32) {
        // 1. Pre-delay (independent per channel)
        let pd_l = self.pre_delay_l.process(input_l);
        let pd_r = self.pre_delay_r.process(input_r);

        // 2. Early reflections (harmony-driven spatial pattern)
        let (er_l, er_r) = self.early_reflections.process(pd_l, pd_r);

        // 3. Late reverb (Freeverb)
        let (late_l, late_r) = self.late_reverb.process_stereo(er_l, er_r);

        // 4. Air absorption (LP filter on reverb tail only, per channel)
        let wet_l = late_l - input_l; // extract wet signal
        let wet_r = late_r - input_r;
        let absorbed_l = self.air_absorption_l.process(wet_l);
        let absorbed_r = self.air_absorption_r.process(wet_r);

        (input_l + absorbed_l, input_r + absorbed_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverb_produces_output() {
        let mut reverb = ConsciousnessReverb::new(44100);
        reverb.update_state(&MusicalState::default());

        let mut has_output = false;
        for i in 0..4410 {
            let input = if i < 10 { 0.5 } else { 0.0 }; // impulse
            let (l, r) = reverb.process_stereo(input, input);
            if i > 100 && (l.abs() > 0.001 || r.abs() > 0.001) {
                has_output = true;
            }
        }
        assert!(
            has_output,
            "reverb should produce decaying output after impulse"
        );
    }

    #[test]
    fn pre_delay_increases_with_consciousness() {
        let mut reverb = ConsciousnessReverb::new(44100);

        // Low consciousness = short pre-delay
        let low_state = MusicalState {
            consciousness_level: 0.1,
            ..Default::default()
        };
        reverb.update_state(&low_state);
        assert!(reverb.pre_delay_l.target_delay < 500.0);
        assert!(reverb.pre_delay_r.target_delay < 500.0);

        // High consciousness = long pre-delay
        let high_state = MusicalState {
            consciousness_level: 0.9,
            ..Default::default()
        };
        reverb.update_state(&high_state);
        assert!(reverb.pre_delay_l.target_delay > 2000.0);
        assert!(reverb.pre_delay_r.target_delay > 2000.0);
    }

    #[test]
    fn harmony_activations_affect_early_reflections() {
        let mut reverb = ConsciousnessReverb::new(44100);

        // Zero harmonies = no early reflections
        let silent = MusicalState {
            harmony_activations: [0.0; 8],
            ..Default::default()
        };
        reverb.update_state(&silent);
        let all_zero = reverb.early_reflections.taps.iter().all(|t| t.gain == 0.0);
        assert!(all_zero, "zero harmonies should mean zero reflection gain");

        // Active harmonies = early reflections
        let active = MusicalState {
            harmony_activations: [0.8; 8],
            ..Default::default()
        };
        reverb.update_state(&active);
        let any_nonzero = reverb.early_reflections.taps.iter().any(|t| t.gain > 0.0);
        assert!(
            any_nonzero,
            "active harmonies should produce reflection gains"
        );
    }

    #[test]
    fn reverb_tail_decays() {
        // Property: after an impulse, the tail's energy must decrease over
        // time and approach silence (a stuck or growing tail means feedback ≥ 1).
        let mut reverb = ConsciousnessReverb::new(44100);
        reverb.update_state(&MusicalState {
            consciousness_level: 0.9, // big room, long tail
            ..Default::default()
        });

        let sr = 44100usize;
        let mut rms_early = 0.0f64; // 0.1-0.5s after impulse
        let mut rms_late = 0.0f64; // 1.5-2.0s after impulse
        for i in 0..(2 * sr) {
            let input = if i < 32 { 0.8 } else { 0.0 };
            let (l, r) = reverb.process_stereo(input, input);
            let p = (l * l + r * r) as f64;
            if i >= sr / 10 && i < sr / 2 {
                rms_early += p;
            }
            if i >= sr * 3 / 2 {
                rms_late += p;
            }
        }
        rms_early = (rms_early / (0.4 * sr as f64)).sqrt();
        rms_late = (rms_late / (0.5 * sr as f64)).sqrt();

        assert!(rms_early > 1e-7, "tail should exist early ({rms_early})");
        assert!(
            rms_late < rms_early * 0.5,
            "tail must decay: early={rms_early} late={rms_late}"
        );
    }

    #[test]
    fn parameter_changes_do_not_click() {
        // Property: yanking consciousness (pre-delay, room size, wet) mid-
        // stream must not produce discontinuities. A click is an ISOLATED
        // outlier in the first-difference sequence — a single diff far above
        // its neighbors. Loud-but-smooth content (comb resonance while
        // feedback glides) legitimately produces LARGE diffs, but they come
        // in runs of similar magnitude, so we test each window's peak diff
        // against that window's mean diff, not an absolute level.
        // (Pre-glide/pre-crossfade, the parameter steps produced window
        // ratios far above 8; the wet/feedback glides and the two-tap
        // pre-delay crossfade brought them into the smooth range.)
        let mut reverb = ConsciousnessReverb::new(44100);
        reverb.update_state(&MusicalState::default());

        let sr = 44100.0f32;
        let mut prev = 0.0f32;
        let mut diffs: Vec<f32> = Vec::with_capacity(44100);
        for i in 0..44100 {
            let x = 0.5 * (std::f32::consts::TAU * 440.0 * i as f32 / sr).sin();
            // Yank the room size every 4410 samples
            if i % 4410 == 0 {
                let psi = if (i / 4410) % 2 == 0 { 0.1 } else { 0.9 };
                reverb.update_state(&MusicalState {
                    consciousness_level: psi,
                    ..Default::default()
                });
            }
            let (l, _r) = reverb.process_stereo(x, x);
            if i > 100 {
                diffs.push((l - prev).abs());
            }
            prev = l;
        }

        let mut worst_ratio = 0.0f32;
        let mut worst_window = 0usize;
        for (w, window) in diffs.chunks(128).enumerate() {
            let mean = window.iter().sum::<f32>() / window.len() as f32;
            let peak = window.iter().fold(0.0f32, |a, &b| a.max(b));
            let ratio = peak / (mean + 1e-6);
            if ratio > worst_ratio {
                worst_ratio = ratio;
                worst_window = w;
            }
        }
        assert!(
            worst_ratio < 8.0,
            "parameter changes should not click: window {worst_window} has a \
             first-difference outlier {worst_ratio:.1}x its window mean"
        );
    }

    #[test]
    fn stereo_output_differs() {
        let mut reverb = ConsciousnessReverb::new(44100);
        let state = MusicalState {
            consciousness_level: 0.8,
            harmony_activations: [0.5, 0.7, 0.3, 0.6, 0.4, 0.8, 0.2, 0.9],
            ..Default::default()
        };
        reverb.update_state(&state);

        // Feed asymmetric input
        let mut diff_count = 0;
        for i in 0..4410 {
            let input = if i % 100 < 5 { 0.3 } else { 0.0 };
            let (l, r) = reverb.process_stereo(input, input * 0.5);
            if (l - r).abs() > 0.001 {
                diff_count += 1;
            }
        }
        assert!(
            diff_count > 100,
            "stereo channels should differ with harmony panning"
        );
    }
}
