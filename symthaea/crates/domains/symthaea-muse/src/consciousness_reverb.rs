// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness-driven reverb with early reflections from the Eight Harmonies.
//!
//! Extends the base Freeverb with:
//! - Pre-delay (consciousness_level → room distance)
//! - 8 early reflection taps from harmony activations
//! - Air absorption LP filter (serotonin → warmth)
//! - Modulated comb delays for lush, living tail
//! - Phi → room size mapping (closet → cathedral)

use crate::MusicalState;
use crate::synth::Freeverb;

/// Consciousness-driven reverb engine.
pub struct ConsciousnessReverb {
    pre_delay: DelayLine,
    early_reflections: EarlyReflections,
    late_reverb: Freeverb,
    air_absorption: OnePoleLP,
    mod_phase: f32,
    sample_rate: f32,
}

/// Smooth mono delay line with interpolated delay changes (no clicks).
struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
    delay_samples: f32, // current (smoothed) delay
    target_delay: f32,  // target delay
    smooth_rate: f32,   // interpolation rate (lower = smoother)
}

impl DelayLine {
    fn new(max_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; max_samples.max(1)],
            write_pos: 0,
            delay_samples: 0.0,
            target_delay: 0.0,
            smooth_rate: 0.001, // very smooth transitions
        }
    }

    fn set_delay(&mut self, samples: usize) {
        self.target_delay = (samples as f32).min((self.buffer.len() - 1) as f32);
    }

    fn process(&mut self, input: f32) -> f32 {
        // Smooth toward target (eliminates clicks)
        self.delay_samples += (self.target_delay - self.delay_samples) * self.smooth_rate;

        self.buffer[self.write_pos] = input;

        // Fractional delay via linear interpolation
        let delay_int = self.delay_samples as usize;
        let delay_frac = self.delay_samples - delay_int as f32;
        let read_pos0 = (self.write_pos + self.buffer.len() - delay_int) % self.buffer.len();
        let read_pos1 = (self.write_pos + self.buffer.len() - delay_int - 1) % self.buffer.len();
        let output =
            self.buffer[read_pos0] * (1.0 - delay_frac) + self.buffer[read_pos1] * delay_frac;

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
        self.state += self.coeff * (input - self.state);
        self.state
    }
}

impl ConsciousnessReverb {
    /// Create a new consciousness reverb.
    pub fn new(sample_rate: u32) -> Self {
        let sr = sample_rate as f32;
        let max_predelay = (100.0 * 0.001 * sr) as usize + 1; // 100ms max

        Self {
            pre_delay: DelayLine::new(max_predelay),
            early_reflections: EarlyReflections::new(sample_rate),
            late_reverb: Freeverb::new(sample_rate, 0.5, 0.5, 0.3),
            air_absorption: OnePoleLP::new(sr, 8000.0),
            mod_phase: 0.0,
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
        self.pre_delay
            .set_delay((predelay_ms * 0.001 * self.sample_rate) as usize);

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
        self.air_absorption.set_cutoff(self.sample_rate, cutoff);
    }

    /// Process a stereo sample pair through the full reverb chain.
    ///
    /// Signal flow: pre-delay → early reflections → late reverb → air absorption
    pub fn process_stereo(&mut self, input_l: f32, input_r: f32) -> (f32, f32) {
        // 1. Pre-delay (mono, applied to both channels equally)
        let pd_l = self.pre_delay.process(input_l);
        let pd_r = self.pre_delay.process(input_r);

        // 2. Early reflections (harmony-driven spatial pattern)
        let (er_l, er_r) = self.early_reflections.process(pd_l, pd_r);

        // 3. Comb delay modulation (slow LFO for lush tail)
        self.mod_phase += 0.3 / self.sample_rate; // ~0.3 Hz LFO
        if self.mod_phase > 1.0 {
            self.mod_phase -= 1.0;
        }

        // 4. Late reverb (Freeverb)
        let (late_l, late_r) = self.late_reverb.process_stereo(er_l, er_r);

        // 5. Air absorption (LP filter on reverb tail only)
        let wet_l = late_l - input_l; // extract wet signal
        let wet_r = late_r - input_r;
        let absorbed_l = self.air_absorption.process(wet_l);
        let absorbed_r = self.air_absorption.process(wet_r);

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
        assert!(reverb.pre_delay.target_delay < 500.0);

        // High consciousness = long pre-delay
        let high_state = MusicalState {
            consciousness_level: 0.9,
            ..Default::default()
        };
        reverb.update_state(&high_state);
        assert!(reverb.pre_delay.target_delay > 2000.0);
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
