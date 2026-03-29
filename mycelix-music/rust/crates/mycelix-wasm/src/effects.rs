// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-time Audio Effects DSP Module
//!
//! Provides browser-side audio effects processing:
//! - Parametric EQ (low shelf, high shelf, peak filters)
//! - Reverb (Schroeder-style convolution)
//! - Spatial Audio (HRTF-based 3D positioning)
//! - Filters (lowpass, highpass, bandpass)
//! - Dynamics (compressor, limiter)

use wasm_bindgen::prelude::*;
use std::f32::consts::PI;

// ============================================================================
// Biquad Filter (foundation for EQ and filters)
// ============================================================================

#[derive(Clone, Copy)]
struct BiquadCoeffs {
    b0: f32, b1: f32, b2: f32,
    a1: f32, a2: f32,
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct BiquadFilter {
    coeffs: BiquadCoeffs,
    x1: f32, x2: f32,
    y1: f32, y2: f32,
    sample_rate: f32,
}

#[wasm_bindgen]
impl BiquadFilter {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            coeffs: BiquadCoeffs { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 },
            x1: 0.0, x2: 0.0,
            y1: 0.0, y2: 0.0,
            sample_rate,
        }
    }

    fn compute_coeffs(&mut self, filter_type: &str, freq: f32, q: f32, gain_db: f32) {
        let omega = 2.0 * PI * freq / self.sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);
        let a = 10.0_f32.powf(gain_db / 40.0);

        let (b0, b1, b2, a0, a1, a2) = match filter_type {
            "lowpass" => (
                (1.0 - cos_omega) / 2.0,
                1.0 - cos_omega,
                (1.0 - cos_omega) / 2.0,
                1.0 + alpha,
                -2.0 * cos_omega,
                1.0 - alpha,
            ),
            "highpass" => (
                (1.0 + cos_omega) / 2.0,
                -(1.0 + cos_omega),
                (1.0 + cos_omega) / 2.0,
                1.0 + alpha,
                -2.0 * cos_omega,
                1.0 - alpha,
            ),
            "bandpass" => (
                alpha,
                0.0,
                -alpha,
                1.0 + alpha,
                -2.0 * cos_omega,
                1.0 - alpha,
            ),
            "notch" => (
                1.0,
                -2.0 * cos_omega,
                1.0,
                1.0 + alpha,
                -2.0 * cos_omega,
                1.0 - alpha,
            ),
            "peak" => (
                1.0 + alpha * a,
                -2.0 * cos_omega,
                1.0 - alpha * a,
                1.0 + alpha / a,
                -2.0 * cos_omega,
                1.0 - alpha / a,
            ),
            "lowshelf" => {
                let sqrt_a = a.sqrt();
                let sqrt_a_alpha = 2.0 * sqrt_a * alpha;
                (
                    a * ((a + 1.0) - (a - 1.0) * cos_omega + sqrt_a_alpha),
                    2.0 * a * ((a - 1.0) - (a + 1.0) * cos_omega),
                    a * ((a + 1.0) - (a - 1.0) * cos_omega - sqrt_a_alpha),
                    (a + 1.0) + (a - 1.0) * cos_omega + sqrt_a_alpha,
                    -2.0 * ((a - 1.0) + (a + 1.0) * cos_omega),
                    (a + 1.0) + (a - 1.0) * cos_omega - sqrt_a_alpha,
                )
            },
            "highshelf" => {
                let sqrt_a = a.sqrt();
                let sqrt_a_alpha = 2.0 * sqrt_a * alpha;
                (
                    a * ((a + 1.0) + (a - 1.0) * cos_omega + sqrt_a_alpha),
                    -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_omega),
                    a * ((a + 1.0) + (a - 1.0) * cos_omega - sqrt_a_alpha),
                    (a + 1.0) - (a - 1.0) * cos_omega + sqrt_a_alpha,
                    2.0 * ((a - 1.0) - (a + 1.0) * cos_omega),
                    (a + 1.0) - (a - 1.0) * cos_omega - sqrt_a_alpha,
                )
            },
            _ => (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        };

        self.coeffs = BiquadCoeffs {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        };
    }

    #[wasm_bindgen]
    pub fn set_lowpass(&mut self, frequency: f32, q: f32) {
        self.compute_coeffs("lowpass", frequency, q, 0.0);
    }

    #[wasm_bindgen]
    pub fn set_highpass(&mut self, frequency: f32, q: f32) {
        self.compute_coeffs("highpass", frequency, q, 0.0);
    }

    #[wasm_bindgen]
    pub fn set_bandpass(&mut self, frequency: f32, q: f32) {
        self.compute_coeffs("bandpass", frequency, q, 0.0);
    }

    #[wasm_bindgen]
    pub fn set_notch(&mut self, frequency: f32, q: f32) {
        self.compute_coeffs("notch", frequency, q, 0.0);
    }

    #[wasm_bindgen]
    pub fn set_peak(&mut self, frequency: f32, q: f32, gain_db: f32) {
        self.compute_coeffs("peak", frequency, q, gain_db);
    }

    #[wasm_bindgen]
    pub fn set_lowshelf(&mut self, frequency: f32, q: f32, gain_db: f32) {
        self.compute_coeffs("lowshelf", frequency, q, gain_db);
    }

    #[wasm_bindgen]
    pub fn set_highshelf(&mut self, frequency: f32, q: f32, gain_db: f32) {
        self.compute_coeffs("highshelf", frequency, q, gain_db);
    }

    #[wasm_bindgen]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let output = self.coeffs.b0 * input
            + self.coeffs.b1 * self.x1
            + self.coeffs.b2 * self.x2
            - self.coeffs.a1 * self.y1
            - self.coeffs.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&s| self.process_sample(s)).collect()
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// ============================================================================
// Parametric EQ (3-band + low/high shelf)
// ============================================================================

#[wasm_bindgen]
pub struct ParametricEQ {
    low_shelf: BiquadFilter,
    low_mid: BiquadFilter,
    mid: BiquadFilter,
    high_mid: BiquadFilter,
    high_shelf: BiquadFilter,
    sample_rate: f32,
}

#[wasm_bindgen]
impl ParametricEQ {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        let mut eq = Self {
            low_shelf: BiquadFilter::new(sample_rate),
            low_mid: BiquadFilter::new(sample_rate),
            mid: BiquadFilter::new(sample_rate),
            high_mid: BiquadFilter::new(sample_rate),
            high_shelf: BiquadFilter::new(sample_rate),
            sample_rate,
        };

        // Default neutral settings
        eq.low_shelf.set_lowshelf(100.0, 0.707, 0.0);
        eq.low_mid.set_peak(250.0, 1.0, 0.0);
        eq.mid.set_peak(1000.0, 1.0, 0.0);
        eq.high_mid.set_peak(4000.0, 1.0, 0.0);
        eq.high_shelf.set_highshelf(10000.0, 0.707, 0.0);

        eq
    }

    #[wasm_bindgen]
    pub fn set_low_shelf(&mut self, frequency: f32, gain_db: f32) {
        self.low_shelf.set_lowshelf(frequency, 0.707, gain_db);
    }

    #[wasm_bindgen]
    pub fn set_low_mid(&mut self, frequency: f32, q: f32, gain_db: f32) {
        self.low_mid.set_peak(frequency, q, gain_db);
    }

    #[wasm_bindgen]
    pub fn set_mid(&mut self, frequency: f32, q: f32, gain_db: f32) {
        self.mid.set_peak(frequency, q, gain_db);
    }

    #[wasm_bindgen]
    pub fn set_high_mid(&mut self, frequency: f32, q: f32, gain_db: f32) {
        self.high_mid.set_peak(frequency, q, gain_db);
    }

    #[wasm_bindgen]
    pub fn set_high_shelf(&mut self, frequency: f32, gain_db: f32) {
        self.high_shelf.set_highshelf(frequency, 0.707, gain_db);
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output = input.to_vec();

        for sample in output.iter_mut() {
            *sample = self.low_shelf.process_sample(*sample);
            *sample = self.low_mid.process_sample(*sample);
            *sample = self.mid.process_sample(*sample);
            *sample = self.high_mid.process_sample(*sample);
            *sample = self.high_shelf.process_sample(*sample);
        }

        output
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.low_shelf.reset();
        self.low_mid.reset();
        self.mid.reset();
        self.high_mid.reset();
        self.high_shelf.reset();
    }
}

// ============================================================================
// Reverb (Schroeder style with comb and allpass filters)
// ============================================================================

struct CombFilter {
    buffer: Vec<f32>,
    write_pos: usize,
    feedback: f32,
    damping: f32,
    filter_store: f32,
}

impl CombFilter {
    fn new(delay_samples: usize, feedback: f32, damping: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples],
            write_pos: 0,
            feedback,
            damping,
            filter_store: 0.0,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.buffer[self.write_pos];

        // Lowpass filter for damping
        self.filter_store = output * (1.0 - self.damping) + self.filter_store * self.damping;

        self.buffer[self.write_pos] = input + self.filter_store * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();

        output
    }
}

struct AllpassFilter {
    buffer: Vec<f32>,
    write_pos: usize,
    feedback: f32,
}

impl AllpassFilter {
    fn new(delay_samples: usize, feedback: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples],
            write_pos: 0,
            feedback,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let delayed = self.buffer[self.write_pos];
        let output = -input + delayed;

        self.buffer[self.write_pos] = input + delayed * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();

        output
    }
}

#[wasm_bindgen]
pub struct Reverb {
    comb_filters: Vec<CombFilter>,
    allpass_filters: Vec<AllpassFilter>,
    wet: f32,
    dry: f32,
    sample_rate: f32,
}

#[wasm_bindgen]
impl Reverb {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        // Schroeder reverb delay times (in samples at 44.1kHz, scaled for sample rate)
        let scale = sample_rate / 44100.0;

        let comb_delays = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116];
        let allpass_delays = [225, 556, 441, 341];

        let comb_filters: Vec<CombFilter> = comb_delays.iter()
            .map(|&d| CombFilter::new((d as f32 * scale) as usize, 0.84, 0.2))
            .collect();

        let allpass_filters: Vec<AllpassFilter> = allpass_delays.iter()
            .map(|&d| AllpassFilter::new((d as f32 * scale) as usize, 0.5))
            .collect();

        Self {
            comb_filters,
            allpass_filters,
            wet: 0.3,
            dry: 0.7,
            sample_rate,
        }
    }

    #[wasm_bindgen]
    pub fn set_room_size(&mut self, size: f32) {
        let feedback = 0.7 + size.clamp(0.0, 1.0) * 0.28;
        for comb in &mut self.comb_filters {
            comb.feedback = feedback;
        }
    }

    #[wasm_bindgen]
    pub fn set_damping(&mut self, damping: f32) {
        let d = damping.clamp(0.0, 1.0);
        for comb in &mut self.comb_filters {
            comb.damping = d;
        }
    }

    #[wasm_bindgen]
    pub fn set_wet(&mut self, wet: f32) {
        self.wet = wet.clamp(0.0, 1.0);
    }

    #[wasm_bindgen]
    pub fn set_dry(&mut self, dry: f32) {
        self.dry = dry.clamp(0.0, 1.0);
    }

    #[wasm_bindgen]
    pub fn set_mix(&mut self, mix: f32) {
        let m = mix.clamp(0.0, 1.0);
        self.wet = m;
        self.dry = 1.0 - m;
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&sample| {
            // Sum comb filter outputs
            let mut comb_out = 0.0f32;
            for comb in &mut self.comb_filters {
                comb_out += comb.process(sample);
            }
            comb_out /= self.comb_filters.len() as f32;

            // Pass through allpass filters
            let mut allpass_out = comb_out;
            for allpass in &mut self.allpass_filters {
                allpass_out = allpass.process(allpass_out);
            }

            // Mix wet/dry
            sample * self.dry + allpass_out * self.wet
        }).collect()
    }

    #[wasm_bindgen]
    pub fn process_stereo(&mut self, left: &[f32], right: &[f32]) -> Vec<f32> {
        // Process mono mix, return interleaved stereo with slight spread
        let mono: Vec<f32> = left.iter().zip(right.iter())
            .map(|(l, r)| (l + r) * 0.5)
            .collect();

        let reverb_out = self.process(&mono);

        let mut stereo = Vec::with_capacity(left.len() * 2);
        for (i, ((&l, &r), &rev)) in left.iter().zip(right.iter()).zip(reverb_out.iter()).enumerate() {
            // Add slight stereo spread to reverb
            let spread = ((i as f32 * 0.01).sin() * 0.1 + 1.0) * 0.5;
            stereo.push(l * self.dry + rev * self.wet * spread);
            stereo.push(r * self.dry + rev * self.wet * (1.0 - spread + 0.5));
        }
        stereo
    }
}

// ============================================================================
// Spatial Audio (simplified HRTF-based 3D positioning)
// ============================================================================

#[wasm_bindgen]
pub struct SpatialAudio {
    azimuth: f32,      // -180 to 180 degrees (0 = front)
    elevation: f32,    // -90 to 90 degrees (0 = ear level)
    distance: f32,     // 0 to infinity (1 = reference distance)
    sample_rate: f32,

    // ITD delay lines
    left_delay: Vec<f32>,
    right_delay: Vec<f32>,
    delay_write_pos: usize,

    // ILD filters
    left_filter: BiquadFilter,
    right_filter: BiquadFilter,
}

#[wasm_bindgen]
impl SpatialAudio {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        let max_delay = (sample_rate * 0.001) as usize; // ~1ms max ITD

        Self {
            azimuth: 0.0,
            elevation: 0.0,
            distance: 1.0,
            sample_rate,
            left_delay: vec![0.0; max_delay],
            right_delay: vec![0.0; max_delay],
            delay_write_pos: 0,
            left_filter: BiquadFilter::new(sample_rate),
            right_filter: BiquadFilter::new(sample_rate),
        }
    }

    #[wasm_bindgen]
    pub fn set_position(&mut self, azimuth: f32, elevation: f32, distance: f32) {
        self.azimuth = azimuth.clamp(-180.0, 180.0);
        self.elevation = elevation.clamp(-90.0, 90.0);
        self.distance = distance.max(0.1);

        self.update_filters();
    }

    #[wasm_bindgen]
    pub fn set_azimuth(&mut self, azimuth: f32) {
        self.azimuth = azimuth.clamp(-180.0, 180.0);
        self.update_filters();
    }

    #[wasm_bindgen]
    pub fn set_elevation(&mut self, elevation: f32) {
        self.elevation = elevation.clamp(-90.0, 90.0);
        self.update_filters();
    }

    #[wasm_bindgen]
    pub fn set_distance(&mut self, distance: f32) {
        self.distance = distance.max(0.1);
    }

    fn update_filters(&mut self) {
        // Simplified head shadow effect - reduce high frequencies on far ear
        let angle_rad = self.azimuth * PI / 180.0;

        // Calculate ILD (Interaural Level Difference) based on azimuth
        let shadow_freq = 2000.0 + (1.0 - angle_rad.abs() / PI) * 6000.0;

        if self.azimuth > 0.0 {
            // Sound from right - shadow left ear
            self.left_filter.set_lowpass(shadow_freq, 0.707);
            self.right_filter.set_lowpass(20000.0, 0.707);
        } else if self.azimuth < 0.0 {
            // Sound from left - shadow right ear
            self.left_filter.set_lowpass(20000.0, 0.707);
            self.right_filter.set_lowpass(shadow_freq, 0.707);
        } else {
            // Sound from center
            self.left_filter.set_lowpass(20000.0, 0.707);
            self.right_filter.set_lowpass(20000.0, 0.707);
        }
    }

    fn calculate_itd_samples(&self) -> (usize, usize) {
        // Interaural Time Difference based on azimuth
        // Max ITD is ~0.7ms for humans
        let max_itd = 0.0007 * self.sample_rate;
        let angle_rad = self.azimuth * PI / 180.0;
        let itd = angle_rad.sin() * max_itd;

        if itd > 0.0 {
            (itd as usize, 0)
        } else {
            (0, (-itd) as usize)
        }
    }

    fn calculate_ild(&self) -> (f32, f32) {
        // Interaural Level Difference based on azimuth
        let angle_rad = self.azimuth * PI / 180.0;
        let pan = angle_rad.sin();

        // Equal power panning with ILD
        let left_gain = ((1.0 - pan) * 0.5).sqrt();
        let right_gain = ((1.0 + pan) * 0.5).sqrt();

        // Distance attenuation (inverse square law)
        let distance_atten = 1.0 / self.distance.max(1.0);

        (left_gain * distance_atten, right_gain * distance_atten)
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let (left_delay_samples, right_delay_samples) = self.calculate_itd_samples();
        let (left_gain, right_gain) = self.calculate_ild();

        let mut output = Vec::with_capacity(input.len() * 2);
        let delay_len = self.left_delay.len();

        for &sample in input.iter() {
            // Write to delay lines
            self.left_delay[self.delay_write_pos] = sample;
            self.right_delay[self.delay_write_pos] = sample;

            // Read from delay lines with ITD offset
            let left_read_pos = (self.delay_write_pos + delay_len - left_delay_samples) % delay_len;
            let right_read_pos = (self.delay_write_pos + delay_len - right_delay_samples) % delay_len;

            let left_sample = self.left_delay[left_read_pos];
            let right_sample = self.right_delay[right_read_pos];

            // Apply ILD and head shadow filtering
            let left_out = self.left_filter.process_sample(left_sample) * left_gain;
            let right_out = self.right_filter.process_sample(right_sample) * right_gain;

            output.push(left_out);
            output.push(right_out);

            self.delay_write_pos = (self.delay_write_pos + 1) % delay_len;
        }

        output
    }
}

// ============================================================================
// Compressor
// ============================================================================

#[wasm_bindgen]
pub struct Compressor {
    threshold_db: f32,
    ratio: f32,
    attack_ms: f32,
    release_ms: f32,
    knee_db: f32,
    makeup_gain_db: f32,

    envelope: f32,
    sample_rate: f32,
}

#[wasm_bindgen]
impl Compressor {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            knee_db: 6.0,
            makeup_gain_db: 0.0,
            envelope: 0.0,
            sample_rate,
        }
    }

    #[wasm_bindgen]
    pub fn set_threshold(&mut self, db: f32) {
        self.threshold_db = db.clamp(-60.0, 0.0);
    }

    #[wasm_bindgen]
    pub fn set_ratio(&mut self, ratio: f32) {
        self.ratio = ratio.clamp(1.0, 20.0);
    }

    #[wasm_bindgen]
    pub fn set_attack(&mut self, ms: f32) {
        self.attack_ms = ms.clamp(0.1, 100.0);
    }

    #[wasm_bindgen]
    pub fn set_release(&mut self, ms: f32) {
        self.release_ms = ms.clamp(10.0, 1000.0);
    }

    #[wasm_bindgen]
    pub fn set_knee(&mut self, db: f32) {
        self.knee_db = db.clamp(0.0, 12.0);
    }

    #[wasm_bindgen]
    pub fn set_makeup_gain(&mut self, db: f32) {
        self.makeup_gain_db = db.clamp(-12.0, 24.0);
    }

    fn compute_gain(&self, input_db: f32) -> f32 {
        let half_knee = self.knee_db / 2.0;

        let output_db = if input_db < self.threshold_db - half_knee {
            // Below knee - no compression
            input_db
        } else if input_db > self.threshold_db + half_knee {
            // Above knee - full compression
            self.threshold_db + (input_db - self.threshold_db) / self.ratio
        } else {
            // In knee region - soft knee
            let knee_input = input_db - self.threshold_db + half_knee;
            input_db + (1.0 / self.ratio - 1.0) * knee_input * knee_input / (2.0 * self.knee_db)
        };

        output_db - input_db + self.makeup_gain_db
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let attack_coef = (-2.2 / (self.attack_ms * 0.001 * self.sample_rate)).exp();
        let release_coef = (-2.2 / (self.release_ms * 0.001 * self.sample_rate)).exp();

        input.iter().map(|&sample| {
            let input_abs = sample.abs();

            // Envelope follower
            let coef = if input_abs > self.envelope { attack_coef } else { release_coef };
            self.envelope = coef * self.envelope + (1.0 - coef) * input_abs;

            // Convert to dB
            let input_db = if self.envelope > 0.0 {
                20.0 * self.envelope.log10()
            } else {
                -96.0
            };

            // Compute gain reduction
            let gain_db = self.compute_gain(input_db);
            let gain_linear = 10.0_f32.powf(gain_db / 20.0);

            sample * gain_linear
        }).collect()
    }

    #[wasm_bindgen]
    pub fn get_gain_reduction(&self) -> f32 {
        let input_db = if self.envelope > 0.0 {
            20.0 * self.envelope.log10()
        } else {
            -96.0
        };

        -self.compute_gain(input_db) + self.makeup_gain_db
    }
}

// ============================================================================
// Limiter
// ============================================================================

#[wasm_bindgen]
pub struct Limiter {
    ceiling_db: f32,
    release_ms: f32,
    envelope: f32,
    sample_rate: f32,
}

#[wasm_bindgen]
impl Limiter {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            ceiling_db: -0.3,
            release_ms: 50.0,
            envelope: 0.0,
            sample_rate,
        }
    }

    #[wasm_bindgen]
    pub fn set_ceiling(&mut self, db: f32) {
        self.ceiling_db = db.clamp(-12.0, 0.0);
    }

    #[wasm_bindgen]
    pub fn set_release(&mut self, ms: f32) {
        self.release_ms = ms.clamp(1.0, 500.0);
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let ceiling_linear = 10.0_f32.powf(self.ceiling_db / 20.0);
        let release_coef = (-2.2 / (self.release_ms * 0.001 * self.sample_rate)).exp();

        input.iter().map(|&sample| {
            let input_abs = sample.abs();

            // Compute required gain reduction
            let target = if input_abs > ceiling_linear {
                ceiling_linear / input_abs
            } else {
                1.0
            };

            // Smooth envelope (instant attack, smooth release)
            if target < self.envelope {
                self.envelope = target;
            } else {
                self.envelope = release_coef * self.envelope + (1.0 - release_coef) * target;
            }

            sample * self.envelope
        }).collect()
    }
}

// ============================================================================
// Effects Chain
// ============================================================================

#[wasm_bindgen]
pub struct EffectsChain {
    eq: ParametricEQ,
    reverb: Reverb,
    compressor: Compressor,
    limiter: Limiter,

    eq_enabled: bool,
    reverb_enabled: bool,
    compressor_enabled: bool,
    limiter_enabled: bool,
}

#[wasm_bindgen]
impl EffectsChain {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            eq: ParametricEQ::new(sample_rate),
            reverb: Reverb::new(sample_rate),
            compressor: Compressor::new(sample_rate),
            limiter: Limiter::new(sample_rate),
            eq_enabled: false,
            reverb_enabled: false,
            compressor_enabled: false,
            limiter_enabled: true,
        }
    }

    #[wasm_bindgen]
    pub fn enable_eq(&mut self, enabled: bool) {
        self.eq_enabled = enabled;
    }

    #[wasm_bindgen]
    pub fn enable_reverb(&mut self, enabled: bool) {
        self.reverb_enabled = enabled;
    }

    #[wasm_bindgen]
    pub fn enable_compressor(&mut self, enabled: bool) {
        self.compressor_enabled = enabled;
    }

    #[wasm_bindgen]
    pub fn enable_limiter(&mut self, enabled: bool) {
        self.limiter_enabled = enabled;
    }

    #[wasm_bindgen]
    pub fn get_eq(&mut self) -> *mut ParametricEQ {
        &mut self.eq
    }

    #[wasm_bindgen]
    pub fn get_reverb(&mut self) -> *mut Reverb {
        &mut self.reverb
    }

    #[wasm_bindgen]
    pub fn get_compressor(&mut self) -> *mut Compressor {
        &mut self.compressor
    }

    #[wasm_bindgen]
    pub fn get_limiter(&mut self) -> *mut Limiter {
        &mut self.limiter
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output = input.to_vec();

        if self.eq_enabled {
            output = self.eq.process(&output);
        }

        if self.compressor_enabled {
            output = self.compressor.process(&output);
        }

        if self.reverb_enabled {
            output = self.reverb.process(&output);
        }

        if self.limiter_enabled {
            output = self.limiter.process(&output);
        }

        output
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biquad_lowpass() {
        let mut filter = BiquadFilter::new(44100.0);
        filter.set_lowpass(1000.0, 0.707);

        // Process a signal
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
        let output = filter.process(&input);

        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_eq() {
        let mut eq = ParametricEQ::new(44100.0);
        eq.set_low_shelf(100.0, 6.0);
        eq.set_mid(1000.0, 1.0, -3.0);

        let input: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
        let output = eq.process(&input);

        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_reverb() {
        let mut reverb = Reverb::new(44100.0);
        reverb.set_room_size(0.8);
        reverb.set_mix(0.3);

        let input: Vec<f32> = (0..4410).map(|i| if i < 100 { 1.0 } else { 0.0 }).collect();
        let output = reverb.process(&input);

        assert_eq!(output.len(), input.len());
        // Reverb tail should have non-zero values after impulse
        assert!(output.iter().skip(1000).any(|&s| s.abs() > 0.001));
    }

    #[test]
    fn test_compressor() {
        let mut comp = Compressor::new(44100.0);
        comp.set_threshold(-20.0);
        comp.set_ratio(4.0);

        let input: Vec<f32> = (0..1000).map(|_| 0.5).collect();
        let output = comp.process(&input);

        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_limiter() {
        let mut limiter = Limiter::new(44100.0);
        limiter.set_ceiling(-3.0);

        let input: Vec<f32> = (0..1000).map(|_| 1.0).collect();
        let output = limiter.process(&input);

        let ceiling_linear = 10.0_f32.powf(-3.0 / 20.0);
        assert!(output.iter().all(|&s| s.abs() <= ceiling_linear + 0.01));
    }
}
