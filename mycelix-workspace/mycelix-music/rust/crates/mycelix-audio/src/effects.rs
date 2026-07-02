// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio effects processors

use crate::{AudioBuffer, AudioError, AudioProcessor, AudioResult};
use crate::dsp::{BiquadFilter, BiquadCoeffs, EnvelopeFollower, db_to_linear, linear_to_db};

/// Gain processor
pub struct Gain {
    gain: f32,
}

impl Gain {
    pub fn new(gain_db: f32) -> Self {
        Self {
            gain: db_to_linear(gain_db),
        }
    }

    pub fn set_gain_db(&mut self, gain_db: f32) {
        self.gain = db_to_linear(gain_db);
    }
}

impl AudioProcessor for Gain {
    fn process(&mut self, buffer: &mut AudioBuffer) -> AudioResult<()> {
        for sample in &mut buffer.samples {
            *sample *= self.gain;
        }
        Ok(())
    }

    fn reset(&mut self) {}

    fn name(&self) -> &str {
        "Gain"
    }
}

/// Hard limiter
pub struct Limiter {
    threshold: f32,
}

impl Limiter {
    pub fn new(threshold_db: f32) -> Self {
        Self {
            threshold: db_to_linear(threshold_db),
        }
    }
}

impl AudioProcessor for Limiter {
    fn process(&mut self, buffer: &mut AudioBuffer) -> AudioResult<()> {
        for sample in &mut buffer.samples {
            if sample.abs() > self.threshold {
                *sample = self.threshold * sample.signum();
            }
        }
        Ok(())
    }

    fn reset(&mut self) {}

    fn name(&self) -> &str {
        "Limiter"
    }
}

/// Simple compressor
pub struct Compressor {
    threshold: f32,
    ratio: f32,
    attack_coeff: f32,
    release_coeff: f32,
    envelope: f32,
    makeup_gain: f32,
}

impl Compressor {
    pub fn new(
        sample_rate: f32,
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        makeup_db: f32,
    ) -> Self {
        Self {
            threshold: threshold_db,
            ratio,
            attack_coeff: (-1.0 / (attack_ms * sample_rate / 1000.0)).exp(),
            release_coeff: (-1.0 / (release_ms * sample_rate / 1000.0)).exp(),
            envelope: 0.0,
            makeup_gain: db_to_linear(makeup_db),
        }
    }

    fn compute_gain(&self, input_db: f32) -> f32 {
        if input_db > self.threshold {
            let over = input_db - self.threshold;
            let compressed_over = over / self.ratio;
            db_to_linear(self.threshold + compressed_over - input_db)
        } else {
            1.0
        }
    }
}

impl AudioProcessor for Compressor {
    fn process(&mut self, buffer: &mut AudioBuffer) -> AudioResult<()> {
        for sample in &mut buffer.samples {
            let input_abs = sample.abs();
            let input_db = linear_to_db(input_abs);

            // Envelope follower
            let coeff = if input_abs > self.envelope {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            self.envelope = coeff * self.envelope + (1.0 - coeff) * input_abs;

            let env_db = linear_to_db(self.envelope);
            let gain = self.compute_gain(env_db);

            *sample *= gain * self.makeup_gain;
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.envelope = 0.0;
    }

    fn name(&self) -> &str {
        "Compressor"
    }
}

/// Simple delay effect
pub struct Delay {
    buffer: Vec<f32>,
    write_pos: usize,
    delay_samples: usize,
    feedback: f32,
    mix: f32,
}

impl Delay {
    pub fn new(sample_rate: f32, delay_ms: f32, feedback: f32, mix: f32) -> Self {
        let delay_samples = (delay_ms * sample_rate / 1000.0) as usize;
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
            delay_samples,
            feedback: feedback.clamp(0.0, 0.99),
            mix: mix.clamp(0.0, 1.0),
        }
    }
}

impl AudioProcessor for Delay {
    fn process(&mut self, buffer: &mut AudioBuffer) -> AudioResult<()> {
        for sample in &mut buffer.samples {
            let read_pos = (self.write_pos + self.buffer.len() - self.delay_samples) % self.buffer.len();
            let delayed = self.buffer[read_pos];

            self.buffer[self.write_pos] = *sample + delayed * self.feedback;
            self.write_pos = (self.write_pos + 1) % self.buffer.len();

            *sample = *sample * (1.0 - self.mix) + delayed * self.mix;
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }

    fn name(&self) -> &str {
        "Delay"
    }

    fn latency(&self) -> usize {
        0 // No latency for this simple implementation
    }
}

/// Parametric EQ band
pub struct EqBand {
    filter: BiquadFilter,
    frequency: f32,
    gain_db: f32,
    q: f32,
}

impl EqBand {
    pub fn new(sample_rate: f32, frequency: f32, gain_db: f32, q: f32, channels: usize) -> Self {
        let coeffs = Self::compute_peaking_coeffs(sample_rate, frequency, gain_db, q);
        Self {
            filter: BiquadFilter::new(coeffs, channels),
            frequency,
            gain_db,
            q,
        }
    }

    fn compute_peaking_coeffs(sample_rate: f32, frequency: f32, gain_db: f32, q: f32) -> BiquadCoeffs {
        let omega = 2.0 * std::f32::consts::PI * frequency / sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q);
        let a = 10.0f32.powf(gain_db / 40.0);

        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha / a;

        BiquadCoeffs {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

impl AudioProcessor for EqBand {
    fn process(&mut self, buffer: &mut AudioBuffer) -> AudioResult<()> {
        self.filter.process(buffer)
    }

    fn reset(&mut self) {
        self.filter.reset();
    }

    fn name(&self) -> &str {
        "EQ Band"
    }
}

/// Soft clipper for saturation
pub struct SoftClipper {
    drive: f32,
}

impl SoftClipper {
    pub fn new(drive_db: f32) -> Self {
        Self {
            drive: db_to_linear(drive_db),
        }
    }

    fn soft_clip(x: f32) -> f32 {
        if x.abs() < 1.0 / 3.0 {
            2.0 * x
        } else if x.abs() < 2.0 / 3.0 {
            let sign = x.signum();
            sign * (3.0 - (2.0 - 3.0 * x.abs()).powi(2)) / 3.0
        } else {
            x.signum()
        }
    }
}

impl AudioProcessor for SoftClipper {
    fn process(&mut self, buffer: &mut AudioBuffer) -> AudioResult<()> {
        for sample in &mut buffer.samples {
            *sample = Self::soft_clip(*sample * self.drive);
        }
        Ok(())
    }

    fn reset(&mut self) {}

    fn name(&self) -> &str {
        "Soft Clipper"
    }
}

/// Stereo width control
pub struct StereoWidth {
    width: f32,
}

impl StereoWidth {
    pub fn new(width: f32) -> Self {
        Self {
            width: width.clamp(0.0, 2.0),
        }
    }
}

impl AudioProcessor for StereoWidth {
    fn process(&mut self, buffer: &mut AudioBuffer) -> AudioResult<()> {
        if buffer.channels != 2 {
            return Ok(());
        }

        for chunk in buffer.samples.chunks_mut(2) {
            let left = chunk[0];
            let right = chunk[1];

            let mid = (left + right) / 2.0;
            let side = (left - right) / 2.0;

            chunk[0] = mid + side * self.width;
            chunk[1] = mid - side * self.width;
        }
        Ok(())
    }

    fn reset(&mut self) {}

    fn name(&self) -> &str {
        "Stereo Width"
    }
}

/// Fade in/out processor
pub struct Fade {
    fade_in_samples: usize,
    fade_out_samples: usize,
    current_sample: usize,
    total_samples: usize,
}

impl Fade {
    pub fn new(sample_rate: f32, fade_in_ms: f32, fade_out_ms: f32, total_samples: usize) -> Self {
        Self {
            fade_in_samples: (fade_in_ms * sample_rate / 1000.0) as usize,
            fade_out_samples: (fade_out_ms * sample_rate / 1000.0) as usize,
            current_sample: 0,
            total_samples,
        }
    }
}

impl AudioProcessor for Fade {
    fn process(&mut self, buffer: &mut AudioBuffer) -> AudioResult<()> {
        let channels = buffer.channels;
        for sample in &mut buffer.samples {
            let frame = self.current_sample / channels;

            let gain = if frame < self.fade_in_samples {
                frame as f32 / self.fade_in_samples as f32
            } else if frame >= self.total_samples / channels - self.fade_out_samples {
                let remaining = (self.total_samples / channels) - frame;
                remaining as f32 / self.fade_out_samples as f32
            } else {
                1.0
            };

            *sample *= gain;
            self.current_sample += 1;
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.current_sample = 0;
    }

    fn name(&self) -> &str {
        "Fade"
    }
}
