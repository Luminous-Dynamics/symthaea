//! # Formant Vocoder
//!
//! Converts formant frames to audio samples using source-filter synthesis.
//!
//! ## Theory
//!
//! Speech synthesis uses the source-filter model:
//! 1. **Source**: Glottal pulse train (voiced) or noise (unvoiced)
//! 2. **Filter**: Cascade of resonators modeling vocal tract formants
//!
//! ```text
//! ┌─────────────────┐     ┌───────────────────────┐     ┌─────────────┐
//! │ LF Glottal +   │────►│ Formant Filters       │────►│   Output    │
//! │ Aspiration +   │     │ (F1–F5, parallel)     │     │   Audio     │
//! │ Noise          │     └───────────────────────┘     └─────────────┘
//! └─────────────────┘
//! ```
//!
//! ## Implementation
//!
//! - Glottal pulse: Liljencrants-Fant (LF) model parameterized by Rd
//! - Aspiration: White noise during glottal open phase (breathiness)
//! - Noise: Pink noise for fricatives
//! - Filters: 5 second-order IIR resonators (F1–F3 articulation, F4/F5 speaker constants)

use crate::voice::articulatory_synthesizer::FormantFrame;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// VOCODER CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Vocoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocoderConfig {
    /// Output sample rate (Hz)
    pub sample_rate: u32,
    /// Number of formant filters (typically 3-5)
    pub num_formants: usize,
    /// Master volume (0.0 to 1.0)
    pub volume: f32,
    /// Glottal pulse shape (0.0 = breathy, 1.0 = pressed)
    pub glottal_shape: f32,
    /// Noise floor for unvoiced sounds
    pub noise_floor: f32,
    /// Anti-aliasing filter cutoff ratio
    pub aa_cutoff: f32,
    /// F4 frequency (Hz) — pharyngeal resonance (speaker-dependent constant)
    pub f4_freq: f32,
    /// F5 frequency (Hz) — nasal cavity (speaker-dependent constant)
    pub f5_freq: f32,
    /// F4 bandwidth (Hz)
    pub f4_bandwidth: f32,
    /// F5 bandwidth (Hz)
    pub f5_bandwidth: f32,
    /// Aspiration noise level (0.0 to 0.1 typical)
    pub aspiration_level: f32,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            num_formants: 5,
            volume: 0.8,
            glottal_shape: 0.5,
            noise_floor: 0.02,
            aa_cutoff: 0.45,
            f4_freq: 3500.0,
            f5_freq: 4500.0,
            f4_bandwidth: 250.0,
            f5_bandwidth: 300.0,
            aspiration_level: 0.03,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BIQUAD RESONATOR FILTER
// ═══════════════════════════════════════════════════════════════════════════════

/// Second-order IIR resonator (biquad)
///
/// Implements a bandpass filter centered at the formant frequency
/// with bandwidth controlled by Q factor.
#[derive(Debug, Clone, Default)]
struct Resonator {
    // Coefficients
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    // State
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Resonator {
    /// Create a new resonator
    fn new() -> Self {
        Self::default()
    }

    /// Set resonator frequency and bandwidth
    fn set_params(&mut self, freq: f32, bandwidth: f32, sample_rate: f32) {
        if freq <= 0.0 || freq >= sample_rate / 2.0 {
            // Invalid frequency - make passthrough
            self.b0 = 1.0;
            self.b1 = 0.0;
            self.b2 = 0.0;
            self.a1 = 0.0;
            self.a2 = 0.0;
            return;
        }

        let omega = 2.0 * std::f32::consts::PI * freq / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();

        // Use resonant lowpass for formant synthesis (better than bandpass cascade)
        // This gives unity gain at DC and resonant peak at formant frequency
        let q = freq / bandwidth.max(20.0);
        let alpha = sin_omega / (2.0 * q);

        // Resonant lowpass coefficients (better for formant synthesis)
        // Using peaking EQ style for formant resonance
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        // Bandpass with unity peak gain (constant peak, not constant skirt)
        let peak_gain = q; // Compensate for Q
        let b0 = alpha * peak_gain;
        let b1 = 0.0;
        let b2 = -alpha * peak_gain;

        // Normalize by a0
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    /// Process one sample
    fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        // Update state
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    /// Reset state
    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GLOTTAL SOURCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Glottal pulse generator using the Liljencrants-Fant (LF) model.
///
/// The LF model is the standard in speech synthesis research, parameterized
/// by Rd (voice quality descriptor from Fant 1995):
/// - Rd = 0.3: Pressed voice (tight, bright)
/// - Rd = 1.0: Modal voice (normal speech)
/// - Rd = 2.7: Breathy voice (soft, airy)
///
/// The `shape` field (0.0–1.0) maps linearly to Rd: `Rd = 0.3 + shape * 2.4`
#[derive(Debug, Clone)]
struct GlottalSource {
    phase: f32,
    sample_rate: f32,
    /// Shape parameter: 0.0 = pressed (Rd=0.3), 1.0 = breathy (Rd=2.7)
    shape: f32,
}

impl GlottalSource {
    fn new(sample_rate: f32, shape: f32) -> Self {
        Self {
            phase: 0.0,
            sample_rate,
            shape: shape.clamp(0.0, 1.0),
        }
    }

    /// Generate one sample at given fundamental frequency using LF model.
    fn tick(&mut self, f0: f32) -> f32 {
        if f0 <= 0.0 {
            return 0.0;
        }

        let period = self.sample_rate / f0;
        let phase_inc = 1.0 / period;

        // Advance phase
        self.phase += phase_inc;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        // Map shape (0–1) to Rd (0.3–2.7)
        let rd = 0.3 + self.shape * 2.4;

        // Derive LF timing parameters from Rd (Fant 1995)
        let tp = 0.1 + 0.22 * rd; // Peak position (fraction of period)
        let te = tp * (1.0 + (5.0 - 4.6 * rd.min(2.7)) * 0.01); // Excitation instant
        let ta = 0.01 * (0.2 + 3.0 * (rd - 1.0).max(0.0)); // Return time constant

        // Closed phase boundary (fraction of period)
        let tc = te + ta * 3.0; // ~3 time constants for return phase

        let t = self.phase; // Current position in glottal cycle (0–1)

        if t < te {
            // Open phase: E(t) = E0 * exp(alpha*t) * sin(omega_g * t)
            let omega_g = std::f32::consts::PI / tp;
            let alpha = if te > 0.01 { 1.0 / te } else { 100.0 }; // Growth rate

            let t_scaled = t;
            let envelope = (alpha * t_scaled * 0.5).exp(); // Gentle exponential growth
            let oscillation = (omega_g * t_scaled).sin();

            -envelope * oscillation * 0.8 // Negative pressure convention
        } else if t < tc.min(1.0) {
            // Return phase: rapid exponential decay
            let epsilon = if ta > 0.001 { 1.0 / ta } else { 1000.0 };
            let t_ret = t - te;

            // E(t) = decay from excitation instant
            let decay = (-epsilon * t_ret * 0.3).exp();
            -decay * 0.4
        } else {
            // Closed phase: vocal folds closed, no airflow
            0.0
        }
    }

    /// Whether the glottal source is in the open phase for the current sample.
    fn is_open(&self) -> bool {
        let rd = 0.3 + self.shape * 2.4;
        let tp = 0.1 + 0.22 * rd;
        let te = tp * (1.0 + (5.0 - 4.6 * rd.min(2.7)) * 0.01);
        self.phase < te
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NOISE SOURCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Noise generator for unvoiced sounds
#[derive(Debug, Clone)]
struct NoiseSource {
    state: u64,
    pink_state: [f32; 7], // For pink noise filtering
}

impl NoiseSource {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
            pink_state: [0.0; 7],
        }
    }

    /// Generate white noise sample
    fn white(&mut self) -> f32 {
        // Xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;

        // Map to [-1, 1]
        (self.state as f32 / u64::MAX as f32) * 2.0 - 1.0
    }

    /// Generate pink noise sample (better for fricatives)
    fn pink(&mut self) -> f32 {
        let white = self.white();

        // Voss-McCartney algorithm approximation
        // Simple IIR low-pass cascade
        self.pink_state[0] = 0.99886 * self.pink_state[0] + white * 0.0555179;
        self.pink_state[1] = 0.99332 * self.pink_state[1] + white * 0.0750759;
        self.pink_state[2] = 0.96900 * self.pink_state[2] + white * 0.153_852;
        self.pink_state[3] = 0.86650 * self.pink_state[3] + white * 0.3104856;
        self.pink_state[4] = 0.55000 * self.pink_state[4] + white * 0.5329522;
        self.pink_state[5] = -0.7616 * self.pink_state[5] - white * 0.0168980;

        let pink = self.pink_state[0]
            + self.pink_state[1]
            + self.pink_state[2]
            + self.pink_state[3]
            + self.pink_state[4]
            + self.pink_state[5]
            + self.pink_state[6]
            + white * 0.5362;

        self.pink_state[6] = white * 0.115926;

        pink * 0.11 // Normalize
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORMANT VOCODER
// ═══════════════════════════════════════════════════════════════════════════════

/// Formant vocoder for speech synthesis
///
/// Converts formant frames to audio using source-filter synthesis.
#[derive(Debug, Clone)]
pub struct FormantVocoder {
    config: VocoderConfig,
    /// Formant resonators (F1, F2, F3, ...)
    resonators: Vec<Resonator>,
    /// Glottal pulse generator
    glottal: GlottalSource,
    /// Noise source for unvoiced
    noise: NoiseSource,
    /// Low-pass filter for smoothing
    lowpass: Resonator,
    /// Current frame index for interpolation
    frame_idx: usize,
    /// Samples since last frame
    samples_in_frame: usize,
}

impl FormantVocoder {
    /// Create a new vocoder with default configuration
    pub fn new() -> Self {
        Self::with_config(VocoderConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: VocoderConfig) -> Self {
        let mut resonators = Vec::with_capacity(config.num_formants);
        for _ in 0..config.num_formants {
            resonators.push(Resonator::new());
        }

        let mut lowpass = Resonator::new();
        lowpass.set_params(
            config.sample_rate as f32 * config.aa_cutoff,
            config.sample_rate as f32 * 0.2,
            config.sample_rate as f32,
        );

        Self {
            glottal: GlottalSource::new(config.sample_rate as f32, config.glottal_shape),
            noise: NoiseSource::new(0xDEADBEEF),
            resonators,
            lowpass,
            config,
            frame_idx: 0,
            samples_in_frame: 0,
        }
    }

    /// Reset vocoder state
    pub fn reset(&mut self) {
        self.glottal.reset();
        for res in &mut self.resonators {
            res.reset();
        }
        self.lowpass.reset();
        self.frame_idx = 0;
        self.samples_in_frame = 0;
    }

    /// Synthesize audio from formant frames
    pub fn synthesize(&mut self, frames: &[FormantFrame]) -> Vec<f32> {
        if frames.is_empty() {
            return Vec::new();
        }

        self.reset();

        // Calculate total samples
        let total_duration = frames.last().expect("frames checked non-empty above").time
            - frames.first().expect("frames checked non-empty above").time;
        let frame_duration = if frames.len() > 1 {
            total_duration / (frames.len() - 1) as f32
        } else {
            0.1
        };

        let samples_per_frame = (frame_duration * self.config.sample_rate as f32) as usize;
        let total_samples = samples_per_frame * frames.len();

        let mut audio = Vec::with_capacity(total_samples);

        // Process frame by frame
        for (i, frame) in frames.iter().enumerate() {
            // Get next frame for interpolation
            let next_frame = frames.get(i + 1).unwrap_or(frame);

            // Update resonator parameters
            self.update_resonators(frame);

            // Generate samples for this frame
            for j in 0..samples_per_frame {
                let t = j as f32 / samples_per_frame as f32;

                // Interpolate parameters
                let current = frame.lerp(next_frame, t);

                // Generate source signal with LF glottal model
                let voiced = self.glottal.tick(current.f0) * current.voicing;
                let unvoiced = self.noise.pink() * (1.0 - current.voicing) * 0.3;

                // Aspiration noise: breathiness during glottal open phase
                let aspiration = if self.glottal.is_open() {
                    self.noise.white()
                        * self.config.aspiration_level
                        * (1.0 - self.glottal.shape * 0.7)
                } else {
                    0.0
                };

                let source = (voiced + aspiration + unvoiced) * 30.0;

                // Apply formant filters (parallel configuration with weighted sum)
                let mut filtered = 0.0;
                if self.resonators.len() >= 3 {
                    filtered += self.resonators[0].process(source) * 1.0;  // F1 (strongest)
                    filtered += self.resonators[1].process(source) * 0.5;  // F2
                    filtered += self.resonators[2].process(source) * 0.25; // F3
                } else {
                    for res in &mut self.resonators {
                        filtered += res.process(source);
                    }
                }
                // F4/F5: speaker-dependent higher formants (subtle presence)
                if self.resonators.len() >= 4 {
                    filtered += self.resonators[3].process(source) * 0.1;  // F4
                }
                if self.resonators.len() >= 5 {
                    filtered += self.resonators[4].process(source) * 0.05; // F5
                }

                // Apply energy envelope
                let output = filtered * current.energy * self.config.volume;

                // Low-pass filter for smoothing
                let smoothed = self.lowpass.process(output);

                // Soft clip to prevent distortion
                let clipped = soft_clip(smoothed);

                audio.push(clipped);
            }
        }

        audio
    }

    /// Update resonator parameters from frame
    fn update_resonators(&mut self, frame: &FormantFrame) {
        let sr = self.config.sample_rate as f32;

        if !self.resonators.is_empty() {
            self.resonators[0].set_params(frame.f1, frame.b1, sr);
        }
        if self.resonators.len() >= 2 {
            self.resonators[1].set_params(frame.f2, frame.b2, sr);
        }
        if self.resonators.len() >= 3 {
            self.resonators[2].set_params(frame.f3, frame.b3, sr);
        }
        // F4/F5 are speaker-dependent constants (not articulation-dependent)
        if self.resonators.len() >= 4 {
            self.resonators[3].set_params(self.config.f4_freq, self.config.f4_bandwidth, sr);
        }
        if self.resonators.len() >= 5 {
            self.resonators[4].set_params(self.config.f5_freq, self.config.f5_bandwidth, sr);
        }
    }

    /// Synthesize audio for a single formant frame (streaming mode).
    ///
    /// Unlike `synthesize()`, this does NOT reset state between calls — resonator
    /// and glottal source states carry over for smooth real-time output.
    /// Call this at your frame rate (e.g., 200Hz) with `samples_per_frame`
    /// samples per call (e.g., 120 at 24kHz/200Hz).
    pub fn synthesize_frame(&mut self, frame: &FormantFrame, samples_per_frame: usize) -> Vec<f32> {
        self.update_resonators(frame);
        let mut audio = Vec::with_capacity(samples_per_frame);

        for _ in 0..samples_per_frame {
            let voiced = self.glottal.tick(frame.f0) * frame.voicing;
            let unvoiced = self.noise.pink() * (1.0 - frame.voicing) * 0.3;

            // Aspiration noise during glottal open phase
            let aspiration = if self.glottal.is_open() {
                self.noise.white()
                    * self.config.aspiration_level
                    * (1.0 - self.glottal.shape * 0.7)
            } else {
                0.0
            };

            let source = (voiced + aspiration + unvoiced) * 30.0;

            let mut filtered = 0.0;
            if self.resonators.len() >= 3 {
                filtered += self.resonators[0].process(source) * 1.0;
                filtered += self.resonators[1].process(source) * 0.5;
                filtered += self.resonators[2].process(source) * 0.25;
            } else {
                for res in &mut self.resonators {
                    filtered += res.process(source);
                }
            }
            if self.resonators.len() >= 4 {
                filtered += self.resonators[3].process(source) * 0.1;
            }
            if self.resonators.len() >= 5 {
                filtered += self.resonators[4].process(source) * 0.05;
            }

            let output = filtered * frame.energy * self.config.volume;
            let smoothed = self.lowpass.process(output);
            audio.push(soft_clip(smoothed));
        }

        audio
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get configuration
    pub fn config(&self) -> &VocoderConfig {
        &self.config
    }
}

impl Default for FormantVocoder {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Soft clipping function to prevent harsh distortion
fn soft_clip(x: f32) -> f32 {
    if x.abs() < 0.5 {
        x
    } else if x > 0.0 {
        0.5 + (1.0 - (-2.0 * (x - 0.5)).exp()) * 0.5
    } else {
        -0.5 - (1.0 - (-2.0 * (-x - 0.5)).exp()) * 0.5
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resonator_basic() {
        let mut res = Resonator::new();
        res.set_params(500.0, 60.0, 24000.0);

        // Process some samples
        let mut output = 0.0;
        for i in 0..1000 {
            let input = if i == 0 { 1.0 } else { 0.0 }; // Impulse
            output = res.process(input);
        }

        // Should have decayed
        assert!(output.abs() < 0.1);
    }

    #[test]
    fn test_glottal_source() {
        let mut glottal = GlottalSource::new(24000.0, 0.5);

        let mut samples = Vec::new();
        for _ in 0..480 {
            // 20ms at 24kHz
            samples.push(glottal.tick(120.0)); // 120 Hz
        }

        // Should have some non-zero samples
        assert!(samples.iter().any(|&s| s.abs() > 0.1));

        // Should be bounded (glottal pulses can have transients up to ~3x)
        assert!(samples.iter().all(|&s| s.abs() < 4.0));
    }

    #[test]
    fn test_noise_source() {
        let mut noise = NoiseSource::new(42);

        let mut samples = Vec::new();
        for _ in 0..1000 {
            samples.push(noise.pink());
        }

        // Should have variance
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance: f32 =
            samples.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / samples.len() as f32;

        assert!(variance > 0.001);
        assert!(variance < 1.0);
    }

    #[test]
    fn test_vocoder_synthesis() {
        let mut vocoder = FormantVocoder::new();

        // Create a simple vowel frame
        let frames = vec![
            FormantFrame {
                f1: 500.0,
                f2: 1500.0,
                f3: 2500.0,
                b1: 60.0,
                b2: 90.0,
                b3: 150.0,
                f0: 120.0,
                energy: 0.8,
                voicing: 1.0,
                time: 0.0,
            },
            FormantFrame {
                f1: 500.0,
                f2: 1500.0,
                f3: 2500.0,
                b1: 60.0,
                b2: 90.0,
                b3: 150.0,
                f0: 120.0,
                energy: 0.8,
                voicing: 1.0,
                time: 0.1,
            },
        ];

        let audio = vocoder.synthesize(&frames);

        assert!(!audio.is_empty());
        assert!(audio.iter().any(|&s| s.abs() > 0.01)); // Has content
        assert!(audio.iter().all(|&s| s.abs() < 1.5)); // Not clipping badly
    }

    #[test]
    fn test_soft_clip() {
        assert!((soft_clip(0.3) - 0.3).abs() < 0.01);
        assert!(soft_clip(2.0) < 1.0);
        assert!(soft_clip(-2.0) > -1.0);
    }

    #[test]
    fn test_five_formant_synthesis() {
        // Default config now uses 5 formants
        let config = VocoderConfig::default();
        assert_eq!(config.num_formants, 5);

        let mut vocoder = FormantVocoder::with_config(config);
        assert_eq!(vocoder.resonators.len(), 5);

        let frames = vec![
            FormantFrame {
                f1: 730.0, f2: 1090.0, f3: 2440.0,
                b1: 60.0, b2: 90.0, b3: 150.0,
                f0: 120.0, energy: 0.8, voicing: 1.0, time: 0.0,
            },
            FormantFrame {
                f1: 730.0, f2: 1090.0, f3: 2440.0,
                b1: 60.0, b2: 90.0, b3: 150.0,
                f0: 120.0, energy: 0.8, voicing: 1.0, time: 0.1,
            },
        ];

        let audio = vocoder.synthesize(&frames);
        assert!(!audio.is_empty(), "5-formant synthesis should produce audio");
        assert!(audio.iter().any(|&s| s.abs() > 0.01), "Should have content");
        assert!(audio.iter().all(|&s| s.abs() < 1.5), "Should not clip");
    }

    #[test]
    fn test_lf_glottal_shape_variation() {
        // Different shapes (Rd values) should produce different waveforms
        let mut glottal_pressed = GlottalSource::new(24000.0, 0.0); // Rd=0.3
        let mut glottal_breathy = GlottalSource::new(24000.0, 1.0); // Rd=2.7

        let mut pressed_samples = Vec::new();
        let mut breathy_samples = Vec::new();
        for _ in 0..480 { // 20ms at 24kHz
            pressed_samples.push(glottal_pressed.tick(120.0));
            breathy_samples.push(glottal_breathy.tick(120.0));
        }

        // Both should produce non-zero output
        assert!(pressed_samples.iter().any(|&s| s.abs() > 0.01), "Pressed should have output");
        assert!(breathy_samples.iter().any(|&s| s.abs() > 0.01), "Breathy should have output");

        // They should differ (different waveform shapes)
        let diff: f32 = pressed_samples.iter().zip(&breathy_samples)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();
        assert!(diff > 0.1, "Different shapes should produce different waveforms, diff={diff}");
    }

    #[test]
    fn test_aspiration_adds_breathiness() {
        let frame = FormantFrame {
            f1: 500.0, f2: 1500.0, f3: 2500.0,
            b1: 60.0, b2: 90.0, b3: 150.0,
            f0: 120.0, energy: 0.8, voicing: 1.0, time: 0.0,
        };

        // Synthesize with aspiration
        let config_with = VocoderConfig {
            aspiration_level: 0.1,
            ..VocoderConfig::default()
        };
        let mut vocoder_with = FormantVocoder::with_config(config_with);
        let audio_with = vocoder_with.synthesize_frame(&frame, 480);

        // Synthesize without aspiration
        let config_without = VocoderConfig {
            aspiration_level: 0.0,
            ..VocoderConfig::default()
        };
        let mut vocoder_without = FormantVocoder::with_config(config_without);
        let audio_without = vocoder_without.synthesize_frame(&frame, 480);

        // Aspiration adds high-frequency energy: compute spectral difference
        // Simple proxy: sum of squared differences should be non-trivial
        let diff_energy: f32 = audio_with.iter().zip(&audio_without)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>();
        assert!(
            diff_energy > 0.001,
            "Aspiration should measurably change the output, diff_energy={diff_energy}"
        );
    }
}
