// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Music → Consciousness feedback: the strange loop.
//!
//! Extracts audio features from rendered PCM, encodes them as a feedback
//! vector that can be injected back into the consciousness pipeline's
//! perception phase. This closes the loop:
//!
//! ```text
//! consciousness → music generation → audio features → perception → consciousness
//! ```
//!
//! The feedback vector modulates:
//! - spectral_centroid → serotonin (brightness ↔ warmth)
//! - spectral_flux → prediction_error (change rate ↔ surprise)
//! - rhythm_entropy → arousal (complexity ↔ activation)
//! - harmonic_tension → harmony_activations (dissonance ↔ resolution)
//! - rms_energy → dopamine (loudness ↔ reward signal)

use crate::MusicalState;

/// Extracted audio features from a PCM chunk.
#[derive(Debug, Clone, Copy, Default)]
pub struct AudioFeatures {
    /// Spectral centroid (brightness), normalized [0, 1].
    pub spectral_centroid: f32,
    /// Spectral flux (rate of spectral change), normalized [0, 1].
    pub spectral_flux: f32,
    /// Rhythm entropy (temporal complexity), normalized [0, 1].
    pub rhythm_entropy: f32,
    /// Harmonic tension (dissonance measure), normalized [0, 1].
    pub harmonic_tension: f32,
    /// RMS energy, normalized [0, 1].
    pub rms_energy: f32,
    /// Zero-crossing rate (proxy for noisiness), normalized [0, 1].
    pub zero_crossing_rate: f32,
}

/// Audio feedback encoder: extracts features and produces modulation signals.
pub struct AudioFeedbackEncoder {
    prev_spectrum: Vec<f32>,
    prev_rms: f32,
    onset_history: Vec<f32>,
    ema_alpha: f32,
    smoothed: AudioFeatures,
}

impl AudioFeedbackEncoder {
    /// Create a new feedback encoder.
    pub fn new() -> Self {
        Self {
            prev_spectrum: Vec::new(),
            prev_rms: 0.0,
            onset_history: Vec::with_capacity(64),
            ema_alpha: 0.15, // smoothing factor (lower = smoother)
            smoothed: AudioFeatures::default(),
        }
    }

    /// Extract audio features from a stereo PCM chunk.
    ///
    /// Returns raw (instantaneous) features. Use `smoothed_features()` for
    /// the EMA-smoothed version suitable for consciousness modulation.
    pub fn extract(&mut self, stereo_chunk: &[[f32; 2]], sample_rate: u32) -> AudioFeatures {
        if stereo_chunk.is_empty() {
            return AudioFeatures::default();
        }

        let sr = sample_rate as f32;
        let mono: Vec<f32> = stereo_chunk.iter().map(|s| (s[0] + s[1]) * 0.5).collect();

        // RMS energy
        let rms = (mono.iter().map(|s| s * s).sum::<f32>() / mono.len() as f32).sqrt();
        let rms_norm = (rms * 3.0).clamp(0.0, 1.0); // scale for typical synth levels

        // Zero-crossing rate
        let zcr = if mono.len() > 1 {
            let crossings = mono
                .windows(2)
                .filter(|w| w[0].signum() != w[1].signum())
                .count();
            crossings as f32 / mono.len() as f32
        } else {
            0.0
        };
        let zcr_norm = (zcr * 5.0).clamp(0.0, 1.0);

        // Spectral centroid via DFT magnitude spectrum
        let spectrum = compute_magnitude_spectrum(&mono);
        let centroid = if spectrum.is_empty() {
            0.0
        } else {
            let total_energy: f32 = spectrum.iter().sum();
            if total_energy > 1e-8 {
                let weighted_sum: f32 = spectrum
                    .iter()
                    .enumerate()
                    .map(|(i, &mag)| i as f32 * mag)
                    .sum();
                let centroid_bin = weighted_sum / total_energy;
                let centroid_hz = centroid_bin * sr / (spectrum.len() as f32 * 2.0);
                (centroid_hz / 8000.0).clamp(0.0, 1.0) // normalize to 8kHz
            } else {
                0.0
            }
        };

        // Spectral flux (change from previous chunk)
        let flux = if !self.prev_spectrum.is_empty() && self.prev_spectrum.len() == spectrum.len() {
            let diff_sum: f32 = spectrum
                .iter()
                .zip(self.prev_spectrum.iter())
                .map(|(&a, &b)| (a - b).max(0.0)) // half-wave rectified
                .sum();
            (diff_sum / spectrum.len().max(1) as f32 * 100.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.prev_spectrum = spectrum;

        // Rhythm entropy: variance of inter-onset intervals
        let onset_strength = (rms - self.prev_rms).max(0.0);
        self.prev_rms = rms;
        self.onset_history.push(onset_strength);
        if self.onset_history.len() > 64 {
            self.onset_history.remove(0);
        }
        let rhythm_entropy = if self.onset_history.len() > 2 {
            let mean: f32 =
                self.onset_history.iter().sum::<f32>() / self.onset_history.len() as f32;
            let variance: f32 = self
                .onset_history
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / self.onset_history.len() as f32;
            (variance.sqrt() * 10.0).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Harmonic tension: ratio of energy in dissonant intervals
        // Approximated by high ZCR relative to centroid (noise vs tone)
        let tension = if centroid > 0.1 {
            (zcr_norm / centroid).clamp(0.0, 1.0)
        } else {
            zcr_norm
        };

        let features = AudioFeatures {
            spectral_centroid: centroid,
            spectral_flux: flux,
            rhythm_entropy,
            harmonic_tension: tension,
            rms_energy: rms_norm,
            zero_crossing_rate: zcr_norm,
        };

        // EMA smoothing
        let a = self.ema_alpha;
        self.smoothed.spectral_centroid +=
            a * (features.spectral_centroid - self.smoothed.spectral_centroid);
        self.smoothed.spectral_flux += a * (features.spectral_flux - self.smoothed.spectral_flux);
        self.smoothed.rhythm_entropy +=
            a * (features.rhythm_entropy - self.smoothed.rhythm_entropy);
        self.smoothed.harmonic_tension +=
            a * (features.harmonic_tension - self.smoothed.harmonic_tension);
        self.smoothed.rms_energy += a * (features.rms_energy - self.smoothed.rms_energy);
        self.smoothed.zero_crossing_rate +=
            a * (features.zero_crossing_rate - self.smoothed.zero_crossing_rate);

        features
    }

    /// Get EMA-smoothed features (suitable for consciousness modulation).
    pub fn smoothed_features(&self) -> &AudioFeatures {
        &self.smoothed
    }

    /// Reset feedback state.
    pub fn reset(&mut self) {
        self.prev_spectrum.clear();
        self.prev_rms = 0.0;
        self.onset_history.clear();
        self.smoothed = AudioFeatures::default();
    }
}

impl AudioFeatures {
    /// Apply audio features as modulation to a MusicalState.
    ///
    /// This creates the strange loop: audio output modulates the consciousness
    /// state that generates the next chunk of audio.
    ///
    /// `strength` [0, 1]: how strongly audio features influence consciousness.
    /// 0.0 = open loop (no feedback), 1.0 = full feedback coupling.
    pub fn modulate_state(&self, state: &mut MusicalState, strength: f32) {
        let s = strength.clamp(0.0, 1.0);

        // Spectral centroid → inverse serotonin (bright audio → less serotonin → brighter next cycle)
        // This creates a self-reinforcing brightness loop dampened by the EMA
        state.serotonin += s * 0.1 * (0.5 - self.spectral_centroid);

        // Spectral flux → prediction error (rapid change → surprise)
        state.prediction_error += s * 0.15 * (self.spectral_flux - state.prediction_error);

        // Rhythm entropy → arousal (complex rhythm → higher arousal)
        state.arousal += s * 0.1 * (self.rhythm_entropy - state.arousal);

        // RMS energy → dopamine (loud → reward)
        state.dopamine += s * 0.08 * (self.rms_energy - state.dopamine);

        // Harmonic tension → attenuate ResonantCoherence harmony
        state.harmony_activations[0] +=
            s * 0.05 * (1.0 - self.harmonic_tension - state.harmony_activations[0]);

        // Clamp all to valid ranges
        state.serotonin = state.serotonin.clamp(0.0, 1.0);
        state.prediction_error = state.prediction_error.clamp(0.0, 1.0);
        state.arousal = state.arousal.clamp(0.0, 1.0);
        state.dopamine = state.dopamine.clamp(0.0, 1.0);
        for h in &mut state.harmony_activations {
            *h = h.clamp(0.0, 1.0);
        }
    }
}

/// Compute magnitude spectrum via DFT (simplified, real-valued input).
///
/// Returns N/2 magnitude bins. For production use, replace with FFT.
fn compute_magnitude_spectrum(samples: &[f32]) -> Vec<f32> {
    // Use a small window for efficiency (256 samples)
    let n = samples.len().min(256);
    if n < 4 {
        return Vec::new();
    }

    let half = n / 2;
    let mut spectrum = Vec::with_capacity(half);

    for k in 0..half {
        let mut real = 0.0f32;
        let mut imag = 0.0f32;
        let freq = std::f32::consts::TAU * k as f32 / n as f32;
        for (i, &sample) in samples.iter().take(n).enumerate() {
            let phase = freq * i as f32;
            real += sample * phase.cos();
            imag += sample * phase.sin();
        }
        spectrum.push((real * real + imag * imag).sqrt() / n as f32);
    }

    spectrum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_chunk(freq: f32, sr: u32, len: usize) -> Vec<[f32; 2]> {
        (0..len)
            .map(|i| {
                let s = (std::f32::consts::TAU * freq * i as f32 / sr as f32).sin() * 0.5;
                [s, s]
            })
            .collect()
    }

    #[test]
    fn extract_features_from_sine() {
        let mut encoder = AudioFeedbackEncoder::new();
        let chunk = sine_chunk(440.0, 44100, 1024);
        let features = encoder.extract(&chunk, 44100);

        assert!(features.rms_energy > 0.0, "should have energy");
        assert!(features.spectral_centroid > 0.0, "should have centroid");
        assert!(
            features.spectral_centroid < 0.5,
            "440Hz centroid should be low"
        );
    }

    #[test]
    fn flux_detects_change() {
        let mut encoder = AudioFeedbackEncoder::new();

        // Two identical chunks → low flux
        let chunk1 = sine_chunk(440.0, 44100, 1024);
        encoder.extract(&chunk1, 44100);
        let features2 = encoder.extract(&chunk1, 44100);
        let low_flux = features2.spectral_flux;

        // Then a different frequency → high flux
        let chunk2 = sine_chunk(2000.0, 44100, 1024);
        let features3 = encoder.extract(&chunk2, 44100);
        let high_flux = features3.spectral_flux;

        assert!(
            high_flux > low_flux,
            "frequency change should increase flux: {high_flux} vs {low_flux}"
        );
    }

    #[test]
    fn smoothing_converges() {
        let mut encoder = AudioFeedbackEncoder::new();
        let chunk = sine_chunk(440.0, 44100, 1024);

        for _ in 0..20 {
            encoder.extract(&chunk, 44100);
        }

        let smoothed = encoder.smoothed_features();
        assert!(smoothed.rms_energy > 0.0);
        assert!(smoothed.spectral_centroid > 0.0);
    }

    #[test]
    fn modulate_state_changes_values() {
        let features = AudioFeatures {
            spectral_centroid: 0.8,
            spectral_flux: 0.5,
            rhythm_entropy: 0.7,
            harmonic_tension: 0.3,
            rms_energy: 0.6,
            zero_crossing_rate: 0.4,
        };

        let mut state = MusicalState::default();
        let original_arousal = state.arousal;
        features.modulate_state(&mut state, 1.0);

        // Arousal should increase (rhythm_entropy 0.7 > default arousal 0.4)
        assert!(
            state.arousal > original_arousal,
            "arousal should increase from rhythm entropy"
        );
        // All values should remain in bounds
        assert!(state.serotonin >= 0.0 && state.serotonin <= 1.0);
        assert!(state.dopamine >= 0.0 && state.dopamine <= 1.0);
    }

    #[test]
    fn zero_strength_no_change() {
        let features = AudioFeatures {
            spectral_centroid: 1.0,
            spectral_flux: 1.0,
            rhythm_entropy: 1.0,
            harmonic_tension: 1.0,
            rms_energy: 1.0,
            zero_crossing_rate: 1.0,
        };

        let mut state = MusicalState::default();
        let original = state.clone();
        features.modulate_state(&mut state, 0.0);

        assert_eq!(state.arousal, original.arousal);
        assert_eq!(state.dopamine, original.dopamine);
    }

    #[test]
    fn empty_chunk_returns_defaults() {
        let mut encoder = AudioFeedbackEncoder::new();
        let features = encoder.extract(&[], 44100);
        assert_eq!(features.rms_energy, 0.0);
        assert_eq!(features.spectral_centroid, 0.0);
    }
}
