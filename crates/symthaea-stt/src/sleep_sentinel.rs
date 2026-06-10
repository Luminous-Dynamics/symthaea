// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sleep Sentinel: Consciousness State Detection via HDC/LTC
//!
//! The Trinity Sentinel processes three physiological signals:
//! - EEG: Brainwave patterns (delta, theta, alpha, beta bands)
//! - EOG: Eye movements (REM detection)
//! - EMG: Muscle tone (REM atonia detection)
//!
//! Uses the same LTC temporal dynamics + HDC projection architecture
//! that powers Symthaea's acoustic processing, adapted for biosignals.

use crate::edf_loader::SleepStage;
use crate::hdc::{HV16, bundle};
use crate::ltc::{LtcCell, LtcConfig};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;

/// Consciousness state as determined by the Sentinel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsciousnessState {
    /// Awake - high beta, alpha activity, voluntary eye/muscle movement
    Awake,
    /// Transitional - drowsy, reduced awareness
    Drowsy,
    /// Light sleep - spindles, K-complexes
    LightSleep,
    /// Deep sleep - high delta, minimal movement, consolidation phase
    DeepSleep,
    /// REM - dream state, rapid eye movements, muscle atonia
    Dreaming,
    /// Unknown / transitional
    Unknown,
}

impl ConsciousnessState {
    /// Convert to the legacy SleepStage enum for compatibility
    pub fn to_sleep_stage(&self) -> SleepStage {
        match self {
            ConsciousnessState::Awake => SleepStage::Wake,
            ConsciousnessState::Drowsy => SleepStage::N1,
            ConsciousnessState::LightSleep => SleepStage::N2,
            ConsciousnessState::DeepSleep => SleepStage::N3,
            ConsciousnessState::Dreaming => SleepStage::REM,
            ConsciousnessState::Unknown => SleepStage::Unknown,
        }
    }

    /// Is this a consolidation-favorable state?
    pub fn is_consolidation_state(&self) -> bool {
        matches!(
            self,
            ConsciousnessState::DeepSleep | ConsciousnessState::Dreaming
        )
    }
}

/// EEG frequency band powers
#[derive(Debug, Clone, Copy, Default)]
pub struct BandPowers {
    /// Delta: 0.5-4 Hz (deep sleep)
    pub delta: f32,
    /// Theta: 4-8 Hz (drowsy, REM)
    pub theta: f32,
    /// Alpha: 8-12 Hz (relaxed awake)
    pub alpha: f32,
    /// Beta: 12-30 Hz (active awake)
    pub beta: f32,
    /// Sigma: 12-15 Hz (sleep spindles)
    pub sigma: f32,
}

impl BandPowers {
    /// Total power across all bands
    pub fn total(&self) -> f32 {
        self.delta + self.theta + self.alpha + self.beta
    }

    /// Delta ratio (indicator of deep sleep)
    pub fn delta_ratio(&self) -> f32 {
        if self.total() > 0.0 {
            self.delta / self.total()
        } else {
            0.0
        }
    }

    /// Alpha ratio (indicator of relaxed wakefulness)
    pub fn alpha_ratio(&self) -> f32 {
        if self.total() > 0.0 {
            self.alpha / self.total()
        } else {
            0.0
        }
    }
}

/// The Trinity Sentinel: Multi-modal consciousness detector
pub struct ConsciousnessSentinel {
    /// LTC cell for EEG processing
    eeg_ltc: LtcCell,
    /// LTC cell for EOG processing
    eog_ltc: LtcCell,
    /// LTC cell for EMG processing
    emg_ltc: LtcCell,

    /// Sample rate (Hz)
    sample_rate: f32,
    /// Frame duration (seconds)
    frame_duration: f32,

    /// Current EEG band powers
    current_bands: BandPowers,
    /// Current EOG activity level
    current_eog_activity: f32,
    /// Current EMG activity level
    current_emg_activity: f32,

    /// Recent HV projections for gradient computation
    recent_hvs: Vec<HV16>,
    /// Current consciousness state
    current_state: ConsciousnessState,

    /// FFT planner
    fft_planner: FftPlanner<f32>,
    /// EEG buffer for spectral analysis
    eeg_buffer: Vec<f32>,
    /// EOG buffer for activity detection
    eog_buffer: Vec<f32>,
    /// EMG buffer for tone analysis
    emg_buffer: Vec<f32>,
}

impl ConsciousnessSentinel {
    /// Create a new consciousness sentinel
    pub fn new() -> Self {
        Self::with_config(100.0, 2.0) // 100 Hz sample rate, 2s frames for proper delta detection
    }

    /// Create with custom configuration
    pub fn with_config(sample_rate: f32, frame_duration: f32) -> Self {
        // LTC config optimized for physiological signals
        // Longer time constants than audio (100ms vs 20ms) for slower dynamics
        let physio_config = LtcConfig {
            hidden_size: 32,
            tau_min: 0.050,  // 50ms - fast response
            tau_max: 0.500,  // 500ms - slow integration
            tau_init: 0.100, // 100ms - typical EEG rhythm
            tau_lr: 0.005,
            adaptive_tau: true,
        };

        // EEG has more features (frequency bands)
        let eeg_ltc = LtcCell::new(5, physio_config.clone()); // 5 band powers

        // EOG/EMG simpler
        let simple_config = LtcConfig {
            hidden_size: 16,
            ..physio_config
        };
        let eog_ltc = LtcCell::new(2, simple_config.clone()); // REM activity, saccades
        let emg_ltc = LtcCell::new(2, simple_config); // tonic, phasic

        let buffer_size = (sample_rate * frame_duration) as usize;

        Self {
            eeg_ltc,
            eog_ltc,
            emg_ltc,
            sample_rate,
            frame_duration,
            current_bands: BandPowers::default(),
            current_eog_activity: 0.0,
            current_emg_activity: 0.0,
            recent_hvs: Vec::with_capacity(100),
            current_state: ConsciousnessState::Unknown,
            fft_planner: FftPlanner::new(),
            eeg_buffer: Vec::with_capacity(buffer_size),
            eog_buffer: Vec::with_capacity(buffer_size),
            emg_buffer: Vec::with_capacity(buffer_size),
        }
    }

    /// Reset the sentinel state
    pub fn reset(&mut self) {
        self.eeg_ltc.reset();
        self.eog_ltc.reset();
        self.emg_ltc.reset();
        self.current_bands = BandPowers::default();
        self.current_eog_activity = 0.0;
        self.current_emg_activity = 0.0;
        self.recent_hvs.clear();
        self.current_state = ConsciousnessState::Unknown;
        self.eeg_buffer.clear();
        self.eog_buffer.clear();
        self.emg_buffer.clear();
    }

    /// Process a single sample from the Trinity channels
    /// Returns the current consciousness state after processing
    pub fn process_sample(&mut self, eeg: f32, eog: f32, emg: f32) -> ConsciousnessState {
        // Add to buffers
        self.eeg_buffer.push(eeg);
        self.eog_buffer.push(eog);
        self.emg_buffer.push(emg);

        let buffer_size = (self.sample_rate * self.frame_duration) as usize;

        // Process when buffer is full
        if self.eeg_buffer.len() >= buffer_size {
            self.process_frame_internal();
            self.eeg_buffer.clear();
            self.eog_buffer.clear();
            self.emg_buffer.clear();
        }

        self.current_state
    }

    /// Process a complete frame (epoch) of data
    /// Typically 30 seconds for standard sleep staging
    pub fn process_frame(&mut self, eeg: f32, eog: f32, emg: f32) -> ConsciousnessState {
        // For single-sample processing, just use the sample directly
        self.eeg_buffer.push(eeg);
        self.eog_buffer.push(eog);
        self.emg_buffer.push(emg);

        // Process with whatever we have
        if self.eeg_buffer.len() >= 30 {
            self.process_frame_internal();
            // Keep some overlap for continuity
            let keep = self.eeg_buffer.len() / 2;
            self.eeg_buffer = self.eeg_buffer[keep..].to_vec();
            self.eog_buffer = self.eog_buffer[keep..].to_vec();
            self.emg_buffer = self.emg_buffer[keep..].to_vec();
        }

        self.current_state
    }

    /// Internal frame processing
    fn process_frame_internal(&mut self) {
        // Clone buffers to avoid borrow conflicts
        let eeg_buf = self.eeg_buffer.clone();
        let eog_buf = self.eog_buffer.clone();
        let emg_buf = self.emg_buffer.clone();

        // 1. Compute EEG band powers (needs &mut self for FFT planner)
        self.current_bands = self.compute_band_powers(&eeg_buf);

        // 2. Compute EOG activity (eye movement detection)
        self.current_eog_activity = Self::compute_eog_activity_static(&eog_buf);

        // 3. Compute EMG activity (muscle tone)
        self.current_emg_activity = Self::compute_emg_activity_static(&emg_buf);

        // 4. Compute additional features before LTC (to avoid borrow conflicts)
        let rem_bursts = Self::detect_rem_bursts_static(&eog_buf);
        let emg_atonia = Self::detect_emg_atonia_static(&emg_buf);

        // 5. Feed through LTC networks
        let eeg_features = vec![
            self.current_bands.delta,
            self.current_bands.theta,
            self.current_bands.alpha,
            self.current_bands.beta,
            self.current_bands.sigma,
        ];
        let eeg_state = self.eeg_ltc.forward(&eeg_features, self.frame_duration);

        let eog_features = vec![self.current_eog_activity, rem_bursts];
        let eog_state = self.eog_ltc.forward(&eog_features, self.frame_duration);

        let emg_features = vec![self.current_emg_activity, emg_atonia];
        let emg_state = self.emg_ltc.forward(&emg_features, self.frame_duration);

        // 6. Project to HDC space (using sign-based LSH like acoustic)
        let hv = Self::state_to_hv_static(eeg_state, eog_state, emg_state);
        self.recent_hvs.push(hv);
        if self.recent_hvs.len() > 100 {
            self.recent_hvs.remove(0);
        }

        // 7. Classify consciousness state
        self.current_state = self.classify_state();
    }

    /// Static EOG activity computation (variance indicates eye movement)
    fn compute_eog_activity_static(eog: &[f32]) -> f32 {
        if eog.is_empty() {
            return 0.0;
        }
        let mean = eog.iter().sum::<f32>() / eog.len() as f32;
        let variance = eog.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / eog.len() as f32;
        variance.sqrt()
    }

    /// Static EMG activity computation (muscle tone)
    fn compute_emg_activity_static(emg: &[f32]) -> f32 {
        if emg.is_empty() {
            return 0.0;
        }
        (emg.iter().map(|x| x.powi(2)).sum::<f32>() / emg.len() as f32).sqrt()
    }

    /// Static REM burst detection (rapid eye movements)
    fn detect_rem_bursts_static(eog: &[f32]) -> f32 {
        if eog.len() < 2 {
            return 0.0;
        }
        let mut crossings = 0;
        for i in 1..eog.len() {
            if (eog[i - 1] > 0.0) != (eog[i] > 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 / eog.len() as f32
    }

    /// Static EMG atonia detection (muscle relaxation during REM)
    fn detect_emg_atonia_static(emg: &[f32]) -> f32 {
        let activity = Self::compute_emg_activity_static(emg);
        1.0 / (1.0 + activity * 10.0)
    }

    /// Static state to HV projection
    fn state_to_hv_static(eeg: &[f32], eog: &[f32], emg: &[f32]) -> HV16 {
        // Combine all states
        let mut combined = Vec::with_capacity(eeg.len() + eog.len() + emg.len());
        combined.extend_from_slice(eeg);
        combined.extend_from_slice(eog);
        combined.extend_from_slice(emg);

        // Sign-based LSH encoding (same as audio projector fix)
        let mut hv = HV16::zero();

        for (dim_idx, &val) in combined.iter().enumerate() {
            let basis = Self::get_dim_basis_static(dim_idx);

            if val > 0.0 {
                for w in 0..16 {
                    hv.words[w] ^= basis.words[w];
                }
            } else {
                let rotated = basis.rotate(1024);
                for w in 0..16 {
                    hv.words[w] ^= rotated.words[w];
                }
            }
        }

        hv
    }

    /// Get deterministic random basis for a dimension (static)
    fn get_dim_basis_static(dim: usize) -> HV16 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"physio_basis_");
        hasher.update(&dim.to_le_bytes());
        let mut output = [0u8; 256];
        hasher.finalize_xof().fill(&mut output);
        HV16::from_bytes(&output)
    }

    /// Compute EEG frequency band powers using FFT
    fn compute_band_powers(&mut self, eeg: &[f32]) -> BandPowers {
        if eeg.len() < 32 {
            return BandPowers::default();
        }

        // Prepare FFT input (power of 2 size)
        let fft_size = eeg.len().next_power_of_two().min(512);
        let mut buffer: Vec<Complex<f32>> = eeg
            .iter()
            .take(fft_size)
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        // Pad with zeros if needed
        buffer.resize(fft_size, Complex::new(0.0, 0.0));

        // Apply Hanning window
        for (i, sample) in buffer.iter_mut().enumerate() {
            let window = 0.5 * (1.0 - (2.0 * PI * i as f32 / fft_size as f32).cos());
            *sample = Complex::new(sample.re * window, 0.0);
        }

        // Perform FFT
        let fft = self.fft_planner.plan_fft_forward(fft_size);
        fft.process(&mut buffer);

        // Compute power spectral density
        let freq_resolution = self.sample_rate / fft_size as f32;

        let mut bands = BandPowers::default();

        for (i, c) in buffer.iter().enumerate().take(fft_size / 2) {
            let freq = i as f32 * freq_resolution;
            let power = c.norm_sqr() / fft_size as f32;

            if (0.5..4.0).contains(&freq) {
                bands.delta += power;
            } else if (4.0..8.0).contains(&freq) {
                bands.theta += power;
            } else if (8.0..12.0).contains(&freq) {
                bands.alpha += power;
            } else if (12.0..15.0).contains(&freq) {
                bands.sigma += power;
            } else if (12.0..30.0).contains(&freq) {
                bands.beta += power;
            }
        }

        // Normalize
        let total = bands.total().max(1e-10);
        bands.delta /= total;
        bands.theta /= total;
        bands.alpha /= total;
        bands.beta /= total;
        bands.sigma /= total;

        bands
    }

    /// Compute EOG activity (variance indicates eye movement)
    /// Reserved for multi-channel polysomnography analysis
    #[allow(dead_code)]
    fn compute_eog_activity(&self, eog: &[f32]) -> f32 {
        if eog.is_empty() {
            return 0.0;
        }
        let mean = eog.iter().sum::<f32>() / eog.len() as f32;
        let variance = eog.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / eog.len() as f32;
        variance.sqrt()
    }

    /// Detect REM burst activity (rapid eye movements)
    #[allow(dead_code)]
    fn detect_rem_bursts(&self, eog: &[f32]) -> f32 {
        if eog.len() < 2 {
            return 0.0;
        }

        // Count zero-crossings (rapid movements)
        let mut crossings = 0;
        for i in 1..eog.len() {
            if (eog[i - 1] > 0.0) != (eog[i] > 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / eog.len() as f32
    }

    /// Compute EMG activity (muscle tone)
    #[allow(dead_code)]
    fn compute_emg_activity(&self, emg: &[f32]) -> f32 {
        if emg.is_empty() {
            return 0.0;
        }
        // RMS of high-passed signal
        let rms = (emg.iter().map(|x| x.powi(2)).sum::<f32>() / emg.len() as f32).sqrt();
        rms
    }

    /// Detect EMG atonia (muscle relaxation during REM)
    #[allow(dead_code)]
    fn detect_emg_atonia(&self, emg: &[f32]) -> f32 {
        let activity = self.compute_emg_activity(emg);
        // Atonia = inverse of activity (normalized)
        1.0 / (1.0 + activity * 10.0)
    }

    /// Project LTC states to HDC hypervector
    #[allow(dead_code)]
    fn state_to_hv(&mut self, eeg: &[f32], eog: &[f32], emg: &[f32]) -> HV16 {
        // Combine all states
        let mut combined = Vec::with_capacity(eeg.len() + eog.len() + emg.len());
        combined.extend_from_slice(eeg);
        combined.extend_from_slice(eog);
        combined.extend_from_slice(emg);

        // Sign-based LSH encoding (same as audio projector fix)
        let mut hv = HV16::zero();

        for (dim_idx, &val) in combined.iter().enumerate() {
            let basis = self.get_dim_basis(dim_idx);

            if val > 0.0 {
                for w in 0..16 {
                    hv.words[w] ^= basis.words[w];
                }
            } else {
                let rotated = basis.rotate(1024);
                for w in 0..16 {
                    hv.words[w] ^= rotated.words[w];
                }
            }
        }

        hv
    }

    /// Get deterministic random basis for a dimension
    #[allow(dead_code)]
    fn get_dim_basis(&self, dim: usize) -> HV16 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"physio_basis_");
        hasher.update(&dim.to_le_bytes());
        let mut output = [0u8; 256];
        hasher.finalize_xof().fill(&mut output);
        HV16::from_bytes(&output)
    }

    /// Classify consciousness state from current features
    fn classify_state(&self) -> ConsciousnessState {
        let bands = &self.current_bands;
        let eog = self.current_eog_activity;
        let emg = self.current_emg_activity;

        // Decision tree based on sleep staging criteria
        // Reference: AASM scoring manual

        // High EMG + high beta/alpha = Awake
        if emg > 0.5 && (bands.beta > 0.2 || bands.alpha > 0.3) {
            return ConsciousnessState::Awake;
        }

        // Low EMG + high EOG + theta = REM
        if emg < 0.2 && eog > 0.3 && bands.theta > 0.3 {
            return ConsciousnessState::Dreaming;
        }

        // Very high delta = Deep Sleep (N3)
        if bands.delta > 0.5 {
            return ConsciousnessState::DeepSleep;
        }

        // Spindles (sigma) + theta = Light Sleep (N2)
        if bands.sigma > 0.1 && bands.theta > 0.2 {
            return ConsciousnessState::LightSleep;
        }

        // Alpha + theta mix = Drowsy (N1)
        if bands.alpha > 0.2 && bands.theta > 0.2 {
            return ConsciousnessState::Drowsy;
        }

        // Default
        if bands.theta > bands.alpha {
            ConsciousnessState::Drowsy
        } else {
            ConsciousnessState::Awake
        }
    }

    /// Generate a gradient vector for federated learning
    /// This represents the current consciousness state as a mathematical object
    pub fn generate_gradient(&self) -> Vec<f32> {
        // K-Vector components based on consciousness state
        let state_encoding = match self.current_state {
            ConsciousnessState::Awake => [1.0, 0.0, 0.0, 0.0, 0.0],
            ConsciousnessState::Drowsy => [0.0, 1.0, 0.0, 0.0, 0.0],
            ConsciousnessState::LightSleep => [0.0, 0.0, 1.0, 0.0, 0.0],
            ConsciousnessState::DeepSleep => [0.0, 0.0, 0.0, 1.0, 0.0],
            ConsciousnessState::Dreaming => [0.0, 0.0, 0.0, 0.0, 1.0],
            ConsciousnessState::Unknown => [0.2, 0.2, 0.2, 0.2, 0.2],
        };

        // Combine with band powers and activity metrics
        let mut gradient = Vec::with_capacity(15);
        gradient.extend_from_slice(&state_encoding);
        gradient.push(self.current_bands.delta);
        gradient.push(self.current_bands.theta);
        gradient.push(self.current_bands.alpha);
        gradient.push(self.current_bands.beta);
        gradient.push(self.current_bands.sigma);
        gradient.push(self.current_eog_activity);
        gradient.push(self.current_emg_activity);

        // Add bundled HV fingerprint
        if !self.recent_hvs.is_empty() {
            let bundled = bundle(&self.recent_hvs);
            // Add first 3 words as fingerprint (normalized)
            for i in 0..3 {
                gradient.push((bundled.words[i].count_ones() as f32) / 128.0);
            }
        } else {
            gradient.extend_from_slice(&[0.5, 0.5, 0.5]);
        }

        gradient
    }

    /// Get the current consciousness state
    pub fn state(&self) -> ConsciousnessState {
        self.current_state
    }

    /// Get current EEG band powers
    pub fn band_powers(&self) -> &BandPowers {
        &self.current_bands
    }

    /// Compute K_Topo (topological closure coefficient)
    /// High K_Topo indicates coherent consciousness
    pub fn k_topo(&self) -> f32 {
        if self.recent_hvs.len() < 2 {
            return 0.5;
        }

        // K_Topo = average similarity of recent HVs
        // High similarity = stable state = coherent consciousness
        let n = self.recent_hvs.len();
        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                total_sim += self.recent_hvs[i].similarity(&self.recent_hvs[j]);
                count += 1;
            }
        }

        if count > 0 {
            (total_sim / count as f32 + 1.0) / 2.0 // Normalize to [0, 1]
        } else {
            0.5
        }
    }
}

impl Default for ConsciousnessSentinel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentinel_creation() {
        let sentinel = ConsciousnessSentinel::new();
        assert_eq!(sentinel.state(), ConsciousnessState::Unknown);
    }

    #[test]
    fn test_band_powers() {
        let mut sentinel = ConsciousnessSentinel::new();

        // Create synthetic delta wave (2 Hz)
        let samples: Vec<f32> = (0..100)
            .map(|i| (2.0 * PI * 2.0 * i as f32 / 100.0).sin())
            .collect();

        let bands = sentinel.compute_band_powers(&samples);
        // Delta should dominate for a 2 Hz signal
        assert!(bands.delta > bands.beta);
    }

    #[test]
    fn test_state_classification() {
        let mut sentinel = ConsciousnessSentinel::new();

        // Simulate deep sleep (high delta, low EMG)
        sentinel.current_bands = BandPowers {
            delta: 0.6,
            theta: 0.2,
            alpha: 0.1,
            beta: 0.05,
            sigma: 0.05,
        };
        sentinel.current_emg_activity = 0.1;
        sentinel.current_eog_activity = 0.1;

        let state = sentinel.classify_state();
        assert_eq!(state, ConsciousnessState::DeepSleep);
    }

    #[test]
    fn test_gradient_generation() {
        let mut sentinel = ConsciousnessSentinel::new();
        sentinel.current_state = ConsciousnessState::DeepSleep;

        let gradient = sentinel.generate_gradient();
        assert_eq!(gradient.len(), 15);
        assert_eq!(gradient[3], 1.0); // DeepSleep encoding
    }
}
