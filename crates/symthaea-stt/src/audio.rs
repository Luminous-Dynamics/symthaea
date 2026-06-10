// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio frontend and feature extraction
//!
//! This module provides:
//! - WAV file loading
//! - Mel spectrogram extraction (using rustfft for O(n log n) performance)
//! - LTC-based audio projection to HDC space

use crate::hdc::HV16;
use crate::ltc::{LtcCell, LtcConfig};
use rustfft::{FftPlanner, num_complex::Complex};
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;

/// Audio frontend configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioConfig {
    /// Sample rate (Hz)
    pub sample_rate: u32,
    /// Frame duration (seconds)
    pub frame_duration: f32,
    /// Hop duration (seconds)
    pub hop_duration: f32,
    /// Number of mel filterbank channels
    pub n_mels: usize,
    /// Minimum frequency for mel filterbank
    pub f_min: f32,
    /// Maximum frequency for mel filterbank
    pub f_max: f32,
    /// FFT size
    pub n_fft: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_duration: 0.025, // 25ms frames
            hop_duration: 0.010,   // 10ms hop
            n_mels: 40,
            f_min: 80.0,
            f_max: 7600.0,
            n_fft: 512,
        }
    }
}

/// Audio frontend for feature extraction
pub struct AudioFrontend {
    config: AudioConfig,
    /// Mel filterbank weights
    mel_filters: Vec<Vec<f32>>,
    /// Hann window
    window: Vec<f32>,
    /// Cached FFT instance
    fft: Arc<dyn rustfft::Fft<f32>>,
    /// Scratch buffer for FFT
    scratch: Vec<Complex<f32>>,
}

impl AudioFrontend {
    /// Create a new audio frontend
    pub fn new(config: AudioConfig) -> Self {
        let mel_filters = Self::create_mel_filterbank(&config);
        let frame_samples = (config.sample_rate as f32 * config.frame_duration) as usize;
        let window = Self::hann_window(frame_samples);

        // Create FFT planner and cache the FFT instance
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(config.n_fft);
        let scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];

        Self {
            config,
            mel_filters,
            window,
            fft,
            scratch,
        }
    }

    /// Load audio from any supported format (WAV, FLAC)
    pub fn load_audio<P: AsRef<Path>>(path: P) -> std::io::Result<(Vec<f32>, u32)> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        match ext.as_str() {
            "wav" => Self::load_wav(path),
            "flac" => Self::load_flac(path),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Unsupported audio format: {ext}"),
            )),
        }
    }

    /// Load audio from WAV file
    pub fn load_wav<P: AsRef<Path>>(path: P) -> std::io::Result<(Vec<f32>, u32)> {
        let reader =
            hound::WavReader::open(path).map_err(|e| std::io::Error::other(e.to_string()))?;
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect(),
            hound::SampleFormat::Int => {
                let max = (1 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / max)
                    .collect()
            }
        };

        // Convert to mono if stereo
        let mono = if spec.channels == 2 {
            samples
                .chunks(2)
                .map(|c| (c[0] + c.get(1).copied().unwrap_or(0.0)) / 2.0)
                .collect()
        } else {
            samples
        };

        Ok((mono, sample_rate))
    }

    /// Load audio from FLAC file
    pub fn load_flac<P: AsRef<Path>>(path: P) -> std::io::Result<(Vec<f32>, u32)> {
        let mut reader =
            claxon::FlacReader::open(path).map_err(|e| std::io::Error::other(e.to_string()))?;

        let info = reader.streaminfo();
        let sample_rate = info.sample_rate;
        let channels = info.channels as usize;
        let bits = info.bits_per_sample;
        let max = (1i64 << (bits - 1)) as f32;

        let samples: Vec<f32> = reader
            .samples()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / max)
            .collect();

        // Convert to mono if stereo
        let mono = if channels == 2 {
            samples
                .chunks(2)
                .map(|c| (c[0] + c.get(1).copied().unwrap_or(0.0)) / 2.0)
                .collect()
        } else if channels > 2 {
            // Mix all channels
            samples
                .chunks(channels)
                .map(|c| c.iter().sum::<f32>() / channels as f32)
                .collect()
        } else {
            samples
        };

        Ok((mono, sample_rate))
    }

    /// Extract mel spectrogram features
    pub fn extract_features(&mut self, audio: &[f32]) -> Vec<Vec<f32>> {
        let frame_samples = (self.config.sample_rate as f32 * self.config.frame_duration) as usize;
        let hop_samples = (self.config.sample_rate as f32 * self.config.hop_duration) as usize;

        let mut features = Vec::new();
        let mut pos = 0;

        while pos + frame_samples <= audio.len() {
            let frame = &audio[pos..pos + frame_samples];
            let mel = self.compute_mel_frame(frame);
            features.push(mel);
            pos += hop_samples;
        }

        features
    }

    /// Extract mel features with delta and delta-delta (velocity + acceleration)
    /// Returns features of dimension n_mels * 3 per frame
    pub fn extract_features_with_deltas(&mut self, audio: &[f32]) -> Vec<Vec<f32>> {
        let base_features = self.extract_features(audio);
        if base_features.is_empty() {
            return Vec::new();
        }

        let n_mels = base_features[0].len();
        let n_frames = base_features.len();

        // Compute deltas (first derivative)
        let deltas = Self::compute_deltas(&base_features, 2);
        // Compute delta-deltas (second derivative)
        let delta_deltas = Self::compute_deltas(&deltas, 2);

        // Concatenate: [mel; delta; delta_delta]
        let mut result = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let mut combined = Vec::with_capacity(n_mels * 3);
            combined.extend_from_slice(&base_features[i]);
            combined.extend_from_slice(&deltas[i]);
            combined.extend_from_slice(&delta_deltas[i]);
            result.push(combined);
        }

        result
    }

    /// Compute delta features using regression over a window
    fn compute_deltas(features: &[Vec<f32>], window: usize) -> Vec<Vec<f32>> {
        let n_frames = features.len();
        let n_dims = if features.is_empty() {
            0
        } else {
            features[0].len()
        };

        let mut deltas = vec![vec![0.0f32; n_dims]; n_frames];

        // Denominator for regression: 2 * sum(n^2) for n=1..window
        let denom: f32 = 2.0 * (1..=window).map(|n| (n * n) as f32).sum::<f32>();

        for t in 0..n_frames {
            for d in 0..n_dims {
                let mut delta = 0.0f32;
                for n in 1..=window {
                    let t_plus = (t + n).min(n_frames - 1);
                    let t_minus = t.saturating_sub(n);
                    delta += n as f32 * (features[t_plus][d] - features[t_minus][d]);
                }
                deltas[t][d] = delta / denom;
            }
        }

        deltas
    }

    /// Compute mel features for a single frame
    fn compute_mel_frame(&mut self, frame: &[f32]) -> Vec<f32> {
        // Apply window
        let windowed: Vec<f32> = frame
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        // Compute power spectrum using FFT (O(n log n))
        let power_spectrum = self.compute_power_spectrum(&windowed);

        // Apply mel filterbank
        let mut mel = vec![0.0; self.config.n_mels];
        for (i, filter) in self.mel_filters.iter().enumerate() {
            let energy = filter
                .iter()
                .zip(power_spectrum.iter())
                .map(|(&f, &p)| f * p)
                .sum::<f32>()
                .max(1e-10);

            // Log-mel with normalization:
            // Raw log values are typically in range [-20, 0] for speech
            // We normalize to approximately [-1, 1] range for stable LTC dynamics
            // Formula: (ln(energy) + 10) / 10  maps [-20, 0] -> [-1, 1]
            mel[i] = (energy.ln() + 10.0) / 10.0;
        }

        mel
    }

    /// Compute power spectrum using rustfft (O(n log n))
    fn compute_power_spectrum(&mut self, frame: &[f32]) -> Vec<f32> {
        let n = self.config.n_fft;

        // Zero-pad frame into complex buffer
        let mut buffer: Vec<Complex<f32>> = frame
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
            .take(n)
            .collect();

        // Perform FFT in-place
        self.fft
            .process_with_scratch(&mut buffer, &mut self.scratch);

        // Compute power spectrum (first half + DC)
        buffer
            .iter()
            .take(n / 2 + 1)
            .map(|c| (c.re * c.re + c.im * c.im) / n as f32)
            .collect()
    }

    /// Create Hann window
    fn hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
            .collect()
    }

    /// Create mel filterbank
    fn create_mel_filterbank(config: &AudioConfig) -> Vec<Vec<f32>> {
        let n_fft_bins = config.n_fft / 2 + 1;

        // Convert Hz to mel scale
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let mel_min = hz_to_mel(config.f_min);
        let mel_max = hz_to_mel(config.f_max);

        // Create mel points
        let mel_points: Vec<f32> = (0..=config.n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (config.n_mels + 1) as f32)
            .collect();

        // Convert back to Hz and then to FFT bin indices
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| {
                ((hz * config.n_fft as f32 / config.sample_rate as f32) as usize)
                    .min(n_fft_bins - 1)
            })
            .collect();

        // Create triangular filters
        let mut filters = Vec::with_capacity(config.n_mels);

        for m in 0..config.n_mels {
            let mut filter = vec![0.0; n_fft_bins];
            let left = bin_points[m];
            let center = bin_points[m + 1];
            let right = bin_points[m + 2];

            // Rising slope
            for k in left..center {
                if center > left {
                    filter[k] = (k - left) as f32 / (center - left) as f32;
                }
            }

            // Falling slope
            for k in center..=right {
                if right > center {
                    filter[k] = (right - k) as f32 / (right - center) as f32;
                }
            }

            filters.push(filter);
        }

        filters
    }

    /// Get configuration
    pub fn config(&self) -> &AudioConfig {
        &self.config
    }
}

/// Audio projector: LTC + HDC encoding
pub struct AudioProjector {
    /// Audio frontend
    frontend: AudioFrontend,
    /// LTC cells
    ltc: LtcCell,
    /// Frame duration
    _frame_duration: f32,
}

impl AudioProjector {
    /// Create a new audio projector
    pub fn new(audio_config: AudioConfig, ltc_config: LtcConfig) -> Self {
        let frame_duration = audio_config.hop_duration;
        let frontend = AudioFrontend::new(audio_config);
        let ltc = LtcCell::new(40, ltc_config); // 40 mel features

        Self {
            frontend,
            ltc,
            _frame_duration: frame_duration,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(AudioConfig::default(), LtcConfig::default())
    }

    /// Reset projector state
    pub fn reset(&mut self) {
        self.ltc.reset();
    }

    /// Project audio to sequence of HDC hypervectors with TEMPORAL CONTEXT
    ///
    /// Uses direct mel encoding for phoneme discrimination, with
    /// a 5-frame sliding window for temporal context.
    ///
    /// This captures phoneme trajectories and eliminates frame-to-frame oscillation.
    pub fn project(&mut self, audio: &[f32]) -> Vec<HV16> {
        self.reset();

        let features = self.frontend.extract_features(audio);
        if features.is_empty() {
            return Vec::new();
        }

        // First pass: encode all individual frames with DIRECT MEL encoding
        // Bypasses LTC to preserve spectral information without random projection
        let mut frame_hvs = Vec::with_capacity(features.len());
        let mut prev_features: Option<Vec<f32>> = None;

        for feature in &features {
            // Use direct mel encoding (no LTC - preserves phoneme-discriminative info)
            let hv = Self::direct_mel_hv(feature, prev_features.as_deref());
            frame_hvs.push(hv);
            prev_features = Some(feature.clone());
        }

        // Second pass: temporal windowing with permutation-based position encoding
        // Window size = 7 (center frame ± 3) for ~70ms context at 10ms hop
        const WINDOW_HALF: usize = 3;
        let mut hvs = Vec::with_capacity(frame_hvs.len());

        for center_idx in 0..frame_hvs.len() {
            let context_hv = Self::bundle_temporal_window(&frame_hvs, center_idx, WINDOW_HALF);
            hvs.push(context_hv);
        }

        hvs
    }

    /// Project audio without temporal windowing - returns raw frame HVs
    ///
    /// Used for diagnostic purposes to check encoding quality
    pub fn project_raw(&mut self, audio: &[f32]) -> Vec<HV16> {
        self.reset();

        let features = self.frontend.extract_features(audio);
        if features.is_empty() {
            return Vec::new();
        }

        let mut frame_hvs = Vec::with_capacity(features.len());
        let mut prev_features: Option<Vec<f32>> = None;

        for feature in &features {
            let hv = Self::direct_mel_hv(feature, prev_features.as_deref());
            frame_hvs.push(hv);
            prev_features = Some(feature.clone());
        }

        frame_hvs
    }

    /// Bundle a temporal window of frames with position-specific permutations
    ///
    /// Position encoding: Each position gets a unique rotation amount.
    /// This preserves temporal order information in the bundle.
    /// Center frame is weighted more heavily (appears twice in bundle).
    fn bundle_temporal_window(frame_hvs: &[HV16], center: usize, half_width: usize) -> HV16 {
        use crate::hdc::BundleAccumulator;

        let mut acc = BundleAccumulator::new();

        // Position rotations for 7-frame window: distinct values
        // Larger rotations for frames further from center
        const POSITION_ROTATIONS: [i32; 7] = [-768, -512, -256, 0, 256, 512, 768];
        let half = half_width as i32;

        for (pos_idx, offset) in (-half..=half).enumerate() {
            let frame_idx = center as i32 + offset;

            // Handle boundary conditions: clamp to valid range
            let frame_idx = frame_idx.clamp(0, frame_hvs.len() as i32 - 1) as usize;

            let frame_hv = &frame_hvs[frame_idx];

            // Apply position-specific rotation
            let rotated = frame_hv.rotate(POSITION_ROTATIONS[pos_idx]);

            // Add to bundle - center frame gets double weight
            acc.add(&rotated);
            if offset == 0 {
                acc.add(&rotated); // Extra weight for center
            }
        }

        acc.finalize()
    }

    /// Project audio and also return salience values (with temporal context)
    pub fn project_with_salience(&mut self, audio: &[f32]) -> (Vec<HV16>, Vec<f32>) {
        self.reset();

        let features = self.frontend.extract_features(audio);
        if features.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // First pass: encode all individual frames with DIRECT MEL encoding
        let mut frame_hvs = Vec::with_capacity(features.len());
        let mut saliences = Vec::with_capacity(features.len());
        let mut prev_features: Option<Vec<f32>> = None;

        for feature in &features {
            // Compute salience from mel energy change
            let salience = if let Some(ref prev) = prev_features {
                feature
                    .iter()
                    .zip(prev.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            } else {
                0.0
            };

            // Use direct mel encoding (no LTC)
            let hv = Self::direct_mel_hv(feature, prev_features.as_deref());
            frame_hvs.push(hv);
            saliences.push(salience);

            prev_features = Some(feature.clone());
        }

        // Second pass: temporal windowing (same as project())
        const WINDOW_HALF: usize = 3;
        let mut hvs = Vec::with_capacity(frame_hvs.len());

        for center_idx in 0..frame_hvs.len() {
            let context_hv = Self::bundle_temporal_window(&frame_hvs, center_idx, WINDOW_HALF);
            hvs.push(context_hv);
        }

        (hvs, saliences)
    }

    /// Convert LTC state to hypervector using STATE + VELOCITY encoding
    ///
    /// Encodes both position (h) and velocity (dh/dt) in phase space.
    /// Sign-based encoding for robustness.
    /// Note: Superseded by hybrid_mel_ltc_hv for better phoneme discrimination.
    #[allow(dead_code)]
    fn state_to_hv_static(state: &[f32], prev_state: &[f32]) -> HV16 {
        let mut hv = HV16::zero();
        let n_dims = state.len();

        // Part 1: Encode STATE (position in phase space)
        for (dim_idx, &val) in state.iter().enumerate() {
            if val.abs() < 0.05 {
                continue;
            }

            let basis = Self::get_dim_basis(dim_idx);
            let rotated = if val > 0.0 { basis } else { basis.rotate(1024) };

            for w in 0..16 {
                hv.words[w] ^= rotated.words[w];
            }
        }

        // Part 2: Encode VELOCITY (change from previous state)
        for (dim_idx, &val) in state.iter().enumerate() {
            let prev_val = prev_state.get(dim_idx).copied().unwrap_or(0.0);
            let delta = val - prev_val;

            if delta.abs() < 0.03 {
                continue;
            }

            let basis = Self::get_dim_basis(n_dims + dim_idx);
            let rotated = if delta > 0.0 {
                basis
            } else {
                basis.rotate(1024)
            };

            for w in 0..16 {
                hv.words[w] ^= rotated.words[w];
            }
        }

        hv
    }

    /// Direct mel encoding: encodes mel features directly into hypervector
    ///
    /// Uses THERMOMETER encoding: for each mel bin, include all basis vectors
    /// from level 0 up to the current level. This ensures similar mel values
    /// produce similar HVs (smooth similarity gradient).
    fn direct_mel_hv(mel_features: &[f32], prev_features: Option<&[f32]>) -> HV16 {
        use crate::hdc::BundleAccumulator;
        let mut acc = BundleAccumulator::new();

        // Part 1: Encode MEL features with WINDOWED level encoding
        // For a value at level N, include levels N-1, N, N+1 (where valid)
        // This creates local overlap for smooth similarity, but not global overlap
        const NUM_LEVELS: usize = 8;

        for (bin_idx, &val) in mel_features.iter().enumerate() {
            // Skip silence-floor bins
            if val < -0.6 {
                continue;
            }

            // Quantize to 8 levels: -0.6 to 1.0 in 0.2 increments
            let level = ((val + 0.6) / 0.2).clamp(0.0, 7.9) as usize;

            // Add the target level with double weight
            let basis_idx = bin_idx * NUM_LEVELS + level;
            let basis = Self::get_dim_basis(basis_idx);
            acc.add(&basis);
            acc.add(&basis); // Double weight for exact level

            // Add adjacent levels with single weight (creates local overlap)
            if level > 0 {
                let basis_idx = bin_idx * NUM_LEVELS + (level - 1);
                let basis = Self::get_dim_basis(basis_idx);
                acc.add(&basis);
            }
            if level < NUM_LEVELS - 1 {
                let basis_idx = bin_idx * NUM_LEVELS + (level + 1);
                let basis = Self::get_dim_basis(basis_idx);
                acc.add(&basis);
            }
        }

        // Part 2: Encode delta (velocity) features if available
        // Use simpler binary encoding for deltas (rising vs falling)
        if let Some(prev) = prev_features {
            for (bin_idx, (&curr, &prev_val)) in mel_features.iter().zip(prev.iter()).enumerate() {
                let delta = curr - prev_val;

                // Skip small deltas
                if delta.abs() < 0.10 {
                    continue;
                }

                // Binary delta: rising (0) or falling (1)
                let delta_level = if delta > 0.0 { 0 } else { 1 };

                // Offset by 500 to avoid collision with static features
                let basis_idx = 500 + bin_idx * 2 + delta_level;
                let basis = Self::get_dim_basis(basis_idx);

                // Lower weight for deltas (they're secondary features)
                acc.add(&basis);
            }
        }

        acc.finalize()
    }

    /// Hybrid encoding: combines MEL spectral features with LTC dynamics
    ///
    /// This preserves raw frequency information (for formants, voicing)
    /// while adding temporal dynamics from LTC.
    /// Uses 8-level quantization for better spectral resolution.
    #[allow(dead_code)]
    fn hybrid_mel_ltc_hv(mel_features: &[f32], state: &[f32], prev_state: &[f32]) -> HV16 {
        let mut hv = HV16::zero();
        let _n_mel = mel_features.len();
        let n_ltc = state.len();

        // Part 1: Encode MEL features with 8-level quantization
        // Mel features are normalized to approximately [-1, 1] range
        // (see compute_mel_frame: (energy.ln() + 10) / 10)
        // Use different basis offset to avoid collision with LTC encoding
        const MEL_BASIS_OFFSET: usize = 256;

        for (bin_idx, &val) in mel_features.iter().enumerate() {
            // Skip very low energy bins (silence floor)
            if val < -0.5 {
                continue;
            }

            let basis = Self::get_dim_basis(MEL_BASIS_OFFSET + bin_idx);

            // 8-level quantization spanning [-0.5, 1.0] range
            // Maps to rotations spread across the hypervector dimension
            // Each level gets 256 bits of rotation (2048 / 8 = 256)
            let level = if val > 0.8 {
                7 // Very high energy (formant peaks)
            } else if val > 0.5 {
                6 // High energy
            } else if val > 0.3 {
                5 // Medium-high energy
            } else if val > 0.1 {
                4 // Medium energy
            } else if val > -0.1 {
                3 // Low-medium energy
            } else if val > -0.3 {
                2 // Low energy
            } else if val > -0.5 {
                1 // Very low energy
            } else {
                0 // Silence
            };

            // Spread rotations across full 2048-bit space for maximum discrimination
            let rotation = (level - 3) * 256; // Center around 0
            let rotated = basis.rotate(rotation);
            for w in 0..16 {
                hv.words[w] ^= rotated.words[w];
            }
        }

        // Part 2: Encode LTC STATE for temporal dynamics (4-level)
        for (dim_idx, &val) in state.iter().enumerate() {
            if val.abs() < 0.1 {
                continue;
            }

            let basis = Self::get_dim_basis(dim_idx);

            // 4-level encoding: strong negative, weak negative, weak positive, strong positive
            let rotation = if val > 0.5 {
                512 // Strong positive
            } else if val > 0.0 {
                256 // Weak positive
            } else if val > -0.5 {
                -256 // Weak negative
            } else {
                -512 // Strong negative
            };

            let rotated = basis.rotate(rotation);
            for w in 0..16 {
                hv.words[w] ^= rotated.words[w];
            }
        }

        // Part 3: Encode LTC VELOCITY for trajectory direction (3-level)
        for (dim_idx, &val) in state.iter().enumerate() {
            let prev_val = prev_state.get(dim_idx).copied().unwrap_or(0.0);
            let delta = val - prev_val;
            if delta.abs() < 0.05 {
                continue;
            }

            let basis = Self::get_dim_basis(n_ltc + dim_idx);

            // 3-level: rising, stable, falling
            let rotation = if delta > 0.1 {
                512 // Rising fast
            } else if delta > 0.0 {
                256 // Rising slow
            } else if delta > -0.1 {
                -256 // Falling slow
            } else {
                -512 // Falling fast
            };

            let rotated = basis.rotate(rotation);
            for w in 0..16 {
                hv.words[w] ^= rotated.words[w];
            }
        }

        hv
    }

    /// Get a deterministic pseudo-random basis HV for a dimension
    fn get_dim_basis(dim: usize) -> HV16 {
        // Use dimension index as seed for deterministic random pattern
        let mut hasher = blake3::Hasher::new();
        hasher.update(&dim.to_le_bytes());
        let mut output = [0u8; 256];
        hasher.finalize_xof().fill(&mut output);
        HV16::from_bytes(&output)
    }

    /// Get LTC state
    pub fn state(&self) -> &[f32] {
        self.ltc.state()
    }

    /// Get current tau values
    pub fn tau(&self) -> &[f32] {
        self.ltc.tau()
    }

    /// Get frontend reference
    pub fn frontend(&self) -> &AudioFrontend {
        &self.frontend
    }
}

/// Prosody features
#[derive(Debug, Clone, Default)]
pub struct ProsodyFeatures {
    /// Pitch (F0) in Hz
    pub pitch: f32,
    /// Energy (RMS)
    pub energy: f32,
    /// Speaking rate estimate
    pub rate: f32,
}

impl ProsodyFeatures {
    /// Extract prosody from audio frame
    pub fn extract(audio: &[f32], sample_rate: u32) -> Self {
        let energy = (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32).sqrt();

        // Simple autocorrelation-based pitch detection
        let pitch = Self::detect_pitch(audio, sample_rate);

        // Rate estimated from zero-crossing rate
        let zcr = audio
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count() as f32
            / audio.len() as f32;
        let rate = zcr * sample_rate as f32 / 2.0;

        Self {
            pitch,
            energy,
            rate,
        }
    }

    /// Probabilistic YIN (PYIN) pitch detection.
    ///
    /// Uses the cumulative mean normalized difference function d'(tau)
    /// instead of raw autocorrelation, with absolute thresholding for
    /// candidate selection and parabolic interpolation for sub-sample
    /// accuracy.
    ///
    /// Reference: de Cheveigne & Kawahara (2002), "YIN, a fundamental
    /// frequency estimator for speech and music"; Mauch & Dixon (2014),
    /// "pYIN: A Fundamental Frequency Estimator Using Probabilistic
    /// Thresholds".
    fn detect_pitch(audio: &[f32], sample_rate: u32) -> f32 {
        let min_lag = sample_rate as usize / 500; // Max 500 Hz
        let max_lag = sample_rate as usize / 50; // Min 50 Hz

        if audio.len() < max_lag * 2 {
            return 0.0;
        }

        let n = audio.len().min(max_lag * 2);

        // Step 1: Compute the difference function d(tau)
        let mut diff = vec![0.0f32; max_lag + 1];
        for tau in 1..=max_lag.min(n / 2) {
            let mut sum = 0.0f32;
            for j in 0..(n - tau) {
                let delta = audio[j] - audio[j + tau];
                sum += delta * delta;
            }
            diff[tau] = sum;
        }

        // Step 2: Cumulative mean normalized difference function d'(tau)
        // d'(0) = 1 by definition; d'(tau) = d(tau) / ((1/tau) * sum(d(j), j=1..tau))
        let mut cmnd = vec![0.0f32; max_lag + 1];
        cmnd[0] = 1.0;
        let mut running_sum = 0.0f32;
        for tau in 1..=max_lag.min(n / 2) {
            running_sum += diff[tau];
            if running_sum > 0.0 {
                cmnd[tau] = diff[tau] * tau as f32 / running_sum;
            } else {
                cmnd[tau] = 1.0;
            }
        }

        // Step 3: Absolute threshold candidate selection (PYIN)
        // Select the first lag whose d'(tau) dips below the threshold.
        // Use multiple thresholds and pick the candidate with the best
        // (lowest) d'(tau) among those that pass any threshold.
        const PYIN_THRESHOLDS: [f32; 3] = [0.10, 0.15, 0.20];

        let search_max = max_lag.min(n / 2);
        let mut best_lag: usize = 0;
        let mut best_score = f32::MAX;

        for &thresh in &PYIN_THRESHOLDS {
            // Find first tau in range that dips below threshold
            for tau in min_lag..search_max {
                if cmnd[tau] < thresh {
                    // Found a candidate — check if it's a local minimum
                    if tau + 1 < search_max && cmnd[tau] <= cmnd[tau + 1] {
                        if cmnd[tau] < best_score {
                            best_score = cmnd[tau];
                            best_lag = tau;
                        }
                        break; // Take first dip per threshold (YIN convention)
                    }
                }
            }
        }

        // Fallback: if no candidate passed threshold, find global minimum
        if best_lag == 0 {
            for tau in min_lag..search_max {
                if cmnd[tau] < best_score {
                    best_score = cmnd[tau];
                    best_lag = tau;
                }
            }
        }

        // Reject if the best score is too high (unvoiced)
        if best_lag == 0 || best_score > 0.50 {
            return 0.0;
        }

        // Step 4: Parabolic interpolation for sub-sample accuracy
        let lag_f = if best_lag > min_lag && best_lag + 1 < search_max {
            let alpha = cmnd[best_lag - 1];
            let beta = cmnd[best_lag];
            let gamma = cmnd[best_lag + 1];
            let denom = 2.0 * (alpha - 2.0 * beta + gamma);
            if denom.abs() > 1e-10 {
                best_lag as f32 + (alpha - gamma) / denom
            } else {
                best_lag as f32
            }
        } else {
            best_lag as f32
        };

        if lag_f > 0.0 {
            sample_rate as f32 / lag_f
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = AudioFrontend::hann_window(100);
        assert_eq!(window.len(), 100);
        assert!((window[0] - 0.0).abs() < 0.01);
        assert!((window[50] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mel_filterbank() {
        let config = AudioConfig::default();
        let filters = AudioFrontend::create_mel_filterbank(&config);
        assert_eq!(filters.len(), config.n_mels);

        // Each filter should have the right size
        for filter in &filters {
            assert_eq!(filter.len(), config.n_fft / 2 + 1);
        }
    }

    #[test]
    fn test_audio_projector() {
        let mut projector = AudioProjector::default_config();

        // Create test audio (1 second of sine wave)
        let sample_rate = 16000;
        let audio: Vec<f32> = (0..sample_rate)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let hvs = projector.project(&audio);

        // Should produce approximately 100 frames (1s / 10ms)
        assert!(hvs.len() > 80 && hvs.len() < 120, "frames = {}", hvs.len());
    }

    #[test]
    fn test_prosody_extraction() {
        let sample_rate = 16000;
        let audio: Vec<f32> = (0..1600)
            .map(|i| (2.0 * PI * 200.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let prosody = ProsodyFeatures::extract(&audio, sample_rate);

        assert!(prosody.energy > 0.0);
        // Pitch should be around 200 Hz
        assert!(
            prosody.pitch > 150.0 && prosody.pitch < 250.0,
            "pitch = {}",
            prosody.pitch
        );
    }

    #[test]
    fn test_audio_config_default() {
        let config = AudioConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert!((config.frame_duration - 0.025).abs() < 1e-6);
        assert!((config.hop_duration - 0.010).abs() < 1e-6);
        assert_eq!(config.n_mels, 40);
        assert_eq!(config.n_fft, 512);
    }

    #[test]
    fn test_hann_window_symmetry() {
        let size = 64;
        let window = AudioFrontend::hann_window(size);
        assert_eq!(window.len(), size);

        // Hann window should be symmetric
        for i in 0..size / 2 {
            let diff = (window[i] - window[size - 1 - i]).abs();
            assert!(diff < 1e-5, "window not symmetric at index {i}");
        }
    }

    #[test]
    fn test_hann_window_endpoints() {
        let window = AudioFrontend::hann_window(50);
        // Endpoints should be near zero
        assert!(window[0].abs() < 0.01, "start = {}", window[0]);
        assert!(window[49].abs() < 0.01, "end = {}", window[49]);
    }

    #[test]
    fn test_hann_window_peak() {
        let window = AudioFrontend::hann_window(101);
        // Center should be near 1.0
        assert!((window[50] - 1.0).abs() < 0.01, "center = {}", window[50]);
    }

    #[test]
    fn test_mel_filterbank_properties() {
        let config = AudioConfig::default();
        let filters = AudioFrontend::create_mel_filterbank(&config);

        assert_eq!(filters.len(), config.n_mels);

        // Each filter should have non-negative values
        for (i, filter) in filters.iter().enumerate() {
            for &val in filter {
                assert!(val >= 0.0, "filter {i} has negative value: {val}");
            }
        }

        // Each filter should have at least some non-zero values
        for (i, filter) in filters.iter().enumerate() {
            let sum: f32 = filter.iter().sum();
            assert!(sum > 0.0, "filter {i} is all zeros");
        }
    }

    #[test]
    fn test_mel_filterbank_triangular() {
        let config = AudioConfig {
            n_mels: 10,
            n_fft: 256,
            ..Default::default()
        };
        let filters = AudioFrontend::create_mel_filterbank(&config);

        // Each filter peak should be <= 1.0 (triangular)
        for (i, filter) in filters.iter().enumerate() {
            let max: f32 = filter.iter().cloned().fold(0.0f32, f32::max);
            assert!(max <= 1.0 + 1e-6, "filter {i} peak exceeds 1.0: {max}");
        }
    }

    #[test]
    fn test_compute_deltas_constant_signal() {
        // Constant features should produce zero deltas
        let features = vec![vec![1.0, 2.0, 3.0]; 10];
        let deltas = AudioFrontend::compute_deltas(&features, 2);

        assert_eq!(deltas.len(), 10);
        for frame in &deltas {
            for &val in frame {
                assert!(val.abs() < 1e-6, "constant signal should have zero deltas");
            }
        }
    }

    #[test]
    fn test_compute_deltas_linear_ramp() {
        // Linear ramp: feature[t] = t
        let features: Vec<Vec<f32>> = (0..10).map(|t| vec![t as f32]).collect();
        let deltas = AudioFrontend::compute_deltas(&features, 2);

        assert_eq!(deltas.len(), 10);
        // Interior frames should have positive delta (increasing signal)
        for t in 2..8 {
            assert!(
                deltas[t][0] > 0.0,
                "delta at t={t} should be positive for linear ramp, got {}",
                deltas[t][0]
            );
        }
    }

    #[test]
    fn test_extract_features_produces_correct_dims() {
        let config = AudioConfig::default();
        let mut frontend = AudioFrontend::new(config);

        // Create 0.5 seconds of audio
        let sample_rate = 16000;
        let audio: Vec<f32> = (0..sample_rate / 2)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let features = frontend.extract_features(&audio);

        assert!(!features.is_empty(), "should produce at least one frame");
        // Each frame should have n_mels dimensions
        for frame in &features {
            assert_eq!(frame.len(), 40, "each frame should have 40 mel features");
        }
    }

    #[test]
    fn test_extract_features_with_deltas_dims() {
        let config = AudioConfig::default();
        let mut frontend = AudioFrontend::new(config);

        let sample_rate = 16000;
        let audio: Vec<f32> = (0..sample_rate / 2)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let features = frontend.extract_features_with_deltas(&audio);

        assert!(!features.is_empty());
        // Each frame should have n_mels * 3 dimensions (mel + delta + delta-delta)
        for frame in &features {
            assert_eq!(
                frame.len(),
                120,
                "each frame should have 120 features (40 mel * 3)"
            );
        }
    }

    #[test]
    fn test_prosody_silence() {
        let audio = vec![0.0; 1600];
        let prosody = ProsodyFeatures::extract(&audio, 16000);

        assert!(prosody.energy.abs() < 1e-6, "silence energy should be 0");
        assert!(prosody.pitch.abs() < 1e-6, "silence pitch should be 0");
    }

    #[test]
    fn test_prosody_default() {
        let prosody = ProsodyFeatures::default();
        assert_eq!(prosody.pitch, 0.0);
        assert_eq!(prosody.energy, 0.0);
        assert_eq!(prosody.rate, 0.0);
    }

    #[test]
    fn test_audio_projector_empty_input() {
        let mut projector = AudioProjector::default_config();
        let hvs = projector.project(&[]);
        assert!(hvs.is_empty(), "empty audio should produce no HVs");
    }

    #[test]
    fn test_audio_projector_with_salience() {
        let mut projector = AudioProjector::default_config();

        let sample_rate = 16000;
        let audio: Vec<f32> = (0..sample_rate)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let (hvs, saliences) = projector.project_with_salience(&audio);

        assert_eq!(hvs.len(), saliences.len());
        assert!(!hvs.is_empty());
        // First salience should be 0.0 (no previous frame)
        assert!(saliences[0].abs() < 1e-6, "first salience should be 0");
    }
}
