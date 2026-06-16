// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio Sentinel - Main pattern recognition engine.
//!
//! This module provides the `AudioSentinel` struct which orchestrates
//! learning and recognition of temporal audio patterns.

use std::collections::HashMap;
use std::f32::consts::PI;

use crate::encoding::{AudioHdcEncoder, AudioHdcVectors, EncoderMode, PremiumHdcEncoder};
use crate::features::{AudioFeatures, CONTROL_RATE, FREQ_BINS};
use crate::hdc::{HDC_DIM, HV};
use crate::patterns::{
    AudioCategory, AudioPattern, MAX_EXEMPLARS, NUM_LTC_LEVELS, PatternSimilarity,
};
use crate::temporal::{HierarchicalLtc, LtcPreset};

// =============================================================================
// Audio Detection Result
// =============================================================================

pub struct AudioDetectionResult {
    pub detected_pattern: String,
    pub detected_category: AudioCategory,
    pub confidence: f32,
    pub similarities: HashMap<String, PatternSimilarity>,
    pub phi_timbre: f64,
    pub phi_rhythm: f64,
    pub current_rhythm_signature: [f32; 8],
    pub current_timbre_signature: [f32; 8],
}

// =============================================================================
// Audio Sentinel
// =============================================================================

pub struct AudioSentinel {
    pub encoder: AudioHdcEncoder,
    pub ltc_timbre: HierarchicalLtc,
    pub ltc_rhythm: HierarchicalLtc,
    pub premium_encoder: Option<PremiumHdcEncoder>,
    pub encoder_mode: EncoderMode,
    pub patterns: HashMap<String, AudioPattern>,

    rhythm_trajectory: Vec<HV>,
    timbre_trajectory: Vec<HV>,
    learning_pattern: Option<String>,
    dt_ms: f32,
}

impl AudioSentinel {
    pub fn new() -> Self {
        Self::with_preset(LtcPreset::Standard)
    }

    pub fn with_preset(preset: LtcPreset) -> Self {
        let dt_ms = 1000.0 / CONTROL_RATE;

        Self {
            encoder: AudioHdcEncoder::new(),
            ltc_timbre: HierarchicalLtc::with_preset(preset),
            ltc_rhythm: HierarchicalLtc::with_preset(preset),
            premium_encoder: None,
            encoder_mode: EncoderMode::Standard,
            patterns: HashMap::new(),
            rhythm_trajectory: Vec::new(),
            timbre_trajectory: Vec::new(),
            learning_pattern: None,
            dt_ms,
        }
    }

    /// Create sentinel with Premium encoder mode
    pub fn premium() -> Self {
        let dt_ms = 1000.0 / CONTROL_RATE;

        Self {
            encoder: AudioHdcEncoder::new(),
            ltc_timbre: HierarchicalLtc::new(),
            ltc_rhythm: HierarchicalLtc::new(),
            premium_encoder: Some(PremiumHdcEncoder::new()),
            encoder_mode: EncoderMode::Premium,
            patterns: HashMap::new(),
            rhythm_trajectory: Vec::new(),
            timbre_trajectory: Vec::new(),
            learning_pattern: None,
            dt_ms,
        }
    }

    pub fn set_encoder_mode(&mut self, mode: EncoderMode) {
        if mode == EncoderMode::Premium && self.premium_encoder.is_none() {
            self.premium_encoder = Some(PremiumHdcEncoder::new());
        }
        self.encoder_mode = mode;
    }

    /// Process one frame of audio features
    pub fn process(&mut self, features: &AudioFeatures) -> AudioDetectionResult {
        let vectors = match self.encoder_mode {
            EncoderMode::Standard => {
                let v = self.encoder.encode(features);
                self.ltc_timbre.step(self.dt_ms, &v.timbre);
                self.ltc_rhythm.step(self.dt_ms, &v.rhythm);
                v
            }
            EncoderMode::Premium => {
                if let Some(ref mut encoder) = self.premium_encoder {
                    encoder.encode(features)
                } else {
                    let v = self.encoder.encode(features);
                    self.ltc_timbre.step(self.dt_ms, &v.timbre);
                    self.ltc_rhythm.step(self.dt_ms, &v.rhythm);
                    v
                }
            }
        };

        // Record trajectories
        self.timbre_trajectory.push(vectors.timbre.clone());
        self.rhythm_trajectory.push(vectors.rhythm.clone());

        if self.timbre_trajectory.len() > 130 {
            self.timbre_trajectory.remove(0);
            self.rhythm_trajectory.remove(0);
        }

        // If learning, accumulate to pattern
        if let Some(name) = self.learning_pattern.clone() {
            self.accumulate_learning(&name, &vectors, features);
        }

        // Compute similarity to learned patterns
        self.compute_similarities(features, &vectors)
    }

    fn accumulate_learning(
        &mut self,
        name: &str,
        vectors: &AudioHdcVectors,
        features: &AudioFeatures,
    ) {
        if let Some(pattern) = self.patterns.get_mut(name) {
            pattern.timbre_prototype = pattern.timbre_prototype.add(&vectors.timbre);
            pattern.rhythm_prototype = pattern.rhythm_prototype.add(&vectors.rhythm);
            pattern.envelope_prototype = pattern.envelope_prototype.add(&vectors.envelope);
            pattern.context_prototype = pattern.context_prototype.add(&vectors.context);

            // Multi-scale prototypes (Standard mode only)
            if self.encoder_mode == EncoderMode::Standard {
                let timbre_velocities = self.ltc_timbre.get_level_velocities();
                let rhythm_velocities = self.ltc_rhythm.get_level_velocities();
                for (i, (tv, rv)) in timbre_velocities
                    .iter()
                    .zip(rhythm_velocities.iter())
                    .enumerate()
                {
                    if i < NUM_LTC_LEVELS {
                        pattern.timbre_scale_prototypes[i] =
                            pattern.timbre_scale_prototypes[i].add(tv);
                        pattern.rhythm_scale_prototypes[i] =
                            pattern.rhythm_scale_prototypes[i].add(rv);
                    }
                }
            }

            // Accumulate spectral features
            pattern.centroid_sum += features.spectral_centroid;
            pattern.flatness_sum += features.spectral_flatness;
            pattern.onset_sum += features.onset_strength;
            pattern.regularity_sum += features.temporal_regularity;
            pattern.envelope_delta_sum += features.envelope_delta;
            pattern.envelope_variance_sum += features.envelope_variance;
            pattern.spectral_flux_sum += features.spectral_flux;
            pattern.harmonic_ratio_sum += features.harmonic_ratio;
            pattern.cfc_theta_gamma_sum += features.cfc_theta_gamma;
            pattern.cfc_delta_beta_sum += features.cfc_delta_beta;
            pattern.attack_sharpness_sum += features.attack_sharpness;
            pattern.decay_roughness_sum += features.decay_roughness;
            pattern.silence_ratio_sum += features.silence_ratio;
            pattern.burst_density_sum += features.burst_density;

            for (i, &band) in features.mel_bands.iter().enumerate() {
                if i < pattern.mel_bands_sum.len() {
                    pattern.mel_bands_sum[i] += band;
                }
            }
            pattern.frame_count += 1;
        }
    }

    fn compute_similarities(
        &self,
        features: &AudioFeatures,
        vectors: &AudioHdcVectors,
    ) -> AudioDetectionResult {
        let rhythm_sig = self.compute_frequency_signature(&self.rhythm_trajectory);
        let timbre_sig = self.compute_frequency_signature(&self.timbre_trajectory);

        let mut similarities: HashMap<String, PatternSimilarity> = HashMap::new();

        const WINDOW_SIZE: usize = 40;

        let current_timbre_velocities = self.ltc_timbre.get_level_velocities();
        let current_rhythm_velocities = self.ltc_rhythm.get_level_velocities();

        for (name, pattern) in &self.patterns {
            if pattern.frame_count == 0 {
                continue;
            }

            if self.encoder_mode == EncoderMode::Premium {
                let sim = self.compute_premium_similarity(
                    features,
                    vectors,
                    pattern,
                    &rhythm_sig,
                    &timbre_sig,
                    WINDOW_SIZE,
                );
                similarities.insert(name.clone(), sim);
            } else {
                let sim = self.compute_standard_similarity(
                    features,
                    vectors,
                    pattern,
                    &rhythm_sig,
                    &timbre_sig,
                    &current_timbre_velocities,
                    &current_rhythm_velocities,
                    WINDOW_SIZE,
                );
                similarities.insert(name.clone(), sim);
            }
        }

        // Find best match
        let (best_name, best_sim) = similarities
            .iter()
            .max_by(|(_, a), (_, b)| a.combined.total_cmp(&b.combined))
            .map(|(n, s)| (n.clone(), s.combined))
            .unwrap_or(("Unknown".to_string(), 0.0));

        let detected_category = self
            .patterns
            .get(&best_name)
            .map(|p| p.category)
            .unwrap_or(AudioCategory::Unknown);

        AudioDetectionResult {
            detected_pattern: if best_sim > 0.4 {
                best_name
            } else {
                "Unknown".to_string()
            },
            detected_category,
            confidence: best_sim,
            similarities,
            phi_timbre: self.ltc_timbre.phi,
            phi_rhythm: self.ltc_rhythm.phi,
            current_rhythm_signature: rhythm_sig,
            current_timbre_signature: timbre_sig,
        }
    }

    fn compute_premium_similarity(
        &self,
        features: &AudioFeatures,
        vectors: &AudioHdcVectors,
        pattern: &AudioPattern,
        rhythm_sig: &[f32; 8],
        timbre_sig: &[f32; 8],
        window_size: usize,
    ) -> PatternSimilarity {
        let timbre_sim = vectors.timbre.similarity(&pattern.timbre_prototype);
        let rhythm_sim = vectors.rhythm.similarity(&pattern.rhythm_prototype);
        let envelope_sim = vectors.envelope.similarity(&pattern.envelope_prototype);

        let timbre_traj_sim = self.windowed_trajectory_similarity(
            &self.timbre_trajectory,
            &pattern.timbre_prototype,
            window_size,
        );
        let rhythm_traj_sim = self.windowed_trajectory_similarity(
            &self.rhythm_trajectory,
            &pattern.rhythm_prototype,
            window_size,
        );

        let timbre_combined = 0.6 * timbre_traj_sim + 0.4 * timbre_sim;
        let rhythm_combined = 0.6 * rhythm_traj_sim + 0.4 * rhythm_sim;

        let rhythm_freq_sim =
            self.freq_signature_similarity(rhythm_sig, &pattern.rhythm_freq_signature);
        let timbre_freq_sim =
            self.freq_signature_similarity(timbre_sig, &pattern.timbre_freq_signature);

        let log_curr = (features.spectral_centroid + 1.0).ln();
        let log_pattern = (pattern.mean_spectral_centroid + 1.0).ln();
        let centroid_sim = 1.0 - ((log_curr - log_pattern).abs() / 2.0).min(1.0);

        let flatness_sim = 1.0
            - ((features.spectral_flatness - pattern.mean_spectral_flatness).abs() * 4.0).min(1.0);
        let regularity_sim = 1.0
            - ((features.temporal_regularity - pattern.mean_temporal_regularity).abs() * 2.0)
                .min(1.0);

        let mel_sim = if !pattern.mean_mel_bands.is_empty() && !features.mel_bands.is_empty() {
            let dot: f32 = features
                .mel_bands
                .iter()
                .zip(&pattern.mean_mel_bands)
                .map(|(a, b)| a * b)
                .sum();
            let norm_a: f32 = features.mel_bands.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = pattern
                .mean_mel_bands
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            if norm_a > 1e-10 && norm_b > 1e-10 {
                (dot / (norm_a * norm_b)).max(0.0)
            } else {
                0.5
            }
        } else {
            0.5
        };

        let silence_sim =
            1.0 - ((features.silence_ratio - pattern.mean_silence_ratio).abs() * 3.0).min(1.0);

        // IOI Variance similarity - discriminates rhythm regularity (Rain vs Clock)
        let burst_density_sim =
            1.0 - ((features.burst_density - pattern.mean_burst_density).abs() * 4.0).min(1.0);

        // Harmonic similarity - discriminates tonal (Birds/Sirens) vs atonal (Clock ticks)
        // Birds sing (high harmonicity), Clocks click (low harmonicity)
        let harmonic_sim =
            1.0 - ((features.harmonic_ratio - pattern.mean_harmonic_ratio).abs() * 3.0).min(1.0);

        let hdc_sim = 0.5 * timbre_combined + 0.5 * rhythm_combined;
        let spectral_sim =
            0.3 * centroid_sim + 0.3 * mel_sim + 0.2 * flatness_sim + 0.2 * regularity_sim;
        let freq_sim = 0.5 * rhythm_freq_sim + 0.5 * timbre_freq_sim;

        // "Tri-Chromat" Weighting: Four sensory channels for complete sound discrimination
        // HDC/Timbre: 50% (primary identity)
        // Silence: 20% (Density - fixes Dog/Clock)
        // Harmonicity: 15% (Tone - fixes Birds/Clock)
        // IOI Variance: 15% (Rhythm - fixes Rain/Clock)
        let combined = 0.23 * hdc_sim
            + 0.14 * spectral_sim
            + 0.13 * freq_sim
            + 0.20 * silence_sim
            + 0.15 * harmonic_sim
            + 0.15 * burst_density_sim;

        PatternSimilarity {
            timbre: timbre_combined,
            rhythm: rhythm_combined,
            envelope: envelope_sim,
            rhythm_freq: rhythm_freq_sim,
            timbre_freq: timbre_freq_sim,
            spectral: spectral_sim,
            multi_scale: hdc_sim,
            combined,
        }
    }

    fn compute_standard_similarity(
        &self,
        features: &AudioFeatures,
        vectors: &AudioHdcVectors,
        pattern: &AudioPattern,
        rhythm_sig: &[f32; 8],
        timbre_sig: &[f32; 8],
        current_timbre_velocities: &[HV],
        current_rhythm_velocities: &[HV],
        window_size: usize,
    ) -> PatternSimilarity {
        let timbre_sim = self.windowed_trajectory_similarity(
            &self.timbre_trajectory,
            &pattern.timbre_prototype,
            window_size,
        );
        let rhythm_sim = self.windowed_trajectory_similarity(
            &self.rhythm_trajectory,
            &pattern.rhythm_prototype,
            window_size,
        );
        let envelope_sim = vectors.envelope.similarity(&pattern.envelope_prototype);

        // Multi-scale similarity
        let mut timbre_scale_sim = 0.0f32;
        let mut rhythm_scale_sim = 0.0f32;
        for i in 0..NUM_LTC_LEVELS {
            let level_weight = 1.0 / (1.0 + 0.3 * i as f32);

            let curr_timbre_mag: f32 = current_timbre_velocities[i]
                .values
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            let proto_timbre_mag: f32 = pattern.timbre_scale_prototypes[i]
                .values
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            let curr_rhythm_mag: f32 = current_rhythm_velocities[i]
                .values
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            let proto_rhythm_mag: f32 = pattern.rhythm_scale_prototypes[i]
                .values
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();

            let timbre_mag_sim = if curr_timbre_mag > 1e-10 && proto_timbre_mag > 1e-10 {
                let ratio = curr_timbre_mag / proto_timbre_mag;
                (-ratio.ln().abs()).exp()
            } else {
                0.0
            };
            let rhythm_mag_sim = if curr_rhythm_mag > 1e-10 && proto_rhythm_mag > 1e-10 {
                let ratio = curr_rhythm_mag / proto_rhythm_mag;
                (-ratio.ln().abs()).exp()
            } else {
                0.0
            };

            timbre_scale_sim += level_weight * timbre_mag_sim;
            rhythm_scale_sim += level_weight * rhythm_mag_sim;
        }
        let weight_sum: f32 = (0..NUM_LTC_LEVELS)
            .map(|i| 1.0 / (1.0 + 0.3 * i as f32))
            .sum();
        timbre_scale_sim /= weight_sum;
        rhythm_scale_sim /= weight_sum;
        let multi_scale_sim = 0.5 * timbre_scale_sim + 0.5 * rhythm_scale_sim;

        let rhythm_freq_sim =
            self.freq_signature_similarity(rhythm_sig, &pattern.rhythm_freq_signature);
        let timbre_freq_sim =
            self.freq_signature_similarity(timbre_sig, &pattern.timbre_freq_signature);

        // Spectral features
        let log_curr = (features.spectral_centroid + 1.0).ln();
        let log_pattern = (pattern.mean_spectral_centroid + 1.0).ln();
        let centroid_diff = (log_curr - log_pattern).abs() / 1.5;
        let centroid_sim = 1.0 - centroid_diff.min(1.0);

        let mel_sim = if !pattern.mean_mel_bands.is_empty() && !features.mel_bands.is_empty() {
            let dot: f32 = features
                .mel_bands
                .iter()
                .zip(&pattern.mean_mel_bands)
                .map(|(a, b)| a * b)
                .sum();
            let norm_a: f32 = features.mel_bands.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = pattern
                .mean_mel_bands
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            if norm_a > 1e-10 && norm_b > 1e-10 {
                (dot / (norm_a * norm_b)).max(0.0)
            } else {
                0.5
            }
        } else {
            0.5
        };

        let flatness_diff = (features.spectral_flatness - pattern.mean_spectral_flatness).abs();
        let flatness_sim = 1.0 - (flatness_diff * 8.0).min(1.0);

        let regularity_diff =
            (features.temporal_regularity - pattern.mean_temporal_regularity).abs();
        let regularity_sim = 1.0 - (regularity_diff * 2.0).min(1.0);

        let base_spectral_sim = 0.30 * centroid_sim + 0.40 * mel_sim + 0.30 * regularity_sim;
        let flatness_gate = 0.3 + 0.7 * flatness_sim;

        let is_noisy = features.spectral_flatness > 0.3 || pattern.mean_spectral_flatness > 0.3;
        let centroid_gate = if is_noisy {
            0.6 + 0.4 * centroid_sim
        } else {
            0.4 + 0.6 * centroid_sim
        };

        let regularity_gate = 0.4 + 0.6 * regularity_sim;
        let spectral_sim = base_spectral_sim * flatness_gate * centroid_gate * regularity_gate;

        let hdc_sim = 0.5 * timbre_sim + 0.5 * rhythm_sim;
        let temporal_freq_avg = 0.35 * rhythm_freq_sim + 0.65 * timbre_freq_sim;
        let hdc_gate = 0.5 + 0.5 * hdc_sim * hdc_sim;

        let base_combined = (0.25 * temporal_freq_avg
            + 0.35 * spectral_sim
            + 0.30 * hdc_sim
            + 0.10 * multi_scale_sim)
            * hdc_gate;

        let both_tonal = features.spectral_flatness < 0.1 && pattern.mean_spectral_flatness < 0.1;
        let temporal_gate = if spectral_sim > 0.7 && both_tonal {
            let t_scaled = temporal_freq_avg * temporal_freq_avg;
            0.3 + 0.7 * t_scaled
        } else {
            1.0
        };

        let combined = base_combined * temporal_gate;

        PatternSimilarity {
            timbre: timbre_sim,
            rhythm: rhythm_sim,
            envelope: envelope_sim,
            rhythm_freq: rhythm_freq_sim,
            timbre_freq: timbre_freq_sim,
            spectral: spectral_sim,
            multi_scale: multi_scale_sim,
            combined,
        }
    }

    fn encode_trajectory(&self, trajectory: &[HV]) -> HV {
        if trajectory.is_empty() {
            return HV::zero();
        }

        let mut encoded = HV::zero();
        for (i, hv) in trajectory.iter().enumerate() {
            let shift = (i * 80) % HDC_DIM;
            encoded = encoded.add(&hv.permute(shift));
        }
        encoded.normalize()
    }

    fn windowed_trajectory_similarity(
        &self,
        trajectory: &[HV],
        prototype: &HV,
        window_size: usize,
    ) -> f32 {
        if trajectory.is_empty() {
            return 0.0;
        }

        if trajectory.len() < window_size || window_size < 5 {
            let mut avg = HV::zero();
            for hv in trajectory {
                avg = avg.add(hv);
            }
            return avg.normalize().similarity(prototype);
        }

        let step = window_size / 2;
        let mut max_sim = 0.0f32;

        let mut pos = 0;
        while pos + window_size <= trajectory.len() {
            let mut avg = HV::zero();
            for hv in &trajectory[pos..pos + window_size] {
                avg = avg.add(hv);
            }
            let window_sim = avg.normalize().similarity(prototype);
            max_sim = max_sim.max(window_sim);
            pos += step;
        }

        max_sim
    }

    fn compute_frequency_signature(&self, trajectory: &[HV]) -> [f32; 8] {
        if trajectory.len() < 10 {
            return [0.0; 8];
        }

        let signal: Vec<f32> = trajectory
            .iter()
            .map(|hv| hv.values.iter().take(1024).sum())
            .collect();

        let n = signal.len();
        let mean: f32 = signal.iter().sum::<f32>() / n as f32;

        let mut signature = [0.0f32; 8];

        for (fi, &freq) in FREQ_BINS.iter().enumerate() {
            let omega = 2.0 * PI * freq / CONTROL_RATE;
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (i, &x) in signal.iter().enumerate() {
                let angle = omega * i as f32;
                real += (x - mean) * angle.cos();
                imag += (x - mean) * angle.sin();
            }

            signature[fi] = (real * real + imag * imag).sqrt() / n as f32;
        }

        signature
    }

    fn freq_signature_similarity(&self, a: &[f32; 8], b: &[f32; 8]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);

        dot / (norm_a * norm_b)
    }

    pub fn start_learning(&mut self, name: &str, category: AudioCategory) {
        self.learning_pattern = Some(name.to_string());
        self.patterns
            .insert(name.to_string(), AudioPattern::new(name, category));
        self.ltc_timbre.reset();
        self.ltc_rhythm.reset();
        if let Some(ref mut encoder) = self.premium_encoder {
            encoder.reset();
        }
        self.timbre_trajectory.clear();
        self.rhythm_trajectory.clear();
    }

    pub fn continue_learning(&mut self, name: &str) {
        self.learning_pattern = Some(name.to_string());
        self.ltc_timbre.reset();
        self.ltc_rhythm.reset();
        if let Some(ref mut encoder) = self.premium_encoder {
            encoder.reset();
        }
    }

    pub fn stop_learning(&mut self) -> Option<String> {
        if let Some(name) = self.learning_pattern.take() {
            let rhythm_freq_sig = self.compute_frequency_signature(&self.rhythm_trajectory);
            let timbre_freq_sig = self.compute_frequency_signature(&self.timbre_trajectory);
            let trajectory_hv = self.encode_trajectory(&self.timbre_trajectory);

            if let Some(pattern) = self.patterns.get_mut(&name) {
                if pattern.frame_count > 0 {
                    let n = pattern.frame_count as f32;
                    pattern.timbre_prototype = pattern.timbre_prototype.scale(1.0 / n).normalize();
                    pattern.rhythm_prototype = pattern.rhythm_prototype.scale(1.0 / n).normalize();
                    pattern.envelope_prototype =
                        pattern.envelope_prototype.scale(1.0 / n).normalize();
                    pattern.context_prototype =
                        pattern.context_prototype.scale(1.0 / n).normalize();

                    if pattern.timbre_exemplars.len() < MAX_EXEMPLARS {
                        pattern
                            .timbre_exemplars
                            .push(pattern.timbre_prototype.clone());
                        pattern
                            .rhythm_exemplars
                            .push(pattern.rhythm_prototype.clone());
                    }

                    pattern.trajectory_prototype = trajectory_hv;
                    if pattern.trajectory_exemplars.len() < MAX_EXEMPLARS {
                        pattern
                            .trajectory_exemplars
                            .push(pattern.trajectory_prototype.clone());
                    }

                    for i in 0..NUM_LTC_LEVELS {
                        pattern.timbre_scale_prototypes[i] =
                            pattern.timbre_scale_prototypes[i].scale(1.0 / n);
                        pattern.rhythm_scale_prototypes[i] =
                            pattern.rhythm_scale_prototypes[i].scale(1.0 / n);
                    }

                    pattern.rhythm_freq_signature = rhythm_freq_sig;
                    pattern.timbre_freq_signature = timbre_freq_sig;

                    pattern.mean_spectral_centroid = pattern.centroid_sum / n;
                    pattern.mean_spectral_flatness = pattern.flatness_sum / n;
                    pattern.mean_onset_strength = pattern.onset_sum / n;
                    pattern.mean_temporal_regularity = pattern.regularity_sum / n;
                    pattern.mean_envelope_delta = pattern.envelope_delta_sum / n;
                    pattern.mean_envelope_variance = pattern.envelope_variance_sum / n;
                    pattern.mean_spectral_flux = pattern.spectral_flux_sum / n;
                    pattern.mean_harmonic_ratio = pattern.harmonic_ratio_sum / n;
                    pattern.mean_cfc_theta_gamma = pattern.cfc_theta_gamma_sum / n;
                    pattern.mean_cfc_delta_beta = pattern.cfc_delta_beta_sum / n;
                    pattern.mean_attack_sharpness = pattern.attack_sharpness_sum / n;
                    pattern.mean_decay_roughness = pattern.decay_roughness_sum / n;
                    pattern.mean_silence_ratio = pattern.silence_ratio_sum / n;
                    pattern.mean_burst_density = pattern.burst_density_sum / n;

                    for i in 0..pattern.mel_bands_sum.len() {
                        pattern.mean_mel_bands[i] = pattern.mel_bands_sum[i] / n;
                    }
                }
            }
            self.timbre_trajectory.clear();
            self.rhythm_trajectory.clear();
            Some(name)
        } else {
            None
        }
    }
}

impl Default for AudioSentinel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::MEL_BANDS;

    #[test]
    fn test_audio_sentinel_learning() {
        let mut sentinel = AudioSentinel::new();

        sentinel.start_learning("TestPattern", AudioCategory::Music);

        for i in 0..50 {
            let features = AudioFeatures {
                mel_bands: vec![0.5; MEL_BANDS],
                mfcc: vec![0.0; 13],
                spectral_centroid: 1000.0,
                spectral_rolloff: 4000.0,
                onset_strength: 0.1,
                rms_energy: 0.3,
                zero_crossing_rate: 0.1,
                spectral_flatness: 0.5,
                temporal_regularity: 0.7,
                envelope_delta: 0.01,
                envelope_variance: 0.001,
                frame_index: i,
                spectral_flux: 0.1,
                harmonic_ratio: 0.5,
                cfc_theta_gamma: 0.5,
                cfc_delta_beta: 0.5,
                attack_sharpness: 0.5,
                decay_roughness: 0.5,
                silence_ratio: 0.5,
                burst_density: 0.5,
            };
            sentinel.process(&features);
        }

        sentinel.stop_learning();

        assert!(sentinel.patterns.contains_key("TestPattern"));
        assert_eq!(sentinel.patterns["TestPattern"].frame_count, 50);
    }
}
