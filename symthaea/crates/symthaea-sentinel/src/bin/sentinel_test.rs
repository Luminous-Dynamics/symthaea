// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regression Test Harness for Audio Sentinel
//!
//! Implements the "Scientific Method" workflow for validating zero-shot
//! temporal pattern discrimination:
//!
//! 1. Control Group: Learn a concept (e.g., "Dog Bark")
//! 2. Variable Group: Test that dissimilar sounds are rejected
//! 3. Generalization: Test that similar sounds are accepted
//!
//! Usage:
//!   cargo run --release --bin regression-test -- --data-dir ./datasets/ESC-50
//!
//! Or run with sample files:
//!   cargo run --release --bin regression-test -- \
//!     --learn dog_bark.wav \
//!     --reject glass_break.wav \
//!     --accept dog_bark_2.wav

use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use symthaea_sentinel::{
    AudioCategory, AudioFeatures, AudioSentinel, CONTROL_RATE, DatasetProcessor, FileAudioConfig,
    FileAudioPump, LtcPreset, MEL_BANDS, compute_ioi_variance, compute_mfcc,
    compute_onset_strength, compute_spectral_centroid, compute_spectral_flatness,
    compute_temporal_regularity, spectrum_to_mel_bands,
};

/// Process a file through the sentinel and get averaged features
/// If finalize_learning is false, stop_learning won't be called (for multi-shot)
fn process_file(
    pump: &mut FileAudioPump,
    sentinel: &mut AudioSentinel,
    learn: bool,
    pattern_name: Option<&str>,
    finalize_learning: bool,
) -> Result<()> {
    let mut prev_spectrum: Vec<f32> = Vec::new();
    let mut onset_history: Vec<f32> = Vec::new();
    // Envelope tracking state
    let mut rms_history: Vec<f32> = Vec::new();
    let mut prev_rms: f32 = 0.0;
    // === TRANSIENT ANALYZER: Track signed envelope deltas ===
    let mut envelope_delta_history: Vec<f32> = Vec::new();
    let mut frame_counter: usize = 0;
    // For new features
    let mut prev_mel_bands: Vec<f32> = Vec::new();
    let mut low_band_history: Vec<f32> = Vec::new();
    let mut high_band_history: Vec<f32> = Vec::new();

    if learn {
        if let Some(name) = pattern_name {
            sentinel.start_learning(name, AudioCategory::Unknown);
        }
    }

    while let Some(spectrum) = pump.next_power_spectrum() {
        // Compute features
        let mel_bands = spectrum_to_mel_bands(&spectrum, MEL_BANDS, pump.sample_rate());

        let spectral_centroid =
            compute_spectral_centroid(&spectrum, pump.sample_rate(), pump.window_size());

        let onset_strength = if prev_spectrum.is_empty() {
            0.0
        } else {
            compute_onset_strength(&prev_spectrum, &spectrum)
        };
        prev_spectrum = spectrum.clone();

        // Track onset history for regularity
        onset_history.push(onset_strength);
        if onset_history.len() > 100 {
            onset_history.remove(0);
        }
        let temporal_regularity = compute_temporal_regularity(&onset_history);

        // Estimate RMS from spectrum
        let rms_energy = (spectrum.iter().sum::<f32>() / spectrum.len() as f32).sqrt();

        // Envelope features
        rms_history.push(rms_energy);
        if rms_history.len() > 32 {
            rms_history.remove(0);
        }
        let envelope_delta = (rms_energy - prev_rms).abs();
        let envelope_variance = if rms_history.len() > 2 {
            let mean: f32 = rms_history.iter().sum::<f32>() / rms_history.len() as f32;
            rms_history.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / rms_history.len() as f32
        } else {
            0.0
        };

        // === TRANSIENT ANALYZER: Compute attack sharpness and decay roughness ===
        let signed_delta = rms_energy - prev_rms;
        prev_rms = rms_energy;
        envelope_delta_history.push(signed_delta);
        if envelope_delta_history.len() > 64 {
            envelope_delta_history.remove(0);
        }

        // Attack sharpness: max positive envelope derivative (how fast sound rises)
        let attack_sharpness = if envelope_delta_history.len() > 4 {
            let max_positive: f32 = envelope_delta_history
                .iter()
                .filter(|&&d| d > 0.0)
                .copied()
                .fold(0.0f32, |max, d| max.max(d));
            (max_positive * 2.0).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // Decay roughness: variance of negative envelope derivatives (smooth vs jagged decay)
        let decay_roughness = if envelope_delta_history.len() > 8 {
            let negative_deltas: Vec<f32> = envelope_delta_history
                .iter()
                .filter(|&&d| d < 0.0)
                .copied()
                .collect();
            if negative_deltas.len() > 2 {
                let mean: f32 = negative_deltas.iter().sum::<f32>() / negative_deltas.len() as f32;
                let variance: f32 = negative_deltas
                    .iter()
                    .map(|&d| (d - mean).powi(2))
                    .sum::<f32>()
                    / negative_deltas.len() as f32;
                (variance * 100.0).sqrt().clamp(0.0, 1.0)
            } else {
                0.5
            }
        } else {
            0.5
        };

        // Spectral rolloff
        let total: f32 = spectrum.iter().sum();
        let mut cumsum = 0.0;
        let mut rolloff_bin = 0;
        for (i, &p) in spectrum.iter().enumerate() {
            cumsum += p;
            if cumsum >= total * 0.85 {
                rolloff_bin = i;
                break;
            }
        }
        let spectral_rolloff =
            rolloff_bin as f32 * (pump.sample_rate() / pump.window_size() as f32);

        let high_freq_energy: f32 = spectrum.iter().skip(spectrum.len() / 2).sum();
        let zero_crossing_rate = (high_freq_energy / total.max(1e-10)).min(0.5);
        let spectral_flatness = compute_spectral_flatness(&spectrum);

        // Compute MFCC from mel bands for spectral shape encoding
        let mfcc = compute_mfcc(&mel_bands);

        // === COMPUTE NEW FEATURES ===
        // Spectral flux: rate of spectral change (L2 norm of difference)
        let spectral_flux = if prev_mel_bands.is_empty() {
            0.1
        } else {
            let diff_sq: f32 = mel_bands
                .iter()
                .zip(&prev_mel_bands)
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (diff_sq.sqrt() / mel_bands.len() as f32).min(1.0)
        };
        prev_mel_bands = mel_bands.clone();

        // Harmonic ratio: approximated from spectral peakiness
        // More peaks = more harmonic content
        let harmonic_ratio = {
            let mean_spec: f32 = spectrum.iter().sum::<f32>() / spectrum.len() as f32;
            let peaks: usize = spectrum
                .windows(3)
                .filter(|w| w[1] > w[0] && w[1] > w[2] && w[1] > mean_spec * 2.0)
                .count();
            (peaks as f32 / 20.0).min(1.0) // Normalize: 20 peaks = max harmonic
        };

        // Cross-frequency coupling (CFC): phase-amplitude coupling approximation
        // Low bands (0-5) modulate high bands (20-32)
        let low_energy: f32 = mel_bands.iter().take(5).sum::<f32>() / 5.0;
        let high_energy: f32 = mel_bands.iter().skip(20).take(12).sum::<f32>() / 12.0;

        low_band_history.push(low_energy);
        high_band_history.push(high_energy);
        if low_band_history.len() > 16 {
            low_band_history.remove(0);
        }
        if high_band_history.len() > 16 {
            high_band_history.remove(0);
        }

        // CFC theta-gamma: correlation between low and high band envelopes
        let cfc_theta_gamma = if low_band_history.len() >= 4 {
            let low_mean: f32 =
                low_band_history.iter().sum::<f32>() / low_band_history.len() as f32;
            let high_mean: f32 =
                high_band_history.iter().sum::<f32>() / high_band_history.len() as f32;
            let cov: f32 = low_band_history
                .iter()
                .zip(&high_band_history)
                .map(|(l, h)| (l - low_mean) * (h - high_mean))
                .sum::<f32>()
                / low_band_history.len() as f32;
            let low_std = (low_band_history
                .iter()
                .map(|x| (x - low_mean).powi(2))
                .sum::<f32>()
                / low_band_history.len() as f32)
                .sqrt();
            let high_std = (high_band_history
                .iter()
                .map(|x| (x - high_mean).powi(2))
                .sum::<f32>()
                / high_band_history.len() as f32)
                .sqrt();
            if low_std > 1e-6 && high_std > 1e-6 {
                ((cov / (low_std * high_std)) * 0.5 + 0.5).clamp(0.0, 1.0) // Map [-1,1] to [0,1]
            } else {
                0.5
            }
        } else {
            0.5
        };

        // CFC delta-beta: mid bands (5-15) to upper-mid (15-25)
        let mid_low: f32 = mel_bands.iter().skip(5).take(10).sum::<f32>() / 10.0;
        let mid_high: f32 = mel_bands.iter().skip(15).take(10).sum::<f32>() / 10.0;
        let cfc_delta_beta = ((mid_low * mid_high).sqrt() * 2.0).min(1.0);

        // Silence ratio: 1.0 if frame is silent (RMS < 2%), 0.0 otherwise
        let silence_ratio = if rms_energy < 0.02 { 1.0 } else { 0.0 };

        // Burst density: rate of significant onset events
        let burst_density = compute_ioi_variance(&onset_history, CONTROL_RATE);

        let features = AudioFeatures {
            mel_bands,
            mfcc,
            spectral_centroid,
            spectral_rolloff,
            onset_strength,
            rms_energy,
            zero_crossing_rate,
            spectral_flatness,
            temporal_regularity,
            envelope_delta,
            envelope_variance,
            frame_index: frame_counter,
            spectral_flux,
            harmonic_ratio,
            cfc_theta_gamma,
            cfc_delta_beta,
            attack_sharpness,
            decay_roughness,
            silence_ratio,
            burst_density,
        };

        sentinel.process(&features);
        frame_counter += 1;
    }

    if learn && finalize_learning {
        sentinel.stop_learning();
    }

    Ok(())
}

/// Get similarity score for a file against learned patterns
/// Detailed similarity breakdown for diagnostics
#[derive(Debug, Clone)]
pub struct SimilarityBreakdown {
    pub combined: f32,
    pub timbre_hdc: f32,
    pub rhythm_hdc: f32,
    pub spectral: f32,
    pub timbre_freq: f32,
    pub rhythm_freq: f32,
    pub multi_scale: f32,
}

fn get_similarity_detailed(
    pump: &mut FileAudioPump,
    sentinel: &mut AudioSentinel,
    target_pattern: &str,
) -> SimilarityBreakdown {
    let mut prev_spectrum: Vec<f32> = Vec::new();
    let mut onset_history: Vec<f32> = Vec::new();
    let mut rms_history: Vec<f32> = Vec::new();
    let mut prev_rms: f32 = 0.0;
    // === TRANSIENT ANALYZER: Track signed envelope deltas ===
    let mut envelope_delta_history: Vec<f32> = Vec::new();
    let mut frame_counter: usize = 0;
    let mut totals = SimilarityBreakdown {
        combined: 0.0,
        timbre_hdc: 0.0,
        rhythm_hdc: 0.0,
        spectral: 0.0,
        timbre_freq: 0.0,
        rhythm_freq: 0.0,
        multi_scale: 0.0,
    };
    let mut count = 0;
    // For new features
    let mut prev_mel_bands: Vec<f32> = Vec::new();
    let mut low_band_history: Vec<f32> = Vec::new();
    let mut high_band_history: Vec<f32> = Vec::new();

    while let Some(spectrum) = pump.next_power_spectrum() {
        let mel_bands = spectrum_to_mel_bands(&spectrum, MEL_BANDS, pump.sample_rate());
        let spectral_centroid =
            compute_spectral_centroid(&spectrum, pump.sample_rate(), pump.window_size());

        let onset_strength = if prev_spectrum.is_empty() {
            0.0
        } else {
            compute_onset_strength(&prev_spectrum, &spectrum)
        };
        prev_spectrum = spectrum.clone();

        onset_history.push(onset_strength);
        if onset_history.len() > 100 {
            onset_history.remove(0);
        }
        let temporal_regularity = compute_temporal_regularity(&onset_history);

        let rms_energy = (spectrum.iter().sum::<f32>() / spectrum.len() as f32).sqrt();
        rms_history.push(rms_energy);
        if rms_history.len() > 32 {
            rms_history.remove(0);
        }
        let envelope_delta = (rms_energy - prev_rms).abs();
        let envelope_variance = if rms_history.len() > 2 {
            let mean: f32 = rms_history.iter().sum::<f32>() / rms_history.len() as f32;
            rms_history.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / rms_history.len() as f32
        } else {
            0.0
        };

        // === TRANSIENT ANALYZER: Compute attack sharpness and decay roughness ===
        let signed_delta = rms_energy - prev_rms;
        prev_rms = rms_energy;
        envelope_delta_history.push(signed_delta);
        if envelope_delta_history.len() > 64 {
            envelope_delta_history.remove(0);
        }

        let attack_sharpness = if envelope_delta_history.len() > 4 {
            let max_positive: f32 = envelope_delta_history
                .iter()
                .filter(|&&d| d > 0.0)
                .copied()
                .fold(0.0f32, |max, d| max.max(d));
            (max_positive * 2.0).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let decay_roughness = if envelope_delta_history.len() > 8 {
            let negative_deltas: Vec<f32> = envelope_delta_history
                .iter()
                .filter(|&&d| d < 0.0)
                .copied()
                .collect();
            if negative_deltas.len() > 2 {
                let mean: f32 = negative_deltas.iter().sum::<f32>() / negative_deltas.len() as f32;
                let variance: f32 = negative_deltas
                    .iter()
                    .map(|&d| (d - mean).powi(2))
                    .sum::<f32>()
                    / negative_deltas.len() as f32;
                (variance * 100.0).sqrt().clamp(0.0, 1.0)
            } else {
                0.5
            }
        } else {
            0.5
        };

        let total: f32 = spectrum.iter().sum();
        let mut cumsum = 0.0;
        let mut rolloff_bin = 0;
        for (i, &p) in spectrum.iter().enumerate() {
            cumsum += p;
            if cumsum >= total * 0.85 {
                rolloff_bin = i;
                break;
            }
        }
        let spectral_rolloff =
            rolloff_bin as f32 * (pump.sample_rate() / pump.window_size() as f32);
        let high_freq_energy: f32 = spectrum.iter().skip(spectrum.len() / 2).sum();
        let zero_crossing_rate = (high_freq_energy / total.max(1e-10)).min(0.5);
        let spectral_flatness = compute_spectral_flatness(&spectrum);
        let mfcc = compute_mfcc(&mel_bands);

        // === COMPUTE NEW FEATURES ===
        // Spectral flux: rate of spectral change
        let spectral_flux = if prev_mel_bands.is_empty() {
            0.1
        } else {
            let diff_sq: f32 = mel_bands
                .iter()
                .zip(&prev_mel_bands)
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (diff_sq.sqrt() / mel_bands.len() as f32).min(1.0)
        };
        prev_mel_bands = mel_bands.clone();

        // Harmonic ratio: approximated from spectral peakiness
        let harmonic_ratio = {
            let mean_spec: f32 = spectrum.iter().sum::<f32>() / spectrum.len() as f32;
            let peaks: usize = spectrum
                .windows(3)
                .filter(|w| w[1] > w[0] && w[1] > w[2] && w[1] > mean_spec * 2.0)
                .count();
            (peaks as f32 / 20.0).min(1.0)
        };

        // CFC: track band envelopes for correlation
        let low_band: f32 = mel_bands.iter().take(5).sum::<f32>() / 5.0;
        let high_band: f32 =
            mel_bands.iter().skip(20).sum::<f32>() / (mel_bands.len() - 20).max(1) as f32;
        low_band_history.push(low_band);
        high_band_history.push(high_band);
        if low_band_history.len() > 16 {
            low_band_history.remove(0);
        }
        if high_band_history.len() > 16 {
            high_band_history.remove(0);
        }

        // CFC theta-gamma: correlation between low and high bands
        let cfc_theta_gamma = if low_band_history.len() >= 4 {
            let low_mean: f32 =
                low_band_history.iter().sum::<f32>() / low_band_history.len() as f32;
            let high_mean: f32 =
                high_band_history.iter().sum::<f32>() / high_band_history.len() as f32;
            let cov: f32 = low_band_history
                .iter()
                .zip(&high_band_history)
                .map(|(l, h)| (l - low_mean) * (h - high_mean))
                .sum::<f32>()
                / low_band_history.len() as f32;
            let low_std = (low_band_history
                .iter()
                .map(|x| (x - low_mean).powi(2))
                .sum::<f32>()
                / low_band_history.len() as f32)
                .sqrt()
                .max(1e-6);
            let high_std = (high_band_history
                .iter()
                .map(|x| (x - high_mean).powi(2))
                .sum::<f32>()
                / high_band_history.len() as f32)
                .sqrt()
                .max(1e-6);
            ((cov / (low_std * high_std)) * 0.5 + 0.5).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // CFC delta-beta: mid bands coupling
        let mid_low: f32 = mel_bands.iter().take(5).copied().sum::<f32>() / 5.0;
        let mid_high: f32 = mel_bands.iter().skip(15).take(10).copied().sum::<f32>() / 10.0;
        let cfc_delta_beta = ((mid_low * mid_high).sqrt() * 2.0).min(1.0);

        // Silence ratio: 1.0 if frame is silent (RMS < 2%), 0.0 otherwise
        let silence_ratio = if rms_energy < 0.02 { 1.0 } else { 0.0 };

        // Burst density: rate of significant onset events
        let burst_density = compute_ioi_variance(&onset_history, CONTROL_RATE);

        let features = AudioFeatures {
            mel_bands,
            mfcc,
            spectral_centroid,
            spectral_rolloff,
            onset_strength,
            rms_energy,
            zero_crossing_rate,
            spectral_flatness,
            temporal_regularity,
            envelope_delta,
            envelope_variance,
            frame_index: frame_counter,
            spectral_flux,
            harmonic_ratio,
            cfc_theta_gamma,
            cfc_delta_beta,
            attack_sharpness,
            decay_roughness,
            silence_ratio,
            burst_density,
        };

        let result = sentinel.process(&features);
        frame_counter += 1;

        if let Some(sim) = result.similarities.get(target_pattern) {
            totals.combined += sim.combined;
            totals.timbre_hdc += sim.timbre;
            totals.rhythm_hdc += sim.rhythm;
            totals.spectral += sim.spectral;
            totals.timbre_freq += sim.timbre_freq;
            totals.rhythm_freq += sim.rhythm_freq;
            totals.multi_scale += sim.multi_scale;
            count += 1;
        }
    }

    if count > 0 {
        let c = count as f32;
        SimilarityBreakdown {
            combined: totals.combined / c,
            timbre_hdc: totals.timbre_hdc / c,
            rhythm_hdc: totals.rhythm_hdc / c,
            spectral: totals.spectral / c,
            timbre_freq: totals.timbre_freq / c,
            rhythm_freq: totals.rhythm_freq / c,
            multi_scale: totals.multi_scale / c,
        }
    } else {
        totals
    }
}

fn get_similarity(
    pump: &mut FileAudioPump,
    sentinel: &mut AudioSentinel,
    target_pattern: &str,
) -> f32 {
    let mut prev_spectrum: Vec<f32> = Vec::new();
    let mut onset_history: Vec<f32> = Vec::new();
    let mut rms_history: Vec<f32> = Vec::new();
    let mut prev_rms: f32 = 0.0;
    // === TRANSIENT ANALYZER: Track signed envelope deltas ===
    let mut envelope_delta_history: Vec<f32> = Vec::new();
    let mut frame_counter: usize = 0;
    let mut total_similarity = 0.0;
    let mut count = 0;
    // For new features
    let mut prev_mel_bands: Vec<f32> = Vec::new();
    let mut low_band_history: Vec<f32> = Vec::new();
    let mut high_band_history: Vec<f32> = Vec::new();

    while let Some(spectrum) = pump.next_power_spectrum() {
        let mel_bands = spectrum_to_mel_bands(&spectrum, MEL_BANDS, pump.sample_rate());
        let spectral_centroid =
            compute_spectral_centroid(&spectrum, pump.sample_rate(), pump.window_size());

        let onset_strength = if prev_spectrum.is_empty() {
            0.0
        } else {
            compute_onset_strength(&prev_spectrum, &spectrum)
        };
        prev_spectrum = spectrum.clone();

        // Track onset history for regularity
        onset_history.push(onset_strength);
        if onset_history.len() > 100 {
            onset_history.remove(0);
        }
        let temporal_regularity = compute_temporal_regularity(&onset_history);

        let rms_energy = (spectrum.iter().sum::<f32>() / spectrum.len() as f32).sqrt();

        // Envelope features
        rms_history.push(rms_energy);
        if rms_history.len() > 32 {
            rms_history.remove(0);
        }
        let envelope_delta = (rms_energy - prev_rms).abs();
        let envelope_variance = if rms_history.len() > 2 {
            let mean: f32 = rms_history.iter().sum::<f32>() / rms_history.len() as f32;
            rms_history.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / rms_history.len() as f32
        } else {
            0.0
        };

        // === TRANSIENT ANALYZER: Compute attack sharpness and decay roughness ===
        let signed_delta = rms_energy - prev_rms;
        prev_rms = rms_energy;
        envelope_delta_history.push(signed_delta);
        if envelope_delta_history.len() > 64 {
            envelope_delta_history.remove(0);
        }

        let attack_sharpness = if envelope_delta_history.len() > 4 {
            let max_positive: f32 = envelope_delta_history
                .iter()
                .filter(|&&d| d > 0.0)
                .copied()
                .fold(0.0f32, |max, d| max.max(d));
            (max_positive * 2.0).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let decay_roughness = if envelope_delta_history.len() > 8 {
            let negative_deltas: Vec<f32> = envelope_delta_history
                .iter()
                .filter(|&&d| d < 0.0)
                .copied()
                .collect();
            if negative_deltas.len() > 2 {
                let mean: f32 = negative_deltas.iter().sum::<f32>() / negative_deltas.len() as f32;
                let variance: f32 = negative_deltas
                    .iter()
                    .map(|&d| (d - mean).powi(2))
                    .sum::<f32>()
                    / negative_deltas.len() as f32;
                (variance * 100.0).sqrt().clamp(0.0, 1.0)
            } else {
                0.5
            }
        } else {
            0.5
        };

        let total: f32 = spectrum.iter().sum();
        let mut cumsum = 0.0;
        let mut rolloff_bin = 0;
        for (i, &p) in spectrum.iter().enumerate() {
            cumsum += p;
            if cumsum >= total * 0.85 {
                rolloff_bin = i;
                break;
            }
        }
        let spectral_rolloff =
            rolloff_bin as f32 * (pump.sample_rate() / pump.window_size() as f32);
        let high_freq_energy: f32 = spectrum.iter().skip(spectrum.len() / 2).sum();
        let zero_crossing_rate = (high_freq_energy / total.max(1e-10)).min(0.5);
        let spectral_flatness = compute_spectral_flatness(&spectrum);

        // Compute MFCC from mel bands for spectral shape encoding
        let mfcc = compute_mfcc(&mel_bands);

        // === COMPUTE NEW FEATURES ===
        // Spectral flux: rate of spectral change
        let spectral_flux = if prev_mel_bands.is_empty() {
            0.1
        } else {
            let diff_sq: f32 = mel_bands
                .iter()
                .zip(&prev_mel_bands)
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (diff_sq.sqrt() / mel_bands.len() as f32).min(1.0)
        };
        prev_mel_bands = mel_bands.clone();

        // Harmonic ratio: approximated from spectral peakiness
        let harmonic_ratio = {
            let mean_spec: f32 = spectrum.iter().sum::<f32>() / spectrum.len() as f32;
            let peaks: usize = spectrum
                .windows(3)
                .filter(|w| w[1] > w[0] && w[1] > w[2] && w[1] > mean_spec * 2.0)
                .count();
            (peaks as f32 / 20.0).min(1.0)
        };

        // CFC: track band envelopes for correlation
        let low_band: f32 = mel_bands.iter().take(5).sum::<f32>() / 5.0;
        let high_band: f32 =
            mel_bands.iter().skip(20).sum::<f32>() / (mel_bands.len() - 20).max(1) as f32;
        low_band_history.push(low_band);
        high_band_history.push(high_band);
        if low_band_history.len() > 16 {
            low_band_history.remove(0);
        }
        if high_band_history.len() > 16 {
            high_band_history.remove(0);
        }

        // CFC theta-gamma: correlation between low and high bands
        let cfc_theta_gamma = if low_band_history.len() >= 4 {
            let low_mean: f32 =
                low_band_history.iter().sum::<f32>() / low_band_history.len() as f32;
            let high_mean: f32 =
                high_band_history.iter().sum::<f32>() / high_band_history.len() as f32;
            let cov: f32 = low_band_history
                .iter()
                .zip(&high_band_history)
                .map(|(l, h)| (l - low_mean) * (h - high_mean))
                .sum::<f32>()
                / low_band_history.len() as f32;
            let low_std = (low_band_history
                .iter()
                .map(|x| (x - low_mean).powi(2))
                .sum::<f32>()
                / low_band_history.len() as f32)
                .sqrt()
                .max(1e-6);
            let high_std = (high_band_history
                .iter()
                .map(|x| (x - high_mean).powi(2))
                .sum::<f32>()
                / high_band_history.len() as f32)
                .sqrt()
                .max(1e-6);
            ((cov / (low_std * high_std)) * 0.5 + 0.5).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // CFC delta-beta: mid bands coupling
        let mid_low: f32 = mel_bands.iter().take(5).copied().sum::<f32>() / 5.0;
        let mid_high: f32 = mel_bands.iter().skip(15).take(10).copied().sum::<f32>() / 10.0;
        let cfc_delta_beta = ((mid_low * mid_high).sqrt() * 2.0).min(1.0);

        // Silence ratio: 1.0 if frame is silent (RMS < 2%), 0.0 otherwise
        let silence_ratio = if rms_energy < 0.02 { 1.0 } else { 0.0 };

        // Burst density: rate of significant onset events
        let burst_density = compute_ioi_variance(&onset_history, CONTROL_RATE);

        let features = AudioFeatures {
            mel_bands,
            mfcc,
            spectral_centroid,
            spectral_rolloff,
            onset_strength,
            rms_energy,
            zero_crossing_rate,
            spectral_flatness,
            temporal_regularity,
            envelope_delta,
            envelope_variance,
            frame_index: frame_counter,
            spectral_flux,
            harmonic_ratio,
            cfc_theta_gamma,
            cfc_delta_beta,
            attack_sharpness,
            decay_roughness,
            silence_ratio,
            burst_density,
        };

        let result = sentinel.process(&features);
        frame_counter += 1;

        if let Some(sim) = result.similarities.get(target_pattern) {
            total_similarity += sim.combined;
            count += 1;
        }
    }

    if count > 0 {
        total_similarity / count as f32
    } else {
        0.0
    }
}

/// Run the "Dog vs Glass" style benchmark
fn run_discrimination_test(
    learn_files: &[&str],
    reject_files: &[&str],
    accept_files: &[&str],
    similarity_threshold: f32,
    preset: LtcPreset,
) -> Result<(usize, usize)> {
    let config = FileAudioConfig::default();
    let mut sentinel = AudioSentinel::with_preset(preset);

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║           REGRESSION TEST - Zero-Shot Discrimination             ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // Phase 1: Learn the concept (multi-shot: all files into same pattern)
    println!("  Phase 1: LEARNING");
    println!("  ─────────────────");

    let pattern_name = "Concept_0";
    for (i, &path) in learn_files.iter().enumerate() {
        match FileAudioPump::new(path, config.clone()) {
            Ok(mut pump) => {
                // First file starts fresh, subsequent files continue learning
                // Don't finalize until all files processed (multi-shot)
                let is_last = i == learn_files.len() - 1;
                if i == 0 {
                    process_file(&mut pump, &mut sentinel, true, Some(pattern_name), is_last)?;
                } else {
                    // Continue learning into same pattern
                    sentinel.continue_learning(pattern_name);
                    process_file(&mut pump, &mut sentinel, true, None, is_last)?;
                }
                println!(
                    "    ✓ Learned '{}' from {} ({}/{})",
                    pattern_name,
                    path,
                    i + 1,
                    learn_files.len()
                );
            }
            Err(e) => {
                println!("    ✗ Failed to load {}: {}", path, e);
            }
        }
    }
    // Note: stop_learning is called by process_file on the last file (is_last=true)

    let mut passed = 0;
    let mut total = 0;

    // Phase 2: Rejection test (should be dissimilar)
    println!(
        "\n  Phase 2: REJECTION TEST (expect similarity < {:.2})",
        similarity_threshold
    );
    println!("  ─────────────────────────────────────────────────────");

    for &path in reject_files {
        total += 1;
        match FileAudioPump::new(path, config.clone()) {
            Ok(mut pump) => {
                let similarity = get_similarity(&mut pump, &mut sentinel, "Concept_0");
                let pass = similarity < similarity_threshold;
                if pass {
                    passed += 1;
                }

                let status = if pass {
                    "\x1b[32m✓ PASS\x1b[0m"
                } else {
                    "\x1b[31m✗ FAIL\x1b[0m"
                };
                println!("    {} {} (similarity: {:.3})", status, path, similarity);
            }
            Err(e) => {
                println!("    \x1b[33m? SKIP\x1b[0m {}: {}", path, e);
            }
        }
    }

    // Phase 3: Acceptance test (should be similar)
    println!(
        "\n  Phase 3: ACCEPTANCE TEST (expect similarity >= {:.2})",
        similarity_threshold
    );
    println!("  ──────────────────────────────────────────────────────");

    for &path in accept_files {
        total += 1;
        match FileAudioPump::new(path, config.clone()) {
            Ok(mut pump) => {
                let similarity = get_similarity(&mut pump, &mut sentinel, "Concept_0");
                let pass = similarity >= similarity_threshold;
                if pass {
                    passed += 1;
                }

                let status = if pass {
                    "\x1b[32m✓ PASS\x1b[0m"
                } else {
                    "\x1b[31m✗ FAIL\x1b[0m"
                };
                println!("    {} {} (similarity: {:.3})", status, path, similarity);
            }
            Err(e) => {
                println!("    \x1b[33m? SKIP\x1b[0m {}: {}", path, e);
            }
        }
    }

    // Summary
    println!("\n  ═══════════════════════════════════════════════════════════════════");
    let pass_rate = passed as f32 / total as f32 * 100.0;
    let color = if pass_rate >= 80.0 {
        "\x1b[32m"
    } else if pass_rate >= 50.0 {
        "\x1b[33m"
    } else {
        "\x1b[31m"
    };
    println!(
        "  RESULT: {}{}/{} tests passed ({:.0}%)\x1b[0m",
        color, passed, total, pass_rate
    );
    println!("  ═══════════════════════════════════════════════════════════════════\n");

    Ok((passed, total))
}

/// Run genre discrimination test on GTZAN-style dataset
fn run_genre_test(data_dir: &str, use_premium: bool) -> Result<()> {
    let config = FileAudioConfig::default();

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    if use_premium {
        println!("║      GENRE DISCRIMINATION TEST (PREMIUM: CfC + RFF + Phase)      ║");
    } else {
        println!("║              GENRE DISCRIMINATION TEST (GTZAN-style)             ║");
    }
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // Try to find genre subdirectories
    let base_path = PathBuf::from(data_dir);

    let genre_dirs: Vec<_> = std::fs::read_dir(&base_path)
        .map_err(|e| anyhow::anyhow!("Cannot read {}: {}", data_dir, e))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();

    if genre_dirs.is_empty() {
        println!("  No genre subdirectories found in {}", data_dir);
        println!(
            "  Expected structure: {}/blues/, {}/metal/, etc.",
            data_dir, data_dir
        );
        return Ok(());
    }

    println!(
        "  Found {} genres: {:?}\n",
        genre_dirs.len(),
        genre_dirs
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy())
            .collect::<Vec<_>>()
    );

    // Create sentinel with appropriate encoder mode
    let mut sentinel = if use_premium {
        AudioSentinel::premium()
    } else {
        AudioSentinel::new()
    };

    // === FEW-SHOT LEARNING: Configure number of samples per class ===
    // 1-shot: baseline (original approach)
    // 2-shot: more stable prototypes, but fewer test samples for small datasets
    const SHOTS_PER_CLASS: usize = 1; // Number of samples to learn per class
    println!(
        "  Phase 1: Learning {} sample(s) per genre...",
        SHOTS_PER_CLASS
    );
    for genre_dir in &genre_dirs {
        let genre_name = genre_dir.file_name().unwrap().to_string_lossy().to_string();

        if let Ok(mut processor) = DatasetProcessor::from_directory(genre_dir) {
            let mut learned_count = 0;

            // Learn multiple samples, accumulating into the same prototype
            while learned_count < SHOTS_PER_CLASS {
                if let Some(Ok(mut pump)) = processor.next_file(config.clone()) {
                    // First sample: start learning, don't finalize
                    // Last sample: finalize (triggers averaging)
                    let is_first = learned_count == 0;
                    let is_last = learned_count == SHOTS_PER_CLASS - 1;

                    if is_first {
                        // Start learning with this pattern name
                        process_file(&mut pump, &mut sentinel, true, Some(&genre_name), false)?;
                    } else if is_last {
                        // Continue accumulating, then finalize
                        process_file(&mut pump, &mut sentinel, true, None, true)?;
                    } else {
                        // Middle samples: just accumulate
                        process_file(&mut pump, &mut sentinel, true, None, false)?;
                    }
                    learned_count += 1;
                } else {
                    break; // No more files available
                }
            }

            if learned_count > 0 {
                println!("    ✓ Learned '{}' ({} samples)", genre_name, learned_count);
            }
        }
    }

    // Test remaining files
    println!("\n  Phase 2: Testing classification...");
    let mut correct = 0;
    let mut total = 0;
    let mut confusion: HashMap<String, HashMap<String, usize>> = HashMap::new();

    for genre_dir in &genre_dirs {
        let true_genre = genre_dir.file_name().unwrap().to_string_lossy().to_string();
        confusion.entry(true_genre.clone()).or_default();

        if let Ok(mut processor) = DatasetProcessor::from_directory(genre_dir) {
            // Skip files used for learning (SHOTS_PER_CLASS samples)
            for _ in 0..SHOTS_PER_CLASS {
                let _ = processor.next_file(config.clone());
            }

            // Test next few files
            let mut tested = 0;
            while tested < 5 {
                if let Some(Ok(mut pump)) = processor.next_file(config.clone()) {
                    // === HUBNESS NORMALIZATION (Z-Score / Local Scaling) ===
                    // Step 1: Collect ALL raw similarity scores
                    let genre_keys: Vec<_> = sentinel.patterns.keys().cloned().collect();
                    let mut raw_scores: Vec<(String, f32)> = Vec::new();

                    for genre in &genre_keys {
                        pump.reset();
                        let sim = get_similarity(&mut pump, &mut sentinel, genre);
                        raw_scores.push((genre.clone(), sim));
                    }

                    // Step 2: Compute mean and std of raw scores
                    let scores: Vec<f32> = raw_scores.iter().map(|(_, s)| *s).collect();
                    let mean: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
                    let variance: f32 = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>()
                        / scores.len() as f32;
                    let std: f32 = variance.sqrt().max(0.001); // Avoid division by zero

                    // Step 3: Z-normalize and find best match
                    let mut best_genre = String::new();
                    let mut best_zscore = f32::NEG_INFINITY;

                    for (genre, raw_sim) in &raw_scores {
                        let zscore = (raw_sim - mean) / std;
                        if zscore > best_zscore {
                            best_zscore = zscore;
                            best_genre = genre.clone();
                        }
                    }

                    total += 1;
                    tested += 1;

                    *confusion
                        .entry(true_genre.clone())
                        .or_default()
                        .entry(best_genre.clone())
                        .or_insert(0) += 1;

                    if best_genre == true_genre {
                        correct += 1;
                    }
                } else {
                    break;
                }
            }
        }
    }

    // Print confusion matrix
    println!("\n  Confusion Matrix:");
    let genres: Vec<_> = confusion.keys().cloned().collect();
    print!("           ");
    for g in &genres {
        print!("{:>8}", &g[..g.len().min(7)]);
    }
    println!();

    for true_g in &genres {
        print!("  {:>8} ", &true_g[..true_g.len().min(7)]);
        for pred_g in &genres {
            let count = confusion
                .get(true_g)
                .and_then(|m| m.get(pred_g))
                .copied()
                .unwrap_or(0);
            if true_g == pred_g {
                print!("\x1b[32m{:>8}\x1b[0m", count);
            } else if count > 0 {
                print!("\x1b[31m{:>8}\x1b[0m", count);
            } else {
                print!("{:>8}", count);
            }
        }
        println!();
    }

    let accuracy = correct as f32 / total.max(1) as f32 * 100.0;
    println!("\n  Accuracy: {}/{} ({:.1}%)\n", correct, total, accuracy);

    Ok(())
}

fn print_usage() {
    println!("Sentinel Regression Test Harness");
    println!();
    println!("Usage:");
    println!("  regression-test --learn <file> [--learn <file2>...] \\");
    println!("                  --reject <file> [--reject <file2>...] \\");
    println!("                  --accept <file> [--accept <file2>...]");
    println!();
    println!("  regression-test --gtzan <directory>");
    println!();
    println!("  regression-test --esc50 <directory>");
    println!();
    println!("Options:");
    println!("  --learn <file>     File(s) to learn as the concept");
    println!("  --reject <file>    File(s) that should be rejected (dissimilar)");
    println!("  --accept <file>    File(s) that should be accepted (similar)");
    println!("  --threshold <val>  Similarity threshold (default: 0.4)");
    println!("  --gtzan <dir>      Run GTZAN genre classification test");
    println!("  --esc50 <dir>      Run ESC-50 environmental sound test");
    println!();
    println!("Examples:");
    println!("  # Test dog bark discrimination");
    println!(
        "  regression-test --learn dog_bark.wav --reject glass_break.wav --accept dog_bark_2.wav"
    );
    println!();
    println!("  # Run on GTZAN dataset");
    println!("  regression-test --gtzan ./datasets/gtzan/genres/");
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return Ok(());
    }

    let mut learn_files: Vec<&str> = Vec::new();
    let mut reject_files: Vec<&str> = Vec::new();
    let mut accept_files: Vec<&str> = Vec::new();
    let mut diagnose_files: Vec<&str> = Vec::new();
    let mut threshold = 0.4;
    let mut gtzan_dir: Option<&str> = None;
    let mut esc50_dir: Option<&str> = None;
    let mut ltc_preset = LtcPreset::Standard;
    let mut use_premium = false; // Premium encoder mode (CfC + RFF)

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--learn" => {
                i += 1;
                if i < args.len() {
                    learn_files.push(&args[i]);
                }
            }
            "--reject" => {
                i += 1;
                if i < args.len() {
                    reject_files.push(&args[i]);
                }
            }
            "--accept" => {
                i += 1;
                if i < args.len() {
                    accept_files.push(&args[i]);
                }
            }
            "--diagnose" => {
                i += 1;
                if i < args.len() {
                    diagnose_files.push(&args[i]);
                }
            }
            "--threshold" => {
                i += 1;
                if i < args.len() {
                    threshold = args[i].parse().unwrap_or(0.4);
                }
            }
            "--gtzan" => {
                i += 1;
                if i < args.len() {
                    gtzan_dir = Some(&args[i]);
                }
            }
            "--esc50" => {
                i += 1;
                if i < args.len() {
                    esc50_dir = Some(&args[i]);
                }
            }
            "--preset" => {
                i += 1;
                if i < args.len() {
                    ltc_preset = match args[i].to_lowercase().as_str() {
                        "bird" | "fast" => LtcPreset::FastBird,
                        "whale" | "slow" => LtcPreset::SlowWhale,
                        _ => LtcPreset::Standard,
                    };
                    println!(
                        "  Using LTC preset: {:?}",
                        match ltc_preset {
                            LtcPreset::Standard => "Standard (environmental)",
                            LtcPreset::FastBird => "FastBird (rapid vocalizations)",
                            LtcPreset::SlowWhale => "SlowWhale (low-frequency)",
                        }
                    );
                }
            }
            "--premium" => {
                use_premium = true;
                println!("  Using PREMIUM encoder (CfC + RFF + Phase Space)");
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    let start = Instant::now();

    // Run appropriate test
    if let Some(dir) = gtzan_dir {
        run_genre_test(dir, use_premium)?;
    } else if let Some(dir) = esc50_dir {
        run_genre_test(dir, use_premium)?; // Same logic works for ESC-50
    } else if !learn_files.is_empty() {
        // First run discrimination test
        let (passed, total) = run_discrimination_test(
            &learn_files,
            &reject_files,
            &accept_files,
            threshold,
            ltc_preset,
        )?;

        // If diagnose files specified, run detailed analysis
        if !diagnose_files.is_empty() {
            println!("\n  ═══════════════════════════════════════════════════════════════════");
            println!("  DETAILED SIMILARITY BREAKDOWN");
            println!("  ═══════════════════════════════════════════════════════════════════\n");

            // Re-learn the concept for diagnosis
            let mut sentinel = AudioSentinel::with_preset(ltc_preset);
            for (i, learn_file) in learn_files.iter().enumerate() {
                let config = FileAudioConfig {
                    window_size: 2048,
                    hop_size: 512,
                    apply_window: true,
                };
                let mut pump =
                    FileAudioPump::new(learn_file, config).map_err(|e| anyhow::anyhow!("{}", e))?;
                println!("  Reference: {}", learn_file);
                let is_last = i == learn_files.len() - 1;
                if i == 0 {
                    process_file(&mut pump, &mut sentinel, true, Some("Concept_0"), is_last)?;
                } else {
                    sentinel.continue_learning("Concept_0");
                    process_file(&mut pump, &mut sentinel, true, None, is_last)?;
                }
            }

            // Diagnose each file
            for diag_file in &diagnose_files {
                let config = FileAudioConfig {
                    window_size: 2048,
                    hop_size: 512,
                    apply_window: true,
                };
                match FileAudioPump::new(diag_file, config) {
                    Ok(mut pump) => {
                        let breakdown =
                            get_similarity_detailed(&mut pump, &mut sentinel, "Concept_0");
                        println!("\n  File: {}", diag_file);
                        println!("    Combined:    {:.3}", breakdown.combined);
                        println!("    ────────────────────────────");
                        println!("    Timbre HDC:  {:.3}", breakdown.timbre_hdc);
                        println!("    Rhythm HDC:  {:.3}", breakdown.rhythm_hdc);
                        println!("    Spectral:    {:.3}", breakdown.spectral);
                        println!("    Timbre Freq: {:.3}", breakdown.timbre_freq);
                        println!("    Rhythm Freq: {:.3}", breakdown.rhythm_freq);
                        println!("    Multi-scale: {:.3}", breakdown.multi_scale);
                    }
                    Err(e) => println!("  Error loading {}: {}", diag_file, e),
                }
            }
            println!();
        }

        // Return exit code for CI
        if passed < total {
            std::process::exit(1);
        }
    } else {
        print_usage();
    }

    println!("  Total time: {:.2}s\n", start.elapsed().as_secs_f32());

    Ok(())
}
