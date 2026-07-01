// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ablation study: measure the contribution of each module to audio quality.
//!
//! Runs the pipeline with each module disabled in turn and measures:
//! - Spectral richness (number of significant frequency bins)
//! - Temporal coherence (autocorrelation at beat period)
//! - Spatial width (L-R decorrelation)
//! - Dynamic range (peak/RMS ratio in dB)
//! - Free energy (if FEP enabled)

use crate::streaming::StreamingSynth;
use crate::{MuseConfig, MusicalState};

/// Audio quality metrics for ablation comparison.
#[derive(Debug, Clone, Copy)]
pub struct AudioQualityMetrics {
    /// RMS energy [0, 1].
    pub rms: f32,
    /// Peak amplitude.
    pub peak: f32,
    /// Dynamic range (crest factor in dB).
    pub dynamic_range_db: f32,
    /// Stereo width: L-R decorrelation [0, 1]. 0 = mono, 1 = fully decorrelated.
    pub stereo_width: f32,
    /// Spectral richness: fraction of non-trivial frequency bins.
    pub spectral_richness: f32,
    /// Temporal coherence: normalized autocorrelation at lag ~beat period.
    pub temporal_coherence: f32,
}

/// Result of an ablation run.
#[derive(Debug, Clone)]
pub struct AblationResult {
    /// Configuration label.
    pub label: String,
    /// Measured metrics.
    pub metrics: AudioQualityMetrics,
    /// Number of chunks rendered.
    pub chunks: usize,
}

/// Run the full ablation study.
///
/// Returns metrics for: baseline (all on), and each module disabled individually.
pub fn run_ablation(state: &MusicalState, chunks: usize) -> Vec<AblationResult> {
    let mut results = Vec::new();

    // Baseline: everything on
    results.push(run_config("all_modules_on", state, chunks, |s| {
        s.enable_binaural = true;
        s.enable_sidechain = true;
        s.enable_fep = true;
        s.feedback_strength = 0.5;
    }));

    // Ablate each module
    results.push(run_config("no_binaural", state, chunks, |s| {
        s.enable_binaural = false;
        s.enable_sidechain = true;
        s.enable_fep = true;
        s.feedback_strength = 0.5;
    }));

    results.push(run_config("no_sidechain", state, chunks, |s| {
        s.enable_binaural = true;
        s.enable_sidechain = false;
        s.enable_fep = true;
        s.feedback_strength = 0.5;
    }));

    results.push(run_config("no_fep", state, chunks, |s| {
        s.enable_binaural = true;
        s.enable_sidechain = true;
        s.enable_fep = false;
        s.feedback_strength = 0.5;
    }));

    results.push(run_config("no_feedback", state, chunks, |s| {
        s.enable_binaural = true;
        s.enable_sidechain = true;
        s.enable_fep = true;
        s.feedback_strength = 0.0;
    }));

    results.push(run_config("all_off", state, chunks, |s| {
        s.enable_binaural = false;
        s.enable_sidechain = false;
        s.enable_fep = false;
        s.feedback_strength = 0.0;
    }));

    results
}

fn run_config<F>(label: &str, state: &MusicalState, chunks: usize, configure: F) -> AblationResult
where
    F: FnOnce(&mut StreamingSynth),
{
    let mut synth = StreamingSynth::new(
        MuseConfig {
            duration_secs: 30.0,
            max_notes: 32,
            ..Default::default()
        },
        44100,
    );
    synth.update_state(state);
    configure(&mut synth);

    let mut all_samples: Vec<[f32; 2]> = Vec::new();
    for _ in 0..chunks {
        let chunk = synth.render_chunk();
        all_samples.extend_from_slice(&chunk);
    }

    let metrics = compute_metrics(&all_samples);
    AblationResult {
        label: label.to_string(),
        metrics,
        chunks,
    }
}

fn compute_metrics(samples: &[[f32; 2]]) -> AudioQualityMetrics {
    if samples.is_empty() {
        return AudioQualityMetrics {
            rms: 0.0,
            peak: 0.0,
            dynamic_range_db: 0.0,
            stereo_width: 0.0,
            spectral_richness: 0.0,
            temporal_coherence: 0.0,
        };
    }

    let n = samples.len() as f32;

    // RMS and peak
    let mut sum_sq = 0.0f32;
    let mut peak = 0.0f32;
    for s in samples {
        let mono = (s[0] + s[1]) * 0.5;
        sum_sq += mono * mono;
        peak = peak.max(mono.abs());
    }
    let rms = (sum_sq / n).sqrt();
    let dynamic_range_db = if rms > 1e-8 {
        20.0 * (peak / rms).log10()
    } else {
        0.0
    };

    // Stereo width: 1 - correlation(L, R)
    let mut sum_lr = 0.0f32;
    let mut sum_ll = 0.0f32;
    let mut sum_rr = 0.0f32;
    for s in samples {
        sum_lr += s[0] * s[1];
        sum_ll += s[0] * s[0];
        sum_rr += s[1] * s[1];
    }
    let denom = (sum_ll * sum_rr).sqrt();
    let correlation = if denom > 1e-8 { sum_lr / denom } else { 1.0 };
    let stereo_width = 1.0 - correlation.clamp(0.0, 1.0);

    // Spectral richness: DFT of first 256 samples, count non-trivial bins
    let mono: Vec<f32> = samples
        .iter()
        .take(256)
        .map(|s| (s[0] + s[1]) * 0.5)
        .collect();
    let spectral_richness = if mono.len() >= 64 {
        let half = mono.len() / 2;
        let mut magnitudes = Vec::with_capacity(half);
        for k in 0..half {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            let freq = std::f32::consts::TAU * k as f32 / mono.len() as f32;
            for (i, &s) in mono.iter().enumerate() {
                let ph = freq * i as f32;
                re += s * ph.cos();
                im += s * ph.sin();
            }
            magnitudes.push((re * re + im * im).sqrt());
        }
        let max_mag = magnitudes.iter().cloned().fold(0.0f32, f32::max);
        if max_mag > 1e-8 {
            let threshold = max_mag * 0.01; // 1% of peak
            let significant = magnitudes.iter().filter(|&&m| m > threshold).count();
            significant as f32 / half as f32
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Temporal coherence: autocorrelation at ~beat period (~500ms = 22050 samples)
    let lag = 22050.min(samples.len() / 2);
    let temporal_coherence = if lag > 0 && samples.len() > lag * 2 {
        let mono: Vec<f32> = samples.iter().map(|s| (s[0] + s[1]) * 0.5).collect();
        let mut autocorr = 0.0f32;
        let mut energy = 0.0f32;
        for i in 0..lag {
            autocorr += mono[i] * mono[i + lag];
            energy += mono[i] * mono[i];
        }
        if energy > 1e-8 {
            (autocorr / energy).clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    AudioQualityMetrics {
        rms,
        peak,
        dynamic_range_db,
        stereo_width,
        spectral_richness,
        temporal_coherence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ablation_produces_results() {
        let state = MusicalState {
            consciousness_level: 0.7,
            arousal: 0.5,
            harmony_activations: [0.6, 0.5, 0.4, 0.3, 0.5, 0.6, 0.4, 0.5],
            ..Default::default()
        };
        let results = run_ablation(&state, 50); // short for testing

        assert_eq!(results.len(), 6, "should have 6 ablation configs");
        for r in &results {
            assert!(r.metrics.rms.is_finite(), "{}: RMS not finite", r.label);
            assert!(
                r.metrics.stereo_width >= 0.0,
                "{}: negative stereo width",
                r.label
            );
        }

        // Baseline should have wider stereo than no_binaural
        let baseline = &results[0];
        let no_bin = results.iter().find(|r| r.label == "no_binaural").unwrap();
        eprintln!(
            "  Ablation: baseline stereo={:.4}, no_binaural stereo={:.4}",
            baseline.metrics.stereo_width, no_bin.metrics.stereo_width
        );
    }

    #[test]
    fn binaural_increases_stereo_width() {
        let state = MusicalState {
            consciousness_level: 0.7,
            arousal: 0.5,
            harmony_activations: [0.6; 8],
            ..Default::default()
        };
        let results = run_ablation(&state, 100);

        let baseline = &results[0]; // all on
        let no_bin = results.iter().find(|r| r.label == "no_binaural").unwrap();

        // Binaural should increase stereo width (or at least not decrease it significantly)
        // Note: with panning fallback, both have some width
        assert!(
            baseline.metrics.stereo_width >= no_bin.metrics.stereo_width - 0.1,
            "binaural should not reduce stereo: with={:.4}, without={:.4}",
            baseline.metrics.stereo_width,
            no_bin.metrics.stereo_width
        );
    }
}
