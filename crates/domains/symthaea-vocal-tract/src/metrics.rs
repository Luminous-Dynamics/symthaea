// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vocal tract quality metrics and WAV output.
//!
//! Provides spectral accuracy metrics for benchmarking the LTC-driven vocal tract
//! pipeline against known formant targets, and WAV file I/O for offline analysis.

use crate::types::{FormantFrame, FormantTarget};

/// Quality metrics for the vocal tract pipeline output.
#[derive(Debug, Clone, Default)]
pub struct VocalTractMetrics {
    /// Mean F1 error (Hz) across all frames.
    pub mean_f1_error: f32,
    /// Mean F2 error (Hz).
    pub mean_f2_error: f32,
    /// Mean F3 error (Hz).
    pub mean_f3_error: f32,
    /// Max frame-to-frame F1 delta (smoothness measure).
    pub max_f1_delta: f32,
    /// F0 RMSE (Hz) — deviation from target pitch.
    pub f0_rmse: f32,
    /// Energy correlation with target (Pearson r, -1 to 1).
    pub energy_correlation: f32,
    /// Number of frames evaluated.
    pub num_frames: usize,
    /// Mel-Cepstral Distortion (dB). Lower is better (<5 dB = excellent).
    pub mcd: f32,
}

impl VocalTractMetrics {
    /// Compute metrics from pipeline output frames vs. target formants.
    ///
    /// `frames`: output from the vocal tract pipeline.
    /// `targets`: one `FormantTarget` per frame (matched by index).
    /// `target_f0`: the intended fundamental frequency.
    pub fn compute(frames: &[FormantFrame], targets: &[FormantTarget], target_f0: f32) -> Self {
        if frames.is_empty() || targets.is_empty() {
            return Self::default();
        }

        let n = frames.len().min(targets.len());
        let mut f1_err_sum = 0.0f32;
        let mut f2_err_sum = 0.0f32;
        let mut f3_err_sum = 0.0f32;
        let mut f0_sq_sum = 0.0f32;
        let mut max_f1_delta = 0.0f32;

        for i in 0..n {
            f1_err_sum += (frames[i].f1 - targets[i].f1).abs();
            f2_err_sum += (frames[i].f2 - targets[i].f2).abs();
            f3_err_sum += (frames[i].f3 - targets[i].f3).abs();
            f0_sq_sum += (frames[i].f0 - target_f0).powi(2);

            if i > 0 {
                let delta = (frames[i].f1 - frames[i - 1].f1).abs();
                if delta > max_f1_delta {
                    max_f1_delta = delta;
                }
            }
        }

        let nf = n as f32;

        // Energy correlation (Pearson r between frame energy and target voicing)
        let target_energies: Vec<f32> = targets[..n]
            .iter()
            .map(|t| if t.is_voiced { 0.7 } else { 0.3 })
            .collect();
        let frame_energies: Vec<f32> = frames[..n].iter().map(|f| f.energy).collect();
        let energy_correlation = pearson_r(&frame_energies, &target_energies);

        let mcd = mel_cepstral_distortion(&frames[..n], &targets[..n]);

        Self {
            mean_f1_error: f1_err_sum / nf,
            mean_f2_error: f2_err_sum / nf,
            mean_f3_error: f3_err_sum / nf,
            max_f1_delta,
            f0_rmse: (f0_sq_sum / nf).sqrt(),
            energy_correlation,
            num_frames: n,
            mcd,
        }
    }

    /// Overall formant error (mean of F1+F2+F3 errors).
    pub fn mean_formant_error(&self) -> f32 {
        (self.mean_f1_error + self.mean_f2_error + self.mean_f3_error) / 3.0
    }
}

/// Save audio samples to a WAV file.
///
/// Writes 16-bit PCM mono at the specified sample rate.
#[cfg(feature = "hound")]
pub fn save_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let sample = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(sample)?;
    }
    writer.finalize()
}

/// Load audio samples from a WAV file (16-bit PCM).
#[cfg(feature = "hound")]
pub fn load_wav(path: &str) -> Result<(Vec<f32>, u32), hound::Error> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .filter_map(|s| s.ok())
        .map(|s| s as f32 / i16::MAX as f32)
        .collect();
    Ok((samples, spec.sample_rate))
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEL-CEPSTRAL DISTORTION (MCD)
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert formant frequencies to a log-mel-cepstral representation.
///
/// Maps F1/F2/F3 through the mel scale and takes the natural log,
/// producing values comparable to real mel-cepstral coefficients.
fn formants_to_mel_cepstrum(f1: f32, f2: f32, f3: f32) -> [f32; 3] {
    // HTK mel scale: mel = 2595 * log10(1 + f/700), then take ln for cepstral domain
    let to_log_mel = |f: f32| (2595.0 * (1.0 + f.max(1.0) / 700.0).log10()).ln();
    [to_log_mel(f1), to_log_mel(f2), to_log_mel(f3)]
}

/// Compute Mel-Cepstral Distortion between synthesized and target formant trajectories.
///
/// MCD (dB) = (10√2 / ln10) × √(Σ (c_synth_i - c_target_i)²)
///
/// Lower is better. Typical speech synthesis values:
/// - <5 dB: excellent (near-natural)
/// - 5-8 dB: good
/// - >8 dB: noticeable artifacts
pub fn mel_cepstral_distortion(
    synth_frames: &[FormantFrame],
    target_frames: &[FormantTarget],
) -> f32 {
    let n = synth_frames.len().min(target_frames.len());
    if n == 0 {
        return 0.0;
    }

    let scale = 10.0 * std::f32::consts::SQRT_2 / std::f32::consts::LN_10;
    let mut total_mcd = 0.0f32;

    for i in 0..n {
        let synth_mc =
            formants_to_mel_cepstrum(synth_frames[i].f1, synth_frames[i].f2, synth_frames[i].f3);
        let target_mc = formants_to_mel_cepstrum(
            target_frames[i].f1,
            target_frames[i].f2,
            target_frames[i].f3,
        );

        let sq_sum: f32 = synth_mc
            .iter()
            .zip(target_mc.iter())
            .map(|(s, t)| (s - t).powi(2))
            .sum();

        total_mcd += scale * sq_sum.sqrt();
    }

    total_mcd / n as f32
}

/// Pearson correlation coefficient.
fn pearson_r(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len()) as f32;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x = x.iter().sum::<f32>() / n;
    let mean_y = y.iter().sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;
    for i in 0..n as usize {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 { 0.0 } else { cov / denom }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERCEPTUAL METRICS — HNR + Spectral Tilt
// ═══════════════════════════════════════════════════════════════════════════════

/// Perceptual quality metrics computed from audio samples.
///
/// Measures voice naturalness beyond formant accuracy:
/// - **HNR**: How periodic (harmonic) vs noisy the voiced signal is
/// - **Spectral Tilt**: How rapidly energy falls with frequency
#[derive(Debug, Clone, Default)]
pub struct PerceptualMetrics {
    /// Harmonic-to-Noise Ratio (dB) for voiced segments.
    /// Higher = cleaner voicing. Typical: 15–25 dB for normal speech.
    pub hnr: f32,
    /// Actual spectral slope (dB/octave).
    pub spectral_tilt: f32,
    /// |actual - expected(-12)| dB/octave — deviation from modal voice target.
    pub spectral_tilt_deviation: f32,
    /// Number of voiced analysis windows.
    pub voiced_frames: usize,
    /// Total analysis windows.
    pub total_frames: usize,
}

/// Compute Harmonic-to-Noise Ratio via autocorrelation.
///
/// For each voiced window of `3 * period` samples, compute the normalized
/// autocorrelation at lag T0 = sample_rate / f0. HNR = 10 log10(r / (1 - r)).
///
/// Returns HNR in dB (higher = cleaner voicing). Capped at 40 dB.
pub fn compute_hnr(samples: &[f32], f0: f32, sample_rate: u32) -> f32 {
    if samples.is_empty() || f0 < 50.0 {
        return 0.0;
    }

    let period = (sample_rate as f32 / f0.max(50.0)).round() as usize;
    if period == 0 {
        return 0.0;
    }

    let window_size = period * 3;
    if samples.len() < window_size {
        return 0.0;
    }

    let mut total_r = 0.0f64;
    let mut count = 0usize;

    // Slide windows across the signal
    let step = window_size / 2; // 50% overlap
    let mut offset = 0;
    while offset + window_size <= samples.len() {
        let window = &samples[offset..offset + window_size];

        // Check if window has sufficient energy (skip silence)
        let energy: f32 = window.iter().map(|s| s * s).sum::<f32>() / window.len() as f32;
        if energy < 1e-8 {
            offset += step;
            continue;
        }

        // Normalized autocorrelation at lag = period
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..window_size - period {
            num += window[i] as f64 * window[i + period] as f64;
            den += window[i] as f64 * window[i] as f64;
        }

        if den > 1e-12 {
            let r = (num / den).clamp(-1.0, 1.0);
            if r > 0.0 && r < 1.0 {
                total_r += r;
                count += 1;
            }
        }

        offset += step;
    }

    if count == 0 {
        return 0.0;
    }

    let avg_r = total_r / count as f64;
    if avg_r <= 0.0 || avg_r >= 1.0 {
        return 0.0;
    }

    // HNR = 10 * log10(r / (1 - r))
    let hnr = 10.0 * (avg_r / (1.0 - avg_r)).log10();
    (hnr as f32).clamp(-10.0, 40.0)
}

/// Compute spectral tilt via formant power regression (Goertzel algorithm).
///
/// Measures power at F1, F2, F3 frequencies using Goertzel's algorithm,
/// then fits a line in log-frequency / log-power space to get dB/octave slope.
///
/// Expected: ~-12 dB/octave for modal voice (LF glottal model).
pub fn compute_spectral_tilt(samples: &[f32], f1: f32, f2: f32, f3: f32, sample_rate: u32) -> f32 {
    if samples.is_empty() || f1 <= 0.0 || f2 <= 0.0 || f3 <= 0.0 {
        return 0.0;
    }

    let sr = sample_rate as f32;
    let freqs = [f1, f2, f3];
    let mut powers = [0.0f32; 3];

    // Goertzel algorithm for each formant frequency
    for (i, &freq) in freqs.iter().enumerate() {
        let k = (freq / sr * samples.len() as f32).round();
        let omega = 2.0 * std::f32::consts::PI * k / samples.len() as f32;
        let coeff = 2.0 * omega.cos();

        let mut s1: f64 = 0.0;
        let mut s2: f64 = 0.0;

        for &sample in samples {
            let s0 = sample as f64 + coeff as f64 * s1 - s2;
            s2 = s1;
            s1 = s0;
        }

        // Power = |X(k)|^2 / N^2
        let real = s1 - s2 * (omega.cos() as f64);
        let imag = s2 * (omega.sin() as f64);
        let power = (real * real + imag * imag) / (samples.len() as f64 * samples.len() as f64);
        powers[i] = power as f32;
    }

    // Linear regression in log-log space: log2(freq) vs 10*log10(power)
    // Slope gives dB/octave
    let log_freqs = [f1.log2(), f2.log2(), f3.log2()];
    let log_powers: [f32; 3] = [
        if powers[0] > 1e-20 {
            10.0 * powers[0].log10()
        } else {
            -200.0
        },
        if powers[1] > 1e-20 {
            10.0 * powers[1].log10()
        } else {
            -200.0
        },
        if powers[2] > 1e-20 {
            10.0 * powers[2].log10()
        } else {
            -200.0
        },
    ];

    // Simple linear regression: slope = (Σxy - n*mx*my) / (Σx² - n*mx²)
    let n = 3.0f32;
    let mx = log_freqs.iter().sum::<f32>() / n;
    let my = log_powers.iter().sum::<f32>() / n;

    let mut sxy = 0.0f32;
    let mut sxx = 0.0f32;
    for i in 0..3 {
        let dx = log_freqs[i] - mx;
        let dy = log_powers[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
    }

    if sxx.abs() < 1e-10 {
        return 0.0;
    }

    sxy / sxx // dB per octave (log2 freq)
}

impl VocalTractMetrics {
    /// Compute metrics including perceptual quality analysis from audio.
    ///
    /// Like `compute()` but also analyzes the synthesized audio waveform
    /// for HNR and spectral tilt.
    pub fn compute_with_audio(
        frames: &[FormantFrame],
        targets: &[FormantTarget],
        target_f0: f32,
        samples: &[f32],
        sample_rate: u32,
    ) -> (Self, PerceptualMetrics) {
        let base = Self::compute(frames, targets, target_f0);

        // Compute perceptual metrics from audio
        let hnr = compute_hnr(samples, target_f0, sample_rate);

        // Use average formant frequencies for spectral tilt
        let n = frames.len().max(1) as f32;
        let avg_f1 = frames.iter().map(|f| f.f1).sum::<f32>() / n;
        let avg_f2 = frames.iter().map(|f| f.f2).sum::<f32>() / n;
        let avg_f3 = frames.iter().map(|f| f.f3).sum::<f32>() / n;
        let tilt = compute_spectral_tilt(samples, avg_f1, avg_f2, avg_f3, sample_rate);

        let perceptual = PerceptualMetrics {
            hnr,
            spectral_tilt: tilt,
            spectral_tilt_deviation: (tilt - (-12.0)).abs(),
            voiced_frames: frames.iter().filter(|f| f.voicing > 0.5).count(),
            total_frames: frames.len(),
        };

        (base, perceptual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_computation() {
        let target = FormantTarget {
            f1: 730.0,
            f2: 1090.0,
            f3: 2440.0,
            b1: 80.0,
            b2: 100.0,
            b3: 120.0,
            is_vowel: true,
            is_voiced: true,
            duration_ms: 80.0,
            manner: crate::types::SourceType::Vowel,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
            f1_offset: 0.0,
            f2_offset: 0.0,
            f3_offset: 0.0,
        };

        // Frames close to target
        let frames: Vec<FormantFrame> = (0..10)
            .map(|i| FormantFrame {
                f1: 730.0 + i as f32 * 2.0,
                f2: 1090.0 - i as f32 * 3.0,
                f3: 2440.0 + i as f32 * 1.0,
                b1: 80.0,
                b2: 100.0,
                b3: 120.0,
                f0: 120.0 + i as f32 * 0.5,
                energy: 0.7,
                voicing: 0.95,
                time: i as f32 * 0.005,
                ..Default::default()
            })
            .collect();

        let targets = vec![target; 10];
        let metrics = VocalTractMetrics::compute(&frames, &targets, 120.0);

        assert_eq!(metrics.num_frames, 10);
        assert!(
            metrics.mean_f1_error < 20.0,
            "F1 error should be small: {}",
            metrics.mean_f1_error
        );
        assert!(metrics.max_f1_delta < 5.0, "F1 delta should be smooth");
        assert!(metrics.f0_rmse < 5.0, "F0 RMSE should be small");
    }

    #[cfg(feature = "hound")]
    #[test]
    fn test_wav_roundtrip() {
        let samples: Vec<f32> = (0..480)
            .map(|i| (i as f32 / 480.0 * std::f32::consts::TAU).sin() * 0.5)
            .collect();

        let path = "/tmp/symthaea_vocal_tract_crate_wav_test.wav";
        save_wav(path, &samples, 24000).expect("WAV write failed");

        let (loaded, sr) = load_wav(path).expect("WAV read failed");
        assert_eq!(sr, 24000);
        assert_eq!(loaded.len(), 480);

        let max_err: f32 = samples
            .iter()
            .zip(loaded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.001, "WAV roundtrip error too large: {max_err}");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_mcd_perfect_match() {
        let target = FormantTarget {
            f1: 520.0,
            f2: 1190.0,
            f3: 2390.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            is_vowel: true,
            is_voiced: true,
            duration_ms: 80.0,
            manner: crate::types::SourceType::Vowel,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
            f1_offset: 0.0,
            f2_offset: 0.0,
            f3_offset: 0.0,
        };
        let frame = FormantFrame {
            f1: 520.0,
            f2: 1190.0,
            f3: 2390.0,
            ..Default::default()
        };

        let mcd = super::mel_cepstral_distortion(&[frame; 10], &[target; 10]);
        assert!(
            mcd < 0.01,
            "Perfect match should have near-zero MCD: {mcd:.4}"
        );
    }

    #[test]
    fn test_mcd_error_scales() {
        let target = FormantTarget {
            f1: 520.0,
            f2: 1190.0,
            f3: 2390.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            is_vowel: true,
            is_voiced: true,
            duration_ms: 80.0,
            manner: crate::types::SourceType::Vowel,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
            f1_offset: 0.0,
            f2_offset: 0.0,
            f3_offset: 0.0,
        };

        // Small error
        let small_err = FormantFrame {
            f1: 530.0,
            f2: 1200.0,
            f3: 2400.0,
            ..Default::default()
        };
        let mcd_small = super::mel_cepstral_distortion(&[small_err; 10], &[target; 10]);

        // Large error (schwa bias)
        let large_err = FormantFrame {
            f1: 500.0,
            f2: 1500.0,
            f3: 2500.0,
            ..Default::default()
        };
        let mcd_large = super::mel_cepstral_distortion(&[large_err; 10], &[target; 10]);

        assert!(
            mcd_large > mcd_small,
            "Larger formant error should produce higher MCD: small={mcd_small:.2}, large={mcd_large:.2}"
        );
    }

    #[test]
    fn test_regression_guard_mean_f1() {
        let target = FormantTarget {
            f1: 730.0,
            f2: 1090.0,
            f3: 2440.0,
            b1: 80.0,
            b2: 100.0,
            b3: 120.0,
            is_vowel: true,
            is_voiced: true,
            duration_ms: 80.0,
            manner: crate::types::SourceType::Vowel,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
            f1_offset: 0.0,
            f2_offset: 0.0,
            f3_offset: 0.0,
        };

        let frames: Vec<FormantFrame> = (0..10)
            .map(|i| FormantFrame {
                f1: 500.0,
                f2: 1500.0,
                f3: 2500.0,
                b1: 60.0,
                b2: 90.0,
                b3: 150.0,
                f0: 120.0,
                energy: 0.5,
                voicing: 0.8,
                time: i as f32 * 0.005,
                ..Default::default()
            })
            .collect();

        let targets = vec![target; 10];
        let metrics = VocalTractMetrics::compute(&frames, &targets, 120.0);

        assert!(
            metrics.mean_f1_error < 300.0,
            "Regression guard: mean F1 error {} Hz exceeds 300 Hz limit",
            metrics.mean_f1_error
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Item 1: Perceptual metrics tests (HNR + spectral tilt)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_hnr_clean_vowel() {
        // Synthesize a clean periodic vowel (no noise) → high HNR
        let sample_rate = 24000u32;
        let f0 = 120.0;
        let duration_samples = sample_rate; // 1 second
        let samples: Vec<f32> = (0..duration_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                // Pure harmonic signal: fundamental + harmonics
                let h1 = (2.0 * std::f32::consts::PI * f0 * t).sin();
                let h2 = 0.5 * (2.0 * std::f32::consts::PI * 2.0 * f0 * t).sin();
                let h3 = 0.25 * (2.0 * std::f32::consts::PI * 3.0 * f0 * t).sin();
                (h1 + h2 + h3) * 0.3
            })
            .collect();

        let hnr = compute_hnr(&samples, f0, sample_rate);
        assert!(
            hnr > 15.0,
            "Clean harmonic signal should have HNR > 15 dB: {hnr:.1}"
        );
    }

    #[test]
    fn test_hnr_noisy_lower() {
        // Signal with added noise → lower HNR
        let sample_rate = 24000u32;
        let f0 = 120.0;
        let duration_samples = sample_rate;

        // Clean signal
        let clean: Vec<f32> = (0..duration_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * f0 * t).sin() * 0.3
            })
            .collect();
        let hnr_clean = compute_hnr(&clean, f0, sample_rate);

        // Noisy signal: add pseudo-random noise (xorshift)
        let mut rng = 12345u64;
        let noisy: Vec<f32> = clean
            .iter()
            .map(|&s| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let noise = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
                s + noise * 0.3
            })
            .collect();
        let hnr_noisy = compute_hnr(&noisy, f0, sample_rate);

        assert!(
            hnr_noisy < hnr_clean,
            "Noisy signal should have lower HNR: clean={hnr_clean:.1}, noisy={hnr_noisy:.1}"
        );
    }

    #[test]
    fn test_spectral_tilt_modal_vowel() {
        // Generate a signal with known spectral rolloff
        let sample_rate = 24000u32;
        let f0 = 120.0;
        let n = sample_rate * 2; // 2 seconds for stable Goertzel

        // Simulate glottal source with ~-12 dB/octave rolloff
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let h1 = (2.0 * std::f32::consts::PI * f0 * t).sin();
                let h2 = 0.25 * (2.0 * std::f32::consts::PI * 2.0 * f0 * t).sin();
                let h3 = 0.0625 * (2.0 * std::f32::consts::PI * 3.0 * f0 * t).sin();
                (h1 + h2 + h3) * 0.3
            })
            .collect();

        let tilt = compute_spectral_tilt(&samples, 500.0, 1500.0, 2500.0, sample_rate);
        // Should be negative (energy decreases with frequency)
        assert!(tilt < 0.0, "Spectral tilt should be negative: {tilt:.1}");
    }

    #[test]
    fn test_spectral_tilt_breathy_steeper() {
        let sample_rate = 24000u32;
        let n = sample_rate * 2;

        // Use formant frequencies as test frequencies directly so Goertzel
        // measures meaningful power differences between modal and breathy.
        let f1 = 500.0f32;
        let f2 = 1500.0f32;
        let f3 = 2500.0f32;

        // Modal voice: moderate rolloff (-6 dB/octave)
        let modal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let h1 = (2.0 * std::f32::consts::PI * f1 * t).sin();
                let h2 = 0.5 * (2.0 * std::f32::consts::PI * f2 * t).sin();
                let h3 = 0.25 * (2.0 * std::f32::consts::PI * f3 * t).sin();
                (h1 + h2 + h3) * 0.3
            })
            .collect();
        let tilt_modal = compute_spectral_tilt(&modal, f1, f2, f3, sample_rate);

        // Breathy voice: steeper rolloff (much less energy at higher formants)
        let breathy: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let h1 = (2.0 * std::f32::consts::PI * f1 * t).sin();
                let h2 = 0.1 * (2.0 * std::f32::consts::PI * f2 * t).sin();
                let h3 = 0.01 * (2.0 * std::f32::consts::PI * f3 * t).sin();
                (h1 + h2 + h3) * 0.3
            })
            .collect();
        let tilt_breathy = compute_spectral_tilt(&breathy, f1, f2, f3, sample_rate);

        assert!(
            tilt_breathy < tilt_modal,
            "Breathy should have steeper tilt: modal={tilt_modal:.1}, breathy={tilt_breathy:.1}"
        );
    }
}
