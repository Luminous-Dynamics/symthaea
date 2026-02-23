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

        Self {
            mean_f1_error: f1_err_sum / nf,
            mean_f2_error: f2_err_sum / nf,
            mean_f3_error: f3_err_sum / nf,
            max_f1_delta,
            f0_rmse: (f0_sq_sum / nf).sqrt(),
            energy_correlation,
            num_frames: n,
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
    if denom < 1e-10 {
        0.0
    } else {
        cov / denom
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
            })
            .collect();

        let targets = vec![target; 10];
        let metrics = VocalTractMetrics::compute(&frames, &targets, 120.0);

        assert_eq!(metrics.num_frames, 10);
        assert!(metrics.mean_f1_error < 20.0, "F1 error should be small: {}", metrics.mean_f1_error);
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
}
