// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compare two rendered WAV files without silently deciding whether they sound alike.
//!
//! Usage:
//!   cargo run --manifest-path crates/domains/symthaea-muse/Cargo.toml --bin sfz_audio_differential -- --candidate muse.wav --reference reference.wav
//!   cargo run --manifest-path crates/domains/symthaea-muse/Cargo.toml --bin sfz_audio_differential -- --candidate muse.wav --reference reference.wav --output comparison.json
//!
//! This is a measurement tool, not an automatic compatibility oracle. It emits
//! reproducible input hashes, signal-level metrics, and windowed spectral metrics.
//! It does not resample, time-align, loudness-normalize, or choose quality thresholds.
//! A successful exit means the comparison ran; it does NOT mean the sounds are equivalent.
//!
//! Supported WAV samples: PCM 8/16/24/32-bit and IEEE float32. Input WAVs must be
//! regular files no larger than 512 MiB. Sample rate and channel count must match;
//! incompatible inputs are reported as non-comparable rather than silently altered.

use hound::{SampleFormat, WavReader, WavSpec};
use rustfft::{num_complex::Complex32, FftPlanner};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
    process::ExitCode,
};

const MAX_INPUT_BYTES: u64 = 512 * 1024 * 1024;
const FFT_SIZE: usize = 4096;
const FFT_HOP: usize = 1024;
const METRIC_EPSILON: f64 = 1.0e-12;

#[derive(Debug)]
struct Arguments {
    candidate: PathBuf,
    reference: PathBuf,
    output: Option<PathBuf>,
}

#[derive(Debug)]
struct LoadedAudio {
    evidence: AudioEvidenceV1,
    samples: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct AudioEvidenceV1 {
    file_name: String,
    sha256: String,
    byte_len: u64,
    sample_rate_hz: u32,
    channels: u16,
    bits_per_sample: u16,
    sample_format: String,
    frames: usize,
    duration_seconds: f64,
    finite_samples: bool,
    peak_absolute: Option<f64>,
    rms: Option<f64>,
    dc_offset: Option<f64>,
}

#[derive(Debug, Serialize)]
struct DifferentialReceiptV1 {
    schema: &'static str,
    comparator_version: &'static str,
    candidate: AudioEvidenceV1,
    reference: AudioEvidenceV1,
    comparison: ComparisonResultV1,
    scope_boundary: &'static str,
}

#[derive(Debug, Serialize)]
struct ComparisonResultV1 {
    status: &'static str,
    sample_rate_matches: bool,
    channel_count_matches: bool,
    candidate_minus_reference_frames: i64,
    compared_frames: Option<usize>,
    compared_duration_seconds: Option<f64>,
    metrics: Option<AudioComparisonMetricsV1>,
    interpretation: &'static str,
}

#[derive(Debug, Serialize)]
struct AudioComparisonMetricsV1 {
    /// Error between samples at the same frame index; no time alignment is applied.
    waveform_rmse: f64,
    /// waveform_rmse divided by reference RMS over the compared prefix.
    waveform_nrmse_reference: f64,
    waveform_max_absolute_error: f64,
    waveform_correlation: Option<f64>,
    rms_delta_db_candidate_minus_reference: f64,
    /// RMS dB difference between average Hann-windowed STFT magnitudes (not mean per-frame LSD).
    log_spectral_distance_db: f64,
    /// L1 distance between unit-sum power spectra; 0 is identical, larger is more different.
    normalized_power_spectrum_l1: f64,
    /// Cosine similarity between average STFT magnitude spectra.
    /// Spectral centroid uses magnitude-weighted frequency bins.
    spectral_magnitude_cosine_similarity: f64,
    spectral_centroid_delta_hz_candidate_minus_reference: f64,
    fft_size: usize,
    fft_hop: usize,
    window: &'static str,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("sfz_audio_differential: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode, Box<dyn Error>> {
    let Some(arguments) = parse_arguments()? else {
        print_usage();
        return Ok(ExitCode::SUCCESS);
    };

    let candidate = load_audio(&arguments.candidate)?;
    let reference = load_audio(&arguments.reference)?;
    let receipt = build_receipt(candidate, reference);
    let json = serde_json::to_string_pretty(&receipt)?;

    if let Some(path) = arguments.output {
        fs::write(path, &json)?;
    }
    println!("{json}");

    // Exit 1 means no defensible comparison could be made, not that audio quality
    // failed. Exit 0 only means metrics were produced; no quality gate is configured.
    Ok(if receipt.comparison.metrics.is_some() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    })
}

fn parse_arguments() -> Result<Option<Arguments>, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let mut candidate = None;
    let mut reference = None;
    let mut output = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--help" | "-h" => return Ok(None),
            "--candidate" => {
                candidate = Some(PathBuf::from(args.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--candidate requires a path")
                })?));
            }
            "--reference" => {
                reference = Some(PathBuf::from(args.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--reference requires a path")
                })?));
            }
            "--output" => {
                output = Some(PathBuf::from(args.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--output requires a path")
                })?));
            }
            unknown => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown argument {unknown:?}; use --help"),
                )
                .into());
            }
        }
    }

    let candidate = candidate.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "missing required --candidate PATH")
    })?;
    let reference = reference.ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "missing required --reference PATH")
    })?;
    Ok(Some(Arguments {
        candidate,
        reference,
        output,
    }))
}

fn print_usage() {
    println!(
        "Usage: sfz_audio_differential --candidate MUSE.wav --reference ENGINE.wav [--output report.json]\n\
         Measures two same-rate, same-channel WAV files. No quality threshold is implied."
    );
}

fn load_audio(path: &Path) -> Result<LoadedAudio, Box<dyn Error>> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("input must be a regular non-symlink file: {}", path.display()),
        )
        .into());
    }
    if metadata.len() > MAX_INPUT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "input {} exceeds the {} MiB size limit",
                path.display(),
                MAX_INPUT_BYTES / (1024 * 1024)
            ),
        )
        .into());
    }

    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate == 0 || spec.channels == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid sample rate or channel count in {}", path.display()),
        )
        .into());
    }

    let samples = decode_samples(&mut reader, spec)?;
    let channels = usize::from(spec.channels);
    if samples.len() % channels != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("decoded sample count is not channel-aligned in {}", path.display()),
        )
        .into());
    }
    let frames = samples.len() / channels;
    let finite_samples = samples.iter().all(|sample| sample.is_finite());
    let (peak_absolute, rms, dc_offset) = signal_summary(&samples, finite_samples);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<non-utf8-filename>")
        .to_owned();

    Ok(LoadedAudio {
        evidence: AudioEvidenceV1 {
            file_name,
            sha256: sha256_file(path)?,
            byte_len: metadata.len(),
            sample_rate_hz: spec.sample_rate,
            channels: spec.channels,
            bits_per_sample: spec.bits_per_sample,
            sample_format: format_name(spec),
            frames,
            duration_seconds: frames as f64 / f64::from(spec.sample_rate),
            finite_samples,
            peak_absolute,
            rms,
            dc_offset,
        },
        samples,
    })
}

fn decode_samples(
    reader: &mut WavReader<std::io::BufReader<File>>,
    spec: WavSpec,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let decoded = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 8) => reader
            .samples::<i8>()
            .map(|sample| sample.map(|value| f32::from(value) / 128.0))
            .collect::<Result<Vec<_>, _>>()?,
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| f32::from(value) / 32_768.0))
            .collect::<Result<Vec<_>, _>>()?,
        (SampleFormat::Int, bits @ (24 | 32)) => {
            let scale = 2.0_f64.powi(i32::from(bits) - 1);
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| (f64::from(value) / scale) as f32))
                .collect::<Result<Vec<_>, _>>()?
        }
        (SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        (format, bits) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported WAV sample format {format:?} with {bits} bits"),
            )
            .into());
        }
    };
    Ok(decoded)
}

fn format_name(spec: WavSpec) -> String {
    match spec.sample_format {
        SampleFormat::Int => format!("pcm_s{}le", spec.bits_per_sample),
        SampleFormat::Float => format!("ieee_float_{}bit", spec.bits_per_sample),
    }
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn signal_summary(samples: &[f32], finite: bool) -> (Option<f64>, Option<f64>, Option<f64>) {
    if samples.is_empty() || !finite {
        return (None, None, None);
    }
    let mut sum = 0.0_f64;
    let mut squares = 0.0_f64;
    let mut peak = 0.0_f64;
    for &sample in samples {
        let value = f64::from(sample);
        sum += value;
        squares += value * value;
        peak = peak.max(value.abs());
    }
    let count = samples.len() as f64;
    (Some(peak), Some((squares / count).sqrt()), Some(sum / count))
}

fn build_receipt(candidate: LoadedAudio, reference: LoadedAudio) -> DifferentialReceiptV1 {
    let sample_rate_matches =
        candidate.evidence.sample_rate_hz == reference.evidence.sample_rate_hz;
    let channel_count_matches = candidate.evidence.channels == reference.evidence.channels;
    let candidate_frames = candidate.evidence.frames;
    let reference_frames = reference.evidence.frames;
    let candidate_minus_reference_frames = candidate_frames as i64 - reference_frames as i64;

    let (status, compared_frames, compared_duration_seconds, metrics) =
        if !sample_rate_matches {
            ("not_comparable_sample_rate_mismatch", None, None, None)
        } else if !channel_count_matches {
            ("not_comparable_channel_count_mismatch", None, None, None)
        } else if !candidate.evidence.finite_samples || !reference.evidence.finite_samples {
            ("invalid_non_finite_samples", None, None, None)
        } else if candidate_frames == 0 || reference_frames == 0 {
            ("not_comparable_empty_audio", None, None, None)
        } else {
            let overlap_frames = candidate_frames.min(reference_frames);
            let metrics = comparison_metrics(
                &candidate.samples,
                &reference.samples,
                usize::from(candidate.evidence.channels),
                overlap_frames,
                candidate.evidence.rms.unwrap_or(0.0),
                reference.evidence.rms.unwrap_or(0.0),
                candidate.evidence.sample_rate_hz,
            );
            (
                "measured_not_quality_qualified",
                Some(overlap_frames),
                Some(overlap_frames as f64 / f64::from(candidate.evidence.sample_rate_hz)),
                Some(metrics),
            )
        };

    DifferentialReceiptV1 {
        schema: "muse-sfz-audio-differential-receipt-v1",
        comparator_version: "sfz-audio-differential-v1",
        candidate: candidate.evidence,
        reference: reference.evidence,
        comparison: ComparisonResultV1 {
            status,
            sample_rate_matches,
            channel_count_matches,
            candidate_minus_reference_frames,
            compared_frames,
            compared_duration_seconds,
            metrics,
            interpretation: "metrics are descriptive only; no alignment, resampling, normalization, thresholds, engine compatibility, or perceptual equivalence is established",
        },
        scope_boundary: "same-sample-rate/same-channel WAV comparison; use engine-specific fixture receipts and separately defined thresholds before any compatibility or quality claim",
    }
}

fn comparison_metrics(
    candidate: &[f32],
    reference: &[f32],
    channels: usize,
    frames: usize,
    candidate_rms_all: f64,
    reference_rms_all: f64,
    sample_rate: u32,
) -> AudioComparisonMetricsV1 {
    let sample_count = frames * channels;
    let candidate = &candidate[..sample_count];
    let reference = &reference[..sample_count];

    let mut squared_error = 0.0_f64;
    let mut maximum_error = 0.0_f64;
    let mut sum_candidate = 0.0_f64;
    let mut sum_reference = 0.0_f64;
    let mut sum_candidate_squared = 0.0_f64;
    let mut sum_reference_squared = 0.0_f64;
    let mut sum_product = 0.0_f64;

    for (&candidate_sample, &reference_sample) in candidate.iter().zip(reference.iter()) {
        let c = f64::from(candidate_sample);
        let r = f64::from(reference_sample);
        let error = c - r;
        squared_error += error * error;
        maximum_error = maximum_error.max(error.abs());
        sum_candidate += c;
        sum_reference += r;
        sum_candidate_squared += c * c;
        sum_reference_squared += r * r;
        sum_product += c * r;
    }

    let count = sample_count as f64;
    let waveform_rmse = (squared_error / count).sqrt();
    let reference_overlap_rms = (sum_reference_squared / count).sqrt();
    let covariance =
        sum_product / count - (sum_candidate / count) * (sum_reference / count);
    let candidate_variance =
        (sum_candidate_squared / count - (sum_candidate / count).powi(2)).max(0.0);
    let reference_variance =
        (sum_reference_squared / count - (sum_reference / count).powi(2)).max(0.0);
    let waveform_correlation = if candidate_variance > METRIC_EPSILON
        && reference_variance > METRIC_EPSILON
    {
        Some((covariance / (candidate_variance * reference_variance).sqrt()).clamp(-1.0, 1.0))
    } else {
        None
    };

    let candidate_spectrum = average_magnitude_spectrum(candidate, channels, frames);
    let reference_spectrum = average_magnitude_spectrum(reference, channels, frames);
    let (log_spectral_distance_db, normalized_power_spectrum_l1, spectral_magnitude_cosine_similarity) =
        spectrum_distances(&candidate_spectrum, &reference_spectrum);
    let candidate_centroid = spectral_centroid_hz(&candidate_spectrum, sample_rate);
    let reference_centroid = spectral_centroid_hz(&reference_spectrum, sample_rate);

    AudioComparisonMetricsV1 {
        waveform_rmse,
        waveform_nrmse_reference: waveform_rmse / reference_overlap_rms.max(METRIC_EPSILON),
        waveform_max_absolute_error: maximum_error,
        waveform_correlation,
        // Whole-file RMS is used consistently for a declared duration difference.
        rms_delta_db_candidate_minus_reference: 20.0
            * ((candidate_rms_all + METRIC_EPSILON) / (reference_rms_all + METRIC_EPSILON)).log10(),
        log_spectral_distance_db,
        normalized_power_spectrum_l1,
        spectral_magnitude_cosine_similarity,
        spectral_centroid_delta_hz_candidate_minus_reference: candidate_centroid - reference_centroid,
        fft_size: FFT_SIZE,
        fft_hop: FFT_HOP,
        window: "hann",
    }
}

fn average_magnitude_spectrum(samples: &[f32], channels: usize, frames: usize) -> Vec<f64> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut aggregate = vec![0.0_f64; FFT_SIZE / 2 + 1];
    let mut transform = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
    let mut window = vec![0.0_f32; FFT_SIZE];
    for (index, coefficient) in window.iter_mut().enumerate() {
        *coefficient = if FFT_SIZE > 1 {
            0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / (FFT_SIZE - 1) as f32).cos()
        } else {
            1.0
        };
    }

    let mut transform_count = 0_u64;
    for channel in 0..channels {
        let mut start = 0_usize;
        while start < frames {
            for index in 0..FFT_SIZE {
                let frame = start + index;
                let value = if frame < frames {
                    samples[frame * channels + channel]
                } else {
                    0.0
                };
                transform[index] = Complex32::new(value * window[index], 0.0);
            }
            fft.process(&mut transform);
            for bin in 0..aggregate.len() {
                aggregate[bin] += f64::from(transform[bin].norm());
            }
            transform_count += 1;
            start = start.saturating_add(FFT_HOP);
        }
    }
    let divisor = transform_count.max(1) as f64;
    for magnitude in &mut aggregate {
        *magnitude /= divisor;
    }
    aggregate
}

fn spectrum_distances(candidate: &[f64], reference: &[f64]) -> (f64, f64, f64) {
    let length = candidate.len().min(reference.len());
    let candidate_max = candidate.iter().copied().fold(0.0_f64, f64::max);
    let reference_max = reference.iter().copied().fold(0.0_f64, f64::max);
    let floor = candidate_max.max(reference_max).max(METRIC_EPSILON) * 1.0e-7;

    let mut log_squared = 0.0_f64;
    let mut candidate_power_sum = 0.0_f64;
    let mut reference_power_sum = 0.0_f64;
    let mut magnitude_dot = 0.0_f64;
    let mut candidate_magnitude_squared = 0.0_f64;
    let mut reference_magnitude_squared = 0.0_f64;

    for index in 0..length {
        let c = candidate[index].max(floor);
        let r = reference[index].max(floor);
        let log_difference = 20.0 * (c / r).log10();
        log_squared += log_difference * log_difference;

        let cp = candidate[index] * candidate[index];
        let rp = reference[index] * reference[index];
        candidate_power_sum += cp;
        reference_power_sum += rp;
        magnitude_dot += candidate[index] * reference[index];
        candidate_magnitude_squared += candidate[index] * candidate[index];
        reference_magnitude_squared += reference[index] * reference[index];
    }

    let bins = length.max(1) as f64;
    let log_spectral_distance_db = (log_squared / bins).sqrt();
    let candidate_power_sum = candidate_power_sum.max(METRIC_EPSILON);
    let reference_power_sum = reference_power_sum.max(METRIC_EPSILON);
    let normalized_power_spectrum_l1 = (0..length)
        .map(|index| {
            let candidate_power = candidate[index] * candidate[index] / candidate_power_sum;
            let reference_power = reference[index] * reference[index] / reference_power_sum;
            (candidate_power - reference_power).abs()
        })
        .sum::<f64>();
    let denominator = (candidate_magnitude_squared * reference_magnitude_squared).sqrt();
    let cosine = if candidate_magnitude_squared <= METRIC_EPSILON
        && reference_magnitude_squared <= METRIC_EPSILON
    {
        1.0
    } else if denominator > METRIC_EPSILON {
        (magnitude_dot / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (log_spectral_distance_db, normalized_power_spectrum_l1, cosine)
}

fn spectral_centroid_hz(spectrum: &[f64], sample_rate: u32) -> f64 {
    let mut weighted_sum = 0.0_f64;
    let mut total = 0.0_f64;
    for (bin, magnitude) in spectrum.iter().copied().enumerate() {
        let frequency = bin as f64 * f64::from(sample_rate) / FFT_SIZE as f64;
        weighted_sum += frequency * magnitude;
        total += magnitude;
    }
    if total <= METRIC_EPSILON {
        0.0
    } else {
        weighted_sum / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frequency_hz: f64, sample_rate: u32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|frame| {
                (0.5 * (std::f64::consts::TAU * frequency_hz * frame as f64
                    / f64::from(sample_rate))
                .sin()) as f32
            })
            .collect()
    }

    fn audio_fixture(samples: Vec<f32>, sample_rate_hz: u32, channels: u16, name: &str) -> LoadedAudio {
        let frames = samples.len() / usize::from(channels);
        let finite_samples = samples.iter().all(|sample| sample.is_finite());
        let (peak_absolute, rms, dc_offset) = signal_summary(&samples, finite_samples);
        LoadedAudio {
            evidence: AudioEvidenceV1 {
                file_name: name.to_owned(),
                sha256: format!("hash-{name}"),
                byte_len: (samples.len() * std::mem::size_of::<f32>()) as u64,
                sample_rate_hz,
                channels,
                bits_per_sample: 32,
                sample_format: "ieee_float_32bit".to_owned(),
                frames,
                duration_seconds: frames as f64 / f64::from(sample_rate_hz),
                finite_samples,
                peak_absolute,
                rms,
                dc_offset,
            },
            samples,
        }
    }

    #[test]
    fn identical_audio_has_zero_waveform_error_and_zero_spectral_distance() {
        let signal = sine(440.0, 44_100, 12_000);
        let metrics = comparison_metrics(
            &signal,
            &signal,
            1,
            signal.len(),
            0.5 / 2.0_f64.sqrt(),
            0.5 / 2.0_f64.sqrt(),
            44_100,
        );
        assert!(metrics.waveform_rmse < 1.0e-7);
        assert!(metrics.waveform_nrmse_reference < 1.0e-6);
        assert!(metrics.log_spectral_distance_db < 1.0e-6);
        assert!(metrics.normalized_power_spectrum_l1 < 1.0e-9);
        assert!((metrics.spectral_magnitude_cosine_similarity - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn spectral_metrics_distinguish_an_octave_change() {
        let candidate = sine(880.0, 44_100, 12_000);
        let reference = sine(440.0, 44_100, 12_000);
        let metrics = comparison_metrics(
            &candidate,
            &reference,
            1,
            candidate.len(),
            0.5 / 2.0_f64.sqrt(),
            0.5 / 2.0_f64.sqrt(),
            44_100,
        );
        assert!(metrics.normalized_power_spectrum_l1 > 0.2);
        assert!(metrics.spectral_magnitude_cosine_similarity < 0.95);
        assert!(metrics.waveform_rmse > 0.1);
    }

    #[test]
    fn zero_variance_waveforms_do_not_create_nan_correlation() {
        let silence = vec![0.0_f32; FFT_SIZE];
        let metrics = comparison_metrics(&silence, &silence, 1, silence.len(), 0.0, 0.0, 44_100);
        assert_eq!(metrics.waveform_correlation, None);
        assert!(metrics.waveform_rmse.is_finite());
        assert!(metrics.log_spectral_distance_db.is_finite());
    }

    #[test]
    fn spectral_similarity_detects_silence_against_non_silent_audio() {
        let candidate = sine(440.0, 44_100, 12_000);
        let reference = vec![0.0_f32; candidate.len()];
        let metrics = comparison_metrics(
            &candidate,
            &reference,
            1,
            candidate.len(),
            0.5 / 2.0_f64.sqrt(),
            0.0,
            44_100,
        );
        assert!(metrics.spectral_magnitude_cosine_similarity < 1.0e-9);
        assert!(metrics.normalized_power_spectrum_l1 > 0.9);
        assert!(metrics.log_spectral_distance_db > 1.0);
    }

    #[test]
    fn sample_rate_mismatch_is_reported_as_not_comparable() {
        let candidate = audio_fixture(vec![0.1, -0.1], 44_100, 1, "candidate.wav");
        let reference = audio_fixture(vec![0.1, -0.1], 48_000, 1, "reference.wav");
        let receipt = build_receipt(candidate, reference);
        assert_eq!(receipt.comparison.status, "not_comparable_sample_rate_mismatch");
        assert_eq!(receipt.comparison.metrics.is_none(), true);
    }

    #[test]
    fn non_finite_samples_are_reported_without_scoring() {
        let candidate = audio_fixture(vec![f32::NAN, 0.1], 44_100, 1, "candidate.wav");
        let reference = audio_fixture(vec![0.0, 0.1], 44_100, 1, "reference.wav");
        let receipt = build_receipt(candidate, reference);
        assert_eq!(receipt.comparison.status, "invalid_non_finite_samples");
        assert_eq!(receipt.comparison.metrics.is_none(), true);
    }

    #[test]
    fn duration_mismatch_records_common_prefix_and_frame_delta() {
        let candidate = audio_fixture(vec![0.1, 0.2, 0.3], 44_100, 1, "candidate.wav");
        let reference = audio_fixture(vec![0.1, 0.2], 44_100, 1, "reference.wav");
        let receipt = build_receipt(candidate, reference);
        assert_eq!(receipt.comparison.status, "measured_not_quality_qualified");
        assert_eq!(receipt.comparison.candidate_minus_reference_frames, 1);
        assert_eq!(receipt.comparison.compared_frames, Some(2));
    }

    #[test]
    fn file_hash_matches_known_sha256_vector() {
        assert_eq!(
            format!("{:x}", Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
