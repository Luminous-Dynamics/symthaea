// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible synthetic-signal qualification for the SFZ one-shot resamplers.
//!
//! Run with:
//!   cargo run --manifest-path crates/domains/symthaea-muse/Cargo.toml --bin sfz_resampler_qualification
//!   cargo run --manifest-path crates/domains/symthaea-muse/Cargo.toml --bin sfz_resampler_qualification -- --output /tmp/sfz-resampler.json
//!
//! The program emits a versioned JSON receipt to stdout (and optionally a file).
//! It is deliberately a synthetic qualification gate, not a real-bank or
//! perceptual-quality claim. A non-zero exit status means at least one declared
//! numerical/performance gate failed.

use serde::Serialize;
use std::{
    env,
    error::Error,
    fs,
    io,
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use symthaea_muse::sfz_manifest::{
    parse_sfz_mapping,
    playback::{
        render_sfz_note_once_with_resampler, SfzPlaybackReceiptV1, SfzResamplerV1,
    },
};

const SAMPLE_RATE: u32 = 44_100;
const SOURCE_FRAMES: usize = SAMPLE_RATE as usize;
const REQUESTED_KEY: u8 = 72;
const PITCH_KEYCENTER: u8 = 60;
const VELOCITY: u8 = 100;
const SINC_RADIUS: usize = 16;
const REPEATS: usize = 3;
const SOURCE_AMPLITUDE: f64 = 16_000.0 / 32_768.0;

#[derive(Debug, Serialize)]
struct ThresholdsV1 {
    minimum_alias_suppression_db: f64,
    maximum_in_band_amplitude_absolute_error: f64,
    maximum_relative_pitch_ratio_error: f64,
    minimum_offline_realtime_factor: f64,
    minimum_non_silent_rms: f64,
    maximum_absolute_peak: f64,
}

impl Default for ThresholdsV1 {
    fn default() -> Self {
        Self {
            minimum_alias_suppression_db: 26.0,
            maximum_in_band_amplitude_absolute_error: 0.04,
            maximum_relative_pitch_ratio_error: 1.0e-6,
            minimum_offline_realtime_factor: 10.0,
            minimum_non_silent_rms: 1.0e-6,
            maximum_absolute_peak: 1.0,
        }
    }
}

#[derive(Debug, Serialize)]
struct EnvironmentV1 {
    operating_system: &'static str,
    architecture: &'static str,
    cpu_model: String,
    sample_rate_hz: u32,
}

#[derive(Debug, Serialize)]
struct PathMeasurementV1 {
    renderer_receipt: SfzPlaybackReceiptV1,
    measured_tone_amplitude: f64,
    finite_samples: bool,
    non_silent: bool,
    peak_within_unity: bool,
    repeated_output_hash_stable: bool,
    median_render_ms: f64,
    offline_realtime_factor: f64,
}

#[derive(Debug, Serialize)]
struct CaseReceiptV1 {
    case_id: String,
    source_tone_cycles_per_frame: f64,
    expected_output_alias_cycles_per_frame: f64,
    expected_source_amplitude: f64,
    linear: PathMeasurementV1,
    lanczos_sinc: PathMeasurementV1,
    alias_suppression_db_vs_linear: f64,
    in_band_amplitude_absolute_error: Option<f64>,
    relative_pitch_ratio_error: f64,
    failed_gates: Vec<String>,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct QualificationReceiptV1 {
    schema: &'static str,
    harness_version: &'static str,
    environment: EnvironmentV1,
    input_fixture_frames: usize,
    requested_key: u8,
    pitch_keycenter: u8,
    requested_velocity: u8,
    sinc_radius: usize,
    repeats_per_path: usize,
    thresholds: ThresholdsV1,
    cases: Vec<CaseReceiptV1>,
    passed: bool,
    scope_boundary: &'static str,
}

fn main() -> ExitCode {
    match run() {
        Ok(passed) => {
            if passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(error) => {
            eprintln!("sfz_resampler_qualification: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool, Box<dyn Error>> {
    let output_path = parse_output_argument()?;
    let thresholds = ThresholdsV1::default();
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let fixture_root = env::temp_dir().join(format!(
        "symthaea-sfz-resampler-qualification-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&fixture_root)?;

    let result = (|| -> Result<QualificationReceiptV1, Box<dyn Error>> {
        let high_name = "above_nyquist.wav";
        let in_band_name = "in_band.wav";
        write_tone_fixture(&fixture_root.join(high_name), 0.40)?;
        write_tone_fixture(&fixture_root.join(in_band_name), 0.10)?;

        let high = qualify_case(
            &fixture_root,
            "above_new_nyquist_alias_rejection",
            high_name,
            0.40,
            None,
            &thresholds,
        )?;
        let in_band = qualify_case(
            &fixture_root,
            "in_band_amplitude_preservation",
            in_band_name,
            0.10,
            Some(SOURCE_AMPLITUDE),
            &thresholds,
        )?;
        let passed = high.passed && in_band.passed;
        Ok(QualificationReceiptV1 {
            schema: "muse-sfz-resampler-qualification-receipt-v1",
            harness_version: env!("CARGO_PKG_VERSION"),
            environment: EnvironmentV1 {
                operating_system: env::consts::OS,
                architecture: env::consts::ARCH,
                cpu_model: cpu_model_name(),
                sample_rate_hz: SAMPLE_RATE,
            },
            input_fixture_frames: SOURCE_FRAMES,
            requested_key: REQUESTED_KEY,
            pitch_keycenter: PITCH_KEYCENTER,
            requested_velocity: VELOCITY,
            sinc_radius: SINC_RADIUS,
            repeats_per_path: REPEATS,
            thresholds,
            cases: vec![high, in_band],
            passed,
            scope_boundary: "synthetic WAV fixtures only; not a real-bank, full-SFZ-compatibility, real-time-callback, or perceptual-quality qualification",
        })
    })();

    let _ = fs::remove_dir_all(&fixture_root);
    let receipt = result?;
    let json = serde_json::to_string_pretty(&receipt)?;
    if let Some(path) = output_path {
        fs::write(path, &json)?;
    }
    println!("{json}");
    Ok(receipt.passed)
}

fn parse_output_argument() -> Result<Option<PathBuf>, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    match args.next() {
        None => Ok(None),
        Some(flag) if flag == "--output" => {
            let path = args.next().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "--output requires a file path")
            })?;
            if args.next().is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "usage: sfz_resampler_qualification [--output PATH]",
                )
                .into());
            }
            Ok(Some(PathBuf::from(path)))
        }
        Some(_) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: sfz_resampler_qualification [--output PATH]",
        )
        .into()),
    }
}

fn write_tone_fixture(path: &Path, cycles_per_frame: f64) -> Result<(), Box<dyn Error>> {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for index in 0..SOURCE_FRAMES {
        let phase = std::f64::consts::TAU * cycles_per_frame * index as f64;
        let sample = (SOURCE_AMPLITUDE * phase.sin() * 32_768.0).round() as i16;
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

fn qualify_case(
    fixture_root: &Path,
    case_id: &str,
    sample_name: &str,
    source_tone_cycles_per_frame: f64,
    expected_source_amplitude_for_passband_case: Option<f64>,
    thresholds: &ThresholdsV1,
) -> Result<CaseReceiptV1, Box<dyn Error>> {
    let sfz_source = format!(
        "<region> key={REQUESTED_KEY} pitch_keycenter={PITCH_KEYCENTER} sample={sample_name}\n"
    );
    let manifest = parse_sfz_mapping(&sfz_source)?;
    let output_frequency = (source_tone_cycles_per_frame * 2.0).rem_euclid(1.0);
    let output_alias_frequency = if output_frequency > 0.5 {
        1.0 - output_frequency
    } else {
        output_frequency
    };

    let linear = measure_path(
        &manifest,
        fixture_root,
        SfzResamplerV1::LinearBaseline,
        output_alias_frequency,
    )?;
    let sinc = measure_path(
        &manifest,
        fixture_root,
        SfzResamplerV1::LanczosSinc { radius: SINC_RADIUS },
        output_alias_frequency,
    )?;

    let alias_suppression_db = 20.0
        * (linear.measured_tone_amplitude.max(1.0e-12)
            / sinc.measured_tone_amplitude.max(1.0e-12))
        .log10();
    let in_band_error = expected_source_amplitude_for_passband_case
        .map(|expected| (sinc.measured_tone_amplitude - expected).abs());
    let ratio_linear = linear.renderer_receipt.frequency_ratio;
    let ratio_sinc = sinc.renderer_receipt.frequency_ratio;
    let relative_pitch_ratio_error = ((ratio_linear - 2.0).abs() / 2.0)
        .max((ratio_sinc - 2.0).abs() / 2.0);

    let mut failed_gates = Vec::new();
    if !linear.finite_samples || !sinc.finite_samples {
        failed_gates.push("all output samples must be finite".to_owned());
    }
    if !linear.non_silent || !sinc.non_silent {
        failed_gates.push("non-silent fixtures must not render as silence".to_owned());
    }
    if !linear.peak_within_unity || !sinc.peak_within_unity {
        failed_gates.push("output peak must remain at or below 1.0".to_owned());
    }
    if !linear.repeated_output_hash_stable || !sinc.repeated_output_hash_stable {
        failed_gates.push("identical same-host renders must have identical output hashes".to_owned());
    }
    if relative_pitch_ratio_error > thresholds.maximum_relative_pitch_ratio_error {
        failed_gates.push(format!(
            "relative pitch-ratio error exceeds {}",
            thresholds.maximum_relative_pitch_ratio_error
        ));
    }
    if linear.offline_realtime_factor < thresholds.minimum_offline_realtime_factor
        || sinc.offline_realtime_factor < thresholds.minimum_offline_realtime_factor
    {
        failed_gates.push(format!(
            "offline throughput is below {}x real time for at least one path",
            thresholds.minimum_offline_realtime_factor
        ));
    }
    if case_id == "above_new_nyquist_alias_rejection"
        && alias_suppression_db < thresholds.minimum_alias_suppression_db
    {
        failed_gates.push(format!(
            "alias suppression {alias_suppression_db:.2} dB is below the {} dB gate",
            thresholds.minimum_alias_suppression_db
        ));
    }
    if let Some(error) = in_band_error {
        if error > thresholds.maximum_in_band_amplitude_absolute_error {
            failed_gates.push(format!(
                "in-band amplitude absolute error {error:.6} exceeds {}",
                thresholds.maximum_in_band_amplitude_absolute_error
            ));
        }
    }
    let expected_resampler_label = format!("lanczos_windowed_sinc_antialias_v1_radius_{SINC_RADIUS}");
    if sinc.renderer_receipt.resampler != expected_resampler_label {
        failed_gates.push("render receipt does not identify the requested sinc kernel".to_owned());
    }
    if linear.renderer_receipt.resampler != "linear_interpolation_unfiltered_v1" {
        failed_gates.push("linear negative-control receipt does not identify its algorithm".to_owned());
    }

    Ok(CaseReceiptV1 {
        case_id: case_id.to_owned(),
        source_tone_cycles_per_frame,
        expected_output_alias_cycles_per_frame: output_alias_frequency,
        expected_source_amplitude: SOURCE_AMPLITUDE,
        linear,
        lanczos_sinc: sinc,
        alias_suppression_db_vs_linear: alias_suppression_db,
        in_band_amplitude_absolute_error: in_band_error,
        relative_pitch_ratio_error,
        passed: failed_gates.is_empty(),
        failed_gates,
    })
}

fn measure_path(
    manifest: &symthaea_muse::sfz_manifest::SfzMappingManifestV1,
    fixture_root: &Path,
    resampler: SfzResamplerV1,
    measured_frequency: f64,
) -> Result<PathMeasurementV1, Box<dyn Error>> {
    let mut elapsed_seconds = Vec::with_capacity(REPEATS);
    let mut output_hashes = Vec::with_capacity(REPEATS);
    let mut final_pair: Option<(SfzPlaybackReceiptV1, Vec<f32>)> = None;

    for _ in 0..REPEATS {
        let start = Instant::now();
        let pair = render_sfz_note_once_with_resampler(
            manifest,
            fixture_root,
            REQUESTED_KEY,
            VELOCITY,
            0,
            SAMPLE_RATE,
            resampler,
        )?;
        elapsed_seconds.push(start.elapsed().as_secs_f64());
        output_hashes.push(pair.0.output_sha256.clone());
        final_pair = Some(pair);
    }

    elapsed_seconds.sort_by(f64::total_cmp);
    let median_seconds = elapsed_seconds[elapsed_seconds.len() / 2];
    let (renderer_receipt, samples) = final_pair.ok_or_else(|| {
        io::Error::other("renderer loop completed without a sample receipt")
    })?;
    let finite_samples = samples.iter().all(|sample| sample.is_finite());
    let peak = samples.iter().fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    let measured_tone_amplitude = tone_amplitude(&samples, measured_frequency);
    let output_duration_seconds = samples.len() as f64 / f64::from(SAMPLE_RATE);
    let realtime_factor = output_duration_seconds / median_seconds.max(f64::MIN_POSITIVE);
    let first_hash = output_hashes.first().map(String::as_str).unwrap_or_default();

    Ok(PathMeasurementV1 {
        non_silent: renderer_receipt.rms > 1.0e-6,
        peak_within_unity: f64::from(peak) <= 1.0,
        repeated_output_hash_stable: output_hashes.iter().all(|hash| hash == first_hash),
        median_render_ms: median_seconds * 1_000.0,
        offline_realtime_factor: realtime_factor,
        renderer_receipt,
        measured_tone_amplitude,
        finite_samples,
    })
}

/// Single-tone quadrature amplitude, excluding 10% of each edge to reduce
/// finite-window effects. The same estimator is used for both algorithms.
fn tone_amplitude(samples: &[f32], cycles_per_frame: f64) -> f64 {
    if samples.len() < 20 {
        return 0.0;
    }
    let start = samples.len() / 10;
    let end = samples.len() - start;
    let count = (end - start) as f64;
    let mut in_phase = 0.0_f64;
    let mut quadrature = 0.0_f64;

    for (index, sample) in samples.iter().enumerate().take(end).skip(start) {
        let phase = std::f64::consts::TAU * cycles_per_frame * index as f64;
        in_phase += f64::from(*sample) * phase.cos();
        quadrature += f64::from(*sample) * phase.sin();
    }
    2.0 * in_phase.hypot(quadrature) / count
}

fn cpu_model_name() -> String {
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if let Some((_, model)) = line.split_once(':') {
                if line.starts_with("model name") || line.starts_with("Hardware") {
                    let model = model.trim();
                    if !model.is_empty() {
                        return model.to_owned();
                    }
                }
            }
        }
    }
    "unreported-by-host".to_owned()
}
