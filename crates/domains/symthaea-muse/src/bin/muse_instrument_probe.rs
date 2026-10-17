// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic instrument signal-health probe for Melothaea.
//!
//! This measures whether the arrangement renderer produced finite, non-silent
//! audio for a small common pitch/velocity grid and records the path selected
//! by the renderer's own dispatch logic. It does NOT prove pitch accuracy or
//! perceptual fidelity of the timbre/source.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::io::{self, Write};
use std::process::ExitCode;
use symthaea_muse::instruments::Instrument;
use symthaea_muse::synth::render_arrangement_with_evidence;
use symthaea_muse::voice::{Arrangement, Voice, VoiceRole};
use symthaea_muse::{AudioData, MuseConfig, MusicalState, Note, OutputFormat, ReverbConfig};

const SAMPLE_RATE: u32 = 22_050;
const NOTE_MS: u64 = 100;
const TAIL_MS: u64 = 100;
const REPEATS: usize = 2;
const PITCH_GRID_HZ: [f32; 3] = [220.0, 440.0, 880.0];
const VELOCITY_GRID: [f32; 3] = [0.2, 0.5, 0.9];

const ALL_INSTRUMENTS: [Instrument; 46] = [
    Instrument::Piano,
    Instrument::PianoPP,
    Instrument::Violin,
    Instrument::Cello,
    Instrument::Flute,
    Instrument::Organ,
    Instrument::ElectricPiano,
    Instrument::Bell,
    Instrument::AcousticGuitar,
    Instrument::Harp,
    Instrument::Pad,
    Instrument::Clarinet,
    Instrument::Trumpet,
    Instrument::Saxophone,
    Instrument::Marimba,
    Instrument::Sitar,
    Instrument::Kalimba,
    Instrument::SawLead,
    Instrument::Koto,
    Instrument::Oud,
    Instrument::Ney,
    Instrument::Recorder,
    Instrument::Ocarina,
    Instrument::Harmonica,
    Instrument::DanTranh,
    Instrument::UprightBass,
    Instrument::Viola,
    Instrument::Oboe,
    Instrument::FrenchHorn,
    Instrument::Bassoon,
    Instrument::Accordion,
    Instrument::Mandolin,
    Instrument::Banjo,
    Instrument::Vibraphone,
    Instrument::Celesta,
    Instrument::ChoirAah,
    Instrument::StringEnsemble,
    Instrument::ElectricBass,
    Instrument::Timpani,
    Instrument::MusicBox,
    Instrument::Harpsichord,
    Instrument::Tuba,
    Instrument::Trombone,
    Instrument::Duduk,
    Instrument::ChoirOoh,
    Instrument::Xylophone,
];

#[derive(Debug)]
struct Options {
    instruments: Vec<Instrument>,
    sample_rate: u32,
    note_ms: u64,
    tail_ms: u64,
    repeats: usize,
}

#[derive(Debug, Serialize)]
struct ProbeReport {
    schema_version: &'static str,
    benchmark: &'static str,
    code_revision: &'static str,
    sample_rate_hz: u32,
    note_duration_ms: u64,
    tail_duration_ms: u64,
    repeat_count: usize,
    pitch_grid_hz: Vec<f32>,
    velocity_grid: Vec<f32>,
    cases_total: usize,
    cases_with_non_finite_samples: usize,
    cases_near_silent: usize,
    cases_with_repeat_hash_mismatch: usize,
    cases_with_missing_receipts: usize,
    cases_with_render_path_mismatch: usize,
    evidence_boundary: &'static str,
    cases: Vec<ProbeCase>,
}

#[derive(Debug, Serialize)]
struct ProbeCase {
    instrument: String,
    gm_program_0_based: u8,
    declared_procedural_family: &'static str,
    selected_render_path_evidence: &'static str,
    frequency_hz: f32,
    velocity: f32,
    frames: usize,
    scalar_sample_count: usize,
    non_finite_sample_count: usize,
    peak: f64,
    rms: f64,
    dc_mean: f64,
    clipped_sample_count_at_or_above_0_999: usize,
    output_sha256: String,
    repeated_hashes: Vec<String>,
    repeat_hashes_match: bool,
    selected_render_path: String,
    sample_attempt_status: String,
    repeated_render_paths: Vec<String>,
    repeated_sample_attempt_statuses: Vec<String>,
    render_path_repeats_match: bool,
    outcome: &'static str,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muse_instrument_probe: {error}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), String> {
    let options = parse_args()?;
    let duration_secs = options.note_ms as f32 / 1_000.0;
    let total_ms = options
        .note_ms
        .checked_add(options.tail_ms)
        .ok_or_else(|| "note plus tail duration overflows".to_string())?;
    let total_frames = ((total_ms as u128) * (options.sample_rate as u128) / 1_000)
        .try_into()
        .map_err(|_| "requested frame count does not fit usize".to_string())?;

    let config = MuseConfig {
        sample_rate: options.sample_rate,
        duration_secs: duration_secs + options.tail_ms as f32 / 1_000.0,
        output_format: OutputFormat::StereoF32,
        reverb: ReverbConfig {
            wet_floor: 0.0,
            ..ReverbConfig::default()
        },
        ..MuseConfig::default()
    };

    let cases_expected = options.instruments.len() * PITCH_GRID_HZ.len() * VELOCITY_GRID.len();
    let mut cases = Vec::with_capacity(cases_expected);
    for instrument in &options.instruments {
        for frequency_hz in PITCH_GRID_HZ {
            for velocity in VELOCITY_GRID {
                let note = Note {
                    frequency: frequency_hz,
                    start_time: 0.0,
                    duration: duration_secs,
                    velocity,
                };
                let arrangement = Arrangement {
                    voices: vec![Voice {
                        role: VoiceRole::Lead,
                        notes: vec![note],
                        pitch_range: (frequency_hz * 0.5, frequency_hz * 2.0),
                        volume: 1.0,
                        pan: 0.0,
                        instrument: Some(*instrument),
                    }],
                };

                let mut hashes = Vec::with_capacity(options.repeats);
                let mut selected_paths = Vec::with_capacity(options.repeats);
                let mut sample_statuses = Vec::with_capacity(options.repeats);
                let mut first_metrics = None;
                for _ in 0..options.repeats {
                    let rendered = render_arrangement_with_evidence(
                        &arrangement,
                        options.sample_rate,
                        total_frames,
                        &MusicalState::default(),
                        &config,
                    );
                    let receipt = rendered.note_receipts.first();
                    selected_paths.push(
                        receipt
                            .map(|item| item.selected_render_path)
                            .unwrap_or("missing_render_receipt")
                            .to_string(),
                    );
                    sample_statuses.push(
                        receipt
                            .map(|item| item.sample_attempt_status)
                            .unwrap_or("missing_render_receipt")
                            .to_string(),
                    );
                    let (samples, frames) = flatten_audio(rendered.audio);
                    let hash = audio_sha256(&samples);
                    if first_metrics.is_none() {
                        first_metrics = Some(measure_signal(&samples, frames));
                    }
                    hashes.push(hash);
                }
                let metrics = first_metrics
                    .ok_or_else(|| "internal error: no render repetitions".to_string())?;
                let repeat_hashes_match = hashes.windows(2).all(|pair| pair[0] == pair[1]);
                let render_path_repeats_match =
                    selected_paths.windows(2).all(|pair| pair[0] == pair[1])
                        && sample_statuses.windows(2).all(|pair| pair[0] == pair[1]);
                let selected_render_path = selected_paths
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "missing_render_receipt".to_string());
                let sample_attempt_status = sample_statuses
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "missing_render_receipt".to_string());
                let outcome = if metrics.non_finite_count > 0 {
                    "non_finite_output"
                } else if selected_render_path == "missing_render_receipt" {
                    "missing_render_receipt"
                } else if !render_path_repeats_match {
                    "render_path_mismatch"
                } else if metrics.peak < 1.0e-5 {
                    "near_silent_output"
                } else if !repeat_hashes_match {
                    "repeat_hash_mismatch"
                } else {
                    "signal_present"
                };
                cases.push(ProbeCase {
                    instrument: format!("{instrument:?}"),
                    gm_program_0_based: instrument.gm_program(),
                    declared_procedural_family: procedural_family(*instrument),
                    // render_arrangement returns audio only. It does not expose
                    // whether local samples or a procedural fallback won.
                    selected_render_path_evidence: "not_exposed_by_legacy_render_api",
                    frequency_hz,
                    velocity,
                    frames: metrics.frames,
                    scalar_sample_count: metrics.scalar_sample_count,
                    non_finite_sample_count: metrics.non_finite_count,
                    peak: metrics.peak,
                    rms: metrics.rms,
                    dc_mean: metrics.dc_mean,
                    clipped_sample_count_at_or_above_0_999: metrics.clipped_count,
                    output_sha256: hashes.first().cloned().unwrap_or_default(),
                    repeated_hashes: hashes,
                    repeat_hashes_match,
                    selected_render_path,
                    sample_attempt_status,
                    repeated_render_paths: selected_paths,
                    repeated_sample_attempt_statuses: sample_statuses,
                    render_path_repeats_match,
                    outcome,
                });
            }
        }
    }

    let report = ProbeReport {
        schema_version: "melothaea.instrument-probe.v1",
        benchmark: "MEL-INST-PROBE-001",
        code_revision: option_env!("MEL_INSTRUMENT_PROBE_REV").unwrap_or("not_embedded"),
        sample_rate_hz: options.sample_rate,
        note_duration_ms: options.note_ms,
        tail_duration_ms: options.tail_ms,
        repeat_count: options.repeats,
        pitch_grid_hz: PITCH_GRID_HZ.to_vec(),
        velocity_grid: VELOCITY_GRID.to_vec(),
        cases_total: cases.len(),
        cases_with_non_finite_samples: cases
            .iter()
            .filter(|case| case.non_finite_sample_count > 0)
            .count(),
        cases_near_silent: cases
            .iter()
            .filter(|case| case.outcome == "near_silent_output")
            .count(),
        cases_with_repeat_hash_mismatch: cases
            .iter()
            .filter(|case| !case.repeat_hashes_match)
            .count(),
        cases_with_missing_receipts: cases
            .iter()
            .filter(|case| case.selected_render_path == "missing_render_receipt")
            .count(),
        cases_with_render_path_mismatch: cases
            .iter()
            .filter(|case| !case.render_path_repeats_match)
            .count(),
        evidence_boundary: "signal health, selected render-path receipt, and same-process repeat determinism only; not pitch accuracy, instrument fidelity, or human preference",
        cases,
    };

    let stdout = io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer_pretty(&mut lock, &report)
        .map_err(|error| format!("cannot serialize probe JSON: {error}"))?;
    writeln!(lock).map_err(|error| format!("cannot flush report: {error}"))?;
    Ok(())
}

struct SignalMetrics {
    frames: usize,
    scalar_sample_count: usize,
    non_finite_count: usize,
    peak: f64,
    rms: f64,
    dc_mean: f64,
    clipped_count: usize,
}

fn flatten_audio(audio: AudioData) -> (Vec<f32>, usize) {
    match audio {
        AudioData::StereoF32(frames) => {
            let frame_count = frames.len();
            (frames.into_iter().flat_map(|frame| [frame[0], frame[1]]).collect(), frame_count)
        }
        AudioData::F32(samples) => {
            let frame_count = samples.len();
            (samples, frame_count)
        }
        AudioData::I16(samples) => {
            let frame_count = samples.len();
            let normalized = samples
                .into_iter()
                .map(|sample| sample as f32 / i16::MAX as f32)
                .collect();
            (normalized, frame_count)
        }
    }
}

fn measure_signal(samples: &[f32], frames: usize) -> SignalMetrics {
    let non_finite_count = samples.iter().filter(|sample| !sample.is_finite()).count();
    let mut peak = 0.0_f64;
    let mut square_sum = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut finite_count = 0_usize;
    let mut clipped_count = 0_usize;
    for sample in samples.iter().copied().filter(|sample| sample.is_finite()) {
        let value = f64::from(sample);
        let absolute = value.abs();
        peak = peak.max(absolute);
        square_sum += value * value;
        sum += value;
        finite_count += 1;
        if absolute >= 0.999 {
            clipped_count += 1;
        }
    }
    let rms = if finite_count > 0 {
        (square_sum / finite_count as f64).sqrt()
    } else {
        0.0
    };
    let dc_mean = if finite_count > 0 {
        sum / finite_count as f64
    } else {
        0.0
    };
    SignalMetrics {
        frames,
        scalar_sample_count: samples.len(),
        non_finite_count,
        peak,
        rms,
        dc_mean,
        clipped_count,
    }
}

fn audio_sha256(samples: &[f32]) -> String {
    let mut hash = Sha256::new();
    for sample in samples {
        hash.update(sample.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn procedural_family(instrument: Instrument) -> &'static str {
    if instrument.uses_karplus_strong() {
        "karplus_strong"
    } else if instrument.uses_fm() {
        "frequency_modulation"
    } else {
        "additive_declared"
    }
}

fn parse_args() -> Result<Options, String> {
    let mut options = Options {
        instruments: ALL_INSTRUMENTS.to_vec(),
        sample_rate: SAMPLE_RATE,
        note_ms: NOTE_MS,
        tail_ms: TAIL_MS,
        repeats: REPEATS,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--instrument" => {
                let name = args
                    .next()
                    .ok_or_else(|| "--instrument requires a name or 'all'".to_string())?;
                if name == "all" {
                    options.instruments = ALL_INSTRUMENTS.to_vec();
                } else {
                    let normalized = name.to_ascii_lowercase();
                    let instrument = Instrument::from_name(&normalized)
                        .ok_or_else(|| format!("unknown instrument name: {name}"))?;
                    options.instruments = vec![instrument];
                }
            }
            "--sample-rate" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--sample-rate requires an integer".to_string())?
                    .parse::<u32>()
                    .map_err(|_| "sample rate must be an integer".to_string())?;
                if !(8_000..=192_000).contains(&value) {
                    return Err("--sample-rate must be between 8000 and 192000 Hz".into());
                }
                options.sample_rate = value;
            }
            "--note-ms" => {
                options.note_ms = parse_bounded_u64(
                    args.next().ok_or_else(|| "--note-ms requires an integer".to_string())?,
                    "--note-ms",
                    20,
                    1_000,
                )?;
            }
            "--tail-ms" => {
                options.tail_ms = parse_bounded_u64(
                    args.next().ok_or_else(|| "--tail-ms requires an integer".to_string())?,
                    "--tail-ms",
                    0,
                    1_000,
                )?;
            }
            "--repeats" => {
                options.repeats = args
                    .next()
                    .ok_or_else(|| "--repeats requires an integer".to_string())?
                    .parse::<usize>()
                    .map_err(|_| "--repeats must be an integer".to_string())?;
                if !(1..=5).contains(&options.repeats) {
                    return Err("--repeats must be between 1 and 5".into());
                }
            }
            _ => return Err(format!("unknown argument: {arg} (use --help)")),
        }
    }
    Ok(options)
}

fn parse_bounded_u64(value: String, label: &str, min: u64, max: u64) -> Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| format!("{label} must be an integer"))?;
    if !(min..=max).contains(&parsed) {
        return Err(format!("{label} must be between {min} and {max}"));
    }
    Ok(parsed)
}

fn print_help() {
    println!(
        "MEL-INST-PROBE-001: deterministic instrument signal-health probe\n\
         Usage: cargo run -p symthaea-muse --bin muse_instrument_probe -- [OPTIONS]\n\
         Options:\n\
           --instrument NAME|all  Probe one registered instrument or all (default: all)\n\
           --sample-rate HZ       Sample rate, 8000..192000 (default: 22050)\n\
           --note-ms MS           Note duration, 20..1000 (default: 100)\n\
           --tail-ms MS           Rendered tail, 0..1000 (default: 100)\n\
           --repeats N            Exact rerenders, 1..5 (default: 2)\n\
           -h, --help             Show this help\n\
         JSON is written to stdout. Redirect it to a file if a local report is desired.\n\
         Per-note render-path receipts are included; they do not prove perceptual fidelity.\n\
         The pitch grid (220/440/880 Hz) is a common smoke grid, not a declared range test."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instrument_roster_is_unique_and_matches_enum_count() {
        let unique: std::collections::HashSet<_> = ALL_INSTRUMENTS
            .iter()
            .map(|instrument| format!("{instrument:?}"))
            .collect();
        assert_eq!(ALL_INSTRUMENTS.len(), 46);
        assert_eq!(unique.len(), ALL_INSTRUMENTS.len());
    }

    #[test]
    fn cli_rejects_unsafe_or_unbounded_probe_settings() {
        assert!(parse_bounded_u64("0".into(), "--note-ms", 20, 1_000).is_err());
        assert!(parse_bounded_u64("1001".into(), "--note-ms", 20, 1_000).is_err());
        assert!(parse_bounded_u64("20".into(), "--note-ms", 20, 1_000).is_ok());
    }

    #[test]
    fn signal_metrics_ignore_non_finite_samples_but_report_them() {
        let metrics = measure_signal(&[0.5, -0.5, f32::NAN, f32::INFINITY], 2);
        assert_eq!(metrics.non_finite_count, 2);
        assert!((metrics.peak - 0.5).abs() < 1.0e-12);
        assert!((metrics.rms - 0.5).abs() < 1.0e-12);
        assert!((metrics.dc_mean).abs() < 1.0e-12);
    }

    #[test]
    fn exact_audio_hash_binds_float_sample_bits() {
        let first = audio_sha256(&[0.0, 0.25, -0.5]);
        assert_eq!(first, audio_sha256(&[0.0, 0.25, -0.5]));
        assert_ne!(first, audio_sha256(&[0.0, 0.25, -0.5001]));
    }
}
