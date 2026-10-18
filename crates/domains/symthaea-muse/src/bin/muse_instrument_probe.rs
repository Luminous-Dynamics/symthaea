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
use symthaea_muse::synth::{
    SelectedSampleEvidence, render_arrangement_with_evidence,
};
use symthaea_muse::voice::{Arrangement, Voice, VoiceRole};
use symthaea_muse::{AudioData, MuseConfig, MusicalState, Note, OutputFormat, ReverbConfig};

const SAMPLE_RATE: u32 = 22_050;
const NOTE_MS: u64 = 100;
const TAIL_MS: u64 = 100;
const REPEATS: usize = 2;
const PITCH_GRID_MIDI: [u8; 3] = [57, 69, 81];
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
    midi_range: Option<(u8, u8)>,
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
    pitch_grid_midi: Vec<u8>,
    pitch_grid_definition: &'static str,
    velocity_grid: Vec<f32>,
    cases_total: usize,
    cases_with_non_finite_samples: usize,
    cases_near_silent: usize,
    cases_with_repeat_hash_mismatch: usize,
    cases_with_missing_receipts: usize,
    cases_with_mismatched_receipts: usize,
    cases_with_render_path_mismatch: usize,
    cases_with_sample_evidence_mismatch: usize,
    velocity_response_group_count: usize,
    velocity_groups_with_non_monotonic_rms: usize,
    evidence_boundary: &'static str,
    cases: Vec<ProbeCase>,
    velocity_response_groups: Vec<VelocityResponseGroup>,
}

#[derive(Debug, Serialize)]
struct ProbeCase {
    instrument: String,
    gm_program_0_based: u8,
    declared_procedural_family: &'static str,
    midi_note: u8,
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
    selected_sample: Option<SelectedSampleEvidence>,
    repeated_selected_samples: Vec<Option<SelectedSampleEvidence>>,
    sample_evidence_repeats_match: bool,
    receipt_matches_request: bool,
    repeated_receipt_matches_request: Vec<bool>,
    render_path_repeats_match: bool,
    estimated_pitch_hz: Option<f64>,
    pitch_error_cents: Option<f64>,
    pitch_periodicity_confidence: Option<f64>,
    pitch_estimate_status: &'static str,
    outcome: &'static str,
}

#[derive(Debug, Serialize)]
struct VelocityResponseGroup {
    instrument: String,
    requested_pitch_hz: f32,
    velocities: Vec<f32>,
    rms_levels: Vec<f64>,
    adjacent_level_steps_db: Vec<Option<f64>>,
    rms_is_monotonic_non_decreasing: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct PitchDiagnostic {
    status: &'static str,
    estimated_pitch_hz: Option<f64>,
    error_cents: Option<f64>,
    periodicity_confidence: Option<f64>,
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
    let pitch_cells = build_pitch_cells(options.midi_range);
    let cases_expected =
        options.instruments.len() * pitch_cells.len() * VELOCITY_GRID.len();
    if cases_expected > 10_000 {
        return Err(format!(
            "requested sweep creates {cases_expected} cells; choose one instrument or narrow --midi-range (limit: 10000 cells)"
        ));
    }
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

    let mut cases = Vec::with_capacity(cases_expected);
    for instrument in &options.instruments {
        for (midi_note, frequency_hz) in pitch_cells.iter().copied() {
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
                let mut selected_samples = Vec::with_capacity(options.repeats);
                let mut receipt_matches = Vec::with_capacity(options.repeats);
                let mut first_metrics = None;
                let mut first_pitch: Option<PitchDiagnostic> = None;
                for _ in 0..options.repeats {
                    let rendered = render_arrangement_with_evidence(
                        &arrangement,
                        options.sample_rate,
                        total_frames,
                        &MusicalState::default(),
                        &config,
                    );
                    let receipt = rendered.note_receipts.first();
                    let matches_request = receipt.is_some_and(|item| {
                        item.voice_index == 0
                            && item.note_index == 0
                            && item.instrument == Some(*instrument)
                            && item.frequency_hz == frequency_hz
                            && item.velocity == velocity
                            && item.duration_secs == duration_secs
                    });
                    receipt_matches.push(matches_request);
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
                    selected_samples.push(receipt.and_then(|item| item.selected_sample.clone()));
                    let (samples, frames) = flatten_audio(rendered.audio);
                    let hash = audio_sha256(&samples);
                    if first_metrics.is_none() {
                        first_metrics = Some(measure_signal(&samples, frames));
                        first_pitch = Some(estimate_pitch_yin(
                            &samples,
                            options.sample_rate,
                            2,
                            options.note_ms,
                            frequency_hz,
                        ));
                    }
                    hashes.push(hash);
                }
                let metrics = first_metrics
                    .ok_or_else(|| "internal error: no render repetitions".to_string())?;
                let pitch = first_pitch.unwrap_or(PitchDiagnostic {
                    status: "missing_pitch_analysis",
                    estimated_pitch_hz: None,
                    error_cents: None,
                    periodicity_confidence: None,
                });
                let repeat_hashes_match = hashes.windows(2).all(|pair| pair[0] == pair[1]);
                let receipt_matches_request = receipt_matches.first().copied().unwrap_or(false);
                let sample_evidence_repeats_match =
                    selected_samples.windows(2).all(|pair| pair[0] == pair[1]);
                let selected_sample = selected_samples.first().cloned().unwrap_or(None);
                let render_path_repeats_match =
                    selected_paths.windows(2).all(|pair| pair[0] == pair[1])
                        && sample_statuses.windows(2).all(|pair| pair[0] == pair[1])
                        && receipt_matches.iter().all(|matches| *matches);
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
                } else if !receipt_matches_request {
                    "render_receipt_mismatch"
                } else if !sample_evidence_repeats_match {
                    "sample_evidence_mismatch"
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
                    midi_note,
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
                    selected_sample,
                    repeated_selected_samples: selected_samples,
                    sample_evidence_repeats_match,
                    receipt_matches_request,
                    repeated_receipt_matches_request: receipt_matches,
                    render_path_repeats_match,
                    estimated_pitch_hz: pitch.estimated_pitch_hz,
                    pitch_error_cents: pitch.error_cents,
                    pitch_periodicity_confidence: pitch.periodicity_confidence,
                    pitch_estimate_status: pitch.status,
                    outcome,
                });
            }
        }
    }

    let velocity_response_groups = build_velocity_response_groups(&cases);
    let velocity_groups_with_non_monotonic_rms = velocity_response_groups
        .iter()
        .filter(|group| !group.rms_is_monotonic_non_decreasing)
        .count();
    let report = ProbeReport {
        schema_version: "melothaea.instrument-probe.v2",
        benchmark: "MEL-INST-PROBE-001",
        code_revision: option_env!("MEL_INSTRUMENT_PROBE_REV").unwrap_or("not_embedded"),
        sample_rate_hz: options.sample_rate,
        note_duration_ms: options.note_ms,
        tail_duration_ms: options.tail_ms,
        repeat_count: options.repeats,
        pitch_grid_hz: pitch_cells.iter().map(|(_, frequency)| *frequency).collect(),
        pitch_grid_midi: pitch_cells.iter().map(|(midi, _)| *midi).collect(),
        pitch_grid_definition: if options.midi_range.is_some() {
            "user_selected_inclusive_midi_semitone_sweep"
        } else {
            "fixed_A3_A4_A5_signal_health_smoke_grid"
        },
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
        cases_with_mismatched_receipts: cases
            .iter()
            .filter(|case| !case.receipt_matches_request)
            .count(),
        cases_with_render_path_mismatch: cases
            .iter()
            .filter(|case| !case.render_path_repeats_match)
            .count(),
        cases_with_sample_evidence_mismatch: cases
            .iter()
            .filter(|case| !case.sample_evidence_repeats_match)
            .count(),
        velocity_response_group_count: velocity_response_groups.len(),
        velocity_groups_with_non_monotonic_rms,
        evidence_boundary: "signal health, renderer/source receipts, exploratory pitch estimate, velocity-response shape, and same-process repeat determinism; pitch diagnostics are not instrument-family calibrated and do not prove fidelity or human preference",
        cases,
        velocity_response_groups,
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


/// Estimate the dominant periodicity in an onset-trimmed mono window with
/// YIN's cumulative-mean-normalized difference function. For inharmonic or
/// noisy timbres this can be low-confidence or octave-ambiguous; callers must
/// retain status/confidence and must not treat a number as a pass by itself.
fn estimate_pitch_yin(
    interleaved_samples: &[f32],
    sample_rate: u32,
    channels: usize,
    note_duration_ms: u64,
    target_hz: f32,
) -> PitchDiagnostic {
    let unavailable = |status| PitchDiagnostic {
        status,
        estimated_pitch_hz: None,
        error_cents: None,
        periodicity_confidence: None,
    };
    if channels == 0 || sample_rate < 1 {
        return unavailable("invalid_audio_shape");
    }
    let frames = interleaved_samples.len() / channels;
    let declared_note_frames = ((note_duration_ms as u128) * (sample_rate as u128) / 1_000)
        .min(frames as u128) as usize;
    let start_frame = ((sample_rate as usize * 20) / 1_000).min(declared_note_frames / 5);
    let end_frame = declared_note_frames;
    if end_frame <= start_frame + 16 {
        return unavailable("window_too_short");
    }

    let mut mono = Vec::with_capacity(end_frame - start_frame);
    for frame in start_frame..end_frame {
        let offset = frame * channels;
        let mut frame_sum = 0.0_f64;
        for channel in 0..channels {
            let sample = interleaved_samples[offset + channel];
            if !sample.is_finite() {
                return unavailable("non_finite_analysis_window");
            }
            frame_sum += f64::from(sample);
        }
        mono.push(frame_sum / channels as f64);
    }
    if mono.is_empty() {
        return unavailable("window_too_short");
    }
    let mean = mono.iter().sum::<f64>() / mono.len() as f64;
    for sample in &mut mono {
        *sample -= mean;
    }
    let rms = (mono.iter().map(|sample| sample * sample).sum::<f64>()
        / mono.len() as f64)
        .sqrt();
    if !rms.is_finite() || rms < 1.0e-7 {
        return unavailable("near_silent_window");
    }

    let min_frequency = 60.0_f64;
    let max_frequency = 1_500.0_f64.min(f64::from(sample_rate) * 0.24);
    let min_lag = (f64::from(sample_rate) / max_frequency).floor().max(2.0) as usize;
    let max_lag = ((f64::from(sample_rate) / min_frequency).floor() as usize)
        .min(mono.len() / 2);
    if max_lag <= min_lag + 1 {
        return unavailable("window_too_short_for_period_range");
    }

    let mut difference = vec![0.0_f64; max_lag + 1];
    for tau in 1..=max_lag {
        let mut sum = 0.0_f64;
        let comparison_len = mono.len() - tau;
        for index in 0..comparison_len {
            let delta = mono[index] - mono[index + tau];
            sum += delta * delta;
        }
        difference[tau] = sum;
    }
    let mut cumulative = 0.0_f64;
    let mut cmnd = vec![1.0_f64; max_lag + 1];
    for tau in 1..=max_lag {
        cumulative += difference[tau];
        cmnd[tau] = if cumulative > 1.0e-18 {
            difference[tau] * tau as f64 / cumulative
        } else {
            1.0
        };
    }

    // Prefer the first trough below the YIN threshold; otherwise retain the
    // best observed lag but label low periodicity rather than declaring a pass.
    const YIN_THRESHOLD: f64 = 0.15;
    let mut chosen = None;
    for tau in min_lag..max_lag {
        if cmnd[tau] < YIN_THRESHOLD {
            let mut trough = tau;
            while trough + 1 < max_lag && cmnd[trough + 1] < cmnd[trough] {
                trough += 1;
            }
            chosen = Some(trough);
            break;
        }
    }
    let tau = chosen.unwrap_or_else(|| {
        (min_lag..=max_lag)
            .min_by(|left, right| cmnd[*left].total_cmp(&cmnd[*right]))
            .unwrap_or(min_lag)
    });
    let left = cmnd[tau.saturating_sub(1)];
    let center = cmnd[tau];
    let right = cmnd[(tau + 1).min(max_lag)];
    let denominator = left - 2.0 * center + right;
    let adjustment = if denominator.abs() > 1.0e-12 {
        (0.5 * (left - right) / denominator).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let refined_lag = tau as f64 + adjustment;
    if refined_lag <= 0.0 || !refined_lag.is_finite() {
        return unavailable("invalid_period_estimate");
    }
    let estimate = f64::from(sample_rate) / refined_lag;
    let confidence = (1.0 - center).clamp(0.0, 1.0);
    let error_cents = if target_hz.is_finite() && target_hz > 0.0 {
        Some(1_200.0 * (estimate / f64::from(target_hz)).log2())
    } else {
        None
    };
    PitchDiagnostic {
        status: if confidence >= 0.60 {
            "periodic_estimate_available"
        } else {
            "low_periodicity"
        },
        estimated_pitch_hz: Some(estimate),
        error_cents,
        periodicity_confidence: Some(confidence),
    }
}

fn build_velocity_response_groups(cases: &[ProbeCase]) -> Vec<VelocityResponseGroup> {
    let mut keys: Vec<(String, f32)> = cases
        .iter()
        .map(|case| (case.instrument.clone(), case.frequency_hz))
        .collect();
    keys.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
    });
    keys.dedup_by(|left, right| left.0 == right.0 && left.1 == right.1);
    keys.into_iter()
        .map(|(instrument, requested_pitch_hz)| {
            let mut matching: Vec<&ProbeCase> = cases
                .iter()
                .filter(|case| {
                    case.instrument == instrument && case.frequency_hz == requested_pitch_hz
                })
                .collect();
            matching.sort_by(|left, right| left.velocity.total_cmp(&right.velocity));
            let velocities: Vec<f32> = matching.iter().map(|case| case.velocity).collect();
            let rms_levels: Vec<f64> = matching.iter().map(|case| case.rms).collect();
            let adjacent_level_steps_db: Vec<Option<f64>> = rms_levels
                .windows(2)
                .map(|pair| {
                    (pair[0].is_finite() && pair[1].is_finite()).then(|| {
                        20.0 * (pair[1].max(1.0e-12) / pair[0].max(1.0e-12)).log10()
                    })
                })
                .collect();
            let rms_is_monotonic_non_decreasing = rms_levels
                .windows(2)
                .all(|pair| pair[1] + 1.0e-12 >= pair[0]);
            VelocityResponseGroup {
                instrument,
                requested_pitch_hz,
                velocities,
                rms_levels,
                adjacent_level_steps_db,
                rms_is_monotonic_non_decreasing,
            }
        })
        .collect()
}

fn parse_args() -> Result<Options, String> {
    let mut options = Options {
        instruments: ALL_INSTRUMENTS.to_vec(),
        sample_rate: SAMPLE_RATE,
        note_ms: NOTE_MS,
        tail_ms: TAIL_MS,
        repeats: REPEATS,
        midi_range: None,
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
            "--midi-range" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--midi-range requires MIN:MAX".to_string())?;
                options.midi_range = Some(parse_midi_range(&value)?);
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

fn parse_midi_range(value: &str) -> Result<(u8, u8), String> {
    let (minimum, maximum) = value
        .split_once(':')
        .ok_or_else(|| "--midi-range must have the form MIN:MAX".to_string())?;
    let minimum = minimum
        .parse::<u8>()
        .map_err(|_| "MIDI minimum must be an integer in 0..=127".to_string())?;
    let maximum = maximum
        .parse::<u8>()
        .map_err(|_| "MIDI maximum must be an integer in 0..=127".to_string())?;
    if minimum > maximum {
        return Err("--midi-range minimum must be <= maximum".into());
    }
    Ok((minimum, maximum))
}

fn build_pitch_cells(midi_range: Option<(u8, u8)>) -> Vec<(u8, f32)> {
    match midi_range {
        Some((minimum, maximum)) => (minimum..=maximum)
            .map(|midi| (midi, midi_to_hz(midi)))
            .collect(),
        None => vec![(57, 220.0), (69, 440.0), (81, 880.0)],
    }
}

fn midi_to_hz(midi: u8) -> f32 {
    440.0 * 2.0_f32.powf((f32::from(midi) - 69.0) / 12.0)
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
           --midi-range MIN:MAX   Inclusive semitone sweep, MIDI 0..127 (default: A3/A4/A5)\n\
           -h, --help             Show this help\n\
         JSON is written to stdout. Redirect it to a file if a local report is desired.\n\
         Per-note render/source receipts are included; they do not prove perceptual fidelity.\n\
         The default pitch grid is a smoke test; a user-selected sweep tests rendered cells, not certified playability."
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
    fn midi_range_is_inclusive_and_validates_bounds() {
        assert_eq!(parse_midi_range("48:72").unwrap(), (48, 72));
        assert_eq!(build_pitch_cells(Some((48, 50))).len(), 3);
        assert_eq!(
            build_pitch_cells(Some((48, 50))).iter().map(|(midi, _)| *midi).collect::<Vec<_>>(),
            vec![48, 49, 50]
        );
        assert!(parse_midi_range("73:60").is_err());
        assert!(parse_midi_range("0:128").is_err());
        assert!(parse_midi_range("C3:72").is_err());
        assert!(parse_midi_range("48").is_err());
    }

    #[test]
    fn default_grid_is_explicitly_a3_a4_a5() {
        assert_eq!(
            build_pitch_cells(None),
            vec![(57, 220.0), (69, 440.0), (81, 880.0)]
        );
        assert!((midi_to_hz(69) - 440.0).abs() < 1.0e-4);
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
    fn yin_recovers_clean_stereo_tone_and_reports_cents() {
        let sample_rate = 22_050_u32;
        let frames = (sample_rate as usize) / 10;
        let samples: Vec<f32> = (0..frames)
            .flat_map(|index| {
                let sample = (2.0 * std::f64::consts::PI * 440.0 * index as f64
                    / sample_rate as f64)
                    .sin() as f32
                    * 0.5;
                [sample, sample]
            })
            .collect();
        let result = estimate_pitch_yin(&samples, sample_rate, 2, 100, 440.0);
        assert_eq!(result.status, "periodic_estimate_available");
        let estimate = result.estimated_pitch_hz.expect("tone should estimate");
        assert!((estimate - 440.0).abs() < 2.0, "estimated {estimate}");
        assert!(
            result.error_cents.expect("error should exist").abs() < 5.0,
            "{result:?}"
        );
        assert!(result.periodicity_confidence.unwrap() >= 0.60);
    }

    #[test]
    fn yin_never_claims_pitch_from_silence() {
        let result = estimate_pitch_yin(&vec![0.0; 4_410], 22_050, 2, 100, 440.0);
        assert_eq!(result.status, "near_silent_window");
        assert!(result.estimated_pitch_hz.is_none());
        assert!(result.error_cents.is_none());
    }

    #[test]
    fn velocity_response_group_reports_monotonic_rms_as_diagnostic() {
        fn sample(velocity: f32, rms: f64) -> ProbeCase {
            ProbeCase {
                instrument: "Harmonica".into(),
                gm_program_0_based: 22,
                declared_procedural_family: "additive_declared",
                midi_note: 69,
                frequency_hz: 440.0,
                velocity,
                frames: 2_205,
                scalar_sample_count: 4_410,
                non_finite_sample_count: 0,
                peak: rms,
                rms,
                dc_mean: 0.0,
                clipped_sample_count_at_or_above_0_999: 0,
                output_sha256: String::new(),
                repeated_hashes: vec![],
                repeat_hashes_match: true,
                selected_render_path: "additive_synthesis".into(),
                sample_attempt_status: "sample_library_not_loaded".into(),
                repeated_render_paths: vec![],
                repeated_sample_attempt_statuses: vec![],
                selected_sample: None,
                repeated_selected_samples: vec![],
                sample_evidence_repeats_match: true,
                receipt_matches_request: true,
                repeated_receipt_matches_request: vec![],
                render_path_repeats_match: true,
                estimated_pitch_hz: Some(440.0),
                pitch_error_cents: Some(0.0),
                pitch_periodicity_confidence: Some(0.9),
                pitch_estimate_status: "periodic_estimate_available",
                outcome: "signal_present",
            }
        }
        let monotonic = build_velocity_response_groups(&[
            sample(0.9, 0.3),
            sample(0.2, 0.1),
            sample(0.5, 0.2),
        ]);
        assert_eq!(monotonic.len(), 1);
        assert_eq!(monotonic[0].velocities, vec![0.2, 0.5, 0.9]);
        assert!(monotonic[0].rms_is_monotonic_non_decreasing);

        let non_monotonic = build_velocity_response_groups(&[
            sample(0.2, 0.1),
            sample(0.5, 0.07),
            sample(0.9, 0.3),
        ]);
        assert!(!non_monotonic[0].rms_is_monotonic_non_decreasing);
        assert_eq!(non_monotonic[0].adjacent_level_steps_db.len(), 2);
    }

    #[test]
    fn exact_audio_hash_binds_float_sample_bits() {
        let first = audio_sha256(&[0.0, 0.25, -0.5]);
        assert_eq!(first, audio_sha256(&[0.0, 0.25, -0.5]));
        assert_ne!(first, audio_sha256(&[0.0, 0.25, -0.5001]));
    }
}
