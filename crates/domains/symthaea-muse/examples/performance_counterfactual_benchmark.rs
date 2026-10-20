// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Counterfactual audit of style-selected performance dialects.
//!
//! This measures performed events, not rendered-audio quality. For each fixed
//! score and seed, it compares the automatic style dialect against an explicit
//! ClassicalRubato control while holding the score, ensemble resolution,
//! swing, return-color setting and cognitive state fixed.
//!
//! Usage:
//!   cargo run --release -p symthaea-muse --features theory --example performance_counterfactual_benchmark
//!   cargo run --release -p symthaea-muse --features theory --example performance_counterfactual_benchmark -- /tmp/melothaea-counterfactual.json

use std::collections::HashSet;
use std::path::PathBuf;

use serde::Serialize;
use sha2::{Digest, Sha256};
use symthaea_muse::theory_realize::{
    PerformanceDialect, PerformedVoice, dialect_for_spec, perform_with_spec,
    perform_with_spec_and_dialect,
};
use symthaea_muse::MusicalState;
use symthaea_music_theory::{MusicalIntent, Style, compose_with_spec};

const SEEDS: [u64; 8] = [0, 1, 7, 13, 29, 42, 73, 101];
const MATCH_WINDOW_SECS: f32 = 0.250;
const REPORT_SCHEMA: u32 = 1;

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    experiment: &'static str,
    status: &'static str,
    source_controls: &'static str,
    human_quality: &'static str,
    audio_rendered: bool,
    seeds: Vec<u64>,
    cases: Vec<CaseReport>,
    summary: Summary,
}

#[derive(Debug, Serialize)]
struct CaseReport {
    style: String,
    seed: u64,
    score_sha256: String,
    selected_dialect: String,
    control_dialect: String,
    replay_exact: bool,
    actual: ArmMetrics,
    control: ArmMetrics,
    matched_events: usize,
    unmatched_actual_events: usize,
    unmatched_control_events: usize,
    changed_onsets: usize,
    changed_onset_fraction: Option<f64>,
    mean_abs_onset_shift_ms: Option<f64>,
    median_abs_onset_shift_ms: Option<f64>,
    p95_abs_onset_shift_ms: Option<f64>,
    status: &'static str,
}

#[derive(Debug, Serialize, Default)]
struct ArmMetrics {
    voices: usize,
    events: usize,
    invalid_events: usize,
    median_duration_ms: Option<f64>,
    median_inter_onset_ms: Option<f64>,
    p95_inter_onset_ms: Option<f64>,
    onset_span_secs: Option<f64>,
}

#[derive(Debug, Serialize)]
struct Summary {
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    total_actual_events: usize,
    total_control_events: usize,
    total_matched_events: usize,
    total_changed_onsets: usize,
    matched_event_fraction: Option<f64>,
    changed_onset_fraction: Option<f64>,
    median_abs_onset_shift_ms: Option<f64>,
    p95_abs_onset_shift_ms: Option<f64>,
}

#[derive(Clone, Copy)]
struct Event {
    voice: String,
    pitch_bits: u32,
    start: f32,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn event_signature(voices: &[PerformedVoice]) -> Vec<(String, String, Vec<(u32, u32, u32, u32)>)> {
    voices.iter().map(|voice| (
        voice.name.clone(),
        voice.instrument.clone(),
        voice.notes.iter().map(|n| (
            n.frequency.to_bits(),
            n.start_time.to_bits(),
            n.duration.to_bits(),
            n.velocity.to_bits(),
        )).collect(),
    )).collect()
}

fn flatten(voices: &[PerformedVoice]) -> Vec<Event> {
    let mut out = Vec::new();
    for voice in voices {
        for note in &voice.notes {
            out.push(Event {
                voice: voice.name.clone(),
                pitch_bits: note.frequency.to_bits(),
                start: note.start_time,
            });
        }
    }
    out
}

fn quantile(values: &mut [f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * q).round() as usize;
    values.get(index).copied()
}

fn metrics(voices: &[PerformedVoice]) -> ArmMetrics {
    let mut durations_ms = Vec::new();
    let mut intervals_ms = Vec::new();
    let mut starts = Vec::new();
    let mut invalid = 0usize;

    for voice in voices {
        let mut voice_starts = Vec::new();
        for note in &voice.notes {
            if !note.frequency.is_finite()
                || note.frequency <= 0.0
                || !note.start_time.is_finite()
                || note.start_time < 0.0
                || !note.duration.is_finite()
                || note.duration <= 0.0
                || !note.velocity.is_finite()
                || !(0.0..=1.0).contains(&note.velocity)
            {
                invalid += 1;
            }
            if note.start_time.is_finite() {
                starts.push(note.start_time as f64);
                voice_starts.push(note.start_time as f64);
            }
            if note.duration.is_finite() && note.duration > 0.0 {
                durations_ms.push(note.duration as f64 * 1000.0);
            }
        }
        voice_starts.sort_by(f64::total_cmp);
        for pair in voice_starts.windows(2) {
            let delta = (pair[1] - pair[0]) * 1000.0;
            if delta > 1e-4 {
                intervals_ms.push(delta);
            }
        }
    }

    let min_start = starts.iter().copied().min_by(f64::total_cmp);
    let max_start = starts.iter().copied().max_by(f64::total_cmp);
    ArmMetrics {
        voices: voices.len(),
        events: voices.iter().map(|v| v.notes.len()).sum(),
        invalid_events: invalid,
        median_duration_ms: quantile(&mut durations_ms, 0.5),
        median_inter_onset_ms: quantile(&mut intervals_ms, 0.5),
        p95_inter_onset_ms: quantile(&mut intervals_ms, 0.95),
        onset_span_secs: min_start.zip(max_start).map(|(lo, hi)| hi - lo),
    }
}

/// Greedy nearest-onset matching within a voice and exact pitch. Candidate
/// edges are sorted by onset error and consumed one-to-one. Unmatched events
/// remain visible; the metric never silently assumes identical note counts.
/// Matching is descriptive, not a proof of notational identity.
fn compare(actual: &[Event], control: &[Event]) -> (usize, usize, usize, Vec<f64>) {
    let mut edges = Vec::<(f64, usize, usize)>::new();
    for (ai, a) in actual.iter().enumerate() {
        for (ci, c) in control.iter().enumerate() {
            if a.voice != c.voice || a.pitch_bits != c.pitch_bits {
                continue;
            }
            let delta = (a.start - c.start).abs() as f64;
            if delta <= MATCH_WINDOW_SECS as f64 {
                edges.push((delta, ai, ci));
            }
        }
    }
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut used_actual = HashSet::new();
    let mut used_control = HashSet::new();
    let mut shifts = Vec::new();
    for (delta, ai, ci) in edges {
        if used_actual.contains(&ai) || used_control.contains(&ci) {
            continue;
        }
        used_actual.insert(ai);
        used_control.insert(ci);
        shifts.push(delta * 1000.0);
    }
    shifts.sort_by(f64::total_cmp);
    (
        shifts.len(),
        actual.len().saturating_sub(used_actual.len()),
        control.len().saturating_sub(used_control.len()),
        shifts,
    )
}

fn case(style: Style, seed: u64, state: &MusicalState) -> CaseReport {
    let spec = style.spec();
    let intent = MusicalIntent {
        seed,
        arousal: 0.7,
        ..Default::default()
    };
    let score = compose_with_spec(&intent, &spec);
    let score_bytes = serde_json::to_vec(&score).expect("score serializes");
    let score_sha256 = sha256_hex(&score_bytes);

    let actual_voices = perform_with_spec(&score, &spec, seed, state);
    let replay_voices = perform_with_spec(&score, &spec, seed, state);
    let control_voices = perform_with_spec_and_dialect(
        &score,
        &spec,
        seed,
        state,
        PerformanceDialect::ClassicalRubato,
    );
    let replay_exact = event_signature(&actual_voices) == event_signature(&replay_voices);
    let actual_metrics = metrics(&actual_voices);
    let control_metrics = metrics(&control_voices);
    let actual_events = flatten(&actual_voices);
    let control_events = flatten(&control_voices);
    let (matched, unmatched_actual, unmatched_control, mut shifts) =
        compare(&actual_events, &control_events);
    let changed_onsets = shifts.iter().filter(|d| **d > 1e-3).count();
    let changed_onset_fraction = (matched > 0).then(|| changed_onsets as f64 / matched as f64);
    let mean_abs_onset_shift_ms = if shifts.is_empty() {
        None
    } else {
        Some(shifts.iter().sum::<f64>() / shifts.len() as f64)
    };
    let median_abs_onset_shift_ms = quantile(&mut shifts, 0.5);
    let p95_abs_onset_shift_ms = quantile(&mut shifts, 0.95);
    let valid = actual_metrics.invalid_events == 0
        && control_metrics.invalid_events == 0
        && actual_metrics.events > 0
        && control_metrics.events > 0;
    let status = if replay_exact && valid && matched > 0 && changed_onsets > 0 {
        "mechanics_pass"
    } else {
        "mechanics_fail"
    };
    CaseReport {
        style: format!("{style:?}"),
        seed,
        score_sha256,
        selected_dialect: format!("{:?}", dialect_for_spec(&spec)),
        control_dialect: "ClassicalRubato".into(),
        replay_exact,
        actual: actual_metrics,
        control: control_metrics,
        matched_events: matched,
        unmatched_actual_events: unmatched_actual,
        unmatched_control_events: unmatched_control,
        changed_onsets,
        changed_onset_fraction,
        mean_abs_onset_shift_ms,
        median_abs_onset_shift_ms,
        p95_abs_onset_shift_ms,
        status,
    }
}

fn summarize(cases: &[CaseReport]) -> Summary {
    let passed_cases = cases.iter().filter(|c| c.status == "mechanics_pass").count();
    let total_actual_events: usize = cases.iter().map(|c| c.actual.events).sum();
    let total_control_events: usize = cases.iter().map(|c| c.control.events).sum();
    let total_matched_events: usize = cases.iter().map(|c| c.matched_events).sum();
    let total_changed_onsets: usize = cases.iter().map(|c| c.changed_onsets).sum();
    let mut shifts: Vec<f64> = cases.iter()
        .flat_map(|c| c.median_abs_onset_shift_ms)
        .collect();
    Summary {
        total_cases: cases.len(),
        passed_cases,
        failed_cases: cases.len().saturating_sub(passed_cases),
        total_actual_events,
        total_control_events,
        total_matched_events,
        total_changed_onsets,
        matched_event_fraction: (total_actual_events > 0)
            .then(|| total_matched_events as f64 / total_actual_events as f64),
        changed_onset_fraction: (total_matched_events > 0)
            .then(|| total_changed_onsets as f64 / total_matched_events as f64),
        median_abs_onset_shift_ms: quantile(&mut shifts, 0.5),
        p95_abs_onset_shift_ms: quantile(&mut shifts, 0.95),
    }
}

fn main() {
    let mut cases = Vec::new();
    let state = MusicalState::default();
    for style in [Style::Mazurka, Style::Toccata] {
        for seed in SEEDS {
            cases.push(case(style, seed, &state));
        }
    }
    let passed = cases.iter().all(|c| c.status == "mechanics_pass");
    let report = Report {
        schema_version: REPORT_SCHEMA,
        experiment: "melothaea_style_performance_counterfactual_v1",
        status: if passed { "mechanics_pass" } else { "mechanics_fail" },
        source_controls: "same score digest, seed, resolved ensemble rule, swing, return-color policy and MusicalState; actual style-derived dialect versus explicit ClassicalRubato",
        human_quality: "not_measured",
        audio_rendered: false,
        seeds: SEEDS.to_vec(),
        summary: summarize(&cases),
        cases,
    };
    let encoded = serde_json::to_string_pretty(&report).expect("report serializes");
    if let Some(path) = std::env::args().nth(1) {
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).expect("create report directory");
        }
        std::fs::write(&path, format!("{encoded}\n")).expect("write report");
        eprintln!("wrote {}", path.display());
    } else {
        println!("{encoded}");
    }
    if !passed {
        std::process::exit(2);
    }
}
