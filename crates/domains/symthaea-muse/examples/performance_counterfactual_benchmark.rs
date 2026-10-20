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

use std::path::PathBuf;
use std::process::Command;

use serde::Serialize;
use sha2::{Digest, Sha256};
use symthaea_muse::theory_realize::{
    PerformanceDialect, PerformedVoice, dialect_for_spec, perform_with_spec,
    perform_with_spec_and_dialect,
};
use symthaea_muse::MusicalState;
use symthaea_music_theory::{MusicalIntent, Style, compose_with_spec};

const SEEDS: [u64; 8] = [0, 1, 7, 13, 29, 42, 73, 101];
const CHANGED_ONSET_EPSILON_MS: f64 = 0.1;
const MINIMUM_EVENT_MATCH_COVERAGE: f64 = 0.90;
const REPORT_SCHEMA: u32 = 1;

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    experiment: &'static str,
    status: &'static str,
    source_controls: &'static str,
    source_revision: Option<String>,
    rustc_version: Option<String>,
    human_quality: &'static str,
    audio_rendered: bool,
    seeds: Vec<u64>,
    minimum_event_match_coverage: f64,
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
    negative_control_replay_exact: bool,
    negative_control_changed_onsets: usize,
    negative_control_unmatched_events: usize,
    actual: ArmMetrics,
    control: ArmMetrics,
    matched_events: usize,
    actual_match_fraction: Option<f64>,
    control_match_fraction: Option<f64>,
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
    actual_matched_event_fraction: Option<f64>,
    control_matched_event_fraction: Option<f64>,
    changed_onset_fraction: Option<f64>,
    median_case_median_abs_onset_shift_ms: Option<f64>,
    p95_case_median_abs_onset_shift_ms: Option<f64>,
}

#[derive(Clone)]
struct Event {
    voice: String,
    pitch_bits: u32,
    start: f32,
}

fn source_revision() -> Option<String> {
    std::env::var("MUSE_SOURCE_REVISION")
        .ok()
        .or_else(|| std::env::var("GITHUB_SHA").ok())
}

fn rustc_version() -> Option<String> {
    let output = Command::new("rustc").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok().map(|s| s.trim().to_string())
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

/// Order-preserving longest-common-subsequence alignment, grouped by voice
/// identity and exact frequency. Unlike a fixed timing window, this continues
/// to match the same note sequence when a dialect changes cumulative rubato by
/// more than a fraction of a second. Unmatched events remain visible. Repeated
/// pitches are disambiguated by preserving order; ties are resolved
/// deterministically by preferring the actual-arm predecessor.
fn compare(actual: &[Event], control: &[Event]) -> (usize, usize, usize, Vec<f64>) {
    let mut voice_names: Vec<String> = actual
        .iter()
        .map(|event| event.voice.clone())
        .chain(control.iter().map(|event| event.voice.clone()))
        .collect();
    voice_names.sort_unstable();
    voice_names.dedup();

    let mut shifts = Vec::new();
    let mut unmatched_actual = 0usize;
    let mut unmatched_control = 0usize;

    for voice_name in voice_names {
        let voice = voice_name.as_str();
        let a: Vec<&Event> = actual
            .iter()
            .filter(|event| event.voice.as_str() == voice)
            .collect();
        let b: Vec<&Event> = control
            .iter()
            .filter(|event| event.voice.as_str() == voice)
            .collect();
        let (n, m) = (a.len(), b.len());
        let width = m + 1;
        let mut dp = vec![0usize; (n + 1) * (m + 1)];

        for i in 1..=n {
            for j in 1..=m {
                let here = i * width + j;
                if a[i - 1].pitch_bits == b[j - 1].pitch_bits {
                    dp[here] = dp[(i - 1) * width + (j - 1)] + 1;
                } else {
                    dp[here] = dp[(i - 1) * width + j].max(dp[i * width + (j - 1)]);
                }
            }
        }

        let mut i = n;
        let mut j = m;
        let mut matched_in_voice = 0usize;
        while i > 0 && j > 0 {
            let here = dp[i * width + j];
            let diagonal = dp[(i - 1) * width + (j - 1)];
            if a[i - 1].pitch_bits == b[j - 1].pitch_bits && here == diagonal + 1 {
                shifts.push((a[i - 1].start - b[j - 1].start).abs() as f64 * 1000.0);
                matched_in_voice += 1;
                i -= 1;
                j -= 1;
            } else {
                let skip_actual = dp[(i - 1) * width + j];
                let skip_control = dp[i * width + (j - 1)];
                if skip_actual >= skip_control {
                    i -= 1;
                } else {
                    j -= 1;
                }
            }
        }
        unmatched_actual += n - matched_in_voice;
        unmatched_control += m - matched_in_voice;
    }

    shifts.sort_by(f64::total_cmp);
    (shifts.len(), unmatched_actual, unmatched_control, shifts)
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
    let control_replay_voices = perform_with_spec_and_dialect(
        &score,
        &spec,
        seed,
        state,
        PerformanceDialect::ClassicalRubato,
    );
    let replay_exact = event_signature(&actual_voices) == event_signature(&replay_voices);
    let negative_control_replay_exact =
        event_signature(&control_voices) == event_signature(&control_replay_voices);
    let actual_metrics = metrics(&actual_voices);
    let control_metrics = metrics(&control_voices);
    let actual_events = flatten(&actual_voices);
    let control_events = flatten(&control_voices);
    let control_replay_events = flatten(&control_replay_voices);
    let (negative_matched, negative_unmatched_control, negative_unmatched_replay, negative_shifts) =
        compare(&control_events, &control_replay_events);
    let negative_control_changed_onsets = negative_shifts
        .iter()
        .filter(|d| **d > CHANGED_ONSET_EPSILON_MS)
        .count();
    let negative_control_unmatched_events =
        negative_unmatched_control + negative_unmatched_replay;
    let negative_control_match_is_clean = negative_control_replay_exact
        && negative_matched == control_events.len()
        && negative_control_unmatched_events == 0
        && negative_control_changed_onsets == 0;
    let (matched, unmatched_actual, unmatched_control, mut shifts) =
        compare(&actual_events, &control_events);
    let changed_onsets = shifts
        .iter()
        .filter(|d| **d > CHANGED_ONSET_EPSILON_MS)
        .count();
    let changed_onset_fraction = (matched > 0).then(|| changed_onsets as f64 / matched as f64);
    let actual_match_fraction = (actual_metrics.events > 0)
        .then(|| matched as f64 / actual_metrics.events as f64);
    let control_match_fraction = (control_metrics.events > 0)
        .then(|| matched as f64 / control_metrics.events as f64);
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
    let sufficient_coverage = actual_match_fraction.is_some_and(|fraction| fraction >= MINIMUM_EVENT_MATCH_COVERAGE)
        && control_match_fraction.is_some_and(|fraction| fraction >= MINIMUM_EVENT_MATCH_COVERAGE);
    let status = if replay_exact && negative_control_match_is_clean
        && valid && sufficient_coverage && matched > 0 && changed_onsets > 0 {
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
        negative_control_replay_exact,
        negative_control_changed_onsets,
        negative_control_unmatched_events,
        actual: actual_metrics,
        control: control_metrics,
        matched_events: matched,
        actual_match_fraction,
        control_match_fraction,
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
    let total_control_events_matched: usize = cases.iter().map(|c| c.matched_events).sum();
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
        actual_matched_event_fraction: (total_actual_events > 0)
            .then(|| total_matched_events as f64 / total_actual_events as f64),
        control_matched_event_fraction: (total_control_events > 0)
            .then(|| total_control_events_matched as f64 / total_control_events as f64),
        changed_onset_fraction: (total_matched_events > 0)
            .then(|| total_changed_onsets as f64 / total_matched_events as f64),
        median_case_median_abs_onset_shift_ms: quantile(&mut shifts, 0.5),
        p95_case_median_abs_onset_shift_ms: quantile(&mut shifts, 0.95),
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
        source_revision: source_revision(),
        rustc_version: rustc_version(),
        human_quality: "not_measured",
        audio_rendered: false,
        seeds: SEEDS.to_vec(),
        minimum_event_match_coverage: MINIMUM_EVENT_MATCH_COVERAGE,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_matcher_requires_same_voice_and_pitch_and_never_reuses_events() {
        let actual = [
            Event { voice: "Melody".into(), pitch_bits: 440.0_f32.to_bits(), start: 0.010 },
            Event { voice: "Melody".into(), pitch_bits: 880.0_f32.to_bits(), start: 1.000 },
        ];
        let control = [
            Event { voice: "Melody".into(), pitch_bits: 440.0_f32.to_bits(), start: 0.012 },
            Event { voice: "Bass".into(), pitch_bits: 440.0_f32.to_bits(), start: 0.013 },
        ];
        let (matched, unmatched_actual, unmatched_control, shifts) = compare(&actual, &control);
        assert_eq!(matched, 1);
        assert_eq!(unmatched_actual, 1);
        assert_eq!(unmatched_control, 1);
        assert!((shifts[0] - 2.0).abs() < 1e-3, "{shifts:?}");
    }

    #[test]
    fn ordered_alignment_keeps_notes_with_large_rubato_shifts() {
        let actual = [
            Event { voice: "Melody".into(), pitch_bits: 440.0_f32.to_bits(), start: 0.0 },
            Event { voice: "Melody".into(), pitch_bits: 660.0_f32.to_bits(), start: 1.0 },
            Event { voice: "Melody".into(), pitch_bits: 440.0_f32.to_bits(), start: 2.0 },
        ];
        let control = [
            Event { voice: "Melody".into(), pitch_bits: 440.0_f32.to_bits(), start: 1.0 },
            Event { voice: "Melody".into(), pitch_bits: 660.0_f32.to_bits(), start: 2.0 },
            Event { voice: "Melody".into(), pitch_bits: 440.0_f32.to_bits(), start: 3.0 },
        ];
        let (matched, unmatched_a, unmatched_b, shifts) = compare(&actual, &control);
        assert_eq!(matched, 3);
        assert_eq!(unmatched_a, 0);
        assert_eq!(unmatched_b, 0);
        assert_eq!(shifts, vec![1000.0; 3]);
    }

    #[test]
    fn identical_event_stream_is_a_zero_effect_negative_control() {
        let stream = [
            Event { voice: "Melody".into(), pitch_bits: 440.0_f32.to_bits(), start: 0.010 },
            Event { voice: "Melody".into(), pitch_bits: 440.0_f32.to_bits(), start: 0.510 },
            Event { voice: "Bass".into(), pitch_bits: 220.0_f32.to_bits(), start: 0.010 },
        ];
        let (matched, unmatched_a, unmatched_b, shifts) = compare(&stream, &stream);
        assert_eq!(matched, stream.len());
        assert_eq!(unmatched_a, 0);
        assert_eq!(unmatched_b, 0);
        assert!(shifts.iter().all(|shift| *shift == 0.0), "{shifts:?}");
    }

    #[test]
    fn quantile_is_explicit_for_empty_samples_and_uses_sorted_values() {
        assert_eq!(quantile(&mut [], 0.5), None);
        let mut values = [9.0, 1.0, 5.0];
        assert_eq!(quantile(&mut values, 0.5), Some(5.0));
    }
}
