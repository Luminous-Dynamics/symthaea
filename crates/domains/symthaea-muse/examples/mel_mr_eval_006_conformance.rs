// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-MR-EVAL-006 synthetic conformance runner.
//!
//! Run with:
//!   cargo run --locked -p symthaea-muse --example mel_mr_eval_006_conformance
//!
//! Emits machine-readable JSON and exits unsuccessfully if any deterministic
//! fixture's expected verdict is not reproduced. The calibration ID and
//! thresholds are explicitly fixture-only; this is not production calibration,
//! an end-to-end editor benchmark, or evidence of perceptual equivalence.

use serde_json::{Value, json};
use std::process::ExitCode;
use symthaea_muse::audio_preservation::{
    AudioView, FrameInterval, OnsetPreservationConfig, PreservationConfig,
    PreservationVerdict, audit_onset_preservation, audit_protected_audio,
};

const SAMPLE_RATE_HZ: u32 = 16_000;
const TOTAL_FRAMES: usize = 8_192;
const SYNTHETIC_CALIBRATION_ID: &str =
    "synthetic-conformance-fixture-only-not-production-calibration";

fn sine(frames: usize, frequency_hz: f32) -> Vec<f32> {
    (0..frames)
        .map(|index| {
            (std::f32::consts::TAU * frequency_hz * index as f32
                / SAMPLE_RATE_HZ as f32)
                .sin()
        })
        .collect()
}

fn burst(frames: usize, start: usize, end: usize) -> Vec<f32> {
    (0..frames)
        .map(|index| {
            if (start..end).contains(&index) {
                (std::f32::consts::TAU * 440.0 * index as f32 / SAMPLE_RATE_HZ as f32).sin()
            } else {
                0.0
            }
        })
        .collect()
}

fn view(samples: &[f32]) -> AudioView<'_> {
    AudioView {
        interleaved_samples: samples,
        sample_rate_hz: SAMPLE_RATE_HZ,
        channels: 1,
    }
}

fn acoustic_config() -> PreservationConfig {
    PreservationConfig {
        frame_size: 256,
        hop_size: 128,
        boundary_guard_frames: 0,
        rms_delta_db_tolerance: 0.5,
        spectral_distance_tolerance: 0.08,
        minimum_protected_windows: 4,
        calibration_id: Some(SYNTHETIC_CALIBRATION_ID.to_owned()),
    }
}

fn onset_config() -> OnsetPreservationConfig {
    OnsetPreservationConfig {
        frame_size: 128,
        hop_size: 32,
        novelty_threshold: 0.12,
        minimum_separation_ms: 8,
        onset_time_tolerance_ms: 4,
        boundary_guard_frames: 0,
        minimum_protected_duration_ms: 100,
        calibration_id: Some(SYNTHETIC_CALIBRATION_ID.to_owned()),
    }
}

fn verdict_name(verdict: PreservationVerdict) -> &'static str {
    match verdict {
        PreservationVerdict::Pass => "pass",
        PreservationVerdict::Fail => "fail",
        PreservationVerdict::Abstain => "abstain",
    }
}

fn acoustic_case(
    id: &str,
    description: &str,
    reference: &[f32],
    candidate: &[f32],
    authorized_edits: &[FrameInterval],
    expected: PreservationVerdict,
) -> Value {
    match audit_protected_audio(
        view(reference),
        view(candidate),
        authorized_edits,
        &acoustic_config(),
    ) {
        Ok(report) => {
            let actual = verdict_name(report.verdict);
            json!({
                "id": id,
                "description": description,
                "expected_verdict": verdict_name(expected),
                "actual_verdict": actual,
                "expected_outcome_reproduced": report.verdict == expected,
                "protected_windows": report.protected_windows,
                "audited_channel_windows": report.audited_channel_windows,
                "violating_channel_windows": report.violating_channel_windows,
                "worst_rms_delta_db": report.worst_rms_delta_db,
                "worst_spectral_distance": report.worst_spectral_distance,
                "calibration_id": report.calibration_id,
                "reference_audio_sha256": report.reference_audio_sha256,
                "candidate_audio_sha256": report.candidate_audio_sha256,
                "reasons": report.reasons,
            })
        }
        Err(error) => json!({
            "id": id,
            "description": description,
            "expected_verdict": verdict_name(expected),
            "input_error": error.to_string(),
            "expected_outcome_reproduced": false,
        }),
    }
}

fn onset_case(
    id: &str,
    description: &str,
    reference: &[f32],
    candidate: &[f32],
    authorized_edits: &[FrameInterval],
    expected: PreservationVerdict,
) -> Value {
    match audit_onset_preservation(
        view(reference),
        view(candidate),
        authorized_edits,
        &onset_config(),
    ) {
        Ok(report) => {
            let actual = verdict_name(report.verdict);
            json!({
                "id": id,
                "description": description,
                "expected_verdict": verdict_name(expected),
                "actual_verdict": actual,
                "expected_outcome_reproduced": report.verdict == expected,
                "reference_onset_count": report.reference_onsets.len(),
                "candidate_onset_count": report.candidate_onsets.len(),
                "matched_events": report.matched_events,
                "missing_reference_events": report.missing_reference_events,
                "extra_candidate_events": report.extra_candidate_events,
                "maximum_matched_displacement_frames": report.maximum_matched_displacement_frames,
                "onset_time_tolerance_frames": report.onset_time_tolerance_frames,
                "effective_boundary_guard_frames": report.effective_boundary_guard_frames,
                "calibration_id": report.calibration_id,
                "reference_audio_sha256": report.reference_audio_sha256,
                "candidate_audio_sha256": report.candidate_audio_sha256,
                "reasons": report.reasons,
            })
        }
        Err(error) => json!({
            "id": id,
            "description": description,
            "expected_verdict": verdict_name(expected),
            "input_error": error.to_string(),
            "expected_outcome_reproduced": false,
        }),
    }
}

fn main() -> ExitCode {
    let reference_sine = sine(TOTAL_FRAMES, 440.0);
    let mut local_damage = reference_sine.clone();
    local_damage[1_024..1_280].fill(0.0);

    let mut authorized_damage = reference_sine.clone();
    authorized_damage[4_096..5_120].fill(0.0);
    let authorized_mask = [FrameInterval {
        start_frame: 4_096,
        end_frame: 5_120,
    }];

    let reference_burst = burst(TOTAL_FRAMES, 1_024, 3_072);
    let delayed_burst = burst(TOTAL_FRAMES, 1_536, 3_584);
    let silent = vec![0.0; TOTAL_FRAMES];
    let onset_mask_hiding_all_events = [FrameInterval {
        start_frame: 512,
        end_frame: 4_096,
    }];

    let cases = vec![
        acoustic_case(
            "AP-IDENTITY-001",
            "Identical audio in all protected windows",
            &reference_sine,
            &reference_sine,
            &authorized_mask,
            PreservationVerdict::Pass,
        ),
        acoustic_case(
            "AP-LOCAL-LEAK-001",
            "A zeroed local window outside the authorized edit",
            &reference_sine,
            &local_damage,
            &authorized_mask,
            PreservationVerdict::Fail,
        ),
        acoustic_case(
            "AP-AUTHORIZED-ONLY-001",
            "A changed window fully contained by the authorized mask",
            &reference_sine,
            &authorized_damage,
            &authorized_mask,
            PreservationVerdict::Pass,
        ),
        onset_case(
            "ONSET-IDENTITY-001",
            "Identical transient timing in protected content",
            &reference_burst,
            &reference_burst,
            &[],
            PreservationVerdict::Pass,
        ),
        onset_case(
            "ONSET-DELAY-001",
            "Protected transient is delayed by 512 frames",
            &reference_burst,
            &delayed_burst,
            &[],
            PreservationVerdict::Fail,
        ),
        onset_case(
            "ONSET-MASK-001",
            "The edit mask and analysis guard hide all available onset evidence",
            &reference_burst,
            &delayed_burst,
            &onset_mask_hiding_all_events,
            PreservationVerdict::Abstain,
        ),
        onset_case(
            "ONSET-UNMEASURABLE-001",
            "Both reference and candidate are silent",
            &silent,
            &silent,
            &[],
            PreservationVerdict::Abstain,
        ),
        onset_case(
            "ONSET-MISSING-001",
            "Candidate removes the reference burst in protected content",
            &reference_burst,
            &silent,
            &[],
            PreservationVerdict::Fail,
        ),
    ];

    let passed = cases
        .iter()
        .filter(|case| case["expected_outcome_reproduced"] == Value::Bool(true))
        .count();
    let failed = cases.len() - passed;

    let output = json!({
        "benchmark": "MEL-MR-EVAL-006",
        "runner_version": "synthetic-conformance-v0.1",
        "fixture_only": true,
        "calibration_id_is_production_calibration": false,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "channels": 1,
        "total_frames_per_case": TOTAL_FRAMES,
        "cases_total": cases.len(),
        "cases_expected_outcome_reproduced": passed,
        "cases_expected_outcome_not_reproduced": failed,
        "status": if failed == 0 { "conformance-fixtures-reproduced" } else { "conformance-fixtures-failed" },
        "limitations": [
            "synthetic fixtures only; not a human listening study",
            "thresholds are fixture settings and are not production-calibrated",
            "not an MMAE or state-of-the-art head-to-head evaluation",
            "does not prove general perceptual preservation",
        ],
        "cases": cases,
    });

    println!("{}", serde_json::to_string_pretty(&output).expect("serialize report"));
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
