// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Run waveform-integrity controls against real Muse rendering and synthetic faults.
//!
//! Usage:
//! cargo run -p symthaea-muse --example creative_audio_audit -- [source-revision-claim]

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::error::Error;
use symthaea_muse::{
    AudioData, MusicalState, Note,
    creative_audio_audit::{
        CreativeAudioAuditConfigV1, CreativeAudioAuditError, CreativeAudioAuditV1,
        audit_creative_audio,
    },
    synth::render_notes,
};

const SAMPLE_RATE: u32 = 44_100;
const DURATION_SECS: usize = 4;
const FRAME_COUNT: usize = SAMPLE_RATE as usize * DURATION_SECS;

#[derive(Debug, Clone, Copy)]
struct Expectations {
    effectively_silent: Option<bool>,
    non_finite_sample_count: Option<usize>,
    out_of_range_sample_count: Option<usize>,
}

#[derive(Debug, Serialize)]
struct AudioCaseReceipt {
    case_id: String,
    description: String,
    sample_representation_sha256: String,
    expected_effectively_silent: Option<bool>,
    expected_non_finite_sample_count: Option<usize>,
    expected_out_of_range_sample_count: Option<usize>,
    expectations_match: bool,
    outcome: AudioAuditOutcome,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", content = "value", rename_all = "snake_case")]
enum AudioAuditOutcome {
    Measured(CreativeAudioAuditV1),
    Error(CreativeAudioAuditError),
}

#[derive(Debug, Serialize)]
struct CreativeAudioAuditReceipt {
    schema_version: u32,
    source_revision_claim: Option<String>,
    renderer: &'static str,
    sample_rate_hz: u32,
    frame_count: usize,
    config: CreativeAudioAuditConfigV1,
    expectation_mismatches: Vec<String>,
    cases: Vec<AudioCaseReceipt>,
}

fn audit_case(
    case_id: &str,
    description: &str,
    audio: AudioData,
    expectations: Expectations,
    config: CreativeAudioAuditConfigV1,
) -> AudioCaseReceipt {
    let sample_representation_sha256 = audio_sha256(&audio);
    let measured = audit_creative_audio(&audio, SAMPLE_RATE, config);
    let expectations_match = match &measured {
        Ok(report) => {
            expectations
                .effectively_silent
                .is_none_or(|expected| expected == report.effectively_silent)
                && expectations
                    .non_finite_sample_count
                    .is_none_or(|expected| expected == report.non_finite_sample_count)
                && expectations
                    .out_of_range_sample_count
                    .is_none_or(|expected| expected == report.out_of_range_sample_count)
        }
        Err(_) => false,
    };
    let outcome = match measured {
        Ok(report) => AudioAuditOutcome::Measured(report),
        Err(error) => AudioAuditOutcome::Error(error),
    };

    AudioCaseReceipt {
        case_id: case_id.to_owned(),
        description: description.to_owned(),
        sample_representation_sha256,
        expected_effectively_silent: expectations.effectively_silent,
        expected_non_finite_sample_count: expectations.non_finite_sample_count,
        expected_out_of_range_sample_count: expectations.out_of_range_sample_count,
        expectations_match,
        outcome,
    }
}

/// Hash exact source representation, including an encoding tag. This is not a
/// hash of exported WAV bytes; it binds the audit to the exact AudioData values.
fn audio_sha256(audio: &AudioData) -> String {
    let mut hasher = Sha256::new();
    match audio {
        AudioData::I16(samples) => {
            hasher.update(b"mono_i16_le\0");
            for sample in samples {
                hasher.update(sample.to_le_bytes());
            }
        }
        AudioData::F32(samples) => {
            hasher.update(b"mono_f32_bits_le\0");
            for sample in samples {
                hasher.update(sample.to_bits().to_le_bytes());
            }
        }
        AudioData::StereoF32(samples) => {
            hasher.update(b"stereo_f32_bits_le\0");
            for [left, right] in samples {
                hasher.update(left.to_bits().to_le_bytes());
                hasher.update(right.to_bits().to_le_bytes());
            }
        }
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let source_revision_claim = std::env::args().nth(1);
    let config = CreativeAudioAuditConfigV1::default();
    let state = MusicalState::default();

    let phrase: Vec<Note> = [
        261.63_f32, 293.66, 329.63, 349.23, 392.0, 349.23, 329.63, 261.63,
    ]
    .into_iter()
    .enumerate()
    .map(|(index, frequency)| Note {
        frequency,
        start_time: index as f32 * 0.5,
        duration: 0.4,
        velocity: 0.7,
    })
    .collect();
    let rendered_phrase = render_notes(&phrase, SAMPLE_RATE, FRAME_COUNT, &state);

    let empty_render = render_notes(&[], SAMPLE_RATE, FRAME_COUNT, &state);

    let silent_note = [Note {
        frequency: 440.0,
        start_time: 0.0,
        duration: 1.0,
        velocity: 0.0,
    }];
    let zero_velocity_render = render_notes(&silent_note, SAMPLE_RATE, FRAME_COUNT, &state);

    let corrupted_signal = AudioData::F32(vec![0.0, 0.5, 1.1, f32::NAN]);

    let cases = vec![
        audit_case(
            "rendered-phrase",
            "Nonzero phrase rendered by the real Muse synthesis path",
            rendered_phrase,
            Expectations {
                effectively_silent: Some(false),
                non_finite_sample_count: Some(0),
                out_of_range_sample_count: Some(0),
            },
            config,
        ),
        audit_case(
            "empty-render",
            "Empty score rendered as a full-length output buffer",
            empty_render,
            Expectations {
                effectively_silent: Some(true),
                non_finite_sample_count: Some(0),
                out_of_range_sample_count: Some(0),
            },
            config,
        ),
        audit_case(
            "zero-velocity-render",
            "A zero-velocity note must not create audible output",
            zero_velocity_render,
            Expectations {
                effectively_silent: Some(true),
                non_finite_sample_count: Some(0),
                out_of_range_sample_count: Some(0),
            },
            config,
        ),
        audit_case(
            "synthetic-corruption-control",
            "Injected NaN and greater-than-full-scale sample test the detector",
            corrupted_signal,
            Expectations {
                effectively_silent: Some(false),
                non_finite_sample_count: Some(1),
                out_of_range_sample_count: Some(1),
            },
            config,
        ),
    ];

    let expectation_mismatches: Vec<String> = cases
        .iter()
        .filter(|case| !case.expectations_match)
        .map(|case| case.case_id.clone())
        .collect();
    let receipt = CreativeAudioAuditReceipt {
        schema_version: 1,
        source_revision_claim,
        renderer: "symthaea_muse::synth::render_notes",
        sample_rate_hz: SAMPLE_RATE,
        frame_count: FRAME_COUNT,
        config,
        expectation_mismatches,
        cases,
    };
    println!("{}", serde_json::to_string_pretty(&receipt)?);

    if !receipt.expectation_mismatches.is_empty() {
        return Err(format!(
            "audio integrity expectations failed for: {}",
            receipt.expectation_mismatches.join(", ")
        )
        .into());
    }
    Ok(())
}
