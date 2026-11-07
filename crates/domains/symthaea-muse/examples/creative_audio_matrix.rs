// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Metamorphic reproducibility and waveform-integrity matrix for Muse.
//!
//! Runs the same seeded creative composition over several sample rates and
//! output encodings, repeats each render, and binds the JSON receipt to the
//! actual Git checkout revision.
//!
//! Usage:
//! cargo run --locked -p symthaea-muse --example creative_audio_matrix -- [expected-git-sha]

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{error::Error, process::Command};
use symthaea_muse::{
    AudioData, MuseConfig, MusicalState, Note, OutputFormat, compose,
    creative_audio_audit::{
        CreativeAudioAuditConfigV1, CreativeAudioAuditError, CreativeAudioAuditV1,
        audit_creative_audio,
    },
};

const SAMPLE_RATES_HZ: [u32; 4] = [22_050, 44_100, 48_000, 96_000];
const OUTPUT_FORMATS: [OutputFormat; 3] = [
    OutputFormat::Mono16,
    OutputFormat::MonoF32,
    OutputFormat::StereoF32,
];
const SEED: u64 = 0x4d55_5345_5f41_5544;
const DURATION_SECS: f32 = 4.0;

#[derive(Debug, Serialize)]
struct AudioMatrixCase {
    case_id: String,
    sample_rate_hz: u32,
    output_format: String,
    duration_secs: f32,
    frame_count: usize,
    note_count: usize,
    renderer_config_sha256: String,
    symbolic_notes_sha256: String,
    symbolic_notes_match_reference: bool,
    audio_representation_sha256: String,
    repeat_audio_representation_sha256: String,
    repeat_render_deterministic: bool,
    expectations_match: bool,
    outcome: AudioMatrixAuditOutcome,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", content = "value", rename_all = "snake_case")]
enum AudioMatrixAuditOutcome {
    Measured(CreativeAudioAuditV1),
    Error(CreativeAudioAuditError),
}

#[derive(Debug, Serialize)]
struct AudioMatrixReceipt {
    schema_version: u32,
    source_revision: String,
    expected_source_revision: Option<String>,
    worktree_clean: bool,
    rustc_version: Option<String>,
    host_os: &'static str,
    host_arch: &'static str,
    state_sha256: String,
    renderer: &'static str,
    seed: u64,
    duration_secs: f32,
    sample_rates_hz: Vec<u32>,
    output_formats: Vec<String>,
    reference_symbolic_notes_sha256: Option<String>,
    expectation_mismatches: Vec<String>,
    cases: Vec<AudioMatrixCase>,
}

/// Read the actual checkout identity instead of treating caller input as proof.
fn git_revision() -> Result<String, Box<dyn Error>> {
    let output = Command::new("git")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["rev-parse", "HEAD"])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "git rev-parse HEAD failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    let revision = String::from_utf8(output.stdout)?.trim().to_ascii_lowercase();
    if revision.is_empty() || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("git rev-parse HEAD returned a malformed revision".into());
    }
    Ok(revision)
}

fn git_worktree_clean() -> Result<bool, Box<dyn Error>> {
    let output = Command::new("git")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(["status", "--porcelain", "--untracked-files=all"])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "git status failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(output.stdout.is_empty())
}

fn rustc_version() -> Option<String> {
    let output = Command::new("rustc").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_owned())
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Digest the exact typed AudioData representation, not exported WAV bytes.
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
    hex_sha256(&hasher.finalize())
}

/// Hash float bit patterns and order; no decimal formatting or lossy JSON.
fn notes_sha256(notes: &[Note]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"muse-symbolic-notes-v1\0");
    hasher.update((notes.len() as u64).to_le_bytes());
    for note in notes {
        hasher.update(note.frequency.to_bits().to_le_bytes());
        hasher.update(note.start_time.to_bits().to_le_bytes());
        hasher.update(note.duration.to_bits().to_le_bytes());
        hasher.update(note.velocity.to_bits().to_le_bytes());
    }
    hex_sha256(&hasher.finalize())
}

fn main() -> Result<(), Box<dyn Error>> {
    let expected_source_revision = std::env::args().nth(1);
    let source_revision = git_revision()?;
    let worktree_clean = git_worktree_clean()?;
    let compiler_version = rustc_version();
    let state = MusicalState {
        consciousness_level: 0.8,
        arousal: 0.35,
        ..MusicalState::default()
    };
    let state_sha256 = hex_sha256(&serde_json::to_vec(&state)?);
    let audit_config = CreativeAudioAuditConfigV1::default();
    let mut cases = Vec::new();
    let mut expectation_mismatches = Vec::new();

    if !worktree_clean {
        expectation_mismatches.push(
            "source checkout is dirty (tracked or untracked changes exist)".to_owned(),
        );
    }
    if compiler_version.is_none() {
        expectation_mismatches.push("rustc --version could not be recorded".to_owned());
    }

    if let Some(expected) = expected_source_revision.as_deref() {
        if expected.to_ascii_lowercase() != source_revision {
            expectation_mismatches.push(format!(
                "source revision mismatch: expected {}, actual {}",
                expected, source_revision
            ));
        }
    }

    for output_format in OUTPUT_FORMATS {
        for sample_rate_hz in SAMPLE_RATES_HZ {
            let config = MuseConfig {
                sample_rate: sample_rate_hz,
                duration_secs: DURATION_SECS,
                max_notes: 12,
                output_format,
                ..MuseConfig::default()
            };
            let config_bytes = serde_json::to_vec(&config)?;
            let renderer_config_sha256 = hex_sha256(&config_bytes);

            let first = compose(&config, &state, SEED);
            let symbolic_notes_sha256 = notes_sha256(&first.notes);
            let audio_representation_sha256 = audio_sha256(&first.audio);
            let frame_count = first.audio.len();
            let first_audit = audit_creative_audio(
                &first.audio,
                first.sample_rate,
                audit_config,
            );

            // Repeat exactly the same inputs to make determinism an executable
            // property, not an assumption inferred from a fixed RNG seed.
            let repeated = compose(&config, &state, SEED);
            let repeat_symbolic_sha256 = notes_sha256(&repeated.notes);
            let repeat_audio_representation_sha256 = audio_sha256(&repeated.audio);
            let repeat_render_deterministic = symbolic_notes_sha256 == repeat_symbolic_sha256
                && audio_representation_sha256 == repeat_audio_representation_sha256;

            let (expectations_match, outcome) = match first_audit {
                Ok(report) => {
                    let matches = report.non_finite_sample_count == 0
                        && report.out_of_range_sample_count == 0
                        && !report.effectively_silent
                        && repeat_render_deterministic
                        && frame_count == (DURATION_SECS * sample_rate_hz as f32) as usize;
                    (matches, AudioMatrixAuditOutcome::Measured(report))
                }
                Err(error) => (false, AudioMatrixAuditOutcome::Error(error)),
            };

            let case_id = format!("{:?}-{}hz", output_format, sample_rate_hz);
            if !expectations_match {
                expectation_mismatches.push(format!(
                    "{case_id}: waveform, frame-count, or repeat-determinism expectation failed"
                ));
            }
            cases.push(AudioMatrixCase {
                case_id,
                sample_rate_hz,
                output_format: format!("{output_format:?}"),
                duration_secs: first.duration_secs,
                frame_count,
                note_count: first.notes.len(),
                renderer_config_sha256,
                symbolic_notes_sha256,
                symbolic_notes_match_reference: false,
                audio_representation_sha256,
                repeat_audio_representation_sha256,
                repeat_render_deterministic,
                expectations_match,
                outcome,
            });
        }
    }

    let reference_symbolic_notes_sha256 = cases.first().map(|case| case.symbolic_notes_sha256.clone());
    if let Some(reference) = reference_symbolic_notes_sha256.as_deref() {
        for case in &mut cases {
            case.symbolic_notes_match_reference = case.symbolic_notes_sha256 == reference;
            if !case.symbolic_notes_match_reference {
                expectation_mismatches.push(format!(
                    "{}: symbolic note sequence changed across output configuration",
                    case.case_id
                ));
            }
        }
    }

    let receipt = AudioMatrixReceipt {
        schema_version: 1,
        source_revision,
        expected_source_revision,
        worktree_clean,
        rustc_version: compiler_version,
        host_os: std::env::consts::OS,
        host_arch: std::env::consts::ARCH,
        state_sha256,
        renderer: "symthaea_muse::compose -> synth::render_arrangement",
        seed: SEED,
        duration_secs: DURATION_SECS,
        sample_rates_hz: SAMPLE_RATES_HZ.to_vec(),
        output_formats: OUTPUT_FORMATS
            .iter()
            .map(|format| format!("{format:?}"))
            .collect(),
        reference_symbolic_notes_sha256,
        expectation_mismatches,
        cases,
    };
    println!("{}", serde_json::to_string_pretty(&receipt)?);

    if !receipt.expectation_mismatches.is_empty() {
        return Err(format!(
            "creative audio matrix failed: {}",
            receipt.expectation_mismatches.join("; ")
        )
        .into());
    }
    Ok(())
}
