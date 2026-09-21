// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//!
//! Execution harness for the frozen ProgSuite contextual-harmony 64-subject
//! lockbox. This file defines no new metric or scientific policy. It wires the
//! already-frozen symbolic comparison, native PCM, section localization,
//! log-mel, pitch-class, symbolic↔acoustic tonal evidence, and streaming panel
//! admission into one memory-bounded execution path.
//!
//! IMPORTANT: this example is an execution subject, not an authorization
//! mechanism. A later wrapper must run the closed-world qualification receipt
//! verifier for exact streaming-library subject d8883413... before invoking
//! this binary. The lightweight receipt checks here are defense-in-depth only.

use serde::Serialize;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use symthaea_muse::evidence_digest::{canonical_json_bytes, canonical_json_sha256, sha256_hex};
use symthaea_muse::evidence_digest::prog_suite_contextual_harmony_audio_evidence::
    measure_prog_suite_native_audio_survival;
use symthaea_muse::evidence_digest::prog_suite_contextual_harmony_audio_protocol::
    predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1;
use symthaea_muse::evidence_digest::prog_suite_contextual_harmony_section_audio::
    measure_prog_suite_section_audio_localization;
use symthaea_muse::evidence_digest::prog_suite_contextual_harmony_section_pitch_class::
    measure_prog_suite_pitch_class_localization;
use symthaea_muse::evidence_digest::prog_suite_contextual_harmony_section_spectral::
    measure_prog_suite_section_spectral_localization;
use symthaea_muse::evidence_digest::prog_suite_contextual_harmony_symbolic_acoustic_tonal::
    measure_prog_suite_symbolic_acoustic_tonal_evidence;
use symthaea_muse::evidence_digest::prog_suite_contextual_harmony_tonal_survival_panel::
    ProgSuiteTonalPanelSubjectInputV1;
use symthaea_muse::evidence_digest::prog_suite_contextual_harmony_tonal_survival_stream::
    ProgSuiteTonalSurvivalPanelStreamBuilderV1;
use symthaea_muse::theory_realize::realize_with_spec;
use symthaea_muse::{AudioData, Composition};
use symthaea_music_theory::prog_suite::plan_prog_suite;
use symthaea_music_theory::prog_suite_contextual_harmony_comparison::
    derive_prog_suite_contextual_harmony_comparison;
use symthaea_music_theory::prog_suite_development_context::
    derive_prog_suite_development_context;

const EXECUTION_ACK: &str = "EXECUTE_FROZEN_64_SUBJECT_LOCKBOX";
const QUALIFIED_STREAM_SUBJECT_SHA: &str = "d888341323b6ee463007cf90d8032153039ec099";
const RUN_SCHEMA: &str = "melothaea-prog-suite-contextual-harmony-lockbox-execution-v1";
const RUN_START_SCHEMA: &str = "melothaea-prog-suite-contextual-harmony-lockbox-run-start-v1";
const RUNNER_SOURCE: &[u8] = include_bytes!("prog_suite_contextual_harmony_lockbox_runner.rs");

#[derive(Debug, Clone, Serialize)]
struct SubjectArtifactReceipt {
    subject_index: usize,
    subject_id: String,
    motif_id: String,
    plan_seed: u64,
    intent_seed: u64,
    comparison_sha256: String,
    tonal_evidence_sha256: String,
    source_score_sha256: String,
    contextual_score_sha256: String,
    source_waveform_sha256: String,
    contextual_waveform_sha256: String,
    audio_outcome: String,
}

#[derive(Debug, Serialize)]
struct RunStartManifest {
    schema: &'static str,
    qualified_stream_subject_sha: &'static str,
    qualification_receipt_sha256: String,
    qualification_provider: String,
    runner_source_sha256: String,
    audio_protocol_sha256: String,
    intended_subject_count: usize,
    raw_pcm_persistence_policy: &'static str,
    authority_scope: &'static str,
}

#[derive(Debug, Serialize)]
struct RunManifest {
    schema: &'static str,
    qualified_stream_subject_sha: &'static str,
    qualification_receipt_sha256: String,
    qualification_provider: String,
    runner_source_sha256: String,
    audio_protocol_sha256: String,
    subject_count: usize,
    subject_artifacts: Vec<SubjectArtifactReceipt>,
    tonal_panel_sha256: String,
    raw_pcm_persisted: bool,
    primary_unit: &'static str,
    scientific_scope: &'static str,
    perceptual_authority: &'static str,
    artistic_quality_authority: &'static str,
    product_authority: &'static str,
}

fn io_debug(label: &'static str, error: impl std::fmt::Debug) -> io::Error {
    io::Error::other(format!("{label}: {error:?}"))
}

fn require_execution_ack() -> io::Result<()> {
    match env::var("MEL_CONTEXTUAL_HARMONY_LOCKBOX_EXECUTION") {
        Ok(value) if value == EXECUTION_ACK => Ok(()),
        _ => Err(io::Error::other(format!(
            "lockbox execution is gated; set MEL_CONTEXTUAL_HARMONY_LOCKBOX_EXECUTION={EXECUTION_ACK} only after exact engineering qualification"
        ))),
    }
}

fn parse_args() -> io::Result<PathBuf> {
    let mut args = env::args().skip(1);
    let mut output = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| io::Error::other("--output-dir requires a path"))?;
                output = Some(PathBuf::from(value));
            }
            other => {
                return Err(io::Error::other(format!("unknown argument: {other}")));
            }
        }
    }
    output.ok_or_else(|| io::Error::other("required: --output-dir <fresh-directory>"))
}

fn prepare_output_dir(path: &Path) -> io::Result<()> {
    if path.exists() {
        let mut entries = fs::read_dir(path)?;
        if entries.next().transpose()?.is_some() {
            return Err(io::Error::other(
                "output directory must be absent or empty; lockbox artifacts are never overwritten",
            ));
        }
    } else {
        fs::create_dir_all(path)?;
    }
    fs::create_dir_all(path.join("subjects"))?;
    Ok(())
}

fn unique_receipt_value<'a>(text: &'a str, key: &str) -> io::Result<&'a str> {
    let values = text
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .filter_map(|(found, value)| (found == key).then_some(value))
        .collect::<Vec<_>>();
    match values.as_slice() {
        [value] if !value.is_empty() => Ok(*value),
        [] => Err(io::Error::other(format!("qualification receipt missing {key}"))),
        _ => Err(io::Error::other(format!(
            "qualification receipt has duplicate {key}"
        ))),
    }
}

fn bind_qualification_receipt() -> io::Result<(String, String)> {
    let path = PathBuf::from(
        env::var("MEL_STREAM_QUAL_RECEIPT")
            .map_err(|_| io::Error::other("MEL_STREAM_QUAL_RECEIPT is required"))?,
    );
    let expected_digest = env::var("MEL_STREAM_QUAL_RECEIPT_SHA256")
        .map_err(|_| io::Error::other("MEL_STREAM_QUAL_RECEIPT_SHA256 is required"))?;
    if expected_digest.len() != 64
        || !expected_digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(io::Error::other(
            "MEL_STREAM_QUAL_RECEIPT_SHA256 must be canonical lowercase SHA-256",
        ));
    }
    let bytes = fs::read(&path)?;
    let actual_digest = sha256_hex(&bytes);
    if actual_digest != expected_digest {
        return Err(io::Error::other("qualification receipt SHA-256 mismatch"));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| io::Error::other("qualification receipt is not UTF-8"))?;
    for (key, expected) in [
        ("schema", "melothaea-tonal-survival-stream-qualification-v1"),
        ("qualifier_id", "melothaea-tonal-survival-stream-qualification-v1"),
        ("status", "PASS"),
        ("exit_code", "0"),
        ("terminal_stage", "none"),
        ("authority_scope", "engineering-software-contract-only"),
        ("scientific_lockbox_execution", "not-performed"),
        ("human_perceptual_authority", "none"),
        ("artistic_quality_authority", "none"),
        ("product_authority", "none"),
        ("subject_sha", QUALIFIED_STREAM_SUBJECT_SHA),
        ("source_state", "clean-exact-subject-checkout-postflight"),
    ] {
        if unique_receipt_value(text, key)? != expected {
            return Err(io::Error::other(format!(
                "qualification receipt field {key} is not canonical"
            )));
        }
    }
    let provider = unique_receipt_value(text, "qualification_provider")?.to_string();
    if provider != "local" && provider != "github-actions" {
        return Err(io::Error::other("unsupported qualification provider"));
    }
    Ok((actual_digest, provider))
}

fn canonical_write<T: Serialize>(path: &Path, value: &T) -> io::Result<String> {
    let bytes = canonical_json_bytes(value).map_err(|e| io_debug("canonical JSON", e))?;
    let digest = sha256_hex(&bytes);
    fs::write(path, bytes)?;
    Ok(digest)
}

fn into_stereo(composition: Composition, label: &str) -> io::Result<Vec<[f32; 2]>> {
    if composition.sample_rate == 0 {
        return Err(io::Error::other(format!("{label}: zero sample rate")));
    }
    match composition.audio {
        AudioData::StereoF32(frames) if !frames.is_empty() => Ok(frames),
        AudioData::StereoF32(_) => Err(io::Error::other(format!("{label}: empty StereoF32"))),
        AudioData::I16(_) | AudioData::F32(_) => Err(io::Error::other(format!(
            "{label}: renderer returned non-StereoF32 output"
        ))),
    }
}

fn main() -> io::Result<()> {
    require_execution_ack()?;
    let output_dir = parse_args()?;
    prepare_output_dir(&output_dir)?;
    let (qualification_receipt_sha256, qualification_provider) =
        bind_qualification_receipt()?;
    let runner_source_sha256 = sha256_hex(RUNNER_SOURCE);

    let protocol = predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1()
        .map_err(|e| io_debug("audio protocol", e))?;
    protocol
        .validate()
        .map_err(|e| io_debug("audio protocol validation", e))?;
    let protocol_sha256 = canonical_write(&output_dir.join("audio-protocol.json"), &protocol)?;
    let run_start = RunStartManifest {
        schema: RUN_START_SCHEMA,
        qualified_stream_subject_sha: QUALIFIED_STREAM_SUBJECT_SHA,
        qualification_receipt_sha256: qualification_receipt_sha256.clone(),
        qualification_provider: qualification_provider.clone(),
        runner_source_sha256: runner_source_sha256.clone(),
        audio_protocol_sha256: protocol_sha256.clone(),
        intended_subject_count: protocol.source_lockbox.subjects.len(),
        raw_pcm_persistence_policy: "never-persist-raw-pcm-drop-after-each-subject",
        authority_scope: "frozen-lockbox-machine-evidence-execution",
    };
    canonical_write(&output_dir.join("run-start.json"), &run_start)?;

    let mut stream = ProgSuiteTonalSurvivalPanelStreamBuilderV1::new(&protocol)
        .map_err(|e| io_debug("stream builder", e))?;
    let mut subject_artifacts = Vec::with_capacity(protocol.source_lockbox.subjects.len());

    for subject in &protocol.source_lockbox.subjects {
        let motif = protocol
            .source_lockbox
            .motifs
            .iter()
            .find(|motif| motif.motif_id == subject.motif_id)
            .ok_or_else(|| io::Error::other("canonical subject motif missing"))?;
        let intent = protocol
            .source_lockbox
            .intent_template
            .materialize(subject.intent_seed);
        let plan = plan_prog_suite(
            protocol.source_lockbox.home_key,
            protocol.source_lockbox.tempo_bpm,
            subject.plan_seed,
            &protocol.source_lockbox.spec,
        )
        .map_err(|e| io_debug("native plan", e))?;
        let context = derive_prog_suite_development_context(&plan)
            .map_err(|e| io_debug("development context", e))?;
        let comparison = derive_prog_suite_contextual_harmony_comparison(
            &context,
            &motif.motif,
            &intent,
            protocol.source_lockbox.profile,
        )
        .map_err(|e| io_debug("symbolic comparison", e))?;
        comparison
            .validate()
            .map_err(|e| io_debug("symbolic comparison validation", e))?;

        let source_score_sha256 = canonical_json_sha256(&comparison.source_realization.score)
            .map_err(|e| io_debug("source score digest", e))?;
        let contextual_score_sha256 =
            canonical_json_sha256(&comparison.contextual_realization.score)
                .map_err(|e| io_debug("contextual score digest", e))?;

        let render = |score| {
            into_stereo(
                realize_with_spec(
                    score,
                    &protocol.source_lockbox.spec,
                    subject.plan_seed,
                    &protocol.render_policy.render_state,
                    protocol.render_policy.sample_rate,
                ),
                "native render",
            )
        };
        let source_render_a = render(&comparison.source_realization.score)?;
        let source_render_b = render(&comparison.source_realization.score)?;
        let contextual_render_a = render(&comparison.contextual_realization.score)?;
        let contextual_render_b = render(&comparison.contextual_realization.score)?;

        let subject_id = format!("{}:seed-{}", subject.motif_id, subject.plan_seed);
        let whole = measure_prog_suite_native_audio_survival(
            &subject_id,
            &subject.motif_id,
            subject.plan_seed,
            subject.intent_seed,
            &source_score_sha256,
            &contextual_score_sha256,
            &source_render_a,
            &source_render_b,
            &contextual_render_a,
            &contextual_render_b,
        )
        .map_err(|e| io_debug("whole-work PCM evidence", e))?;
        let section_audio = measure_prog_suite_section_audio_localization(
            &protocol,
            subject.subject_index,
            &comparison,
            &whole,
            &source_render_a,
            &source_render_b,
            &contextual_render_a,
            &contextual_render_b,
        )
        .map_err(|e| io_debug("section PCM localization", e))?;
        let spectral = measure_prog_suite_section_spectral_localization(
            &protocol,
            &comparison,
            &section_audio,
            &source_render_a,
            &source_render_b,
            &contextual_render_a,
            &contextual_render_b,
        )
        .map_err(|e| io_debug("section spectral evidence", e))?;
        let pitch_class = measure_prog_suite_pitch_class_localization(
            &protocol,
            &comparison,
            &spectral,
            &source_render_a,
            &source_render_b,
            &contextual_render_a,
            &contextual_render_b,
        )
        .map_err(|e| io_debug("pitch-class evidence", e))?;
        let tonal = measure_prog_suite_symbolic_acoustic_tonal_evidence(
            &protocol,
            &comparison,
            &pitch_class,
            &source_render_a,
            &source_render_b,
            &contextual_render_a,
            &contextual_render_b,
        )
        .map_err(|e| io_debug("symbolic-acoustic tonal evidence", e))?;

        let input = ProgSuiteTonalPanelSubjectInputV1 {
            comparison: &comparison,
            evidence: &tonal,
            source_render_a: &source_render_a,
            source_render_b: &source_render_b,
            contextual_render_a: &contextual_render_a,
            contextual_render_b: &contextual_render_b,
        };
        stream
            .admit_next(&input)
            .map_err(|e| io_debug("stream admission", e))?;

        let subject_dir = output_dir
            .join("subjects")
            .join(format!("{:02}", subject.subject_index));
        fs::create_dir(&subject_dir)?;
        let comparison_sha256 =
            canonical_write(&subject_dir.join("symbolic-comparison.json"), &comparison)?;
        let tonal_evidence_sha256 =
            canonical_write(&subject_dir.join("tonal-evidence.json"), &tonal)?;
        let receipt = SubjectArtifactReceipt {
            subject_index: subject.subject_index,
            subject_id,
            motif_id: subject.motif_id.clone(),
            plan_seed: subject.plan_seed,
            intent_seed: subject.intent_seed,
            comparison_sha256,
            tonal_evidence_sha256,
            source_score_sha256,
            contextual_score_sha256,
            source_waveform_sha256: whole.source_audio.sha256.clone(),
            contextual_waveform_sha256: whole.contextual_audio.sha256.clone(),
            audio_outcome: format!("{:?}", whole.outcome),
        };
        canonical_write(&subject_dir.join("subject-receipt.json"), &receipt)?;
        subject_artifacts.push(receipt);

        // All four raw render buffers and detailed in-memory evidence are
        // dropped at the end of this iteration. Only canonical compact files
        // and the streaming panel projection persist.
        eprintln!(
            "admitted lockbox subject {}/{}",
            stream.admitted_subject_count(),
            protocol.source_lockbox.subjects.len()
        );
    }

    let panel = stream.finish().map_err(|e| io_debug("final panel", e))?;
    let panel_sha256 = canonical_write(&output_dir.join("tonal-survival-panel.json"), &panel)?;

    let manifest = RunManifest {
        schema: RUN_SCHEMA,
        qualified_stream_subject_sha: QUALIFIED_STREAM_SUBJECT_SHA,
        qualification_receipt_sha256,
        qualification_provider,
        runner_source_sha256,
        audio_protocol_sha256: protocol_sha256,
        subject_count: subject_artifacts.len(),
        subject_artifacts,
        tonal_panel_sha256: panel_sha256,
        raw_pcm_persisted: false,
        primary_unit: "motif-seed-subject",
        scientific_scope: "descriptive-frozen-lockbox-machine-evidence-only",
        perceptual_authority: "none",
        artistic_quality_authority: "none",
        product_authority: "none",
    };
    let manifest_sha256 = canonical_write(&output_dir.join("run-manifest.json"), &manifest)?;
    fs::write(
        output_dir.join("run-manifest.sha256"),
        format!("{manifest_sha256}  run-manifest.json\n"),
    )?;

    eprintln!("lockbox execution complete; manifest_sha256={manifest_sha256}");
    Ok(())
}
