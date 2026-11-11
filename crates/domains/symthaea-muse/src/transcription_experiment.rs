// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Provenance-bound receipts for reproducible automatic music transcription
//! experiments. A receipt binds exact input audio, model identity/artifacts,
//! preprocessing, conditioning, runtime, rights, MIDI output and evaluator
//! report. It records evidence; it does not certify that the model is accurate.

use crate::transcription_eval::{
    MidiTranscriptionEvaluationReportV1,
    midi_transcription_evaluation_commitment_v1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const TRANSCRIPTION_EXPERIMENT_RECEIPT_SCHEMA_V1: &str =
    "muse-transcription-experiment-receipt-v1";
pub const TRANSCRIPTION_EXPERIMENT_RECEIPT_ALGORITHM_V1: &str =
    "sha256-domain-separated-experiment-receipt-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranscriptionModelDeliveryV1 {
    /// Versioned model weights are available to the experiment owner.
    OpenWeights,
    /// Inference occurred through a hosted service; weights are not locally
    /// available for hashing or independent replay.
    HostedApi,
    /// A locally run binary/container was used, but the model weights are not
    /// separately available as an independently hashable artifact.
    OpaqueLocalRuntime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranscriptionConditioningSourceV1 {
    /// No instrument/source conditioning was supplied.
    None,
    /// Conditioning labels were derived from the trusted benchmark reference.
    /// This is an oracle-conditioned experiment and must not be compared as
    /// an audio-only result.
    GroundTruthInstrumentPresence,
    /// Conditioning labels came from a separate prediction system.
    PredictedInstrumentPresence,
    /// Conditioning labels were explicitly supplied by a human/operator.
    UserProvidedInstrumentPresence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranscriptionRightsBasisV1 {
    PublicDomain,
    OwnedOrLicensed,
    ConsentedResearch,
    ResearchOnly,
    NonCommercialOnly,
    Unresolved,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptionAudioPreprocessingV1 {
    pub source_sample_rate_hz: u32,
    pub model_sample_rate_hz: u32,
    pub source_channel_count: u16,
    /// Explicit description such as "preserve_stereo", "average_mono",
    /// "left_only", or a named channel-selection policy.
    pub channel_policy: String,
    /// Explicit resampler/version identifier. Do not use an empty value for
    /// "whatever the current library default is".
    pub resampler_id: String,
    /// Explicit normalization policy identifier.
    pub normalization_id: String,
    /// Zero means unchunked. Otherwise overlap must be smaller than duration.
    pub chunk_duration_ms: u32,
    pub chunk_overlap_ms: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptionRuntimeReceiptV1 {
    pub provider: String,
    pub device: String,
    pub accelerator: String,
    pub runtime_id: String,
    pub precision: String,
    pub batch_size: u32,
    pub wall_time_ms: u64,
    pub peak_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptionExperimentReceiptV1 {
    pub schema: String,
    pub algorithm: String,
    pub experiment_id: String,
    pub benchmark_id: String,
    pub dataset_id: String,
    pub dataset_revision: String,
    pub sample_id: String,
    pub split: String,

    /// SHA-256 of the exact audio bytes passed to preprocessing/inference.
    pub input_audio_sha256: String,
    pub input_rights_basis: TranscriptionRightsBasisV1,
    pub input_rights_reference: Option<String>,
    pub input_usage_scope: String,

    pub model_name: String,
    pub model_revision: String,
    pub model_delivery: TranscriptionModelDeliveryV1,
    /// Required for OpenWeights; optional for hosted/opaque runtimes.
    pub model_artifact_sha256: Option<String>,
    /// Exact code/container revision when available.
    pub model_code_revision: Option<String>,
    pub model_provider: String,
    pub model_license: String,
    /// Digest of the complete inference/decode configuration. The full config
    /// may be stored externally; this receipt binds the exact snapshot.
    pub inference_config_sha256: String,
    pub random_seed: Option<u64>,

    pub preprocessing: TranscriptionAudioPreprocessingV1,
    pub conditioning_source: TranscriptionConditioningSourceV1,
    /// Digest of the exact instrument labels/conditioning payload. Required
    /// for every non-None conditioning source.
    pub conditioning_payload_sha256: Option<String>,

    pub runtime: TranscriptionRuntimeReceiptV1,

    /// SHA-256 of the exact predicted Standard MIDI bytes.
    pub prediction_midi_sha256: String,
    pub reference_manifest_sha256: String,
    pub evaluator_schema: String,
    pub evaluator_algorithm: String,
    pub evaluator_report_sha256: String,

    pub created_unix_ms: Option<u64>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptionExperimentReceiptIssueV1 {
    InvalidSchema,
    MissingField { field: String },
    InvalidDigest { field: String },
    AudioDigestMismatch,
    PredictionMidiDigestMismatch,
    InvalidModelArtifact,
    InvalidConditioningBinding,
    InvalidPreprocessing { field: String },
    InvalidRuntime { field: String },
    EvaluatorReportBindingMismatch,
    EvaluatorReportInvalid,
    ReceiptDigestMismatch,
    SerializationFailed,
}

#[derive(Serialize)]
struct ReceiptCommitmentV1<'a> {
    schema: &'a str,
    algorithm: &'a str,
    experiment_id: &'a str,
    benchmark_id: &'a str,
    dataset_id: &'a str,
    dataset_revision: &'a str,
    sample_id: &'a str,
    split: &'a str,
    input_audio_sha256: &'a str,
    input_rights_basis: TranscriptionRightsBasisV1,
    input_rights_reference: &'a Option<String>,
    input_usage_scope: &'a str,
    model_name: &'a str,
    model_revision: &'a str,
    model_delivery: TranscriptionModelDeliveryV1,
    model_artifact_sha256: &'a Option<String>,
    model_code_revision: &'a Option<String>,
    model_provider: &'a str,
    model_license: &'a str,
    inference_config_sha256: &'a str,
    random_seed: Option<u64>,
    preprocessing: &'a TranscriptionAudioPreprocessingV1,
    conditioning_source: TranscriptionConditioningSourceV1,
    conditioning_payload_sha256: &'a Option<String>,
    runtime: &'a TranscriptionRuntimeReceiptV1,
    prediction_midi_sha256: &'a str,
    reference_manifest_sha256: &'a str,
    evaluator_schema: &'a str,
    evaluator_algorithm: &'a str,
    evaluator_report_sha256: &'a str,
    created_unix_ms: Option<u64>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn domain_separated_sha256(schema: &str, bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(schema.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn nonempty(field: &str, value: &str) -> Result<(), TranscriptionExperimentReceiptIssueV1> {
    if value.trim().is_empty() {
        Err(TranscriptionExperimentReceiptIssueV1::MissingField {
            field: field.into(),
        })
    } else {
        Ok(())
    }
}

fn validate_digest(
    field: &str,
    value: &str,
) -> Result<(), TranscriptionExperimentReceiptIssueV1> {
    if !valid_sha256(value) {
        Err(TranscriptionExperimentReceiptIssueV1::InvalidDigest {
            field: field.into(),
        })
    } else {
        Ok(())
    }
}

pub fn transcription_experiment_receipt_commitment_v1(
    receipt: &TranscriptionExperimentReceiptV1,
) -> Result<String, serde_json::Error> {
    let commitment = ReceiptCommitmentV1 {
        schema: &receipt.schema,
        algorithm: &receipt.algorithm,
        experiment_id: &receipt.experiment_id,
        benchmark_id: &receipt.benchmark_id,
        dataset_id: &receipt.dataset_id,
        dataset_revision: &receipt.dataset_revision,
        sample_id: &receipt.sample_id,
        split: &receipt.split,
        input_audio_sha256: &receipt.input_audio_sha256,
        input_rights_basis: receipt.input_rights_basis,
        input_rights_reference: &receipt.input_rights_reference,
        input_usage_scope: &receipt.input_usage_scope,
        model_name: &receipt.model_name,
        model_revision: &receipt.model_revision,
        model_delivery: receipt.model_delivery,
        model_artifact_sha256: &receipt.model_artifact_sha256,
        model_code_revision: &receipt.model_code_revision,
        model_provider: &receipt.model_provider,
        model_license: &receipt.model_license,
        inference_config_sha256: &receipt.inference_config_sha256,
        random_seed: receipt.random_seed,
        preprocessing: &receipt.preprocessing,
        conditioning_source: receipt.conditioning_source,
        conditioning_payload_sha256: &receipt.conditioning_payload_sha256,
        runtime: &receipt.runtime,
        prediction_midi_sha256: &receipt.prediction_midi_sha256,
        reference_manifest_sha256: &receipt.reference_manifest_sha256,
        evaluator_schema: &receipt.evaluator_schema,
        evaluator_algorithm: &receipt.evaluator_algorithm,
        evaluator_report_sha256: &receipt.evaluator_report_sha256,
        created_unix_ms: receipt.created_unix_ms,
    };
    Ok(domain_separated_sha256(
        TRANSCRIPTION_EXPERIMENT_RECEIPT_SCHEMA_V1,
        &serde_json::to_vec(&commitment)?,
    ))
}

/// A receipt is public-benchmark eligible only when input rights are resolved
/// and not restricted to non-commercial research. This is a policy signal,
/// not a legal opinion or a substitute for checking the actual license.
pub fn transcription_experiment_public_benchmark_eligible_v1(
    receipt: &TranscriptionExperimentReceiptV1,
) -> bool {
    matches!(
        receipt.input_rights_basis,
        TranscriptionRightsBasisV1::PublicDomain
            | TranscriptionRightsBasisV1::OwnedOrLicensed
            | TranscriptionRightsBasisV1::ConsentedResearch
    )
}

/// Check receipt structure, exact audio/MIDI bytes, and the nested evaluator
/// report. The report's own commitment is recomputed; it must bind the same
/// predicted MIDI and reference manifest as this experiment receipt.
pub fn validate_transcription_experiment_receipt_v1(
    receipt: &TranscriptionExperimentReceiptV1,
    audio_bytes: &[u8],
    prediction_midi_bytes: &[u8],
    report: &MidiTranscriptionEvaluationReportV1,
) -> Vec<TranscriptionExperimentReceiptIssueV1> {
    let mut issues = Vec::new();
    if receipt.schema != TRANSCRIPTION_EXPERIMENT_RECEIPT_SCHEMA_V1
        || receipt.algorithm != TRANSCRIPTION_EXPERIMENT_RECEIPT_ALGORITHM_V1
    {
        issues.push(TranscriptionExperimentReceiptIssueV1::InvalidSchema);
    }

    for (field, value) in [
        ("experiment_id", receipt.experiment_id.as_str()),
        ("benchmark_id", receipt.benchmark_id.as_str()),
        ("dataset_id", receipt.dataset_id.as_str()),
        ("dataset_revision", receipt.dataset_revision.as_str()),
        ("sample_id", receipt.sample_id.as_str()),
        ("split", receipt.split.as_str()),
        ("input_usage_scope", receipt.input_usage_scope.as_str()),
        ("model_name", receipt.model_name.as_str()),
        ("model_revision", receipt.model_revision.as_str()),
        ("model_provider", receipt.model_provider.as_str()),
        ("model_license", receipt.model_license.as_str()),
        ("runtime.provider", receipt.runtime.provider.as_str()),
        ("runtime.device", receipt.runtime.device.as_str()),
        ("runtime.accelerator", receipt.runtime.accelerator.as_str()),
        ("runtime.runtime_id", receipt.runtime.runtime_id.as_str()),
        ("runtime.precision", receipt.runtime.precision.as_str()),
        ("preprocessing.channel_policy", receipt.preprocessing.channel_policy.as_str()),
        ("preprocessing.resampler_id", receipt.preprocessing.resampler_id.as_str()),
        ("preprocessing.normalization_id", receipt.preprocessing.normalization_id.as_str()),
        ("evaluator_schema", receipt.evaluator_schema.as_str()),
        ("evaluator_algorithm", receipt.evaluator_algorithm.as_str()),
    ] {
        if let Err(issue) = nonempty(field, value) {
            issues.push(issue);
        }
    }

    for (field, value) in [
        ("input_audio_sha256", receipt.input_audio_sha256.as_str()),
        ("inference_config_sha256", receipt.inference_config_sha256.as_str()),
        ("prediction_midi_sha256", receipt.prediction_midi_sha256.as_str()),
        ("reference_manifest_sha256", receipt.reference_manifest_sha256.as_str()),
        ("evaluator_report_sha256", receipt.evaluator_report_sha256.as_str()),
        ("receipt_sha256", receipt.receipt_sha256.as_str()),
    ] {
        if let Err(issue) = validate_digest(field, value) {
            issues.push(issue);
        }
    }

    if let Some(digest) = receipt.model_artifact_sha256.as_deref() {
        if let Err(issue) = validate_digest("model_artifact_sha256", digest) {
            issues.push(issue);
        }
    }
    if let Some(revision) = receipt.model_code_revision.as_deref() {
        if revision.trim().is_empty() {
            issues.push(TranscriptionExperimentReceiptIssueV1::MissingField {
                field: "model_code_revision".into(),
            });
        }
    }

    match receipt.model_delivery {
        TranscriptionModelDeliveryV1::OpenWeights
        | TranscriptionModelDeliveryV1::OpaqueLocalRuntime => {
            if receipt.model_artifact_sha256.is_none() {
                issues.push(TranscriptionExperimentReceiptIssueV1::InvalidModelArtifact);
            }
        }
        TranscriptionModelDeliveryV1::HostedApi => {
            if receipt.model_provider.trim().is_empty() {
                issues.push(TranscriptionExperimentReceiptIssueV1::InvalidModelArtifact);
            }
        }
    }

    match receipt.conditioning_source {
        TranscriptionConditioningSourceV1::None => {
            if receipt.conditioning_payload_sha256.is_some() {
                issues.push(TranscriptionExperimentReceiptIssueV1::InvalidConditioningBinding);
            }
        }
        _ => match receipt.conditioning_payload_sha256.as_deref() {
            Some(digest) if valid_sha256(digest) => {}
            _ => issues.push(TranscriptionExperimentReceiptIssueV1::InvalidConditioningBinding),
        },
    }

    let preprocessing = &receipt.preprocessing;
    if preprocessing.source_sample_rate_hz == 0 || preprocessing.model_sample_rate_hz == 0 {
        issues.push(TranscriptionExperimentReceiptIssueV1::InvalidPreprocessing {
            field: "sample_rate_hz".into(),
        });
    }
    if preprocessing.source_channel_count == 0 {
        issues.push(TranscriptionExperimentReceiptIssueV1::InvalidPreprocessing {
            field: "source_channel_count".into(),
        });
    }
    if preprocessing.chunk_duration_ms == 0 && preprocessing.chunk_overlap_ms != 0 {
        issues.push(TranscriptionExperimentReceiptIssueV1::InvalidPreprocessing {
            field: "overlap_without_chunking".into(),
        });
    }
    if preprocessing.chunk_duration_ms > 0
        && preprocessing.chunk_overlap_ms >= preprocessing.chunk_duration_ms
    {
        issues.push(TranscriptionExperimentReceiptIssueV1::InvalidPreprocessing {
            field: "chunk_overlap_ms".into(),
        });
    }

    if receipt.runtime.batch_size == 0 {
        issues.push(TranscriptionExperimentReceiptIssueV1::InvalidRuntime {
            field: "batch_size".into(),
        });
    }

    if receipt.input_audio_sha256 != sha256_hex(audio_bytes) {
        issues.push(TranscriptionExperimentReceiptIssueV1::AudioDigestMismatch);
    }
    if receipt.prediction_midi_sha256 != sha256_hex(prediction_midi_bytes) {
        issues.push(TranscriptionExperimentReceiptIssueV1::PredictionMidiDigestMismatch);
    }

    if report.prediction_midi_sha256 != receipt.prediction_midi_sha256
        || report.reference_manifest_sha256 != receipt.reference_manifest_sha256
        || report.schema != receipt.evaluator_schema
        || report.algorithm != receipt.evaluator_algorithm
        || report.report_sha256 != receipt.evaluator_report_sha256
    {
        issues.push(TranscriptionExperimentReceiptIssueV1::EvaluatorReportBindingMismatch);
    }
    match midi_transcription_evaluation_commitment_v1(report) {
        Ok(actual) if actual == report.report_sha256 => {}
        _ => issues.push(TranscriptionExperimentReceiptIssueV1::EvaluatorReportInvalid),
    }

    match transcription_experiment_receipt_commitment_v1(receipt) {
        Ok(actual) if actual == receipt.receipt_sha256 => {}
        _ => issues.push(TranscriptionExperimentReceiptIssueV1::ReceiptDigestMismatch),
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcription_eval::{
        MidiTranscriptionEvaluationConfigV1,
        MidiTranscriptionEvaluationMetricV1,
        MidiTranscriptionMetricV1,
        MidiTranscriptionTimingErrorSummaryV1,
        MidiTranscriptionValueErrorSummaryV1,
        MIDI_TRANSCRIPTION_EVALUATION_ALGORITHM_V1,
        MIDI_TRANSCRIPTION_EVALUATION_SCHEMA_V1,
    };

    fn metric() -> MidiTranscriptionMetricV1 {
        MidiTranscriptionMetricV1 {
            true_positives: 0,
            false_positives: 0,
            false_negatives: 0,
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
        }
    }

    fn report(midi_bytes: &[u8]) -> MidiTranscriptionEvaluationReportV1 {
        let mut report = MidiTranscriptionEvaluationReportV1 {
            schema: MIDI_TRANSCRIPTION_EVALUATION_SCHEMA_V1.into(),
            algorithm: MIDI_TRANSCRIPTION_EVALUATION_ALGORITHM_V1.into(),
            reference_score_sha256: "a".repeat(64),
            reference_midi_sha256: "b".repeat(64),
            reference_manifest_sha256: "c".repeat(64),
            prediction_midi_sha256: sha256_hex(midi_bytes),
            config: MidiTranscriptionEvaluationConfigV1::default(),
            reference_note_count: 0,
            prediction_note_count: 0,
            prediction_unclosed_note_count: 0,
            prediction_orphan_note_off_count: 0,
            prediction_tempo_point_count: 1,
            prediction_overlapping_same_pitch_note_on_count: 0,
            onset_only_metrics: metric(),
            offset_only_metrics: metric(),
            onset_pitch_metrics: metric(),
            full_note_metrics: metric(),
            average_overlap_ratio: 0.0,
            exact_program_onset_metrics: metric(),
            exact_program_full_note_metrics: metric(),
            velocity_aware_full_note_metrics: None,
            frame_pitch_metrics: metric(),
            frame_program_pitch_metrics: metric(),
            frame_step_ms: 10,
            velocity_error: None,
            onset_error_ms: MidiTranscriptionTimingErrorSummaryV1 {
                sample_count: 0,
                mean_signed_error_ms: None,
                mean_absolute_error_ms: None,
                median_absolute_error_ms: None,
                p95_absolute_error_ms: None,
                p99_absolute_error_ms: None,
            },
            offset_error_ms: MidiTranscriptionTimingErrorSummaryV1 {
                sample_count: 0,
                mean_signed_error_ms: None,
                mean_absolute_error_ms: None,
                median_absolute_error_ms: None,
                p95_absolute_error_ms: None,
                p99_absolute_error_ms: None,
            },
            report_sha256: String::new(),
        };
        report.report_sha256 = midi_transcription_evaluation_commitment_v1(&report).unwrap();
        report
    }

    fn valid_receipt(audio: &[u8], midi: &[u8], report: &MidiTranscriptionEvaluationReportV1) -> TranscriptionExperimentReceiptV1 {
        let mut receipt = TranscriptionExperimentReceiptV1 {
            schema: TRANSCRIPTION_EXPERIMENT_RECEIPT_SCHEMA_V1.into(),
            algorithm: TRANSCRIPTION_EXPERIMENT_RECEIPT_ALGORITHM_V1.into(),
            experiment_id: "exp-001".into(),
            benchmark_id: "muse-mir-001".into(),
            dataset_id: "commissioned-test-corpus".into(),
            dataset_revision: "sha256:corpus-v1".into(),
            sample_id: "sample-0001".into(),
            split: "heldout".into(),
            input_audio_sha256: sha256_hex(audio),
            input_rights_basis: TranscriptionRightsBasisV1::ConsentedResearch,
            input_rights_reference: Some("consent-register:corpus-v1".into()),
            input_usage_scope: "noncommercial_research".into(),
            model_name: "FixtureAMT".into(),
            model_revision: "v1.2.3".into(),
            model_delivery: TranscriptionModelDeliveryV1::OpenWeights,
            model_artifact_sha256: Some("d".repeat(64)),
            model_code_revision: Some("e".repeat(40)),
            model_provider: "local".into(),
            model_license: "Apache-2.0".into(),
            inference_config_sha256: "f".repeat(64),
            random_seed: Some(7),
            preprocessing: TranscriptionAudioPreprocessingV1 {
                source_sample_rate_hz: 48_000,
                model_sample_rate_hz: 16_000,
                source_channel_count: 2,
                channel_policy: "average_mono".into(),
                resampler_id: "soxr-hq-1.0".into(),
                normalization_id: "peak-normalize-1".into(),
                chunk_duration_ms: 5_000,
                chunk_overlap_ms: 500,
            },
            conditioning_source: TranscriptionConditioningSourceV1::None,
            conditioning_payload_sha256: None,
            runtime: TranscriptionRuntimeReceiptV1 {
                provider: "local".into(),
                device: "RTX 2070 Laptop".into(),
                accelerator: "CUDA".into(),
                runtime_id: "torch-2.7".into(),
                precision: "fp16".into(),
                batch_size: 1,
                wall_time_ms: 12_000,
                peak_memory_bytes: Some(7_000_000_000),
            },
            prediction_midi_sha256: sha256_hex(midi),
            reference_manifest_sha256: report.reference_manifest_sha256.clone(),
            evaluator_schema: report.schema.clone(),
            evaluator_algorithm: report.algorithm.clone(),
            evaluator_report_sha256: report.report_sha256.clone(),
            created_unix_ms: Some(1_800_000_000_000),
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = transcription_experiment_receipt_commitment_v1(&receipt).unwrap();
        receipt
    }

    #[test]
    fn valid_open_weights_experiment_receipt_binds_all_artifacts() {
        let audio = b"audio-experiment-fixture";
        let midi = b"midi-experiment-fixture";
        let report = report(midi);
        let receipt = valid_receipt(audio, midi, &report);
        assert!(validate_transcription_experiment_receipt_v1(
            &receipt,
            audio,
            midi,
            &report,
        )
        .is_empty());
        assert!(transcription_experiment_public_benchmark_eligible_v1(&receipt));
    }

    #[test]
    fn audio_byte_substitution_invalidates_experiment_receipt() {
        let audio = b"audio-fixture";
        let midi = b"midi-fixture";
        let report = report(midi);
        let receipt = valid_receipt(audio, midi, &report);
        let issues = validate_transcription_experiment_receipt_v1(
            &receipt,
            b"replaced-audio",
            midi,
            &report,
        );
        assert!(issues.contains(&TranscriptionExperimentReceiptIssueV1::AudioDigestMismatch));
    }

    #[test]
    fn midi_output_substitution_invalidates_experiment_receipt() {
        let audio = b"audio-fixture";
        let midi = b"midi-fixture";
        let report = report(midi);
        let receipt = valid_receipt(audio, midi, &report);
        let issues = validate_transcription_experiment_receipt_v1(
            &receipt,
            audio,
            b"replaced-midi",
            &report,
        );
        assert!(issues.contains(&TranscriptionExperimentReceiptIssueV1::PredictionMidiDigestMismatch));
    }

    #[test]
    fn edited_evaluator_report_is_rejected_even_when_receipt_was_sealed() {
        let audio = b"audio-fixture";
        let midi = b"midi-fixture";
        let mut report = report(midi);
        let receipt = valid_receipt(audio, midi, &report);
        report.onset_pitch_metrics.f1 = 1.0;
        let issues = validate_transcription_experiment_receipt_v1(
            &receipt,
            audio,
            midi,
            &report,
        );
        assert!(issues.contains(&TranscriptionExperimentReceiptIssueV1::EvaluatorReportInvalid));
    }

    #[test]
    fn open_weights_require_a_hashable_model_artifact() {
        let audio = b"audio";
        let midi = b"midi";
        let report = report(midi);
        let mut receipt = valid_receipt(audio, midi, &report);
        receipt.model_artifact_sha256 = None;
        receipt.receipt_sha256 = transcription_experiment_receipt_commitment_v1(&receipt).unwrap();
        let issues = validate_transcription_experiment_receipt_v1(
            &receipt,
            audio,
            midi,
            &report,
        );
        assert!(issues.contains(&TranscriptionExperimentReceiptIssueV1::InvalidModelArtifact));
    }

    #[test]
    fn hosted_api_receipt_can_record_opaque_weights_with_service_identity() {
        let audio = b"audio";
        let midi = b"midi";
        let report = report(midi);
        let mut receipt = valid_receipt(audio, midi, &report);
        receipt.model_delivery = TranscriptionModelDeliveryV1::HostedApi;
        receipt.model_artifact_sha256 = None;
        receipt.model_provider = "VendorCloud".into();
        receipt.receipt_sha256 = transcription_experiment_receipt_commitment_v1(&receipt).unwrap();
        assert!(validate_transcription_experiment_receipt_v1(
            &receipt,
            audio,
            midi,
            &report,
        )
        .is_empty());
    }

    #[test]
    fn conditioning_requires_a_digest_of_the_exact_conditioning_payload() {
        let audio = b"audio";
        let midi = b"midi";
        let report = report(midi);
        let mut receipt = valid_receipt(audio, midi, &report);
        receipt.conditioning_source = TranscriptionConditioningSourceV1::GroundTruthInstrumentPresence;
        receipt.conditioning_payload_sha256 = None;
        receipt.receipt_sha256 = transcription_experiment_receipt_commitment_v1(&receipt).unwrap();
        let issues = validate_transcription_experiment_receipt_v1(
            &receipt,
            audio,
            midi,
            &report,
        );
        assert!(issues.contains(&TranscriptionExperimentReceiptIssueV1::InvalidConditioningBinding));
    }

    #[test]
    fn ground_truth_conditioning_is_explicit_and_binds_payload() {
        let audio = b"audio";
        let midi = b"midi";
        let report = report(midi);
        let mut receipt = valid_receipt(audio, midi, &report);
        receipt.conditioning_source = TranscriptionConditioningSourceV1::GroundTruthInstrumentPresence;
        receipt.conditioning_payload_sha256 = Some("1".repeat(64));
        receipt.receipt_sha256 = transcription_experiment_receipt_commitment_v1(&receipt).unwrap();
        assert!(validate_transcription_experiment_receipt_v1(
            &receipt,
            audio,
            midi,
            &report,
        )
        .is_empty());
    }

    #[test]
    fn unresolved_rights_are_recorded_but_not_public_benchmark_eligible() {
        let audio = b"audio";
        let midi = b"midi";
        let report = report(midi);
        let mut receipt = valid_receipt(audio, midi, &report);
        receipt.input_rights_basis = TranscriptionRightsBasisV1::Unresolved;
        receipt.receipt_sha256 = transcription_experiment_receipt_commitment_v1(&receipt).unwrap();
        assert!(validate_transcription_experiment_receipt_v1(
            &receipt,
            audio,
            midi,
            &report,
        )
        .is_empty());
        assert!(!transcription_experiment_public_benchmark_eligible_v1(&receipt));
    }

    #[test]
    fn noncommercial_rights_do_not_pass_public_benchmark_eligibility() {
        let audio = b"audio";
        let midi = b"midi";
        let report = report(midi);
        let mut receipt = valid_receipt(audio, midi, &report);
        receipt.input_rights_basis = TranscriptionRightsBasisV1::NonCommercialOnly;
        receipt.receipt_sha256 = transcription_experiment_receipt_commitment_v1(&receipt).unwrap();
        assert!(!transcription_experiment_public_benchmark_eligible_v1(&receipt));
    }

    #[test]
    fn invalid_chunk_overlap_is_rejected() {
        let audio = b"audio";
        let midi = b"midi";
        let report = report(midi);
        let mut receipt = valid_receipt(audio, midi, &report);
        receipt.preprocessing.chunk_overlap_ms = 5_000;
        receipt.receipt_sha256 = transcription_experiment_receipt_commitment_v1(&receipt).unwrap();
        let issues = validate_transcription_experiment_receipt_v1(
            &receipt,
            audio,
            midi,
            &report,
        );
        assert!(issues.contains(&TranscriptionExperimentReceiptIssueV1::InvalidPreprocessing {
            field: "chunk_overlap_ms".into()
        }));
    }

    #[test]
    fn tampered_receipt_fields_fail_commitment_validation() {
        let audio = b"audio";
        let midi = b"midi";
        let report = report(midi);
        let mut receipt = valid_receipt(audio, midi, &report);
        receipt.model_revision = "another-checkpoint".into();
        let issues = validate_transcription_experiment_receipt_v1(
            &receipt,
            audio,
            midi,
            &report,
        );
        assert!(issues.contains(&TranscriptionExperimentReceiptIssueV1::ReceiptDigestMismatch));
    }
}
