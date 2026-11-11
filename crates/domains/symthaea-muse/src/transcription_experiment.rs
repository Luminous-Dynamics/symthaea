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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranscriptionUsageScopeV1 {
    InternalOnly,
    ResearchOnly,
    NonCommercialResearch,
    /// Permission expressly covers publication of aggregate benchmark results.
    PublicBenchmarkResults,
    /// Permission expressly covers redistribution of the underlying audio.
    PublicDatasetRedistribution,
    CommercialProductUse,
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

    /// SHA-256 of the exact source audio bytes before preprocessing.
    pub input_audio_sha256: String,
    /// SHA-256 of the exact canonical model-input bytes after resampling,
    /// channel conversion, normalization and chunking. For tensor-based APIs,
    /// hash a documented canonical byte encoding with shape and dtype.
    pub model_input_audio_sha256: String,
    pub dataset_manifest_sha256: String,
    pub dataset_manifest_reference: String,
    pub input_rights_basis: TranscriptionRightsBasisV1,
    pub input_rights_reference: Option<String>,
    pub input_usage_scope: TranscriptionUsageScopeV1,

    pub model_name: String,
    pub model_revision: String,
    pub model_delivery: TranscriptionModelDeliveryV1,
    /// Required for OpenWeights; optional for hosted/opaque runtimes.
    pub model_artifact_sha256: Option<String>,
    /// Exact code/container revision when available.
    pub model_code_revision: Option<String>,
    pub adapter_id: String,
    pub adapter_revision: String,
    pub model_provider: String,
    pub model_license: String,
    pub model_license_reference: Option<String>,
    pub model_usage_scope: TranscriptionUsageScopeV1,
    /// Digest and retrievable reference to the complete inference/decode config.
    pub inference_config_sha256: String,
    pub inference_config_reference: String,
    pub random_seed: Option<u64>,

    pub preprocessing: TranscriptionAudioPreprocessingV1,
    /// Digest of any additional preprocessing parameters not represented as
    /// first-class fields above (e.g. filter coefficients or tensor layout).
    pub preprocessing_config_sha256: String,
    pub preprocessing_config_reference: String,
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
    ModelArtifactDigestMismatch,
    ArtifactDigestMismatch { field: String },
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
    dataset_manifest_sha256: &'a str,
    dataset_manifest_reference: &'a str,
    sample_id: &'a str,
    split: &'a str,
    input_audio_sha256: &'a str,
    model_input_audio_sha256: &'a str,
    input_rights_basis: TranscriptionRightsBasisV1,
    input_rights_reference: &'a Option<String>,
    input_usage_scope: TranscriptionUsageScopeV1,
    model_name: &'a str,
    model_revision: &'a str,
    model_delivery: TranscriptionModelDeliveryV1,
    model_artifact_sha256: &'a Option<String>,
    model_code_revision: &'a Option<String>,
    adapter_id: &'a str,
    adapter_revision: &'a str,
    model_provider: &'a str,
    model_license: &'a str,
    model_license_reference: &'a Option<String>,
    model_usage_scope: TranscriptionUsageScopeV1,
    inference_config_sha256: &'a str,
    inference_config_reference: &'a str,
    random_seed: Option<u64>,
    preprocessing: &'a TranscriptionAudioPreprocessingV1,
    preprocessing_config_sha256: &'a str,
    preprocessing_config_reference: &'a str,
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
        dataset_manifest_sha256: &receipt.dataset_manifest_sha256,
        dataset_manifest_reference: &receipt.dataset_manifest_reference,
        sample_id: &receipt.sample_id,
        split: &receipt.split,
        input_audio_sha256: &receipt.input_audio_sha256,
        model_input_audio_sha256: &receipt.model_input_audio_sha256,
        input_rights_basis: receipt.input_rights_basis,
        input_rights_reference: &receipt.input_rights_reference,
        input_usage_scope: receipt.input_usage_scope,
        model_name: &receipt.model_name,
        model_revision: &receipt.model_revision,
        model_delivery: receipt.model_delivery,
        model_artifact_sha256: &receipt.model_artifact_sha256,
        model_code_revision: &receipt.model_code_revision,
        adapter_id: &receipt.adapter_id,
        adapter_revision: &receipt.adapter_revision,
        model_provider: &receipt.model_provider,
        model_license: &receipt.model_license,
        model_license_reference: &receipt.model_license_reference,
        model_usage_scope: receipt.model_usage_scope,
        inference_config_sha256: &receipt.inference_config_sha256,
        inference_config_reference: &receipt.inference_config_reference,
        random_seed: receipt.random_seed,
        preprocessing: &receipt.preprocessing,
        preprocessing_config_sha256: &receipt.preprocessing_config_sha256,
        preprocessing_config_reference: &receipt.preprocessing_config_reference,
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

/// A receipt is public-benchmark-result eligible only when both the input
/// rights and model terms expressly permit publication of benchmark results
/// (or wider redistribution), and non-public-domain permissions have a
/// reference. This is a policy signal, not a legal opinion or license parser.
pub fn transcription_experiment_public_benchmark_eligible_v1(
    receipt: &TranscriptionExperimentReceiptV1,
) -> bool {
    let input_basis_ok = match receipt.input_rights_basis {
        TranscriptionRightsBasisV1::PublicDomain => true,
        TranscriptionRightsBasisV1::OwnedOrLicensed
        | TranscriptionRightsBasisV1::ConsentedResearch => {
            receipt.input_rights_reference.as_deref().is_some_and(|value| !value.trim().is_empty())
        }
        TranscriptionRightsBasisV1::ResearchOnly
        | TranscriptionRightsBasisV1::NonCommercialOnly
        | TranscriptionRightsBasisV1::Unresolved => false,
    };
    let public_scope = |scope| {
        matches!(
            scope,
            TranscriptionUsageScopeV1::PublicBenchmarkResults
                | TranscriptionUsageScopeV1::PublicDatasetRedistribution
        )
    };
    input_basis_ok
        && public_scope(receipt.input_usage_scope)
        && public_scope(receipt.model_usage_scope)
        && receipt.model_license_reference.as_deref().is_some_and(|value| !value.trim().is_empty())
}

/// Actual artifacts available to the validator. The input bytes for every
/// supplied digest are re-hashed; a digest string alone is not treated as
/// proof that the artifact matches.
pub struct TranscriptionExperimentEvidenceV1<'a> {
    pub source_audio_bytes: &'a [u8],
    pub model_input_audio_bytes: &'a [u8],
    pub dataset_manifest_bytes: &'a [u8],
    pub inference_config_bytes: &'a [u8],
    pub preprocessing_config_bytes: &'a [u8],
    /// Required for OpenWeights. Optional for hosted APIs and opaque runtimes.
    pub model_artifact_bytes: Option<&'a [u8]>,
    pub prediction_midi_bytes: &'a [u8],
}

/// Check receipt structure, exact source/preprocessed audio, dataset and
/// configuration artifacts, MIDI bytes, and nested evaluator report. A
/// receipt digest is tamper-evident only when its expected value is held by a
/// trusted party; this hash is not a producer signature or attestation.
pub fn validate_transcription_experiment_receipt_v1(
    receipt: &TranscriptionExperimentReceiptV1,
    evidence: &TranscriptionExperimentEvidenceV1<'_>,
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
        ("dataset_manifest_reference", receipt.dataset_manifest_reference.as_str()),
        ("preprocessing_config_reference", receipt.preprocessing_config_reference.as_str()),
        ("sample_id", receipt.sample_id.as_str()),
        ("split", receipt.split.as_str()),
        ("adapter_id", receipt.adapter_id.as_str()),
        ("adapter_revision", receipt.adapter_revision.as_str()),
        ("inference_config_reference", receipt.inference_config_reference.as_str()),
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
        ("model_input_audio_sha256", receipt.model_input_audio_sha256.as_str()),
        ("dataset_manifest_sha256", receipt.dataset_manifest_sha256.as_str()),
        ("inference_config_sha256", receipt.inference_config_sha256.as_str()),
        ("preprocessing_config_sha256", receipt.preprocessing_config_sha256.as_str()),
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
    if let Some(reference) = receipt.input_rights_reference.as_deref() {
        if reference.trim().is_empty() {
            issues.push(TranscriptionExperimentReceiptIssueV1::MissingField {
                field: "input_rights_reference".into(),
            });
        }
    }
    if let Some(reference) = receipt.model_license_reference.as_deref() {
        if reference.trim().is_empty() {
            issues.push(TranscriptionExperimentReceiptIssueV1::MissingField {
                field: "model_license_reference".into(),
            });
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
        TranscriptionModelDeliveryV1::OpenWeights => {
            if receipt.model_artifact_sha256.is_none()
                || evidence.model_artifact_bytes.is_none()
            {
                issues.push(TranscriptionExperimentReceiptIssueV1::InvalidModelArtifact);
            }
        }
        TranscriptionModelDeliveryV1::OpaqueLocalRuntime
        | TranscriptionModelDeliveryV1::HostedApi => {}
    }
    if let (Some(expected), Some(actual)) = (
        receipt.model_artifact_sha256.as_deref(),
        evidence.model_artifact_bytes,
    ) {
        if expected != sha256_hex(actual) {
            issues.push(TranscriptionExperimentReceiptIssueV1::ModelArtifactDigestMismatch);
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
    if receipt.runtime.wall_time_ms == 0 {
        issues.push(TranscriptionExperimentReceiptIssueV1::InvalidRuntime {
            field: "wall_time_ms".into(),
        });
    }
    if receipt.created_unix_ms.is_none_or(|timestamp| timestamp == 0) {
        issues.push(TranscriptionExperimentReceiptIssueV1::MissingField {
            field: "created_unix_ms".into(),
        });
    }

    for (field, expected, actual) in [
        (
            "input_audio_sha256",
            receipt.input_audio_sha256.as_str(),
            evidence.source_audio_bytes,
        ),
        (
            "model_input_audio_sha256",
            receipt.model_input_audio_sha256.as_str(),
            evidence.model_input_audio_bytes,
        ),
        (
            "dataset_manifest_sha256",
            receipt.dataset_manifest_sha256.as_str(),
            evidence.dataset_manifest_bytes,
        ),
        (
            "inference_config_sha256",
            receipt.inference_config_sha256.as_str(),
            evidence.inference_config_bytes,
        ),
        (
            "preprocessing_config_sha256",
            receipt.preprocessing_config_sha256.as_str(),
            evidence.preprocessing_config_bytes,
        ),
        (
            "prediction_midi_sha256",
            receipt.prediction_midi_sha256.as_str(),
            evidence.prediction_midi_bytes,
        ),
    ] {
        if expected != sha256_hex(actual) {
            if field == "input_audio_sha256" {
                issues.push(TranscriptionExperimentReceiptIssueV1::AudioDigestMismatch);
            } else if field == "prediction_midi_sha256" {
                issues.push(TranscriptionExperimentReceiptIssueV1::PredictionMidiDigestMismatch);
            } else {
                issues.push(TranscriptionExperimentReceiptIssueV1::ArtifactDigestMismatch {
                    field: field.into(),
                });
            }
        }
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
            model_input_audio_sha256: sha256_hex(b"canonical-model-input"),
            dataset_manifest_sha256: sha256_hex(b"dataset-manifest"),
            dataset_manifest_reference: "artifact://dataset-manifest/corpus-v1".into(),
            input_rights_basis: TranscriptionRightsBasisV1::ConsentedResearch,
            input_rights_reference: Some("consent-register:corpus-v1".into()),
            input_usage_scope: TranscriptionUsageScopeV1::PublicBenchmarkResults,
            model_name: "FixtureAMT".into(),
            model_revision: "v1.2.3".into(),
            model_delivery: TranscriptionModelDeliveryV1::OpenWeights,
            model_artifact_sha256: Some(sha256_hex(b"model-weights")),
            model_code_revision: Some("e".repeat(40)),
            adapter_id: "symthaea-muse-amt-adapter".into(),
            adapter_revision: "git:adapter-fixture-v1".into(),
            model_provider: "local".into(),
            model_license: "Apache-2.0".into(),
            model_license_reference: Some("https://example.invalid/model-license".into()),
            model_usage_scope: TranscriptionUsageScopeV1::PublicBenchmarkResults,
            inference_config_sha256: sha256_hex(b"inference-config"),
            inference_config_reference: "artifact://experiment-config/exp-001".into(),
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
            preprocessing_config_sha256: sha256_hex(b"preprocessing-config"),
            preprocessing_config_reference: "artifact://preprocessing-config/preprocess-v1".into(),
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

    fn validate_fixture(
        receipt: &TranscriptionExperimentReceiptV1,
        audio: &[u8],
        midi: &[u8],
        report: &MidiTranscriptionEvaluationReportV1,
    ) -> Vec<TranscriptionExperimentReceiptIssueV1> {
        let evidence = TranscriptionExperimentEvidenceV1 {
            source_audio_bytes: audio,
            model_input_audio_bytes: b"canonical-model-input",
            dataset_manifest_bytes: b"dataset-manifest",
            inference_config_bytes: b"inference-config",
            preprocessing_config_bytes: b"preprocessing-config",
            model_artifact_bytes: Some(b"model-weights"),
            prediction_midi_bytes: midi,
        };
        validate_transcription_experiment_receipt_v1(receipt, &evidence, report)
    }

    #[test]
    fn valid_open_weights_experiment_receipt_binds_all_artifacts() {
        let audio = b"audio-experiment-fixture";
        let midi = b"midi-experiment-fixture";
        let report = report(midi);
        let receipt = valid_receipt(audio, midi, &report);
        assert!(validate_fixture(
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
        let issues = validate_fixture(
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
        let issues = validate_fixture(
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
        let issues = validate_fixture(
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
        let issues = validate_fixture(
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
        assert!(validate_fixture(
            &receipt,
            audio,
            midi,
            &report,
        )
        .is_empty());
    }

    #[test]
    fn open_weights_artifact_bytes_must_match_the_model_digest() {
        let audio = b"audio";
        let midi = b"midi";
        let report = report(midi);
        let receipt = valid_receipt(audio, midi, &report);
        let evidence = TranscriptionExperimentEvidenceV1 {
            source_audio_bytes: audio,
            model_input_audio_bytes: b"canonical-model-input",
            dataset_manifest_bytes: b"dataset-manifest",
            inference_config_bytes: b"inference-config",
            preprocessing_config_bytes: b"preprocessing-config",
            model_artifact_bytes: Some(b"different-model-weights"),
            prediction_midi_bytes: midi,
        };
        let issues = validate_transcription_experiment_receipt_v1(
            &receipt, &evidence, &report,
        );
        assert!(issues.contains(&TranscriptionExperimentReceiptIssueV1::ModelArtifactDigestMismatch));
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
        let issues = validate_fixture(
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
        assert!(validate_fixture(
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
        receipt.input_usage_scope = TranscriptionUsageScopeV1::ResearchOnly;
        receipt.receipt_sha256 = transcription_experiment_receipt_commitment_v1(&receipt).unwrap();
        assert!(validate_fixture(
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
        receipt.input_usage_scope = TranscriptionUsageScopeV1::NonCommercialResearch;
        receipt.model_usage_scope = TranscriptionUsageScopeV1::NonCommercialResearch;
        receipt.receipt_sha256 = transcription_experiment_receipt_commitment_v1(&receipt).unwrap();
        assert!(!transcription_experiment_public_benchmark_eligible_v1(&receipt));
    }

    #[test]
    fn preprocessed_audio_and_config_artifacts_are_verified_by_hash() {
        let audio = b"audio";
        let midi = b"midi";
        let report = report(midi);
        let receipt = valid_receipt(audio, midi, &report);
        let mut evidence = TranscriptionExperimentEvidenceV1 {
            source_audio_bytes: audio,
            model_input_audio_bytes: b"replaced-model-input",
            dataset_manifest_bytes: b"dataset-manifest",
            inference_config_bytes: b"inference-config",
            preprocessing_config_bytes: b"preprocessing-config",
            model_artifact_bytes: Some(b"model-weights"),
            prediction_midi_bytes: midi,
        };
        let issues = validate_transcription_experiment_receipt_v1(
            &receipt, &evidence, &report
        );
        assert!(issues.contains(&TranscriptionExperimentReceiptIssueV1::ArtifactDigestMismatch {
            field: "model_input_audio_sha256".into()
        }));
        evidence.model_input_audio_bytes = b"canonical-model-input";
        evidence.inference_config_bytes = b"replaced-config";
        let issues = validate_transcription_experiment_receipt_v1(
            &receipt, &evidence, &report
        );
        assert!(issues.contains(&TranscriptionExperimentReceiptIssueV1::ArtifactDigestMismatch {
            field: "inference_config_sha256".into()
        }));
    }

    #[test]
    fn invalid_chunk_overlap_is_rejected() {
        let audio = b"audio";
        let midi = b"midi";
        let report = report(midi);
        let mut receipt = valid_receipt(audio, midi, &report);
        receipt.preprocessing.chunk_overlap_ms = 5_000;
        receipt.receipt_sha256 = transcription_experiment_receipt_commitment_v1(&receipt).unwrap();
        let issues = validate_fixture(
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
        let issues = validate_fixture(
            &receipt,
            audio,
            midi,
            &report,
        );
        assert!(issues.contains(&TranscriptionExperimentReceiptIssueV1::ReceiptDigestMismatch));
    }
}
