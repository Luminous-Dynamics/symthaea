#![forbid(unsafe_code)]

//! Byte-bound contracts for controlled, causal acoustic interventions.
//!
//! This crate validates manifests and execution receipts. It does not execute
//! audio transformations or claim that an intervention's expected effect is
//! acoustically or perceptually correct; independent evaluation owns that.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt;

const MANIFEST_VERSION: &str = "mel-hear-012-intervention-manifest-v0.1";
const RECEIPT_VERSION: &str = "mel-hear-012-intervention-receipt-v0.1";
const BENCHMARK_ID: &str = "MEL-CAUSAL-ACOUSTIC-INTERVENTION";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationSplit {
    Train,
    Development,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionFactor {
    MusicalContent,
    Performance,
    InstrumentSource,
    RoomChannel,
    ProductionProcessing,
    Corruption,
    ShamControl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedEffect {
    Invariant,
    Equivariant,
    NonInvariant,
    NotEstimable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampleSupport {
    pub start_sample: u64,
    pub end_sample: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterventionManifestV1 {
    pub schema_version: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub case_id: String,
    pub pair_id: String,
    pub split: EvaluationSplit,
    pub source_work_lineage_sha256: String,
    pub performer_session_cluster_sha256: String,
    pub baseline_audio_sha256: String,
    pub target_property: String,
    pub factor: InterventionFactor,
    pub intervention_family: String,
    pub parameter_manifest_sha256: String,
    pub implementation_sha256: String,
    pub seed: u64,
    pub deterministic: bool,
    pub expected_effect: ExpectedEffect,
    pub reference_annotation_sha256: String,
    pub rights_manifest_sha256: String,
    pub reference_hidden_from_model: bool,
    pub source_support: SampleSupport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterventionReceiptV1 {
    pub schema_version: String,
    pub case_id: String,
    pub pair_id: String,
    pub manifest_sha256: String,
    pub input_audio_sha256: String,
    pub output_audio_sha256: String,
    pub parameter_manifest_sha256: String,
    pub implementation_sha256: String,
    pub execution_environment_sha256: String,
    pub seed: u64,
    pub input_sample_rate_hz: u32,
    pub output_sample_rate_hz: u32,
    pub input_frame_count: u64,
    pub output_frame_count: u64,
    pub applied_start_sample: u64,
    pub applied_end_sample: u64,
    pub completed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterventionError {
    InvalidJson(String),
    UnsupportedVersion,
    InvalidDigest(&'static str),
    EmptyIdentifier(&'static str),
    NonDeterministicCase,
    ReferenceLeakage,
    EmptySupport,
    SupportExceedsInput,
    InvalidSampleGeometry,
    ManifestReceiptIdentityMismatch,
    ManifestDigestMismatch,
    InputArtifactMismatch,
    OutputArtifactMismatch,
    ParameterBindingMismatch,
    ImplementationBindingMismatch,
    SeedBindingMismatch,
    ScopeBindingMismatch,
    ShamMustPreserveTarget,
}

impl fmt::Display for InterventionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(reason) => write!(formatter, "invalid intervention JSON: {reason}"),
            Self::UnsupportedVersion => write!(formatter, "unsupported intervention schema version"),
            Self::InvalidDigest(field) => write!(formatter, "{field} must be a lowercase SHA-256 digest"),
            Self::EmptyIdentifier(field) => write!(formatter, "{field} must not be empty"),
            Self::NonDeterministicCase => write!(formatter, "benchmark interventions must declare deterministic execution"),
            Self::ReferenceLeakage => write!(formatter, "reference annotation must remain hidden from the model"),
            Self::EmptySupport => write!(formatter, "intervention sample support must be a non-empty half-open interval"),
            Self::SupportExceedsInput => write!(formatter, "intervention support exceeds the input audio frame count"),
            Self::InvalidSampleGeometry => write!(formatter, "input/output sample geometry is invalid"),
            Self::ManifestReceiptIdentityMismatch => write!(formatter, "manifest and receipt case/pair identities differ"),
            Self::ManifestDigestMismatch => write!(formatter, "receipt does not bind the exact manifest bytes"),
            Self::InputArtifactMismatch => write!(formatter, "receipt input digest does not match the supplied source bytes"),
            Self::OutputArtifactMismatch => write!(formatter, "receipt output digest does not match the supplied output bytes"),
            Self::ParameterBindingMismatch => write!(formatter, "receipt parameter digest differs from the manifest"),
            Self::ImplementationBindingMismatch => write!(formatter, "receipt implementation digest differs from the manifest"),
            Self::SeedBindingMismatch => write!(formatter, "receipt seed differs from the manifest"),
            Self::ScopeBindingMismatch => write!(formatter, "receipt applied scope differs from the declared source support"),
            Self::ShamMustPreserveTarget => write!(formatter, "sham controls must declare an invariant target property"),
        }
    }
}

impl Error for InterventionError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InterventionValidation {
    pub case_id: String,
    pub pair_id: String,
    pub factor: InterventionFactor,
    pub expected_effect: ExpectedEffect,
    pub input_audio_sha256: String,
    pub output_audio_sha256: String,
    pub manifest_sha256: String,
    pub source_support: SampleSupport,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_manifest(manifest: &InterventionManifestV1) -> Result<(), InterventionError> {
    if manifest.schema_version != MANIFEST_VERSION || manifest.benchmark_id != BENCHMARK_ID {
        return Err(InterventionError::UnsupportedVersion);
    }
    for (value, field) in [
        (manifest.source_work_lineage_sha256.as_str(), "source_work_lineage_sha256"),
        (manifest.performer_session_cluster_sha256.as_str(), "performer_session_cluster_sha256"),
        (manifest.baseline_audio_sha256.as_str(), "baseline_audio_sha256"),
        (manifest.parameter_manifest_sha256.as_str(), "parameter_manifest_sha256"),
        (manifest.implementation_sha256.as_str(), "implementation_sha256"),
        (manifest.reference_annotation_sha256.as_str(), "reference_annotation_sha256"),
        (manifest.rights_manifest_sha256.as_str(), "rights_manifest_sha256"),
    ] {
        if !valid_digest(value) {
            return Err(InterventionError::InvalidDigest(field));
        }
    }
    for (value, field) in [
        (manifest.benchmark_version.as_str(), "benchmark_version"),
        (manifest.case_id.as_str(), "case_id"),
        (manifest.pair_id.as_str(), "pair_id"),
        (manifest.target_property.as_str(), "target_property"),
        (manifest.intervention_family.as_str(), "intervention_family"),
    ] {
        if value.trim().is_empty() {
            return Err(InterventionError::EmptyIdentifier(field));
        }
    }
    if !manifest.deterministic {
        return Err(InterventionError::NonDeterministicCase);
    }
    if !manifest.reference_hidden_from_model {
        return Err(InterventionError::ReferenceLeakage);
    }
    if manifest.source_support.start_sample >= manifest.source_support.end_sample {
        return Err(InterventionError::EmptySupport);
    }
    if manifest.factor == InterventionFactor::ShamControl
        && manifest.expected_effect != ExpectedEffect::Invariant
    {
        return Err(InterventionError::ShamMustPreserveTarget);
    }
    Ok(())
}

/// Validate a manifest and receipt against the exact input and output audio bytes.
///
/// The receipt's manifest digest is computed over the exact serialized manifest
/// bytes supplied by the caller. The annotation digest is provenance only: the
/// reference annotation itself is not passed to this function or exposed to a
/// model input.
pub fn validate_intervention_pair(
    manifest_bytes: &[u8],
    receipt_bytes: &[u8],
    input_audio: &[u8],
    output_audio: &[u8],
) -> Result<InterventionValidation, InterventionError> {
    let manifest: InterventionManifestV1 = serde_json::from_slice(manifest_bytes)
        .map_err(|error| InterventionError::InvalidJson(error.to_string()))?;
    let receipt: InterventionReceiptV1 = serde_json::from_slice(receipt_bytes)
        .map_err(|error| InterventionError::InvalidJson(error.to_string()))?;

    validate_manifest(&manifest)?;

    if receipt.schema_version != RECEIPT_VERSION {
        return Err(InterventionError::UnsupportedVersion);
    }
    if !receipt.completed {
        return Err(InterventionError::InvalidSampleGeometry);
    }
    for (value, field) in [
        (receipt.manifest_sha256.as_str(), "manifest_sha256"),
        (receipt.input_audio_sha256.as_str(), "input_audio_sha256"),
        (receipt.output_audio_sha256.as_str(), "output_audio_sha256"),
        (receipt.parameter_manifest_sha256.as_str(), "parameter_manifest_sha256"),
        (receipt.implementation_sha256.as_str(), "implementation_sha256"),
        (receipt.execution_environment_sha256.as_str(), "execution_environment_sha256"),
    ] {
        if !valid_digest(value) {
            return Err(InterventionError::InvalidDigest(field));
        }
    }

    if receipt.case_id != manifest.case_id || receipt.pair_id != manifest.pair_id {
        return Err(InterventionError::ManifestReceiptIdentityMismatch);
    }
    if receipt.manifest_sha256 != sha256_hex(manifest_bytes) {
        return Err(InterventionError::ManifestDigestMismatch);
    }
    if receipt.input_audio_sha256 != sha256_hex(input_audio)
        || receipt.input_audio_sha256 != manifest.baseline_audio_sha256
    {
        return Err(InterventionError::InputArtifactMismatch);
    }
    if receipt.output_audio_sha256 != sha256_hex(output_audio) {
        return Err(InterventionError::OutputArtifactMismatch);
    }
    if receipt.parameter_manifest_sha256 != manifest.parameter_manifest_sha256 {
        return Err(InterventionError::ParameterBindingMismatch);
    }
    if receipt.implementation_sha256 != manifest.implementation_sha256 {
        return Err(InterventionError::ImplementationBindingMismatch);
    }
    if receipt.seed != manifest.seed {
        return Err(InterventionError::SeedBindingMismatch);
    }
    if receipt.applied_start_sample != manifest.source_support.start_sample
        || receipt.applied_end_sample != manifest.source_support.end_sample
    {
        return Err(InterventionError::ScopeBindingMismatch);
    }
    if receipt.input_sample_rate_hz == 0
        || receipt.output_sample_rate_hz == 0
        || receipt.input_frame_count == 0
        || receipt.output_frame_count == 0
        || receipt.applied_end_sample > receipt.input_frame_count
        || receipt.input_frame_count as usize > input_audio.len()
    {
        return Err(InterventionError::InvalidSampleGeometry);
    }
    if manifest.source_support.end_sample > receipt.input_frame_count {
        return Err(InterventionError::SupportExceedsInput);
    }

    Ok(InterventionValidation {
        case_id: manifest.case_id,
        pair_id: manifest.pair_id,
        factor: manifest.factor,
        expected_effect: manifest.expected_effect,
        input_audio_sha256: receipt.input_audio_sha256,
        output_audio_sha256: receipt.output_audio_sha256,
        manifest_sha256: receipt.manifest_sha256,
        source_support: manifest.source_support,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        validate_intervention_pair, ExpectedEffect, InterventionError, InterventionFactor,
        InterventionManifestV1, InterventionReceiptV1, EvaluationSplit, SampleSupport,
    };
    use serde_json::to_vec;
    use sha2::{Digest, Sha256};

    fn digest(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn valid_case() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let input_audio = b"baseline synthetic audio fixture".to_vec();
        let output_audio = b"room-response intervention output".to_vec();
        let mut manifest = InterventionManifestV1 {
            schema_version: "mel-hear-012-intervention-manifest-v0.1".to_owned(),
            benchmark_id: "MEL-CAUSAL-ACOUSTIC-INTERVENTION".to_owned(),
            benchmark_version: "0.1.0".to_owned(),
            case_id: "room-reverb-stay-001".to_owned(),
            pair_id: "pair-001".to_owned(),
            split: EvaluationSplit::Development,
            source_work_lineage_sha256: digest(b"work-lineage"),
            performer_session_cluster_sha256: digest(b"performer-session"),
            baseline_audio_sha256: digest(&input_audio),
            target_property: "chord_quality".to_owned(),
            factor: InterventionFactor::RoomChannel,
            intervention_family: "convolutional_room_response".to_owned(),
            parameter_manifest_sha256: digest(b"reverb-parameters"),
            implementation_sha256: digest(b"renderer-binary"),
            seed: 11,
            deterministic: true,
            expected_effect: ExpectedEffect::Invariant,
            reference_annotation_sha256: digest(b"hidden-reference"),
            rights_manifest_sha256: digest(b"rights-and-consent"),
            reference_hidden_from_model: true,
            source_support: SampleSupport {
                start_sample: 0,
                end_sample: 64,
            },
        };
        let manifest_bytes = to_vec(&manifest).unwrap();
        let receipt = InterventionReceiptV1 {
            schema_version: "mel-hear-012-intervention-receipt-v0.1".to_owned(),
            case_id: manifest.case_id.clone(),
            pair_id: manifest.pair_id.clone(),
            manifest_sha256: digest(&manifest_bytes),
            input_audio_sha256: digest(&input_audio),
            output_audio_sha256: digest(&output_audio),
            parameter_manifest_sha256: manifest.parameter_manifest_sha256.clone(),
            implementation_sha256: manifest.implementation_sha256.clone(),
            execution_environment_sha256: digest(b"pinned-execution-environment"),
            seed: manifest.seed,
            input_sample_rate_hz: 8000,
            output_sample_rate_hz: 8000,
            input_frame_count: 128,
            output_frame_count: 128,
            applied_start_sample: manifest.source_support.start_sample,
            applied_end_sample: manifest.source_support.end_sample,
            completed: true,
        };
        let receipt_bytes = to_vec(&receipt).unwrap();
        (manifest_bytes, receipt_bytes, input_audio, output_audio)
    }

    #[test]
    fn exact_manifest_receipt_and_audio_bytes_validate() {
        let (manifest, receipt, input, output) = valid_case();
        let result = validate_intervention_pair(&manifest, &receipt, &input, &output).unwrap();
        assert_eq!(result.case_id, "room-reverb-stay-001");
        assert_eq!(result.factor, InterventionFactor::RoomChannel);
        assert_eq!(result.expected_effect, ExpectedEffect::Invariant);
    }

    #[test]
    fn source_byte_substitution_rejects() {
        let (manifest, receipt, mut input, output) = valid_case();
        input[0] ^= 1;
        assert_eq!(
            validate_intervention_pair(&manifest, &receipt, &input, &output),
            Err(InterventionError::InputArtifactMismatch)
        );
    }

    #[test]
    fn output_byte_substitution_rejects() {
        let (manifest, receipt, input, mut output) = valid_case();
        output[0] ^= 1;
        assert_eq!(
            validate_intervention_pair(&manifest, &receipt, &input, &output),
            Err(InterventionError::OutputArtifactMismatch)
        );
    }

    #[test]
    fn receipt_cannot_be_replayed_against_another_manifest() {
        let (manifest, receipt, input, output) = valid_case();
        let mut changed = manifest.clone();
        let mut changed_json: serde_json::Value = serde_json::from_slice(&changed).unwrap();
        changed_json["target_property"] = serde_json::json!("root_pitch_class");
        changed = to_vec(&changed_json).unwrap();
        assert_eq!(
            validate_intervention_pair(&changed, &receipt, &input, &output),
            Err(InterventionError::ManifestDigestMismatch)
        );
        let _ = manifest;
    }

    #[test]
    fn sham_control_must_declare_an_invariant_target() {
        let (manifest, receipt, input, output) = valid_case();
        let mut json: serde_json::Value = serde_json::from_slice(&manifest).unwrap();
        json["factor"] = serde_json::json!("sham_control");
        json["expected_effect"] = serde_json::json!("non_invariant");
        let bad_manifest = to_vec(&json).unwrap();
        assert_eq!(
            validate_intervention_pair(&bad_manifest, &receipt, &input, &output),
            Err(InterventionError::ShamMustPreserveTarget)
        );
    }

    #[test]
    fn unblinded_reference_annotation_rejects() {
        let (manifest, receipt, input, output) = valid_case();
        let mut json: serde_json::Value = serde_json::from_slice(&manifest).unwrap();
        json["reference_hidden_from_model"] = serde_json::json!(false);
        let bad_manifest = to_vec(&json).unwrap();
        assert_eq!(
            validate_intervention_pair(&bad_manifest, &receipt, &input, &output),
            Err(InterventionError::ReferenceLeakage)
        );
    }

    #[test]
    fn receipt_must_bind_the_declared_input_support() {
        let (manifest, receipt, input, output) = valid_case();
        let mut json: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        json["applied_end_sample"] = serde_json::json!(63);
        let bad_receipt = to_vec(&json).unwrap();
        assert_eq!(
            validate_intervention_pair(&manifest, &bad_receipt, &input, &output),
            Err(InterventionError::ScopeBindingMismatch)
        );
    }
}
