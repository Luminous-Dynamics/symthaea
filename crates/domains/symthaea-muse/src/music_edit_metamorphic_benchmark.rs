// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pair contracts for metamorphic music-edit invariance and equivariance.
//! This module validates declarations; it does not execute transforms or editors.

use crate::music_edit_benchmark_adapter::{MusicEditOperationV1, MusicEditSplitV1};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MUSIC_EDIT_METAMORPHIC_PAIR_SCHEMA_V1: &str = "mel-music-edit-metamorphic-pair-v1";
const DOMAIN: &[u8] = b"luminous.mel.music-edit-metamorphic-pair.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditMetamorphicTransformFamilyV1 {
    MetadataOnly, LosslessReencode, GainHeadroom, ChannelPermutation,
    BenignCropPadding, RendererChange, GlobalTransposition, TempoScaling,
    KeyScaleShift, MeterPreservingRhythmScale, SpatialPermutation,
    TargetCrop, TargetMasking, AbsolutePitchShift, DynamicCompression,
    SeparationTopologyChange, InstructionScopeChange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditMetamorphicRelationV1 {
    Invariant, Equivariant, NonInvariant, AbstainExpected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditMetamorphicPairV1 {
    pub schema_version: String,
    pub pair_id: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub benchmark_track_id: String,
    pub base_sample_id: String,
    pub transformed_sample_id: String,
    pub base_source_lineage_sha256: String,
    pub transformed_source_lineage_sha256: String,
    pub base_split: MusicEditSplitV1,
    pub transformed_split: MusicEditSplitV1,
    pub operation: MusicEditOperationV1,
    pub base_source_audio_sha256: String,
    pub transformed_source_audio_sha256: String,
    pub base_instruction_sha256: String,
    pub transformed_instruction_sha256: String,
    pub base_validity_envelope_sha256: String,
    pub transformed_validity_envelope_sha256: String,
    pub transform_family: MusicEditMetamorphicTransformFamilyV1,
    pub transform_implementation_sha256: String,
    pub transform_receipt_sha256: String,
    pub transform_version: String,
    pub transform_parameters_sha256: String,
    pub expected_relation: MusicEditMetamorphicRelationV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub property_mapping_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admissibility_reason: Option<String>,
}

pub const MUSIC_EDIT_METAMORPHIC_TRANSFORM_RECEIPT_SCHEMA_V1: &str =
    "mel-music-edit-metamorphic-transform-receipt-v1";

/// Content-addressed summary of the transformation that produced a paired input.
/// The hashes must be computed from the actual input/output bytes by the transform
/// executor; this structure alone does not prove the executor read those bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditMetamorphicTransformReceiptV1 {
    pub schema_version: String,
    pub pair_id: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub benchmark_track_id: String,
    pub base_sample_id: String,
    pub transformed_sample_id: String,
    pub source_lineage_sha256: String,
    pub split: MusicEditSplitV1,
    pub transform_family: MusicEditMetamorphicTransformFamilyV1,
    pub input_audio_sha256: String,
    pub output_audio_sha256: String,
    pub base_instruction_sha256: String,
    pub transformed_instruction_sha256: String,
    pub transform_implementation_sha256: String,
    pub transform_version: String,
    pub transform_parameters_sha256: String,
    /// SHA-256 of the canonical receipt commitment with this field empty.
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct MusicEditMetamorphicTransformReceiptCommitmentV1<'a> {
    schema_version: &'a str,
    pair_id: &'a str,
    dataset_id: &'a str,
    dataset_version: &'a str,
    benchmark_track_id: &'a str,
    base_sample_id: &'a str,
    transformed_sample_id: &'a str,
    source_lineage_sha256: &'a str,
    split: MusicEditSplitV1,
    transform_family: MusicEditMetamorphicTransformFamilyV1,
    input_audio_sha256: &'a str,
    output_audio_sha256: &'a str,
    base_instruction_sha256: &'a str,
    transformed_instruction_sha256: &'a str,
    transform_implementation_sha256: &'a str,
    transform_version: &'a str,
    transform_parameters_sha256: &'a str,
}

pub fn music_edit_metamorphic_transform_receipt_digest_v1(
    receipt: &MusicEditMetamorphicTransformReceiptV1,
) -> Result<String, MusicEditMetamorphicIssueV1> {
    let commitment = MusicEditMetamorphicTransformReceiptCommitmentV1 {
        schema_version: &receipt.schema_version,
        pair_id: &receipt.pair_id,
        dataset_id: &receipt.dataset_id,
        dataset_version: &receipt.dataset_version,
        benchmark_track_id: &receipt.benchmark_track_id,
        base_sample_id: &receipt.base_sample_id,
        transformed_sample_id: &receipt.transformed_sample_id,
        source_lineage_sha256: &receipt.source_lineage_sha256,
        split: receipt.split,
        transform_family: receipt.transform_family,
        input_audio_sha256: &receipt.input_audio_sha256,
        output_audio_sha256: &receipt.output_audio_sha256,
        base_instruction_sha256: &receipt.base_instruction_sha256,
        transformed_instruction_sha256: &receipt.transformed_instruction_sha256,
        transform_implementation_sha256: &receipt.transform_implementation_sha256,
        transform_version: &receipt.transform_version,
        transform_parameters_sha256: &receipt.transform_parameters_sha256,
    };
    let bytes = serde_json::to_vec(&commitment)
        .map_err(|_| MusicEditMetamorphicIssueV1::SerializationError)?;
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN);
    hasher.update(b"\0transform-receipt\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Bind a transform receipt to the exact pair declarations. This validates the
/// receipt commitment and field correspondence; actual-byte hashing remains the
/// responsibility of the materializer/transform executor.
pub fn verify_music_edit_metamorphic_transform_receipt_v1(
    pair: &MusicEditMetamorphicPairV1,
    receipt: &MusicEditMetamorphicTransformReceiptV1,
) -> Vec<MusicEditMetamorphicIssueV1> {
    let mut issues = Vec::new();
    if receipt.schema_version != MUSIC_EDIT_METAMORPHIC_TRANSFORM_RECEIPT_SCHEMA_V1 {
        issues.push(MusicEditMetamorphicIssueV1::WrongTransformReceiptSchema);
    }
    if receipt.pair_id != pair.pair_id
        || receipt.dataset_id != pair.dataset_id
        || receipt.dataset_version != pair.dataset_version
        || receipt.benchmark_track_id != pair.benchmark_track_id
        || receipt.base_sample_id != pair.base_sample_id
        || receipt.transformed_sample_id != pair.transformed_sample_id
    {
        issues.push(MusicEditMetamorphicIssueV1::TransformReceiptIdentityMismatch);
    }
    if receipt.source_lineage_sha256 != pair.base_source_lineage_sha256 {
        issues.push(MusicEditMetamorphicIssueV1::SourceLineageMismatch);
    }
    if receipt.split != pair.base_split || pair.base_split != pair.transformed_split {
        issues.push(MusicEditMetamorphicIssueV1::SplitMismatch);
    }
    if receipt.transform_family != pair.transform_family
        || receipt.transform_implementation_sha256 != pair.transform_implementation_sha256
        || receipt.transform_version != pair.transform_version
        || receipt.transform_parameters_sha256 != pair.transform_parameters_sha256
    {
        issues.push(MusicEditMetamorphicIssueV1::TransformReceiptConfigurationMismatch);
    }
    if receipt.input_audio_sha256 != pair.base_source_audio_sha256
        || receipt.output_audio_sha256 != pair.transformed_source_audio_sha256
        || receipt.base_instruction_sha256 != pair.base_instruction_sha256
        || receipt.transformed_instruction_sha256 != pair.transformed_instruction_sha256
    {
        issues.push(MusicEditMetamorphicIssueV1::TransformReceiptArtifactMismatch);
    }
    if !is_sha256(&receipt.input_audio_sha256)
        || !is_sha256(&receipt.output_audio_sha256)
        || !is_sha256(&receipt.source_lineage_sha256)
        || !is_sha256(&receipt.transform_implementation_sha256)
        || !is_sha256(&receipt.transform_parameters_sha256)
        || !is_sha256(&receipt.base_instruction_sha256)
        || !is_sha256(&receipt.transformed_instruction_sha256)
    {
        issues.push(MusicEditMetamorphicIssueV1::TransformReceiptInvalidDigest);
    }
    match music_edit_metamorphic_transform_receipt_digest_v1(receipt) {
        Ok(digest) if digest == receipt.receipt_sha256 && digest == pair.transform_receipt_sha256 => {}
        _ => issues.push(MusicEditMetamorphicIssueV1::TransformReceiptDigestMismatch),
    }
    issues
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicEditMetamorphicIssueV1 {
    WrongSchema, EmptyField(&'static str), MutableVersion, InvalidDigest(&'static str),
    SameSampleIdentity, SourceLineageMismatch, SplitMismatch,
    InvariantInstructionChanged, InvariantEnvelopeChanged, RelationMapMissing,
    AdmissibilityReasonMissing, UnexpectedAdmissibilityReason,
    WrongTransformReceiptSchema, TransformReceiptIdentityMismatch,
    TransformReceiptConfigurationMismatch, TransformReceiptArtifactMismatch,
    TransformReceiptInvalidDigest, TransformReceiptDigestMismatch,
    InvalidPair, SerializationError,
}

impl std::fmt::Display for MusicEditMetamorphicIssueV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for MusicEditMetamorphicIssueV1 {}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn require(issues: &mut Vec<MusicEditMetamorphicIssueV1>, field: &'static str, value: &str) {
    if value.trim().is_empty() {
        issues.push(MusicEditMetamorphicIssueV1::EmptyField(field));
    }
}

fn check_digest(issues: &mut Vec<MusicEditMetamorphicIssueV1>, field: &'static str, value: &str) {
    if !is_sha256(value) {
        issues.push(MusicEditMetamorphicIssueV1::InvalidDigest(field));
    }
}

/// Invariant: instruction and validity envelope are unchanged.
/// Equivariant: transformed task and envelope are bound to a property map.
/// NonInvariant/AbstainExpected: a reason must explain why equality is not expected.
pub fn validate_music_edit_metamorphic_pair_v1(
    pair: &MusicEditMetamorphicPairV1,
) -> Vec<MusicEditMetamorphicIssueV1> {
    use MusicEditMetamorphicRelationV1 as R;
    let mut issues = Vec::new();

    if pair.schema_version != MUSIC_EDIT_METAMORPHIC_PAIR_SCHEMA_V1 {
        issues.push(MusicEditMetamorphicIssueV1::WrongSchema);
    }
    for (field, value) in [
        ("pair_id", pair.pair_id.as_str()), ("dataset_id", pair.dataset_id.as_str()),
        ("dataset_version", pair.dataset_version.as_str()),
        ("benchmark_track_id", pair.benchmark_track_id.as_str()),
        ("base_sample_id", pair.base_sample_id.as_str()),
        ("transformed_sample_id", pair.transformed_sample_id.as_str()),
        ("transform_version", pair.transform_version.as_str()),
    ] { require(&mut issues, field, value); }

    if ["latest", "current", "stable"].contains(&pair.dataset_version.to_ascii_lowercase().as_str())
        || ["latest", "current", "stable"].contains(&pair.transform_version.to_ascii_lowercase().as_str())
    {
        issues.push(MusicEditMetamorphicIssueV1::MutableVersion);
    }
    if pair.base_sample_id == pair.transformed_sample_id {
        issues.push(MusicEditMetamorphicIssueV1::SameSampleIdentity);
    }
    if pair.base_source_lineage_sha256 != pair.transformed_source_lineage_sha256 {
        issues.push(MusicEditMetamorphicIssueV1::SourceLineageMismatch);
    }
    if pair.base_split != pair.transformed_split {
        issues.push(MusicEditMetamorphicIssueV1::SplitMismatch);
    }

    for (field, digest) in [
        ("base_source_lineage_sha256", pair.base_source_lineage_sha256.as_str()),
        ("transformed_source_lineage_sha256", pair.transformed_source_lineage_sha256.as_str()),
        ("base_source_audio_sha256", pair.base_source_audio_sha256.as_str()),
        ("transformed_source_audio_sha256", pair.transformed_source_audio_sha256.as_str()),
        ("base_instruction_sha256", pair.base_instruction_sha256.as_str()),
        ("transformed_instruction_sha256", pair.transformed_instruction_sha256.as_str()),
        ("base_validity_envelope_sha256", pair.base_validity_envelope_sha256.as_str()),
        ("transformed_validity_envelope_sha256", pair.transformed_validity_envelope_sha256.as_str()),
        ("transform_implementation_sha256", pair.transform_implementation_sha256.as_str()),
        ("transform_receipt_sha256", pair.transform_receipt_sha256.as_str()),
        ("transform_parameters_sha256", pair.transform_parameters_sha256.as_str()),
    ] { check_digest(&mut issues, field, digest); }
    if let Some(digest) = pair.property_mapping_sha256.as_deref() {
        check_digest(&mut issues, "property_mapping_sha256", digest);
    }

    match pair.expected_relation {
        R::Invariant => {
            if pair.base_instruction_sha256 != pair.transformed_instruction_sha256 {
                issues.push(MusicEditMetamorphicIssueV1::InvariantInstructionChanged);
            }
            if pair.base_validity_envelope_sha256 != pair.transformed_validity_envelope_sha256 {
                issues.push(MusicEditMetamorphicIssueV1::InvariantEnvelopeChanged);
            }
            if pair.property_mapping_sha256.is_none() {
                issues.push(MusicEditMetamorphicIssueV1::RelationMapMissing);
            }
            if pair.admissibility_reason.as_ref().is_some_and(|v| !v.trim().is_empty()) {
                issues.push(MusicEditMetamorphicIssueV1::UnexpectedAdmissibilityReason);
            }
        }
        R::Equivariant => {
            if pair.property_mapping_sha256.is_none() {
                issues.push(MusicEditMetamorphicIssueV1::RelationMapMissing);
            }
            if pair.admissibility_reason.as_ref().is_some_and(|v| !v.trim().is_empty()) {
                issues.push(MusicEditMetamorphicIssueV1::UnexpectedAdmissibilityReason);
            }
        }
        R::NonInvariant | R::AbstainExpected => {
            if !pair.admissibility_reason.as_ref().is_some_and(|v| !v.trim().is_empty()) {
                issues.push(MusicEditMetamorphicIssueV1::AdmissibilityReasonMissing);
            }
        }
    }
    issues
}

pub fn music_edit_metamorphic_pair_digest_v1(
    pair: &MusicEditMetamorphicPairV1,
) -> Result<String, MusicEditMetamorphicIssueV1> {
    if !validate_music_edit_metamorphic_pair_v1(pair).is_empty() {
        return Err(MusicEditMetamorphicIssueV1::InvalidPair);
    }
    let bytes = serde_json::to_vec(pair).map_err(|_| MusicEditMetamorphicIssueV1::SerializationError)?;
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN);
    hasher.update(b"\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditMetamorphicEvidenceV1 { Symbolic, Acoustic, HumanAdjudication }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MusicEditMetamorphicValueV1 {
    Numeric { value: f64, unit: String },
    Category { value: String },
    Boolean { value: bool },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditMetamorphicPropertyObservationV1 {
    pub property_id: String,
    pub evidence: MusicEditMetamorphicEvidenceV1,
    pub value: MusicEditMetamorphicValueV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditMetamorphicObservationV1 {
    pub schema_version: String,
    pub pair_sha256: String,
    pub sample_id: String,
    pub source_lineage_sha256: String,
    pub split: MusicEditSplitV1,
    /// Must be derived from the independent operation verifier, not a model self-report.
    pub obligation_satisfied: bool,
    /// Must be derived from the independent protected-content verifier.
    pub protected_invariants_satisfied: bool,
    pub operation_report_sha256: String,
    pub validity_report_sha256: String,
    pub properties: Vec<MusicEditMetamorphicPropertyObservationV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MusicEditMetamorphicPropertyRuleV1 {
    NumericAffine { scale: f64, offset: f64, tolerance: f64, unit: String },
    CategoryMap { values: Vec<MusicEditMetamorphicCategoryMapV1> },
    BooleanIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditMetamorphicCategoryMapV1 {
    pub source_value: String,
    pub transformed_value: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditMetamorphicPropertyMappingV1 {
    pub schema_version: String,
    pub properties: Vec<MusicEditMetamorphicPropertyRuleEntryV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditMetamorphicPropertyRuleEntryV1 {
    pub property_id: String,
    pub rule: MusicEditMetamorphicPropertyRuleV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicEditMetamorphicResidualV1 {
    pub property_id: String,
    pub numeric_residual: Option<f64>,
    pub tolerance: Option<f64>,
    pub matched: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicEditMetamorphicAssessmentV1 {
    pub schema_version: String,
    pub pair_sha256: String,
    pub base_observation_sha256: String,
    pub transformed_observation_sha256: String,
    pub property_mapping_sha256: String,
    pub expected_relation: MusicEditMetamorphicRelationV1,
    pub base_obligation_satisfied: bool,
    pub transformed_obligation_satisfied: bool,
    pub base_preservation_satisfied: bool,
    pub transformed_preservation_satisfied: bool,
    pub relation_satisfied: bool,
    pub passed: bool,
    pub residuals: Vec<MusicEditMetamorphicResidualV1>,
    pub assessment_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicEditMetamorphicAssessmentIssueV1 {
    NotEvaluableRelation,
    WrongObservationSchema,
    ObservationPairMismatch,
    ObservationSampleMismatch,
    ObservationLineageMismatch,
    ObservationSplitMismatch,
    InvalidEvidenceDigest(&'static str),
    EmptyPropertyId,
    DuplicateOrUnorderedProperties,
    NonFiniteNumericValue(String),
    MissingProperty(String),
    PropertyEvidenceMismatch(String),
    PropertyTypeMismatch(String),
    NumericUnitMismatch(String),
    InvalidPropertyMapping,
    PropertyMappingDigestMismatch,
    InvalidNumericRule(String),
    NonIdentityRuleForInvariant(String),
    MissingCategoryMapping(String),
    UnmappedObservationProperty(String),
    InvalidCategoryMap(String),
    SerializationError,
}

fn canonical_metamorphic_json_digest<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<String, MusicEditMetamorphicAssessmentIssueV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| MusicEditMetamorphicAssessmentIssueV1::SerializationError)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous.mel.metamorphic-assessment.v1\0");
    hasher.update(domain);
    hasher.update(b"\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn music_edit_metamorphic_property_mapping_digest_v1(
    mapping: &MusicEditMetamorphicPropertyMappingV1,
) -> Result<String, MusicEditMetamorphicAssessmentIssueV1> {
    canonical_metamorphic_json_digest(b"property-map", mapping)
}

fn validate_metamorphic_observation_v1(
    pair: &MusicEditMetamorphicPairV1,
    pair_sha256: &str,
    observation: &MusicEditMetamorphicObservationV1,
    base: bool,
) -> Vec<MusicEditMetamorphicAssessmentIssueV1> {
    let mut issues = Vec::new();
    if observation.schema_version != "mel-music-edit-metamorphic-observation-v1" {
        issues.push(MusicEditMetamorphicAssessmentIssueV1::WrongObservationSchema);
    }
    if observation.pair_sha256 != pair_sha256 {
        issues.push(MusicEditMetamorphicAssessmentIssueV1::ObservationPairMismatch);
    }
    let expected_sample = if base { &pair.base_sample_id } else { &pair.transformed_sample_id };
    if &observation.sample_id != expected_sample {
        issues.push(MusicEditMetamorphicAssessmentIssueV1::ObservationSampleMismatch);
    }
    if observation.source_lineage_sha256 != pair.base_source_lineage_sha256 {
        issues.push(MusicEditMetamorphicAssessmentIssueV1::ObservationLineageMismatch);
    }
    let expected_split = pair.base_split;
    if observation.split != expected_split {
        issues.push(MusicEditMetamorphicAssessmentIssueV1::ObservationSplitMismatch);
    }
    for (field, digest) in [
        ("operation_report_sha256", observation.operation_report_sha256.as_str()),
        ("validity_report_sha256", observation.validity_report_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(MusicEditMetamorphicAssessmentIssueV1::InvalidEvidenceDigest(field));
        }
    }
    if !observation.properties.windows(2).all(|w| w[0].property_id < w[1].property_id) {
        issues.push(MusicEditMetamorphicAssessmentIssueV1::DuplicateOrUnorderedProperties);
    }
    for property in &observation.properties {
        if property.property_id.trim().is_empty() {
            issues.push(MusicEditMetamorphicAssessmentIssueV1::EmptyPropertyId);
        }
        if let MusicEditMetamorphicValueV1::Numeric { value, unit } = &property.value {
            if !value.is_finite() || unit.trim().is_empty() {
                issues.push(MusicEditMetamorphicAssessmentIssueV1::NonFiniteNumericValue(
                    property.property_id.clone(),
                ));
            }
        }
        if let MusicEditMetamorphicValueV1::Category { value } = &property.value {
            if value.trim().is_empty() {
                issues.push(MusicEditMetamorphicAssessmentIssueV1::EmptyPropertyId);
            }
        }
    }
    issues
}

/// Compare independently verified base/transformed edit observations under a typed
/// property map. The relation can pass only if both edits satisfy their requested
/// obligations and both preserve their declared protected content.
/// Recompute the content commitment and verify that the pass/fail summary
/// agrees with the recorded per-property residuals and independent gates.
pub fn verify_music_edit_metamorphic_assessment_v1(
    assessment: &MusicEditMetamorphicAssessmentV1,
) -> Result<(), MusicEditMetamorphicAssessmentIssueV1> {
    let relation_satisfied = assessment.residuals.iter().all(|residual| residual.matched);
    let expected_pass = assessment.base_obligation_satisfied
        && assessment.transformed_obligation_satisfied
        && assessment.base_preservation_satisfied
        && assessment.transformed_preservation_satisfied
        && relation_satisfied;
    if relation_satisfied != assessment.relation_satisfied || expected_pass != assessment.passed {
        return Err(MusicEditMetamorphicAssessmentIssueV1::SerializationError);
    }
    let mut commitment = assessment.clone();
    commitment.assessment_sha256.clear();
    let expected = canonical_metamorphic_json_digest(b"assessment", &commitment)?;
    if expected != assessment.assessment_sha256 {
        return Err(MusicEditMetamorphicAssessmentIssueV1::SerializationError);
    }
    Ok(())
}

pub fn evaluate_music_edit_metamorphic_pair_v1(
    pair: &MusicEditMetamorphicPairV1,
    mapping: &MusicEditMetamorphicPropertyMappingV1,
    base: &MusicEditMetamorphicObservationV1,
    transformed: &MusicEditMetamorphicObservationV1,
) -> Result<MusicEditMetamorphicAssessmentV1, Vec<MusicEditMetamorphicAssessmentIssueV1>> {
    use MusicEditMetamorphicRelationV1 as Relation;
    use MusicEditMetamorphicValueV1 as Value;
    use MusicEditMetamorphicPropertyRuleV1 as Rule;

    let mut issues = Vec::new();
    if !matches!(pair.expected_relation, Relation::Invariant | Relation::Equivariant) {
        issues.push(MusicEditMetamorphicAssessmentIssueV1::NotEvaluableRelation);
        return Err(issues);
    }
    let pair_sha256 = music_edit_metamorphic_pair_digest_v1(pair)
        .map_err(|_| vec![MusicEditMetamorphicAssessmentIssueV1::SerializationError])?;
    issues.extend(validate_metamorphic_observation_v1(pair, &pair_sha256, base, true));
    issues.extend(validate_metamorphic_observation_v1(pair, &pair_sha256, transformed, false));

    if mapping.schema_version != "mel-music-edit-metamorphic-property-map-v1"
        || mapping.properties.is_empty()
        || !mapping.properties.windows(2).all(|w| {
            !w[0].property_id.trim().is_empty()
                && w[0].property_id < w[1].property_id
        })
        || mapping.properties.last().is_some_and(|entry| entry.property_id.trim().is_empty())
    {
        issues.push(MusicEditMetamorphicAssessmentIssueV1::InvalidPropertyMapping);
    }

    let mapped_ids: std::collections::BTreeSet<&str> =
        mapping.properties.iter().map(|entry| entry.property_id.as_str()).collect();
    for observation in [base, transformed] {
        for property in &observation.properties {
            if !mapped_ids.contains(property.property_id.as_str()) {
                issues.push(MusicEditMetamorphicAssessmentIssueV1::UnmappedObservationProperty(
                    property.property_id.clone(),
                ));
            }
        }
    }
    for entry in &mapping.properties {
        if let Rule::CategoryMap { values } = &entry.rule {
            if values.is_empty()
                || !values.windows(2).all(|window| {
                    window[0].source_value < window[1].source_value
                })
                || values.iter().any(|mapping| {
                    mapping.source_value.trim().is_empty()
                        || mapping.transformed_value.trim().is_empty()
                })
            {
                issues.push(MusicEditMetamorphicAssessmentIssueV1::InvalidCategoryMap(
                    entry.property_id.clone(),
                ));
            }
            if pair.expected_relation == Relation::Invariant
                && values.iter().any(|mapping| {
                    mapping.source_value != mapping.transformed_value
                })
            {
                issues.push(MusicEditMetamorphicAssessmentIssueV1::NonIdentityRuleForInvariant(
                    entry.property_id.clone(),
                ));
            }
        }
    }

    let mapping_sha256 = music_edit_metamorphic_property_mapping_digest_v1(mapping)
        .map_err(|_| vec![MusicEditMetamorphicAssessmentIssueV1::SerializationError])?;
    if pair.property_mapping_sha256.as_deref() != Some(mapping_sha256.as_str()) {
        issues.push(MusicEditMetamorphicAssessmentIssueV1::PropertyMappingDigestMismatch);
    }

    let mut residuals = Vec::new();
    for entry in &mapping.properties {
        let left = base.properties.iter().find(|p| p.property_id == entry.property_id);
        let right = transformed.properties.iter().find(|p| p.property_id == entry.property_id);
        let (Some(left), Some(right)) = (left, right) else {
            issues.push(MusicEditMetamorphicAssessmentIssueV1::MissingProperty(
                entry.property_id.clone(),
            ));
            continue;
        };
        if left.evidence != right.evidence {
            issues.push(MusicEditMetamorphicAssessmentIssueV1::PropertyEvidenceMismatch(
                entry.property_id.clone(),
            ));
            continue;
        }

        let mut matched = false;
        let mut numeric_residual = None;
        let mut tolerance = None;
        match (&entry.rule, &left.value, &right.value) {
            (Rule::NumericAffine { scale, offset, tolerance: tol, unit },
             Value::Numeric { value: source, unit: source_unit },
             Value::Numeric { value: actual, unit: actual_unit }) => {
                if !scale.is_finite() || !offset.is_finite() || !tol.is_finite()
                    || *tol < 0.0 || unit.trim().is_empty()
                {
                    issues.push(MusicEditMetamorphicAssessmentIssueV1::InvalidNumericRule(
                        entry.property_id.clone(),
                    ));
                    continue;
                }
                if source_unit != unit || actual_unit != unit {
                    issues.push(MusicEditMetamorphicAssessmentIssueV1::NumericUnitMismatch(
                        entry.property_id.clone(),
                    ));
                    continue;
                }
                if pair.expected_relation == Relation::Invariant && (*scale != 1.0 || *offset != 0.0) {
                    issues.push(MusicEditMetamorphicAssessmentIssueV1::NonIdentityRuleForInvariant(
                        entry.property_id.clone(),
                    ));
                    continue;
                }
                let residual = *actual - (*source * *scale + *offset);
                numeric_residual = Some(residual);
                tolerance = Some(*tol);
                matched = residual.abs() <= *tol;
            }
            (Rule::CategoryMap { values }, Value::Category { value: source }, Value::Category { value: actual }) => {
                let expected = values.iter().find(|mapping| mapping.source_value == *source);
                match expected {
                    Some(mapping) => {
                        if pair.expected_relation == Relation::Invariant
                            && mapping.source_value != mapping.transformed_value
                        {
                            issues.push(MusicEditMetamorphicAssessmentIssueV1::NonIdentityRuleForInvariant(
                                entry.property_id.clone(),
                            ));
                            continue;
                        }
                        matched = mapping.transformed_value == *actual;
                    }
                    None => {
                        issues.push(MusicEditMetamorphicAssessmentIssueV1::MissingCategoryMapping(
                            entry.property_id.clone(),
                        ));
                        continue;
                    }
                }
            }
            (Rule::BooleanIdentity, Value::Boolean { value: source }, Value::Boolean { value: actual }) => {
                if *source != *actual {
                    matched = false;
                } else {
                    matched = true;
                }
            }
            _ => {
                issues.push(MusicEditMetamorphicAssessmentIssueV1::PropertyTypeMismatch(
                    entry.property_id.clone(),
                ));
                continue;
            }
        }
        residuals.push(MusicEditMetamorphicResidualV1 {
            property_id: entry.property_id.clone(),
            numeric_residual,
            tolerance,
            matched,
        });
    }

    if !issues.is_empty() {
        return Err(issues);
    }
    let relation_satisfied = residuals.iter().all(|residual| residual.matched);
    let mut assessment = MusicEditMetamorphicAssessmentV1 {
        schema_version: "mel-music-edit-metamorphic-assessment-v1".to_owned(),
        pair_sha256,
        base_observation_sha256: canonical_metamorphic_json_digest(b"base-observation", base)
            .map_err(|issue| vec![issue])?,
        transformed_observation_sha256: canonical_metamorphic_json_digest(b"transformed-observation", transformed)
            .map_err(|issue| vec![issue])?,
        property_mapping_sha256: mapping_sha256,
        expected_relation: pair.expected_relation,
        base_obligation_satisfied: base.obligation_satisfied,
        transformed_obligation_satisfied: transformed.obligation_satisfied,
        base_preservation_satisfied: base.protected_invariants_satisfied,
        transformed_preservation_satisfied: transformed.protected_invariants_satisfied,
        relation_satisfied,
        passed: base.obligation_satisfied
            && transformed.obligation_satisfied
            && base.protected_invariants_satisfied
            && transformed.protected_invariants_satisfied
            && relation_satisfied,
        residuals,
        assessment_sha256: String::new(),
    };
    assessment.assessment_sha256 =
        canonical_metamorphic_json_digest(b"assessment", &assessment)
            .map_err(|issue| vec![issue])?;
    Ok(assessment)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(n: u8) -> String { format!("{n:02x}").repeat(32) }

    fn pair(relation: MusicEditMetamorphicRelationV1) -> MusicEditMetamorphicPairV1 {
        let invariant = relation == MusicEditMetamorphicRelationV1::Invariant;
        MusicEditMetamorphicPairV1 {
            schema_version: MUSIC_EDIT_METAMORPHIC_PAIR_SCHEMA_V1.to_owned(),
            pair_id: "synthetic-pair".to_owned(), dataset_id: "synthetic".to_owned(),
            dataset_version: "v1".to_owned(), benchmark_track_id: "development".to_owned(),
            base_sample_id: "base".to_owned(), transformed_sample_id: "variant".to_owned(),
            base_source_lineage_sha256: d(1), transformed_source_lineage_sha256: d(1),
            base_split: MusicEditSplitV1::Development,
            transformed_split: MusicEditSplitV1::Development,
            operation: MusicEditOperationV1::Staccato,
            base_source_audio_sha256: d(2), transformed_source_audio_sha256: d(3),
            base_instruction_sha256: d(4),
            transformed_instruction_sha256: if invariant { d(4) } else { d(5) },
            base_validity_envelope_sha256: d(6),
            transformed_validity_envelope_sha256: if invariant { d(6) } else { d(7) },
            transform_family: if relation == MusicEditMetamorphicRelationV1::Equivariant {
                MusicEditMetamorphicTransformFamilyV1::GlobalTransposition
            } else { MusicEditMetamorphicTransformFamilyV1::GainHeadroom },
            transform_implementation_sha256: d(8), transform_receipt_sha256: d(9),
            transform_version: "v1".to_owned(), transform_parameters_sha256: d(10),
            expected_relation: relation, property_mapping_sha256: Some(d(11)),
            admissibility_reason: matches!(relation,
                MusicEditMetamorphicRelationV1::NonInvariant | MusicEditMetamorphicRelationV1::AbstainExpected)
                .then(|| "synthetic reason".to_owned()),
        }
    }


    fn matching_transform_receipt(
        pair: &MusicEditMetamorphicPairV1,
    ) -> MusicEditMetamorphicTransformReceiptV1 {
        let mut receipt = MusicEditMetamorphicTransformReceiptV1 {
            schema_version: MUSIC_EDIT_METAMORPHIC_TRANSFORM_RECEIPT_SCHEMA_V1.to_owned(),
            pair_id: pair.pair_id.clone(),
            dataset_id: pair.dataset_id.clone(),
            dataset_version: pair.dataset_version.clone(),
            benchmark_track_id: pair.benchmark_track_id.clone(),
            base_sample_id: pair.base_sample_id.clone(),
            transformed_sample_id: pair.transformed_sample_id.clone(),
            source_lineage_sha256: pair.base_source_lineage_sha256.clone(),
            split: pair.base_split,
            transform_family: pair.transform_family,
            input_audio_sha256: pair.base_source_audio_sha256.clone(),
            output_audio_sha256: pair.transformed_source_audio_sha256.clone(),
            base_instruction_sha256: pair.base_instruction_sha256.clone(),
            transformed_instruction_sha256: pair.transformed_instruction_sha256.clone(),
            transform_implementation_sha256: pair.transform_implementation_sha256.clone(),
            transform_version: pair.transform_version.clone(),
            transform_parameters_sha256: pair.transform_parameters_sha256.clone(),
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 =
            music_edit_metamorphic_transform_receipt_digest_v1(&receipt).unwrap();
        receipt
    }

    fn observation(
        pair: &MusicEditMetamorphicPairV1,
        pair_sha: &str,
        base: bool,
        pitch: f64,
    ) -> MusicEditMetamorphicObservationV1 {
        MusicEditMetamorphicObservationV1 {
            schema_version: "mel-music-edit-metamorphic-observation-v1".to_owned(),
            pair_sha256: pair_sha.to_owned(),
            sample_id: if base { pair.base_sample_id.clone() } else { pair.transformed_sample_id.clone() },
            source_lineage_sha256: pair.base_source_lineage_sha256.clone(),
            split: pair.base_split,
            obligation_satisfied: true,
            protected_invariants_satisfied: true,
            operation_report_sha256: d(31),
            validity_report_sha256: d(32),
            properties: vec![
                MusicEditMetamorphicPropertyObservationV1 {
                    property_id: "target_pitch".to_owned(),
                    evidence: MusicEditMetamorphicEvidenceV1::Symbolic,
                    value: MusicEditMetamorphicValueV1::Numeric { value: pitch, unit: "midi".to_owned() },
                },
                MusicEditMetamorphicPropertyObservationV1 {
                    property_id: "target_part_present".to_owned(),
                    evidence: MusicEditMetamorphicEvidenceV1::Acoustic,
                    value: MusicEditMetamorphicValueV1::Boolean { value: true },
                },
            ],
        }
    }

    fn mapping(scale: f64, offset: f64, tolerance: f64) -> MusicEditMetamorphicPropertyMappingV1 {
        MusicEditMetamorphicPropertyMappingV1 {
            schema_version: "mel-music-edit-metamorphic-property-map-v1".to_owned(),
            properties: vec![
                MusicEditMetamorphicPropertyRuleEntryV1 {
                    property_id: "target_part_present".to_owned(),
                    rule: MusicEditMetamorphicPropertyRuleV1::BooleanIdentity,
                },
                MusicEditMetamorphicPropertyRuleEntryV1 {
                    property_id: "target_pitch".to_owned(),
                    rule: MusicEditMetamorphicPropertyRuleV1::NumericAffine {
                        scale, offset, tolerance, unit: "midi".to_owned(),
                    },
                },
            ],
        }
    }

    #[test]
    fn invariant_evaluator_requires_both_edits_and_property_relations_to_pass() {
        let mut pair = pair(MusicEditMetamorphicRelationV1::Invariant);
        let map = mapping(1.0, 0.0, 0.0);
        pair.property_mapping_sha256 =
            Some(music_edit_metamorphic_property_mapping_digest_v1(&map).unwrap());
        let pair_sha = music_edit_metamorphic_pair_digest_v1(&pair).unwrap();
        let base = observation(&pair, &pair_sha, true, 64.0);
        let transformed = observation(&pair, &pair_sha, false, 64.0);
        let result = evaluate_music_edit_metamorphic_pair_v1(&pair, &map, &base, &transformed).unwrap();
        assert!(result.passed);
        assert!(result.relation_satisfied);
        assert!(verify_music_edit_metamorphic_assessment_v1(&result).is_ok());
        let mut tampered = result;
        tampered.passed = false;
        assert!(verify_music_edit_metamorphic_assessment_v1(&tampered).is_err());
    }

    #[test]
    fn equivariant_evaluator_checks_declared_transposition_map() {
        let mut pair = pair(MusicEditMetamorphicRelationV1::Equivariant);
        let map = mapping(1.0, 3.0, 0.01);
        pair.property_mapping_sha256 =
            Some(music_edit_metamorphic_property_mapping_digest_v1(&map).unwrap());
        let pair_sha = music_edit_metamorphic_pair_digest_v1(&pair).unwrap();
        let base = observation(&pair, &pair_sha, true, 60.0);
        let transformed = observation(&pair, &pair_sha, false, 63.0);
        let result = evaluate_music_edit_metamorphic_pair_v1(&pair, &map, &base, &transformed).unwrap();
        assert!(result.passed);
        assert_eq!(result.residuals[1].numeric_residual, Some(0.0));
    }

    #[test]
    fn relation_can_fail_even_when_both_edits_pass_individually() {
        let mut pair = pair(MusicEditMetamorphicRelationV1::Equivariant);
        let map = mapping(1.0, 3.0, 0.01);
        pair.property_mapping_sha256 =
            Some(music_edit_metamorphic_property_mapping_digest_v1(&map).unwrap());
        let pair_sha = music_edit_metamorphic_pair_digest_v1(&pair).unwrap();
        let base = observation(&pair, &pair_sha, true, 60.0);
        let transformed = observation(&pair, &pair_sha, false, 64.0);
        let result = evaluate_music_edit_metamorphic_pair_v1(&pair, &map, &base, &transformed).unwrap();
        assert!(!result.passed);
        assert!(!result.relation_satisfied);
    }

    #[test]
    fn failed_protected_content_gate_prevents_metamorphic_pass() {
        let mut pair = pair(MusicEditMetamorphicRelationV1::Invariant);
        let map = mapping(1.0, 0.0, 0.0);
        pair.property_mapping_sha256 =
            Some(music_edit_metamorphic_property_mapping_digest_v1(&map).unwrap());
        let pair_sha = music_edit_metamorphic_pair_digest_v1(&pair).unwrap();
        let base = observation(&pair, &pair_sha, true, 64.0);
        let mut transformed = observation(&pair, &pair_sha, false, 64.0);
        transformed.protected_invariants_satisfied = false;
        let result = evaluate_music_edit_metamorphic_pair_v1(&pair, &map, &base, &transformed).unwrap();
        assert!(!result.passed);
    }

    #[test]
    fn evaluator_rejects_an_observed_property_omitted_from_mapping() {
        let mut pair = pair(MusicEditMetamorphicRelationV1::Invariant);
        let map = mapping(1.0, 0.0, 0.0);
        pair.property_mapping_sha256 =
            Some(music_edit_metamorphic_property_mapping_digest_v1(&map).unwrap());
        let pair_sha = music_edit_metamorphic_pair_digest_v1(&pair).unwrap();
        let base = observation(&pair, &pair_sha, true, 64.0);
        let mut transformed = observation(&pair, &pair_sha, false, 64.0);
        transformed.properties.push(MusicEditMetamorphicPropertyObservationV1 {
            property_id: "unmapped-loudness".to_owned(),
            evidence: MusicEditMetamorphicEvidenceV1::Acoustic,
            value: MusicEditMetamorphicValueV1::Numeric {
                value: -20.0,
                unit: "dBFS".to_owned(),
            },
        });
        transformed.properties.sort_by(|a, b| a.property_id.cmp(&b.property_id));
        assert!(evaluate_music_edit_metamorphic_pair_v1(&pair, &map, &base, &transformed)
            .unwrap_err()
            .contains(&MusicEditMetamorphicAssessmentIssueV1::UnmappedObservationProperty(
                "unmapped-loudness".to_owned(),
            )));
    }

    #[test]
    fn invariant_category_maps_must_be_identity_and_canonical() {
        let mut pair = pair(MusicEditMetamorphicRelationV1::Invariant);
        let map = MusicEditMetamorphicPropertyMappingV1 {
            schema_version: "mel-music-edit-metamorphic-property-map-v1".to_owned(),
            properties: vec![MusicEditMetamorphicPropertyRuleEntryV1 {
                property_id: "instrument".to_owned(),
                rule: MusicEditMetamorphicPropertyRuleV1::CategoryMap {
                    values: vec![
                        MusicEditMetamorphicCategoryMapV1 {
                            source_value: "flute".to_owned(),
                            transformed_value: "oboe".to_owned(),
                        },
                        MusicEditMetamorphicCategoryMapV1 {
                            source_value: "oboe".to_owned(),
                            transformed_value: "oboe".to_owned(),
                        },
                    ],
                },
            }],
        };
        pair.property_mapping_sha256 =
            Some(music_edit_metamorphic_property_mapping_digest_v1(&map).unwrap());
        let pair_sha = music_edit_metamorphic_pair_digest_v1(&pair).unwrap();
        let base = observation(&pair, &pair_sha, true, 64.0);
        let transformed = observation(&pair, &pair_sha, false, 64.0);
        assert!(evaluate_music_edit_metamorphic_pair_v1(&pair, &map, &base, &transformed)
            .unwrap_err()
            .iter()
            .any(|issue| matches!(
                issue,
                MusicEditMetamorphicAssessmentIssueV1::NonIdentityRuleForInvariant(_)
            )));
    }

    #[test]
    fn noninvariant_pairs_are_not_scored_as_invariance() {
        let pair = pair(MusicEditMetamorphicRelationV1::NonInvariant);
        let map = mapping(1.0, 0.0, 0.0);
        let pair_sha = music_edit_metamorphic_pair_digest_v1(&pair).unwrap();
        let base = observation(&pair, &pair_sha, true, 440.0);
        let transformed = observation(&pair, &pair_sha, false, 445.0);
        assert!(evaluate_music_edit_metamorphic_pair_v1(&pair, &map, &base, &transformed).is_err());
    }

    #[test]
    fn transform_receipt_binds_pair_artifacts_and_configuration() {
        let mut pair = pair(MusicEditMetamorphicRelationV1::Invariant);
        let receipt = matching_transform_receipt(&pair);
        pair.transform_receipt_sha256 = receipt.receipt_sha256.clone();
        assert!(validate_music_edit_metamorphic_pair_v1(&pair).is_empty());
        assert!(verify_music_edit_metamorphic_transform_receipt_v1(&pair, &receipt).is_empty());
    }

    #[test]
    fn transform_receipt_rejects_output_substitution_and_digest_tampering() {
        let mut pair = pair(MusicEditMetamorphicRelationV1::Invariant);
        let mut receipt = matching_transform_receipt(&pair);
        pair.transform_receipt_sha256 = receipt.receipt_sha256.clone();
        receipt.output_audio_sha256 = d(99);
        let issues = verify_music_edit_metamorphic_transform_receipt_v1(&pair, &receipt);
        assert!(issues.contains(&MusicEditMetamorphicIssueV1::TransformReceiptArtifactMismatch));
        assert!(issues.contains(&MusicEditMetamorphicIssueV1::TransformReceiptDigestMismatch));
    }

    #[test]
    fn transform_receipt_rejects_wrong_implementation_identity() {
        let mut pair = pair(MusicEditMetamorphicRelationV1::Invariant);
        let mut receipt = matching_transform_receipt(&pair);
        pair.transform_receipt_sha256 = receipt.receipt_sha256.clone();
        receipt.transform_implementation_sha256 = d(55);
        assert!(verify_music_edit_metamorphic_transform_receipt_v1(&pair, &receipt)
            .contains(&MusicEditMetamorphicIssueV1::TransformReceiptConfigurationMismatch));
    }

    #[test] fn invariant_pair_passes_and_commits() {
        let p = pair(MusicEditMetamorphicRelationV1::Invariant);
        assert!(validate_music_edit_metamorphic_pair_v1(&p).is_empty());
        assert!(music_edit_metamorphic_pair_digest_v1(&p).is_ok());
    }
    #[test] fn invariant_instruction_drift_rejects() {
        let mut p = pair(MusicEditMetamorphicRelationV1::Invariant);
        p.transformed_instruction_sha256 = d(42);
        assert!(validate_music_edit_metamorphic_pair_v1(&p).contains(
            &MusicEditMetamorphicIssueV1::InvariantInstructionChanged));
    }
    #[test] fn invariant_envelope_drift_rejects() {
        let mut p = pair(MusicEditMetamorphicRelationV1::Invariant);
        p.transformed_validity_envelope_sha256 = d(42);
        assert!(validate_music_edit_metamorphic_pair_v1(&p).contains(
            &MusicEditMetamorphicIssueV1::InvariantEnvelopeChanged));
    }
    #[test] fn equivariant_requires_mapping() {
        let mut p = pair(MusicEditMetamorphicRelationV1::Equivariant);
        assert!(validate_music_edit_metamorphic_pair_v1(&p).is_empty());
        p.property_mapping_sha256 = None;
        assert!(validate_music_edit_metamorphic_pair_v1(&p).contains(
            &MusicEditMetamorphicIssueV1::RelationMapMissing));
    }
    #[test] fn noninvariant_requires_reason() {
        let mut p = pair(MusicEditMetamorphicRelationV1::NonInvariant);
        assert!(validate_music_edit_metamorphic_pair_v1(&p).is_empty());
        p.admissibility_reason = None;
        assert!(validate_music_edit_metamorphic_pair_v1(&p).contains(
            &MusicEditMetamorphicIssueV1::AdmissibilityReasonMissing));
    }
    #[test] fn abstention_requires_reason() {
        let mut p = pair(MusicEditMetamorphicRelationV1::AbstainExpected);
        assert!(validate_music_edit_metamorphic_pair_v1(&p).is_empty());
        p.admissibility_reason = Some(" ".to_owned());
        assert!(validate_music_edit_metamorphic_pair_v1(&p).contains(
            &MusicEditMetamorphicIssueV1::AdmissibilityReasonMissing));
    }
    #[test] fn lineage_and_split_mismatch_reject() {
        let mut p = pair(MusicEditMetamorphicRelationV1::Invariant);
        p.transformed_source_lineage_sha256 = d(99);
        assert!(validate_music_edit_metamorphic_pair_v1(&p).contains(
            &MusicEditMetamorphicIssueV1::SourceLineageMismatch));
        let mut p = pair(MusicEditMetamorphicRelationV1::Invariant);
        p.transformed_split = MusicEditSplitV1::Test;
        assert!(validate_music_edit_metamorphic_pair_v1(&p).contains(
            &MusicEditMetamorphicIssueV1::SplitMismatch));
    }
    #[test] fn mutable_versions_and_sample_identity_reject() {
        let mut p = pair(MusicEditMetamorphicRelationV1::Invariant);
        p.transform_version = "latest".to_owned();
        p.transformed_sample_id = p.base_sample_id.clone();
        let issues = validate_music_edit_metamorphic_pair_v1(&p);
        assert!(issues.contains(&MusicEditMetamorphicIssueV1::MutableVersion));
        assert!(issues.contains(&MusicEditMetamorphicIssueV1::SameSampleIdentity));
    }
    #[test] fn digest_changes_when_transform_parameters_change() {
        let p = pair(MusicEditMetamorphicRelationV1::Invariant);
        let before = music_edit_metamorphic_pair_digest_v1(&p).unwrap();
        let mut q = p;
        q.transform_parameters_sha256 = d(77);
        let after = music_edit_metamorphic_pair_digest_v1(&q).unwrap();
        assert_ne!(before, after);
    }
}
