// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composition and commutator contracts for metamorphic musical transformations.
//!
//! This module composes existing pair, transform-receipt and property-map
//! authorities. It does not execute audio transforms or establish perception.

use crate::music_edit_metamorphic_benchmark::{
    music_edit_metamorphic_pair_digest_v1,
    music_edit_metamorphic_property_mapping_digest_v1,
    validate_music_edit_metamorphic_pair_v1,
    verify_music_edit_metamorphic_transform_receipt_v1,
    MusicEditMetamorphicCategoryMapV1,
    MusicEditMetamorphicPairV1,
    MusicEditMetamorphicPropertyMappingV1,
    MusicEditMetamorphicPropertyRuleEntryV1,
    MusicEditMetamorphicPropertyRuleV1,
    MusicEditMetamorphicRelationV1,
    MusicEditMetamorphicTransformReceiptV1,
};
use crate::music_edit_benchmark_adapter::MusicEditSplitV1;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MUSIC_EDIT_METAMORPHIC_PATH_SCHEMA_V1: &str =
    "mel-music-edit-metamorphic-path-v1";
pub const MUSIC_EDIT_METAMORPHIC_PATH_REPORT_SCHEMA_V1: &str =
    "mel-music-edit-metamorphic-path-report-v1";
pub const MUSIC_EDIT_METAMORPHIC_COMMUTATOR_CASE_SCHEMA_V1: &str =
    "mel-music-edit-metamorphic-commutator-case-v1";
pub const MUSIC_EDIT_METAMORPHIC_COMMUTATOR_REPORT_SCHEMA_V1: &str =
    "mel-music-edit-metamorphic-commutator-report-v1";

const PATH_DOMAIN: &[u8] = b"luminous.mel.music-edit-metamorphic-path.v1";
const EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditMetamorphicPathStepV1 {
    pub pair: MusicEditMetamorphicPairV1,
    pub transform_receipt: MusicEditMetamorphicTransformReceiptV1,
    pub property_mapping: MusicEditMetamorphicPropertyMappingV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditMetamorphicPathV1 {
    pub schema_version: String,
    pub path_id: String,
    /// Ordered in execution order: the first step is applied first.
    pub steps: Vec<MusicEditMetamorphicPathStepV1>,
}

/// Preregistered expectation for a pair of transformation paths.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditMetamorphicCommutatorCaseV1 {
    pub schema_version: String,
    pub case_id: String,
    pub preregistration_sha256: String,
    pub evaluation_split: String,
    pub left_path_sha256: String,
    pub right_path_sha256: String,
    pub expected_to_commute: bool,
    pub rationale: String,
    pub case_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicEditMetamorphicCompositionIssueV1 {
    WrongPathSchema,
    EmptyPathId,
    EmptyPath,
    PairRejected(String),
    TransformReceiptRejected(String),
    PropertyMapRejected(String),
    PropertyMapDigestMismatch,
    NonComposableRelation,
    DuplicatePairIdentity,
    PathContextMismatch,
    BrokenSampleChain,
    BrokenAudioChain,
    BrokenInstructionChain,
    BrokenValidityEnvelopeChain,
    PropertySetMismatch,
    PropertyRuleTypeMismatch(String),
    NumericUnitMismatch(String),
    InvalidNumericRule(String),
    InvalidCategoryMap(String),
    MissingCategoryIntermediate(String),
    SerializationError,
    PathsDoNotShareStartState,
}

impl std::fmt::Display for MusicEditMetamorphicCompositionIssueV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for MusicEditMetamorphicCompositionIssueV1 {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditMetamorphicPathReportV1 {
    pub schema_version: String,
    pub path_id: String,
    pub path_sha256: String,
    pub step_count: usize,
    pub dataset_id: String,
    pub dataset_version: String,
    pub benchmark_track_id: String,
    pub operation: String,
    pub source_lineage_sha256: String,
    pub split: String,
    pub start_sample_id: String,
    pub end_sample_id: String,
    pub start_audio_sha256: String,
    pub end_audio_sha256: String,
    pub start_instruction_sha256: String,
    pub end_instruction_sha256: String,
    pub start_validity_envelope_sha256: String,
    pub end_validity_envelope_sha256: String,
    pub composed_relation: MusicEditMetamorphicRelationV1,
    pub composed_property_mapping_sha256: String,
    pub composed_property_mapping: MusicEditMetamorphicPropertyMappingV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditMetamorphicCommutatorReportV1 {
    pub schema_version: String,
    pub case_id: String,
    pub case_sha256: String,
    pub left_path_sha256: String,
    pub right_path_sha256: String,
    pub expected_to_commute: bool,
    pub mapping_commutes: bool,
    pub terminal_instruction_matches: bool,
    pub terminal_validity_envelope_matches: bool,
    pub observed_commutation: bool,
    pub passed: bool,
}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= EPSILON
}

fn canonical_split_label(split: MusicEditSplitV1) -> &'static str {
    match split {
        MusicEditSplitV1::Train => "train",
        MusicEditSplitV1::Development => "development",
        MusicEditSplitV1::Test => "test",
    }
}

fn compose_numeric_affine(
    before: &MusicEditMetamorphicPropertyRuleV1,
    after: &MusicEditMetamorphicPropertyRuleV1,
    property_id: &str,
) -> Result<MusicEditMetamorphicPropertyRuleV1, MusicEditMetamorphicCompositionIssueV1> {
    let (
        MusicEditMetamorphicPropertyRuleV1::NumericAffine {
            scale: s1,
            offset: b1,
            tolerance: t1,
            unit: u1,
        },
        MusicEditMetamorphicPropertyRuleV1::NumericAffine {
            scale: s2,
            offset: b2,
            tolerance: t2,
            unit: u2,
        },
    ) = (before, after)
    else {
        return Err(
            MusicEditMetamorphicCompositionIssueV1::PropertyRuleTypeMismatch(
                property_id.to_owned(),
            ),
        );
    };

    if u1 != u2 {
        return Err(MusicEditMetamorphicCompositionIssueV1::NumericUnitMismatch(
            property_id.to_owned(),
        ));
    }
    if !s1.is_finite()
        || !b1.is_finite()
        || !t1.is_finite()
        || *t1 < 0.0
        || !s2.is_finite()
        || !b2.is_finite()
        || !t2.is_finite()
        || *t2 < 0.0
    {
        return Err(MusicEditMetamorphicCompositionIssueV1::InvalidNumericRule(
            property_id.to_owned(),
        ));
    }

    // after(before(x)); propagate the prior bound through the later scale.
    let scale = *s2 * *s1;
    let offset = *s2 * *b1 + *b2;
    let tolerance = s2.abs() * *t1 + *t2;
    if !scale.is_finite() || !offset.is_finite() || !tolerance.is_finite() {
        return Err(MusicEditMetamorphicCompositionIssueV1::InvalidNumericRule(
            property_id.to_owned(),
        ));
    }
    Ok(MusicEditMetamorphicPropertyRuleV1::NumericAffine {
        scale,
        offset,
        tolerance,
        unit: u1.clone(),
    })
}

fn validate_property_mapping(
    mapping: &MusicEditMetamorphicPropertyMappingV1,
    relation: MusicEditMetamorphicRelationV1,
) -> Result<(), MusicEditMetamorphicCompositionIssueV1> {
    if mapping.schema_version != "mel-music-edit-metamorphic-property-map-v1"
        || mapping.properties.is_empty()
        || !mapping.properties.windows(2).all(|w| {
            !w[0].property_id.trim().is_empty() && w[0].property_id < w[1].property_id
        })
        || mapping.properties.last().is_some_and(|entry| entry.property_id.trim().is_empty())
    {
        return Err(MusicEditMetamorphicCompositionIssueV1::PropertyMapRejected(
            "schema, empty property set, or non-canonical property ordering".to_owned(),
        ));
    }

    for entry in &mapping.properties {
        let identity = match &entry.rule {
            MusicEditMetamorphicPropertyRuleV1::NumericAffine {
                scale,
                offset,
                tolerance,
                unit,
            } => {
                if !scale.is_finite()
                    || !offset.is_finite()
                    || !tolerance.is_finite()
                    || *tolerance < 0.0
                    || unit.trim().is_empty()
                {
                    return Err(
                        MusicEditMetamorphicCompositionIssueV1::InvalidNumericRule(
                            entry.property_id.clone(),
                        ),
                    );
                }
                approximately_equal(*scale, 1.0) && approximately_equal(*offset, 0.0)
            }
            MusicEditMetamorphicPropertyRuleV1::CategoryMap { values } => {
                if values.is_empty()
                    || !values.windows(2).all(|w| w[0].source_value < w[1].source_value)
                    || values.iter().any(|v| {
                        v.source_value.trim().is_empty() || v.transformed_value.trim().is_empty()
                    })
                {
                    return Err(
                        MusicEditMetamorphicCompositionIssueV1::InvalidCategoryMap(
                            entry.property_id.clone(),
                        ),
                    );
                }
                values.iter().all(|v| v.source_value == v.transformed_value)
            }
            MusicEditMetamorphicPropertyRuleV1::BooleanIdentity => true,
        };
        if relation == MusicEditMetamorphicRelationV1::Invariant && !identity {
            return Err(MusicEditMetamorphicCompositionIssueV1::PropertyMapRejected(
                format!("invariant step changes property {}", entry.property_id),
            ));
        }
    }
    Ok(())
}

fn compose_category_maps(
    before: &[MusicEditMetamorphicCategoryMapV1],
    after: &[MusicEditMetamorphicCategoryMapV1],
    property_id: &str,
) -> Result<Vec<MusicEditMetamorphicCategoryMapV1>, MusicEditMetamorphicCompositionIssueV1> {
    let mut values = Vec::with_capacity(before.len());
    for first in before {
        let next = after
            .iter()
            .find(|candidate| candidate.source_value == first.transformed_value)
            .ok_or_else(|| {
                MusicEditMetamorphicCompositionIssueV1::MissingCategoryIntermediate(
                    property_id.to_owned(),
                )
            })?;
        values.push(MusicEditMetamorphicCategoryMapV1 {
            source_value: first.source_value.clone(),
            transformed_value: next.transformed_value.clone(),
        });
    }
    Ok(values)
}

fn compose_rules(
    before: &MusicEditMetamorphicPropertyRuleV1,
    after: &MusicEditMetamorphicPropertyRuleV1,
    property_id: &str,
) -> Result<MusicEditMetamorphicPropertyRuleV1, MusicEditMetamorphicCompositionIssueV1> {
    use MusicEditMetamorphicPropertyRuleV1 as Rule;
    match (before, after) {
        (Rule::NumericAffine { .. }, Rule::NumericAffine { .. }) => {
            compose_numeric_affine(before, after, property_id)
        }
        (Rule::CategoryMap { values: a }, Rule::CategoryMap { values: b }) => {
            Ok(Rule::CategoryMap {
                values: compose_category_maps(a, b, property_id)?,
            })
        }
        (Rule::BooleanIdentity, Rule::BooleanIdentity) => Ok(Rule::BooleanIdentity),
        _ => Err(
            MusicEditMetamorphicCompositionIssueV1::PropertyRuleTypeMismatch(
                property_id.to_owned(),
            ),
        ),
    }
}

/// Compose property mappings in execution order. Every step must cover exactly
/// the same property set; an omitted property is never silently treated as identity.
pub fn compose_music_edit_metamorphic_property_mappings_v1(
    mappings: &[MusicEditMetamorphicPropertyMappingV1],
) -> Result<MusicEditMetamorphicPropertyMappingV1, MusicEditMetamorphicCompositionIssueV1> {
    let Some(first) = mappings.first() else {
        return Err(MusicEditMetamorphicCompositionIssueV1::EmptyPath);
    };
    validate_property_mapping(first, MusicEditMetamorphicRelationV1::Equivariant)?;
    let mut composed = first.clone();

    for next in mappings.iter().skip(1) {
        validate_property_mapping(next, MusicEditMetamorphicRelationV1::Equivariant)?;
        if composed.properties.len() != next.properties.len()
            || !composed
                .properties
                .iter()
                .zip(&next.properties)
                .all(|(a, b)| a.property_id == b.property_id)
        {
            return Err(MusicEditMetamorphicCompositionIssueV1::PropertySetMismatch);
        }
        let mut properties = Vec::with_capacity(composed.properties.len());
        for (before, after) in composed.properties.iter().zip(&next.properties) {
            properties.push(MusicEditMetamorphicPropertyRuleEntryV1 {
                property_id: before.property_id.clone(),
                rule: compose_rules(&before.rule, &after.rule, &before.property_id)?,
            });
        }
        composed.properties = properties;
    }
    Ok(composed)
}

fn mapping_is_identity(mapping: &MusicEditMetamorphicPropertyMappingV1) -> bool {
    mapping.properties.iter().all(|entry| match &entry.rule {
        MusicEditMetamorphicPropertyRuleV1::NumericAffine { scale, offset, .. } => {
            approximately_equal(*scale, 1.0) && approximately_equal(*offset, 0.0)
        }
        MusicEditMetamorphicPropertyRuleV1::CategoryMap { values } => {
            values.iter().all(|v| v.source_value == v.transformed_value)
        }
        MusicEditMetamorphicPropertyRuleV1::BooleanIdentity => true,
    })
}

fn mappings_have_same_semantics(
    left: &MusicEditMetamorphicPropertyMappingV1,
    right: &MusicEditMetamorphicPropertyMappingV1,
) -> bool {
    if left.properties.len() != right.properties.len() {
        return false;
    }
    left.properties.iter().zip(&right.properties).all(|(a, b)| {
        if a.property_id != b.property_id {
            return false;
        }
        match (&a.rule, &b.rule) {
            (
                MusicEditMetamorphicPropertyRuleV1::NumericAffine {
                    scale: sa,
                    offset: oa,
                    unit: ua,
                    ..
                },
                MusicEditMetamorphicPropertyRuleV1::NumericAffine {
                    scale: sb,
                    offset: ob,
                    unit: ub,
                    ..
                },
            ) => {
                ua == ub && approximately_equal(*sa, *sb) && approximately_equal(*oa, *ob)
            }
            (
                MusicEditMetamorphicPropertyRuleV1::CategoryMap { values: a },
                MusicEditMetamorphicPropertyRuleV1::CategoryMap { values: b },
            ) => a == b,
            (
                MusicEditMetamorphicPropertyRuleV1::BooleanIdentity,
                MusicEditMetamorphicPropertyRuleV1::BooleanIdentity,
            ) => true,
            _ => false,
        }
    })
}

fn calculate_path_digest(
    path: &MusicEditMetamorphicPathV1,
) -> Result<String, MusicEditMetamorphicCompositionIssueV1> {
    let bytes = serde_json::to_vec(path)
        .map_err(|_| MusicEditMetamorphicCompositionIssueV1::SerializationError)?;
    let mut hasher = Sha256::new();
    hasher.update(PATH_DOMAIN);
    hasher.update(b"\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}


const COMMUTATOR_CASE_DOMAIN: &[u8] =
    b"luminous.mel.music-edit-metamorphic-commutator-case.v1";

pub fn music_edit_metamorphic_commutator_case_digest_v1(
    case: &MusicEditMetamorphicCommutatorCaseV1,
) -> Result<String, MusicEditMetamorphicCompositionIssueV1> {
    let mut commitment = case.clone();
    commitment.case_sha256.clear();
    let bytes = serde_json::to_vec(&commitment)
        .map_err(|_| MusicEditMetamorphicCompositionIssueV1::SerializationError)?;
    let mut hasher = Sha256::new();
    hasher.update(COMMUTATOR_CASE_DOMAIN);
    hasher.update(b"\\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn verify_commutator_case(
    case: &MusicEditMetamorphicCommutatorCaseV1,
    left: &MusicEditMetamorphicPathReportV1,
    right: &MusicEditMetamorphicPathReportV1,
) -> Result<(), MusicEditMetamorphicCompositionIssueV1> {
    let valid_sha = |value: &str| {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    };
    if case.schema_version != MUSIC_EDIT_METAMORPHIC_COMMUTATOR_CASE_SCHEMA_V1
        || case.case_id.trim().is_empty()
        || case.rationale.trim().is_empty()
        || case.evaluation_split.trim().is_empty()
        || !valid_sha(&case.preregistration_sha256)
    {
        return Err(MusicEditMetamorphicCompositionIssueV1::PropertyMapRejected(
            "commutator case identity, rationale, split or preregistration digest is invalid".to_owned(),
        ));
    }
    if case.left_path_sha256 != left.path_sha256
        || case.right_path_sha256 != right.path_sha256
        || case.evaluation_split != left.split
        || case.evaluation_split != right.split
    {
        return Err(MusicEditMetamorphicCompositionIssueV1::PathsDoNotShareStartState);
    }
    if music_edit_metamorphic_commutator_case_digest_v1(case)? != case.case_sha256 {
        return Err(MusicEditMetamorphicCompositionIssueV1::SerializationError);
    }
    Ok(())
}

/// Revalidate each existing pair and receipt, enforce exact edge continuity,
/// and compose the property mappings along the path.
pub fn compose_music_edit_metamorphic_path_v1(
    path: &MusicEditMetamorphicPathV1,
) -> Result<MusicEditMetamorphicPathReportV1, Vec<MusicEditMetamorphicCompositionIssueV1>> {
    let mut issues = Vec::new();
    if path.schema_version != MUSIC_EDIT_METAMORPHIC_PATH_SCHEMA_V1 {
        issues.push(MusicEditMetamorphicCompositionIssueV1::WrongPathSchema);
    }
    if path.path_id.trim().is_empty() {
        issues.push(MusicEditMetamorphicCompositionIssueV1::EmptyPathId);
    }
    if path.steps.is_empty() {
        issues.push(MusicEditMetamorphicCompositionIssueV1::EmptyPath);
        return Err(issues);
    }

    let mut seen_pairs = std::collections::BTreeSet::new();
    for (index, step) in path.steps.iter().enumerate() {
        if !seen_pairs.insert(step.pair.pair_id.as_str()) {
            issues.push(MusicEditMetamorphicCompositionIssueV1::DuplicatePairIdentity);
        }
        let pair_issues = validate_music_edit_metamorphic_pair_v1(&step.pair);
        if !pair_issues.is_empty() {
            issues.push(MusicEditMetamorphicCompositionIssueV1::PairRejected(format!(
                "step {index}: {pair_issues:?}"
            )));
        }
        if !matches!(
            step.pair.expected_relation,
            MusicEditMetamorphicRelationV1::Invariant
                | MusicEditMetamorphicRelationV1::Equivariant
        ) {
            issues.push(MusicEditMetamorphicCompositionIssueV1::NonComposableRelation);
        }
        if !verify_music_edit_metamorphic_transform_receipt_v1(
            &step.pair,
            &step.transform_receipt,
        )
        .is_empty()
        {
            issues.push(
                MusicEditMetamorphicCompositionIssueV1::TransformReceiptRejected(format!(
                    "step {index}"
                )),
            );
        }
        if let Err(issue) =
            validate_property_mapping(&step.property_mapping, step.pair.expected_relation)
        {
            issues.push(issue);
        }
        let mapping_digest =
            music_edit_metamorphic_property_mapping_digest_v1(&step.property_mapping);
        if mapping_digest.ok().as_deref() != step.pair.property_mapping_sha256.as_deref() {
            issues.push(MusicEditMetamorphicCompositionIssueV1::PropertyMapDigestMismatch);
        }
        if music_edit_metamorphic_pair_digest_v1(&step.pair).is_err() {
            issues.push(MusicEditMetamorphicCompositionIssueV1::SerializationError);
        }
    }

    let first = &path.steps[0].pair;
    for window in path.steps.windows(2) {
        let previous = &window[0].pair;
        let next = &window[1].pair;
        if previous.transformed_sample_id != next.base_sample_id {
            issues.push(MusicEditMetamorphicCompositionIssueV1::BrokenSampleChain);
        }
        if previous.transformed_source_audio_sha256 != next.base_source_audio_sha256 {
            issues.push(MusicEditMetamorphicCompositionIssueV1::BrokenAudioChain);
        }
        if previous.transformed_instruction_sha256 != next.base_instruction_sha256 {
            issues.push(MusicEditMetamorphicCompositionIssueV1::BrokenInstructionChain);
        }
        if previous.transformed_validity_envelope_sha256 != next.base_validity_envelope_sha256 {
            issues.push(
                MusicEditMetamorphicCompositionIssueV1::BrokenValidityEnvelopeChain,
            );
        }
        if previous.dataset_id != next.dataset_id
            || previous.dataset_version != next.dataset_version
            || previous.benchmark_track_id != next.benchmark_track_id
            || previous.operation != next.operation
            || previous.base_source_lineage_sha256 != next.base_source_lineage_sha256
            || previous.base_split != next.base_split
        {
            issues.push(MusicEditMetamorphicCompositionIssueV1::PathContextMismatch);
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let mappings = path
        .steps
        .iter()
        .map(|step| step.property_mapping.clone())
        .collect::<Vec<_>>();
    let composed_property_mapping = compose_music_edit_metamorphic_property_mappings_v1(&mappings)
        .map_err(|issue| vec![issue])?;
    let composed_relation = if mapping_is_identity(&composed_property_mapping) {
        MusicEditMetamorphicRelationV1::Invariant
    } else {
        MusicEditMetamorphicRelationV1::Equivariant
    };
    let composed_property_mapping_sha256 =
        music_edit_metamorphic_property_mapping_digest_v1(&composed_property_mapping)
            .map_err(|_| vec![MusicEditMetamorphicCompositionIssueV1::SerializationError])?;
    let last = &path.steps[path.steps.len() - 1].pair;
    let path_sha256 = calculate_path_digest(path).map_err(|issue| vec![issue])?;

    Ok(MusicEditMetamorphicPathReportV1 {
        schema_version: MUSIC_EDIT_METAMORPHIC_PATH_REPORT_SCHEMA_V1.to_owned(),
        path_id: path.path_id.clone(),
        path_sha256,
        step_count: path.steps.len(),
        dataset_id: first.dataset_id.clone(),
        dataset_version: first.dataset_version.clone(),
        benchmark_track_id: first.benchmark_track_id.clone(),
        operation: format!("{:?}", first.operation),
        source_lineage_sha256: first.base_source_lineage_sha256.clone(),
        split: canonical_split_label(first.base_split).to_owned(),
        start_sample_id: first.base_sample_id.clone(),
        end_sample_id: last.transformed_sample_id.clone(),
        start_audio_sha256: first.base_source_audio_sha256.clone(),
        end_audio_sha256: last.transformed_source_audio_sha256.clone(),
        start_instruction_sha256: first.base_instruction_sha256.clone(),
        end_instruction_sha256: last.transformed_instruction_sha256.clone(),
        start_validity_envelope_sha256: first.base_validity_envelope_sha256.clone(),
        end_validity_envelope_sha256: last.transformed_validity_envelope_sha256.clone(),
        composed_relation,
        composed_property_mapping_sha256,
        composed_property_mapping,
    })
}

/// Compare two paths from the same exact start state. The result qualifies
/// property-map and task-contract commutation, not waveform equivalence.
pub fn compare_music_edit_metamorphic_paths_v1(
    case: &MusicEditMetamorphicCommutatorCaseV1,
    left_path: &MusicEditMetamorphicPathV1,
    right_path: &MusicEditMetamorphicPathV1,
) -> Result<MusicEditMetamorphicCommutatorReportV1, Vec<MusicEditMetamorphicCompositionIssueV1>> {
    let left = compose_music_edit_metamorphic_path_v1(left_path)?;
    let right = compose_music_edit_metamorphic_path_v1(right_path)?;

    let same_start = left.dataset_id == right.dataset_id
        && left.dataset_version == right.dataset_version
        && left.benchmark_track_id == right.benchmark_track_id
        && left.operation == right.operation
        && left.source_lineage_sha256 == right.source_lineage_sha256
        && left.split == right.split
        && left.start_sample_id == right.start_sample_id
        && left.start_audio_sha256 == right.start_audio_sha256
        && left.start_instruction_sha256 == right.start_instruction_sha256
        && left.start_validity_envelope_sha256 == right.start_validity_envelope_sha256;
    if !same_start {
        return Err(vec![
            MusicEditMetamorphicCompositionIssueV1::PathsDoNotShareStartState,
        ]);
    }
    if let Err(issue) = verify_commutator_case(case, &left, &right) {
        return Err(vec![issue]);
    }

    let mapping_commutes = mappings_have_same_semantics(
        &left.composed_property_mapping,
        &right.composed_property_mapping,
    );
    let terminal_instruction_matches =
        left.end_instruction_sha256 == right.end_instruction_sha256;
    let terminal_validity_envelope_matches =
        left.end_validity_envelope_sha256 == right.end_validity_envelope_sha256;
    let observed_commutation =
        mapping_commutes && terminal_instruction_matches && terminal_validity_envelope_matches;

    Ok(MusicEditMetamorphicCommutatorReportV1 {
        schema_version: MUSIC_EDIT_METAMORPHIC_COMMUTATOR_REPORT_SCHEMA_V1.to_owned(),
        case_id: case.case_id.clone(),
        case_sha256: case.case_sha256.clone(),
        left_path_sha256: left.path_sha256,
        right_path_sha256: right.path_sha256,
        expected_to_commute: case.expected_to_commute,
        mapping_commutes,
        terminal_instruction_matches,
        terminal_validity_envelope_matches,
        observed_commutation,
        passed: observed_commutation == case.expected_to_commute,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music_edit_benchmark_adapter::{
        MusicEditOperationV1, MusicEditSplitV1,
    };
    use crate::music_edit_metamorphic_benchmark::{
        music_edit_metamorphic_transform_receipt_digest_v1,
        MusicEditMetamorphicTransformFamilyV1,
        MusicEditMetamorphicTransformReceiptV1,
        MUSIC_EDIT_METAMORPHIC_TRANSFORM_RECEIPT_SCHEMA_V1,
    };

    fn d(n: u8) -> String {
        format!("{n:02x}").repeat(32)
    }

    fn mapping(
        pitch_scale: f64,
        pitch_offset: f64,
        pitch_tolerance: f64,
        duration_scale: f64,
        duration_offset: f64,
        duration_tolerance: f64,
    ) -> MusicEditMetamorphicPropertyMappingV1 {
        use MusicEditMetamorphicPropertyRuleV1 as Rule;
        MusicEditMetamorphicPropertyMappingV1 {
            schema_version: "mel-music-edit-metamorphic-property-map-v1".to_owned(),
            properties: vec![
                MusicEditMetamorphicPropertyRuleEntryV1 {
                    property_id: "duration_ms".to_owned(),
                    rule: Rule::NumericAffine {
                        scale: duration_scale,
                        offset: duration_offset,
                        tolerance: duration_tolerance,
                        unit: "ms".to_owned(),
                    },
                },
                MusicEditMetamorphicPropertyRuleEntryV1 {
                    property_id: "pitch_midi".to_owned(),
                    rule: Rule::NumericAffine {
                        scale: pitch_scale,
                        offset: pitch_offset,
                        tolerance: pitch_tolerance,
                        unit: "semitones".to_owned(),
                    },
                },
            ],
        }
    }

    fn category_mapping(
        entries: &[(&str, &str)],
    ) -> MusicEditMetamorphicPropertyMappingV1 {
        MusicEditMetamorphicPropertyMappingV1 {
            schema_version: "mel-music-edit-metamorphic-property-map-v1".to_owned(),
            properties: vec![MusicEditMetamorphicPropertyRuleEntryV1 {
                property_id: "instrument_family".to_owned(),
                rule: MusicEditMetamorphicPropertyRuleV1::CategoryMap {
                    values: entries
                        .iter()
                        .map(|(source, transformed)| MusicEditMetamorphicCategoryMapV1 {
                            source_value: (*source).to_owned(),
                            transformed_value: (*transformed).to_owned(),
                        })
                        .collect(),
                },
            }],
        }
    }

    fn step(
        pair_id: &str,
        base_sample: &str,
        output_sample: &str,
        base_audio: &str,
        output_audio: &str,
        base_instruction: &str,
        output_instruction: &str,
        base_envelope: &str,
        output_envelope: &str,
        family: MusicEditMetamorphicTransformFamilyV1,
        property_mapping: MusicEditMetamorphicPropertyMappingV1,
    ) -> MusicEditMetamorphicPathStepV1 {
        let identity = mapping_is_identity(&property_mapping);
        let relation = if identity {
            MusicEditMetamorphicRelationV1::Invariant
        } else {
            MusicEditMetamorphicRelationV1::Equivariant
        };
        let mut pair = MusicEditMetamorphicPairV1 {
            schema_version: crate::music_edit_metamorphic_benchmark::MUSIC_EDIT_METAMORPHIC_PAIR_SCHEMA_V1.to_owned(),
            pair_id: pair_id.to_owned(),
            dataset_id: "synthetic-composition".to_owned(),
            dataset_version: "v1".to_owned(),
            benchmark_track_id: "development".to_owned(),
            base_sample_id: base_sample.to_owned(),
            transformed_sample_id: output_sample.to_owned(),
            base_source_lineage_sha256: d(1),
            transformed_source_lineage_sha256: d(1),
            base_split: MusicEditSplitV1::Development,
            transformed_split: MusicEditSplitV1::Development,
            operation: MusicEditOperationV1::Staccato,
            base_source_audio_sha256: base_audio.to_owned(),
            transformed_source_audio_sha256: output_audio.to_owned(),
            base_instruction_sha256: base_instruction.to_owned(),
            transformed_instruction_sha256: output_instruction.to_owned(),
            base_validity_envelope_sha256: base_envelope.to_owned(),
            transformed_validity_envelope_sha256: output_envelope.to_owned(),
            transform_family: family,
            transform_implementation_sha256: d(8),
            transform_receipt_sha256: d(9),
            transform_version: "synthetic-v1".to_owned(),
            transform_parameters_sha256: d(10),
            expected_relation: relation,
            property_mapping_sha256: Some(
                music_edit_metamorphic_property_mapping_digest_v1(&property_mapping).unwrap(),
            ),
            admissibility_reason: None,
        };
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
        pair.transform_receipt_sha256 = receipt.receipt_sha256.clone();

        MusicEditMetamorphicPathStepV1 {
            pair,
            transform_receipt: receipt,
            property_mapping,
        }
    }

    fn path(path_id: &str, steps: Vec<MusicEditMetamorphicPathStepV1>) -> MusicEditMetamorphicPathV1 {
        MusicEditMetamorphicPathV1 {
            schema_version: MUSIC_EDIT_METAMORPHIC_PATH_SCHEMA_V1.to_owned(),
            path_id: path_id.to_owned(),
            steps,
        }
    }

    fn disjoint_transform_paths() -> (MusicEditMetamorphicPathV1, MusicEditMetamorphicPathV1) {
        let origin_audio = d(2);
        let origin_instruction = d(3);
        let origin_envelope = d(4);
        let final_instruction = d(9);
        let final_envelope = d(10);
        let transpose = mapping(1.0, 2.0, 0.0, 1.0, 0.0, 0.0);
        let tempo = mapping(1.0, 0.0, 0.0, 2.0, 0.0, 0.0);

        let left = path(
            "transpose-then-tempo",
            vec![
                step(
                    "left-1", "origin", "left-pitch", &origin_audio, &d(5),
                    &origin_instruction, &d(6), &origin_envelope, &d(7),
                    MusicEditMetamorphicTransformFamilyV1::GlobalTransposition,
                    transpose.clone(),
                ),
                step(
                    "left-2", "left-pitch", "left-final", &d(5), &d(8),
                    &d(6), &final_instruction, &d(7), &final_envelope,
                    MusicEditMetamorphicTransformFamilyV1::TempoScaling, tempo.clone(),
                ),
            ],
        );
        let right = path(
            "tempo-then-transpose",
            vec![
                step(
                    "right-1", "origin", "right-tempo", &origin_audio, &d(11),
                    &origin_instruction, &d(12), &origin_envelope, &d(13),
                    MusicEditMetamorphicTransformFamilyV1::TempoScaling, tempo,
                ),
                step(
                    "right-2", "right-tempo", "right-final", &d(11), &d(14),
                    &d(12), &final_instruction, &d(13), &final_envelope,
                    MusicEditMetamorphicTransformFamilyV1::GlobalTransposition, transpose,
                ),
            ],
        );
        (left, right)
    }

    #[test]
    fn affine_maps_compose_in_execution_order_and_propagate_tolerance() {
        let before = mapping(1.0, 2.0, 0.1, 1.0, 0.0, 0.0);
        let after = mapping(2.0, 3.0, 0.2, 1.0, 0.0, 0.0);
        let result =
            compose_music_edit_metamorphic_property_mappings_v1(&[before, after]).unwrap();
        let MusicEditMetamorphicPropertyRuleV1::NumericAffine {
            scale, offset, tolerance, ..
        } = &result.properties[1].rule else {
            panic!("pitch mapping should remain numeric");
        };
        assert_eq!((*scale, *offset, *tolerance), (2.0, 7.0, 0.4));
    }

    fn preregistered_case(
        case_id: &str,
        left: &MusicEditMetamorphicPathV1,
        right: &MusicEditMetamorphicPathV1,
        expected_to_commute: bool,
    ) -> MusicEditMetamorphicCommutatorCaseV1 {
        let left_report = compose_music_edit_metamorphic_path_v1(left).unwrap();
        let right_report = compose_music_edit_metamorphic_path_v1(right).unwrap();
        let mut case = MusicEditMetamorphicCommutatorCaseV1 {
            schema_version: MUSIC_EDIT_METAMORPHIC_COMMUTATOR_CASE_SCHEMA_V1.to_owned(),
            case_id: case_id.to_owned(),
            preregistration_sha256: d(29),
            evaluation_split: canonical_split_label(left.steps[0].pair.base_split).to_owned(),
            left_path_sha256: left_report.path_sha256,
            right_path_sha256: right_report.path_sha256,
            expected_to_commute,
            rationale: "Frozen synthetic algebra known-answer case.".to_owned(),
            case_sha256: String::new(),
        };
        case.case_sha256 = music_edit_metamorphic_commutator_case_digest_v1(&case).unwrap();
        case
    }

    #[test]
    fn transposition_and_tempo_scaling_commute_on_disjoint_properties() {
        let (left, right) = disjoint_transform_paths();
        let left_report = compose_music_edit_metamorphic_path_v1(&left).unwrap();
        let right_report = compose_music_edit_metamorphic_path_v1(&right).unwrap();
        assert_eq!(
            left_report.composed_property_mapping,
            right_report.composed_property_mapping
        );
        let case = preregistered_case("pitch-tempo-commute", &left, &right, true);
        let report = compare_music_edit_metamorphic_paths_v1(&case, &left, &right).unwrap();
        assert!(report.mapping_commutes);
        assert!(report.observed_commutation);
        assert!(report.passed);
    }

    #[test]
    fn commutator_expectation_is_digest_bound() {
        let (left, right) = disjoint_transform_paths();
        let mut case = preregistered_case("tamper-control", &left, &right, true);
        case.expected_to_commute = false;
        assert!(compare_music_edit_metamorphic_paths_v1(&case, &left, &right).is_err());
    }

    #[test]
    fn scale_and_offset_on_one_property_are_noncommuting() {
        let start_audio = d(2);
        let start_instruction = d(3);
        let start_envelope = d(4);
        let end_instruction = d(9);
        let end_envelope = d(10);
        let scale = mapping(2.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let offset = mapping(1.0, 3.0, 0.0, 1.0, 0.0, 0.0);

        let left = path("scale-then-offset", vec![
            step("left-scale", "origin", "scaled", &start_audio, &d(5),
                &start_instruction, &d(6), &start_envelope, &d(7),
                MusicEditMetamorphicTransformFamilyV1::GlobalTransposition, scale.clone()),
            step("left-offset", "scaled", "left-end", &d(5), &d(8),
                &d(6), &end_instruction, &d(7), &end_envelope,
                MusicEditMetamorphicTransformFamilyV1::KeyScaleShift, offset.clone()),
        ]);
        let right = path("offset-then-scale", vec![
            step("right-offset", "origin", "offset", &start_audio, &d(11),
                &start_instruction, &d(12), &start_envelope, &d(13),
                MusicEditMetamorphicTransformFamilyV1::KeyScaleShift, offset),
            step("right-scale", "offset", "right-end", &d(11), &d(14),
                &d(12), &end_instruction, &d(13), &end_envelope,
                MusicEditMetamorphicTransformFamilyV1::GlobalTransposition, scale),
        ]);
        let case = preregistered_case("scale-offset-noncommute", &left, &right, false);
        let report = compare_music_edit_metamorphic_paths_v1(&case, &left, &right).unwrap();
        assert!(!report.mapping_commutes);
        assert!(!report.observed_commutation);
        assert!(report.passed);
    }

    #[test]
    fn inverse_transpositions_compose_to_identity() {
        let plus = mapping(1.0, 2.0, 0.0, 1.0, 0.0, 0.0);
        let minus = mapping(1.0, -2.0, 0.0, 1.0, 0.0, 0.0);
        let result =
            compose_music_edit_metamorphic_property_mappings_v1(&[plus, minus]).unwrap();
        assert!(mapping_is_identity(&result));
    }

    #[test]
    fn category_maps_compose_and_missing_intermediates_are_rejected() {
        let first = category_mapping(&[("brass", "brass_section"), ("strings", "string_section")]);
        let second = category_mapping(&[
            ("brass_section", "woodwind"),
            ("string_section", "percussion"),
        ]);
        let result =
            compose_music_edit_metamorphic_property_mappings_v1(&[first.clone(), second]).unwrap();
        let MusicEditMetamorphicPropertyRuleV1::CategoryMap { values } =
            &result.properties[0].rule else {
            panic!("instrument family mapping should remain categorical");
        };
        assert_eq!(values[0].transformed_value, "woodwind");
        assert_eq!(values[1].transformed_value, "percussion");

        let incomplete = category_mapping(&[("brass_section", "woodwind")]);
        assert_eq!(
            compose_music_edit_metamorphic_property_mappings_v1(&[first, incomplete]),
            Err(MusicEditMetamorphicCompositionIssueV1::MissingCategoryIntermediate(
                "instrument_family".to_owned()
            ))
        );
    }

    #[test]
    fn boolean_identity_composes_to_identity() {
        let mapping = MusicEditMetamorphicPropertyMappingV1 {
            schema_version: "mel-music-edit-metamorphic-property-map-v1".to_owned(),
            properties: vec![MusicEditMetamorphicPropertyRuleEntryV1 {
                property_id: "protected_material".to_owned(),
                rule: MusicEditMetamorphicPropertyRuleV1::BooleanIdentity,
            }],
        };
        let result = compose_music_edit_metamorphic_property_mappings_v1(&[
            mapping.clone(), mapping,
        ]).unwrap();
        assert!(mapping_is_identity(&result));
    }

    #[test]
    fn broken_audio_chain_is_rejected() {
        let first = mapping(1.0, 1.0, 0.0, 1.0, 0.0, 0.0);
        let second = mapping(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let left = step("step-1", "origin", "middle", &d(2), &d(3),
            &d(4), &d(5), &d(6), &d(7),
            MusicEditMetamorphicTransformFamilyV1::GlobalTransposition, first);
        let right = step("step-2", "middle", "end", &d(99), &d(8),
            &d(5), &d(9), &d(7), &d(10),
            MusicEditMetamorphicTransformFamilyV1::GainHeadroom, second);
        let error = compose_music_edit_metamorphic_path_v1(&path("broken-audio", vec![left, right]))
            .unwrap_err();
        assert!(error.contains(&MusicEditMetamorphicCompositionIssueV1::BrokenAudioChain));
    }

    #[test]
    fn noninvariant_and_abstain_relations_are_not_composable() {
        let mut step = step(
            "noninvariant", "origin", "changed", &d(2), &d(3), &d(4), &d(5), &d(6), &d(7),
            MusicEditMetamorphicTransformFamilyV1::AbsolutePitchShift,
            mapping(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        );
        step.pair.expected_relation = MusicEditMetamorphicRelationV1::NonInvariant;
        step.pair.admissibility_reason = Some("absolute target changed".to_owned());
        let error = compose_music_edit_metamorphic_path_v1(&path("noninvariant", vec![step]))
            .unwrap_err();
        assert!(error.contains(&MusicEditMetamorphicCompositionIssueV1::NonComposableRelation));
    }

    #[test]
    fn path_digests_are_deterministic() {
        let (left, _) = disjoint_transform_paths();
        let first = compose_music_edit_metamorphic_path_v1(&left).unwrap();
        let second = compose_music_edit_metamorphic_path_v1(&left).unwrap();
        assert_eq!(first.path_sha256, second.path_sha256);
        assert_eq!(
            first.composed_property_mapping_sha256,
            second.composed_property_mapping_sha256
        );
    }
}
