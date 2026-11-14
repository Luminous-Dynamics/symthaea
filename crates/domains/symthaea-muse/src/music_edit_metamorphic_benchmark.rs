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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicEditMetamorphicIssueV1 {
    WrongSchema, EmptyField(&'static str), MutableVersion, InvalidDigest(&'static str),
    SameSampleIdentity, SourceLineageMismatch, SplitMismatch,
    InvariantInstructionChanged, InvariantEnvelopeChanged, RelationMapMissing,
    AdmissibilityReasonMissing, UnexpectedAdmissibilityReason,
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
