// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validation boundary for persisted/imported blind-comparison evidence.
//!
//! `ArtifactBoundBlindComparisonEvidenceV1` is intentionally a Serde wire/data
//! object. Successfully deserializing it proves only that bytes matched the JSON
//! shape. It does not prove that the schema is supported, hashes are canonical,
//! blind/resolved choices agree, exposure really satisfies the recorded policy,
//! or the stored identity relation follows from the two exact artifacts.
//!
//! This module revalidates those invariants and promotes a borrowed wire record
//! into a private-field runtime wrapper. The wrapper is deliberately not
//! serializable or deserializable: persisted bytes never mint validation state.
//!
//! Structural validation still does not prove authorship, listening attention,
//! randomization quality, study validity, causal effect, or musical quality.

use std::fmt;

use symthaea_muse_protocol::ArtifactIdentity;

use crate::comparison_evidence_record::{
    ARTIFACT_BOUND_COMPARISON_EVIDENCE_SCHEMA_VERSION,
    MAX_ASSIGNMENT_METHOD_UTF8_BYTES, MAX_ASSIGNMENT_REFERENCE_UTF8_BYTES,
    MAX_COMPARISON_NOTE_UTF8_BYTES, ArtifactBoundBlindComparisonEvidenceV1,
    BlindAssignmentProvenanceV1, BlindHumanComparisonChoiceV1,
    ComparisonEvidenceAuthorityV1, ComparisonIdentityRelationV1, EvidenceSideV1,
    ResolvedHumanComparisonChoiceV1,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComparisonEvidenceValidationError {
    UnsupportedSchemaVersion { found: u32 },
    InvalidArtifactIdentity { side: EvidenceSideV1 },
    SameArtifact,
    IdentityConflict,
    IdentityRelationMismatch,
    InvalidAssignment,
    InvalidAnchor,
    InvalidExposurePolicy,
    InvalidDurationEvidence,
    ExposureNotQualified,
    MissingListeningExposure,
    BlindResolvedChoiceMismatch,
    InvalidConfidence,
    NoteTooLarge,
    InvalidAssignmentProvenance,
    UnsupportedAuthority,
}

impl fmt::Display for ComparisonEvidenceValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported artifact-bound comparison evidence schema version {found}"
            ),
            Self::InvalidArtifactIdentity { side } => write!(
                f,
                "comparison evidence side {side:?} contains a non-canonical SHA-256 artifact identity"
            ),
            Self::SameArtifact => write!(
                f,
                "comparison evidence cannot bind both sides to the exact same artifact"
            ),
            Self::IdentityConflict => write!(
                f,
                "comparison evidence artifact identities are internally contradictory"
            ),
            Self::IdentityRelationMismatch => write!(
                f,
                "stored comparison identity relation does not match the exact artifact identities"
            ),
            Self::InvalidAssignment => write!(
                f,
                "revealed assignment must map visible A/B bijectively onto underlying A/B"
            ),
            Self::InvalidAnchor => write!(
                f,
                "comparison musical anchor must have a finite non-negative beat offset"
            ),
            Self::InvalidExposurePolicy => write!(
                f,
                "comparison exposure policy values must be finite and positive"
            ),
            Self::InvalidDurationEvidence => write!(
                f,
                "comparison duration evidence must be finite and non-negative"
            ),
            Self::ExposureNotQualified => write!(
                f,
                "comparison duration evidence does not satisfy its recorded exposure policy"
            ),
            Self::MissingListeningExposure => write!(
                f,
                "comparison evidence requires at least one admitted audition of each visible side"
            ),
            Self::BlindResolvedChoiceMismatch => write!(
                f,
                "blind choice does not resolve through the stored revealed assignment to the stored underlying choice"
            ),
            Self::InvalidConfidence => write!(
                f,
                "self-reported confidence must be finite and within [0, 1]"
            ),
            Self::NoteTooLarge => write!(
                f,
                "comparison note exceeds the supported V1 UTF-8 byte budget"
            ),
            Self::InvalidAssignmentProvenance => write!(
                f,
                "blind-assignment provenance is empty or exceeds the supported V1 byte budget"
            ),
            Self::UnsupportedAuthority => write!(
                f,
                "comparison evidence authority is not supported by this validator"
            ),
        }
    }
}

impl std::error::Error for ComparisonEvidenceValidationError {}

/// Opaque runtime proof that one V1 wire record passed every structural
/// admission check implemented by [`validate_artifact_bound_comparison_evidence`].
///
/// This is not a cryptographic signature, authenticity proof, or scientific
/// authority token. It only prevents downstream in-process code from confusing
/// unchecked Serde data with structurally validated evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedArtifactBoundBlindComparisonEvidenceV1 {
    record: ArtifactBoundBlindComparisonEvidenceV1,
}

impl ValidatedArtifactBoundBlindComparisonEvidenceV1 {
    pub fn record(&self) -> &ArtifactBoundBlindComparisonEvidenceV1 {
        &self.record
    }

    pub fn into_record(self) -> ArtifactBoundBlindComparisonEvidenceV1 {
        self.record
    }
}

/// Revalidate a deserialized/imported V1 comparison record and promote it into
/// an opaque runtime wrapper only if every cross-field invariant holds.
pub fn validate_artifact_bound_comparison_evidence(
    record: ArtifactBoundBlindComparisonEvidenceV1,
) -> Result<ValidatedArtifactBoundBlindComparisonEvidenceV1, ComparisonEvidenceValidationError> {
    if record.schema_version != ARTIFACT_BOUND_COMPARISON_EVIDENCE_SCHEMA_VERSION {
        return Err(ComparisonEvidenceValidationError::UnsupportedSchemaVersion {
            found: record.schema_version,
        });
    }

    validate_identity(&record.side_a_identity, EvidenceSideV1::A)?;
    validate_identity(&record.side_b_identity, EvidenceSideV1::B)?;
    let relation = derive_identity_relation(&record.side_a_identity, &record.side_b_identity)?;
    if relation != record.identity_relation {
        return Err(ComparisonEvidenceValidationError::IdentityRelationMismatch);
    }

    validate_assignment(&record)?;
    validate_anchor(&record)?;
    validate_exposure(&record)?;
    validate_judgment(&record)?;
    validate_assignment_provenance(&record.assignment_provenance)?;

    if record.authority != ComparisonEvidenceAuthorityV1::HumanSelfReport {
        return Err(ComparisonEvidenceValidationError::UnsupportedAuthority);
    }

    Ok(ValidatedArtifactBoundBlindComparisonEvidenceV1 { record })
}

fn validate_identity(
    identity: &ArtifactIdentity,
    side: EvidenceSideV1,
) -> Result<(), ComparisonEvidenceValidationError> {
    if valid_sha256_hex(&identity.score_content.0)
        && valid_sha256_hex(&identity.composition.0)
        && valid_sha256_hex(&identity.rendition.0)
    {
        Ok(())
    } else {
        Err(ComparisonEvidenceValidationError::InvalidArtifactIdentity { side })
    }
}

fn derive_identity_relation(
    a: &ArtifactIdentity,
    b: &ArtifactIdentity,
) -> Result<ComparisonIdentityRelationV1, ComparisonEvidenceValidationError> {
    if a == b {
        return Err(ComparisonEvidenceValidationError::SameArtifact);
    }

    let same_score = a.score_content == b.score_content;
    let same_composition = a.composition == b.composition;
    let same_rendition = a.rendition == b.rendition;

    // A rendition commitment names exact rendered bytes. The same bytes cannot
    // honestly be attached to conflicting upstream score/composition identity.
    // Likewise one composition commitment cannot name two symbolic scores.
    if same_rendition && (!same_score || !same_composition)
        || same_composition && !same_score
    {
        return Err(ComparisonEvidenceValidationError::IdentityConflict);
    }

    match (same_score, same_composition, same_rendition) {
        (true, true, false) => Ok(ComparisonIdentityRelationV1::AlternateRendition),
        (true, false, false) => Ok(ComparisonIdentityRelationV1::SameScoreDifferentComposition),
        (false, false, false) => Ok(ComparisonIdentityRelationV1::DifferentComposition),
        // `(true, true, true)` was rejected as SameArtifact. Every other
        // remaining combination violates the artifact identity model.
        _ => Err(ComparisonEvidenceValidationError::IdentityConflict),
    }
}

fn validate_assignment(
    record: &ArtifactBoundBlindComparisonEvidenceV1,
) -> Result<(), ComparisonEvidenceValidationError> {
    match (
        record.revealed_assignment.visible_a_side,
        record.revealed_assignment.visible_b_side,
    ) {
        (EvidenceSideV1::A, EvidenceSideV1::B)
        | (EvidenceSideV1::B, EvidenceSideV1::A) => Ok(()),
        _ => Err(ComparisonEvidenceValidationError::InvalidAssignment),
    }
}

fn validate_anchor(
    record: &ArtifactBoundBlindComparisonEvidenceV1,
) -> Result<(), ComparisonEvidenceValidationError> {
    if record.anchor.beat_offset.is_finite() && record.anchor.beat_offset >= 0.0 {
        Ok(())
    } else {
        Err(ComparisonEvidenceValidationError::InvalidAnchor)
    }
}

fn validate_exposure(
    record: &ArtifactBoundBlindComparisonEvidenceV1,
) -> Result<(), ComparisonEvidenceValidationError> {
    let policy = record.exposure_policy;
    if !policy.minimum_seconds_per_side.is_finite()
        || policy.minimum_seconds_per_side <= 0.0
        || !policy.max_contiguous_step_seconds.is_finite()
        || policy.max_contiguous_step_seconds <= 0.0
    {
        return Err(ComparisonEvidenceValidationError::InvalidExposurePolicy);
    }

    let duration = record.duration_evidence_at_judgment;
    if !duration.visible_a_seconds.is_finite()
        || duration.visible_a_seconds < 0.0
        || !duration.visible_b_seconds.is_finite()
        || duration.visible_b_seconds < 0.0
    {
        return Err(ComparisonEvidenceValidationError::InvalidDurationEvidence);
    }

    const EPS: f64 = 1e-6;
    if duration.visible_a_seconds + EPS < policy.minimum_seconds_per_side
        || duration.visible_b_seconds + EPS < policy.minimum_seconds_per_side
    {
        return Err(ComparisonEvidenceValidationError::ExposureNotQualified);
    }
    Ok(())
}

fn validate_judgment(
    record: &ArtifactBoundBlindComparisonEvidenceV1,
) -> Result<(), ComparisonEvidenceValidationError> {
    let exposure = record.judgment.listening_exposure;
    if exposure.visible_a_auditions == 0 || exposure.visible_b_auditions == 0 {
        return Err(ComparisonEvidenceValidationError::MissingListeningExposure);
    }

    if let Some(confidence) = record.judgment.self_reported_confidence
        && (!confidence.is_finite() || !(0.0..=1.0).contains(&confidence))
    {
        return Err(ComparisonEvidenceValidationError::InvalidConfidence);
    }
    if record.judgment.note.len() > MAX_COMPARISON_NOTE_UTF8_BYTES {
        return Err(ComparisonEvidenceValidationError::NoteTooLarge);
    }

    let expected_resolved = match record.judgment.blind_choice {
        BlindHumanComparisonChoiceV1::PreferVisibleA => match record.revealed_assignment.visible_a_side {
            EvidenceSideV1::A => ResolvedHumanComparisonChoiceV1::PreferSideA,
            EvidenceSideV1::B => ResolvedHumanComparisonChoiceV1::PreferSideB,
        },
        BlindHumanComparisonChoiceV1::PreferVisibleB => match record.revealed_assignment.visible_b_side {
            EvidenceSideV1::A => ResolvedHumanComparisonChoiceV1::PreferSideA,
            EvidenceSideV1::B => ResolvedHumanComparisonChoiceV1::PreferSideB,
        },
        BlindHumanComparisonChoiceV1::NoPreference => {
            ResolvedHumanComparisonChoiceV1::NoPreference
        }
    };
    if expected_resolved != record.judgment.resolved_choice {
        return Err(ComparisonEvidenceValidationError::BlindResolvedChoiceMismatch);
    }
    Ok(())
}

fn validate_assignment_provenance(
    provenance: &BlindAssignmentProvenanceV1,
) -> Result<(), ComparisonEvidenceValidationError> {
    match provenance {
        BlindAssignmentProvenanceV1::Unspecified => Ok(()),
        BlindAssignmentProvenanceV1::ExternallySupplied { method, reference } => {
            if method.trim().is_empty() || method.len() > MAX_ASSIGNMENT_METHOD_UTF8_BYTES {
                return Err(ComparisonEvidenceValidationError::InvalidAssignmentProvenance);
            }
            if let Some(reference) = reference
                && (reference.trim().is_empty()
                    || reference.len() > MAX_ASSIGNMENT_REFERENCE_UTF8_BYTES)
            {
                return Err(ComparisonEvidenceValidationError::InvalidAssignmentProvenance);
            }
            Ok(())
        }
    }
}

fn valid_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison_evidence_record::{
        BlindAssignmentProvenanceV1, BlindHumanComparisonChoiceV1,
        ComparisonEvidenceAuthorityV1, DurationEvidenceV1, EvidenceSideV1,
        ExposurePolicyEvidenceV1, HumanComparisonJudgmentV1,
        ListeningExposureEvidenceV1, MusicalAnchorEvidenceV1,
        ResolvedHumanComparisonChoiceV1, RevealedAssignmentEvidenceV1,
    };
    use symthaea_muse_protocol::{
        CompositionArtifactId, RenditionArtifactId, ScoreContentArtifactId,
    };

    fn identity(score: char, composition: char, rendition: char) -> ArtifactIdentity {
        ArtifactIdentity {
            score_content: ScoreContentArtifactId(score.to_string().repeat(64)),
            composition: CompositionArtifactId(composition.to_string().repeat(64)),
            rendition: RenditionArtifactId(rendition.to_string().repeat(64)),
        }
    }

    fn valid_record() -> ArtifactBoundBlindComparisonEvidenceV1 {
        ArtifactBoundBlindComparisonEvidenceV1 {
            schema_version: ARTIFACT_BOUND_COMPARISON_EVIDENCE_SCHEMA_VERSION,
            trial_id: 202,
            recorded_at_unix_ms: 1_800_000_000_000,
            side_a_identity: identity('a', 'b', 'c'),
            side_b_identity: identity('d', 'e', 'f'),
            revealed_assignment: RevealedAssignmentEvidenceV1 {
                visible_a_side: EvidenceSideV1::B,
                visible_b_side: EvidenceSideV1::A,
            },
            anchor: MusicalAnchorEvidenceV1 {
                bar_index: 2,
                beat_offset: 0.5,
            },
            exposure_policy: ExposurePolicyEvidenceV1 {
                minimum_seconds_per_side: 0.5,
                max_contiguous_step_seconds: 0.6,
            },
            duration_evidence_at_judgment: DurationEvidenceV1 {
                visible_a_seconds: 0.5,
                visible_b_seconds: 0.5,
            },
            judgment: HumanComparisonJudgmentV1 {
                blind_choice: BlindHumanComparisonChoiceV1::PreferVisibleA,
                resolved_choice: ResolvedHumanComparisonChoiceV1::PreferSideB,
                listening_exposure: ListeningExposureEvidenceV1 {
                    visible_a_auditions: 1,
                    visible_b_auditions: 1,
                },
                self_reported_confidence: Some(0.75),
                note: "blind report".into(),
            },
            identity_relation: ComparisonIdentityRelationV1::DifferentComposition,
            assignment_provenance: BlindAssignmentProvenanceV1::Unspecified,
            authority: ComparisonEvidenceAuthorityV1::HumanSelfReport,
        }
    }

    #[test]
    fn exact_v1_record_promotes_to_non_serde_validated_wrapper() {
        let record = valid_record();
        let validated = validate_artifact_bound_comparison_evidence(record.clone()).unwrap();
        assert_eq!(validated.record(), &record);
        assert_eq!(validated.into_record(), record);
    }

    #[test]
    fn serde_shape_success_does_not_bypass_validation() {
        let mut record = valid_record();
        record.schema_version = 99;
        let bytes = serde_json::to_vec(&record).unwrap();
        let decoded: ArtifactBoundBlindComparisonEvidenceV1 =
            serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            validate_artifact_bound_comparison_evidence(decoded),
            Err(ComparisonEvidenceValidationError::UnsupportedSchemaVersion { found: 99 })
        );
    }

    #[test]
    fn tampered_hash_and_identity_relation_fail_closed() {
        let mut bad_hash = valid_record();
        bad_hash.side_a_identity.score_content.0 = "NOT-A-HASH".into();
        assert_eq!(
            validate_artifact_bound_comparison_evidence(bad_hash),
            Err(ComparisonEvidenceValidationError::InvalidArtifactIdentity {
                side: EvidenceSideV1::A,
            })
        );

        let mut bad_relation = valid_record();
        bad_relation.identity_relation = ComparisonIdentityRelationV1::AlternateRendition;
        assert_eq!(
            validate_artifact_bound_comparison_evidence(bad_relation),
            Err(ComparisonEvidenceValidationError::IdentityRelationMismatch)
        );
    }

    #[test]
    fn same_artifact_and_conflicting_identity_are_rejected() {
        let mut same = valid_record();
        same.side_b_identity = same.side_a_identity.clone();
        assert_eq!(
            validate_artifact_bound_comparison_evidence(same),
            Err(ComparisonEvidenceValidationError::SameArtifact)
        );

        let mut conflict = valid_record();
        conflict.side_b_identity.composition = conflict.side_a_identity.composition.clone();
        assert_eq!(
            validate_artifact_bound_comparison_evidence(conflict),
            Err(ComparisonEvidenceValidationError::IdentityConflict)
        );
    }

    #[test]
    fn blind_choice_must_resolve_through_stored_assignment() {
        let mut record = valid_record();
        record.judgment.resolved_choice = ResolvedHumanComparisonChoiceV1::PreferSideA;
        assert_eq!(
            validate_artifact_bound_comparison_evidence(record),
            Err(ComparisonEvidenceValidationError::BlindResolvedChoiceMismatch)
        );
    }

    #[test]
    fn duration_must_satisfy_the_recorded_policy() {
        let mut record = valid_record();
        record.duration_evidence_at_judgment.visible_b_seconds = 0.49;
        assert_eq!(
            validate_artifact_bound_comparison_evidence(record),
            Err(ComparisonEvidenceValidationError::ExposureNotQualified)
        );
    }

    #[test]
    fn partial_audition_and_posthoc_bad_provenance_are_rejected() {
        let mut partial = valid_record();
        partial.judgment.listening_exposure.visible_b_auditions = 0;
        assert_eq!(
            validate_artifact_bound_comparison_evidence(partial),
            Err(ComparisonEvidenceValidationError::MissingListeningExposure)
        );

        let mut bad_provenance = valid_record();
        bad_provenance.assignment_provenance = BlindAssignmentProvenanceV1::ExternallySupplied {
            method: String::new(),
            reference: None,
        };
        assert_eq!(
            validate_artifact_bound_comparison_evidence(bad_provenance),
            Err(ComparisonEvidenceValidationError::InvalidAssignmentProvenance)
        );
    }
}
