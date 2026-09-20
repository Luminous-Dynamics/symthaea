// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Backend-independent create-once storage contract for private comparison evidence.
//!
//! This module deliberately performs no browser I/O and contains no network path.
//! It defines the admission rules an IndexedDB adapter must obey:
//!
//! ```text
//! stored JSON
//!     != valid evidence
//!
//! deserialize
//!   -> validate envelope
//!   -> compare against validated incoming trial
//!   -> insert | exact replay | conflict
//! ```
//!
//! Existing malformed evidence is corruption, never overwrite permission. The
//! browser adapter must use a create-only primitive (`IDBObjectStore::add`) and
//! invoke this contract only to prepare a candidate or resolve a uniqueness
//! conflict. It must never downgrade to `localStorage` or `put` semantics.

use std::fmt;

use crate::comparison_evidence_envelope::{
    ArtifactBoundBlindComparisonEnvelopeV1, ComparisonEvidenceEnvelopeError,
    ComparisonEvidenceReplayDisposition, ValidatedArtifactBoundBlindComparisonEnvelopeV1,
    classify_validated_replay, validate_artifact_bound_comparison_envelope,
};

/// A validated envelope plus the exact JSON text a backend may attempt to store.
///
/// The fields remain private so callers cannot construct a write intent without
/// crossing both Serde serialization and the full envelope validation boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedComparisonEvidenceWrite {
    envelope: ArtifactBoundBlindComparisonEnvelopeV1,
    validated: ValidatedArtifactBoundBlindComparisonEnvelopeV1,
    json: String,
}

impl PreparedComparisonEvidenceWrite {
    pub fn trial_instance_id(&self) -> &str {
        self.validated.trial_instance_id()
    }

    pub fn json(&self) -> &str {
        &self.json
    }

    pub fn envelope(&self) -> &ArtifactBoundBlindComparisonEnvelopeV1 {
        &self.envelope
    }

    pub fn validated(&self) -> &ValidatedArtifactBoundBlindComparisonEnvelopeV1 {
        &self.validated
    }
}

/// Result of resolving one create-once write against storage state.
#[derive(Clone, Debug, PartialEq)]
pub enum ComparisonEvidenceCreateResolution {
    /// No existing record is present. The adapter may attempt an atomic
    /// create-only insert using the enclosed already-validated write intent.
    Insert(PreparedComparisonEvidenceWrite),
    /// The durable trial key already names exactly the same validated evidence.
    /// A retry is therefore idempotent and requires no mutation.
    ExactReplay,
}

#[derive(Debug, PartialEq)]
pub enum ComparisonEvidenceStoreContractError {
    CandidateInvalid(ComparisonEvidenceEnvelopeError),
    CandidateSerialization(String),
    ExistingMalformedJson(String),
    ExistingInvalid(ComparisonEvidenceEnvelopeError),
    ExistingKeyMismatch {
        expected_trial_instance_id: String,
        found_trial_instance_id: String,
    },
    ConflictingTrialReuse {
        trial_instance_id: String,
    },
}

impl fmt::Display for ComparisonEvidenceStoreContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CandidateInvalid(error) => {
                write!(f, "comparison evidence candidate is invalid: {error}")
            }
            Self::CandidateSerialization(message) => {
                write!(f, "comparison evidence candidate could not be serialized: {message}")
            }
            Self::ExistingMalformedJson(message) => write!(
                f,
                "stored comparison evidence is malformed JSON and must not be overwritten: {message}"
            ),
            Self::ExistingInvalid(error) => write!(
                f,
                "stored comparison evidence failed structural validation and must not be overwritten: {error}"
            ),
            Self::ExistingKeyMismatch {
                expected_trial_instance_id,
                found_trial_instance_id,
            } => write!(
                f,
                "stored comparison evidence key mismatch: requested {expected_trial_instance_id}, value contains {found_trial_instance_id}"
            ),
            Self::ConflictingTrialReuse { trial_instance_id } => write!(
                f,
                "comparison trial_instance_id {trial_instance_id} is already bound to different evidence"
            ),
        }
    }
}

impl std::error::Error for ComparisonEvidenceStoreContractError {}

/// Validate and serialize one new envelope before any storage call is attempted.
///
/// Validation intentionally happens before serialization becomes a write intent:
/// callers cannot use this helper to persist an envelope that merely happens to
/// implement `Serialize` but violates the evidence theorem.
pub fn prepare_comparison_evidence_write(
    envelope: ArtifactBoundBlindComparisonEnvelopeV1,
) -> Result<PreparedComparisonEvidenceWrite, ComparisonEvidenceStoreContractError> {
    let validated = validate_artifact_bound_comparison_envelope(envelope.clone())
        .map_err(ComparisonEvidenceStoreContractError::CandidateInvalid)?;
    let json = serde_json::to_string(&envelope).map_err(|error| {
        ComparisonEvidenceStoreContractError::CandidateSerialization(error.to_string())
    })?;
    Ok(PreparedComparisonEvidenceWrite {
        envelope,
        validated,
        json,
    })
}

/// Decode and structurally validate one stored JSON record.
///
/// This is the only ordinary read boundary a browser adapter should expose to
/// evidence consumers. Returning the raw DTO would allow storage corruption or
/// arbitrary injected JSON to regain the constructor's authority by shape alone.
pub fn validate_stored_comparison_evidence_json(
    json: &str,
) -> Result<ValidatedArtifactBoundBlindComparisonEnvelopeV1, ComparisonEvidenceStoreContractError>
{
    let envelope: ArtifactBoundBlindComparisonEnvelopeV1 = serde_json::from_str(json).map_err(
        |error| ComparisonEvidenceStoreContractError::ExistingMalformedJson(error.to_string()),
    )?;
    validate_artifact_bound_comparison_envelope(envelope)
        .map_err(ComparisonEvidenceStoreContractError::ExistingInvalid)
}

/// Resolve one candidate against the storage state observed after a failed
/// create-only insert or during a preflight/read path.
///
/// `None` means there is no existing record and returns an insert intent. When an
/// existing record is supplied, it is revalidated before comparison. Corrupt
/// data never becomes permission to replace the record.
pub fn resolve_create_once_comparison_evidence(
    existing_json: Option<&str>,
    candidate: PreparedComparisonEvidenceWrite,
) -> Result<ComparisonEvidenceCreateResolution, ComparisonEvidenceStoreContractError> {
    let Some(existing_json) = existing_json else {
        return Ok(ComparisonEvidenceCreateResolution::Insert(candidate));
    };

    let existing = validate_stored_comparison_evidence_json(existing_json)?;
    if existing.trial_instance_id() != candidate.trial_instance_id() {
        return Err(ComparisonEvidenceStoreContractError::ExistingKeyMismatch {
            expected_trial_instance_id: candidate.trial_instance_id().to_string(),
            found_trial_instance_id: existing.trial_instance_id().to_string(),
        });
    }

    match classify_validated_replay(Some(&existing), candidate.validated()) {
        ComparisonEvidenceReplayDisposition::ExactReplay => {
            Ok(ComparisonEvidenceCreateResolution::ExactReplay)
        }
        ComparisonEvidenceReplayDisposition::Conflict => {
            Err(ComparisonEvidenceStoreContractError::ConflictingTrialReuse {
                trial_instance_id: candidate.trial_instance_id().to_string(),
            })
        }
        ComparisonEvidenceReplayDisposition::NewTrial => {
            unreachable!("equal validated trial IDs cannot classify as NewTrial")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison_evidence_envelope::ARTIFACT_BOUND_COMPARISON_ENVELOPE_SCHEMA_VERSION;
    use crate::comparison_evidence_record::{
        ARTIFACT_BOUND_COMPARISON_EVIDENCE_SCHEMA_VERSION, BlindAssignmentProvenanceV1,
        BlindHumanComparisonChoiceV1, ComparisonEvidenceAuthorityV1,
        ComparisonIdentityRelationV1, DurationEvidenceV1, EvidenceSideV1,
        ExposurePolicyEvidenceV1, HumanComparisonJudgmentV1,
        ListeningExposureEvidenceV1, MusicalAnchorEvidenceV1,
        ResolvedHumanComparisonChoiceV1, RevealedAssignmentEvidenceV1,
    };
    use symthaea_muse_protocol::{
        ArtifactIdentity, CompositionArtifactId, RenditionArtifactId, ScoreContentArtifactId,
    };

    const TRIAL: &str = "0123456789abcdef0123456789abcdef";
    const OTHER_TRIAL: &str = "fedcba9876543210fedcba9876543210";

    fn identity(score: char, composition: char, rendition: char) -> ArtifactIdentity {
        ArtifactIdentity {
            score_content: ScoreContentArtifactId(score.to_string().repeat(64)),
            composition: CompositionArtifactId(composition.to_string().repeat(64)),
            rendition: RenditionArtifactId(rendition.to_string().repeat(64)),
        }
    }

    fn envelope_for(id: &str, note: &str) -> ArtifactBoundBlindComparisonEnvelopeV1 {
        ArtifactBoundBlindComparisonEnvelopeV1 {
            schema_version: ARTIFACT_BOUND_COMPARISON_ENVELOPE_SCHEMA_VERSION,
            trial_instance_id: id.into(),
            evidence: crate::comparison_evidence_record::ArtifactBoundBlindComparisonEvidenceV1 {
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
                    note: note.into(),
                },
                identity_relation: ComparisonIdentityRelationV1::DifferentComposition,
                assignment_provenance: BlindAssignmentProvenanceV1::Unspecified,
                authority: ComparisonEvidenceAuthorityV1::HumanSelfReport,
            },
        }
    }

    fn envelope(note: &str) -> ArtifactBoundBlindComparisonEnvelopeV1 {
        envelope_for(TRIAL, note)
    }

    #[test]
    fn valid_candidate_becomes_create_only_write_intent() {
        let candidate = prepare_comparison_evidence_write(envelope("blind report")).unwrap();
        assert_eq!(candidate.trial_instance_id(), TRIAL);
        let decoded: ArtifactBoundBlindComparisonEnvelopeV1 =
            serde_json::from_str(candidate.json()).unwrap();
        assert_eq!(decoded, *candidate.envelope());

        let resolution = resolve_create_once_comparison_evidence(None, candidate).unwrap();
        assert!(matches!(
            resolution,
            ComparisonEvidenceCreateResolution::Insert(_)
        ));
    }

    #[test]
    fn exact_existing_record_is_idempotent_replay() {
        let candidate = prepare_comparison_evidence_write(envelope("blind report")).unwrap();
        let existing_json = candidate.json().to_string();
        assert_eq!(
            resolve_create_once_comparison_evidence(Some(&existing_json), candidate).unwrap(),
            ComparisonEvidenceCreateResolution::ExactReplay
        );
    }

    #[test]
    fn changed_evidence_under_same_trial_id_is_conflict() {
        let existing = prepare_comparison_evidence_write(envelope("original")).unwrap();
        let candidate = prepare_comparison_evidence_write(envelope("rewritten")).unwrap();
        assert_eq!(
            resolve_create_once_comparison_evidence(Some(existing.json()), candidate),
            Err(ComparisonEvidenceStoreContractError::ConflictingTrialReuse {
                trial_instance_id: TRIAL.into(),
            })
        );
    }

    #[test]
    fn malformed_existing_json_is_corruption_not_overwrite_permission() {
        let candidate = prepare_comparison_evidence_write(envelope("blind report")).unwrap();
        let error = resolve_create_once_comparison_evidence(Some("{not-json"), candidate)
            .expect_err("corrupt existing evidence must fail closed");
        assert!(matches!(
            error,
            ComparisonEvidenceStoreContractError::ExistingMalformedJson(_)
        ));
    }

    #[test]
    fn structurally_invalid_existing_record_is_not_replaced() {
        let mut existing = envelope("blind report");
        existing.schema_version = 99;
        let existing_json = serde_json::to_string(&existing).unwrap();
        let candidate = prepare_comparison_evidence_write(envelope("blind report")).unwrap();
        assert!(matches!(
            resolve_create_once_comparison_evidence(Some(&existing_json), candidate),
            Err(ComparisonEvidenceStoreContractError::ExistingInvalid(_))
        ));
    }

    #[test]
    fn valid_value_under_wrong_storage_key_is_explicit_corruption() {
        let existing = prepare_comparison_evidence_write(envelope_for(OTHER_TRIAL, "blind report"))
            .unwrap();
        let candidate = prepare_comparison_evidence_write(envelope("blind report")).unwrap();
        assert_eq!(
            resolve_create_once_comparison_evidence(Some(existing.json()), candidate),
            Err(ComparisonEvidenceStoreContractError::ExistingKeyMismatch {
                expected_trial_instance_id: TRIAL.into(),
                found_trial_instance_id: OTHER_TRIAL.into(),
            })
        );
    }

    #[test]
    fn ordinary_read_returns_validated_wrapper_only_after_revalidation() {
        let prepared = prepare_comparison_evidence_write(envelope("blind report")).unwrap();
        let validated = validate_stored_comparison_evidence_json(prepared.json()).unwrap();
        assert_eq!(validated.trial_instance_id(), TRIAL);
        assert_eq!(
            validated.evidence().record(),
            &prepared.envelope().evidence
        );
    }

    #[test]
    fn candidate_validation_happens_before_a_write_intent_exists() {
        let mut invalid = envelope("blind report");
        invalid.trial_instance_id = "NOT-CANONICAL".into();
        assert!(matches!(
            prepare_comparison_evidence_write(invalid),
            Err(ComparisonEvidenceStoreContractError::CandidateInvalid(_))
        ));
    }
}
