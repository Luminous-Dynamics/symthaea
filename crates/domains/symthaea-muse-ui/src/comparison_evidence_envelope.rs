// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Durable trial-instance envelope for artifact-bound comparison evidence.
//!
//! `BlindTrialId(u64)` is intentionally local runtime bookkeeping. It is not a
//! durable cross-browser/device namespace and should not become a persistence
//! primary key merely because it appears inside the comparison state machine.
//!
//! This layer adds a caller-supplied 128-bit trial-instance identifier, encoded
//! as exactly 32 lower-case hexadecimal characters and bound before the strict
//! comparison workspace starts. The identifier is a namespace token only: this
//! module does not claim that it was randomly generated, globally unique, or
//! cryptographically unpredictable.
//!
//! A future create-once store can use the envelope to distinguish exact replay
//! from conflicting reuse of one durable trial-instance identifier.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::comparison::{ComparisonSession, ComparisonTransportIntent};
use crate::comparison_blind::{BlindLabel, BlindSubjectView, BlindTrialId};
use crate::comparison_browser_events::{ComparisonBrowserEvent, ComparisonBrowserUpdate};
use crate::comparison_duration_exposure::BlindExposurePolicy;
use crate::comparison_evidence_record::{
    ArtifactBoundBlindComparisonEvidenceV1, ArtifactBoundBlindComparisonWorkspace,
    BlindAssignmentProvenanceV1, ComparisonEvidenceRecordError,
};
use crate::comparison_evidence_validation::{
    ComparisonEvidenceValidationError, ValidatedArtifactBoundBlindComparisonEvidenceV1,
    validate_artifact_bound_comparison_evidence,
};
use crate::comparison_judgment::{BlindComparisonChoice, BlindComparisonJudgment};
use crate::comparison_timeline::ResolvedComparisonAnchor;
use crate::comparison_workspace_runtime::{
    BlindComparisonRevealReport, BlindComparisonWorkspaceStatus,
};
use crate::playback::{PlaybackEffect, PlaybackState};

pub const ARTIFACT_BOUND_COMPARISON_ENVELOPE_SCHEMA_VERSION: u32 = 1;
pub const TRIAL_INSTANCE_ID_HEX_LEN: usize = 32;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtifactBoundBlindComparisonEnvelopeV1 {
    pub schema_version: u32,
    pub trial_instance_id: String,
    pub evidence: ArtifactBoundBlindComparisonEvidenceV1,
}

#[derive(Debug, PartialEq)]
pub enum ComparisonEvidenceEnvelopeError {
    InvalidTrialInstanceId,
    UnsupportedEnvelopeSchema { found: u32 },
    Record(ComparisonEvidenceRecordError),
    Validation(ComparisonEvidenceValidationError),
}

impl fmt::Display for ComparisonEvidenceEnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTrialInstanceId => write!(
                f,
                "comparison trial_instance_id must be exactly 32 lower-case hexadecimal characters"
            ),
            Self::UnsupportedEnvelopeSchema { found } => write!(
                f,
                "unsupported artifact-bound comparison envelope schema version {found}"
            ),
            Self::Record(error) => error.fmt(f),
            Self::Validation(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ComparisonEvidenceEnvelopeError {}

impl From<ComparisonEvidenceRecordError> for ComparisonEvidenceEnvelopeError {
    fn from(value: ComparisonEvidenceRecordError) -> Self {
        Self::Record(value)
    }
}

impl From<ComparisonEvidenceValidationError> for ComparisonEvidenceEnvelopeError {
    fn from(value: ComparisonEvidenceValidationError) -> Self {
        Self::Validation(value)
    }
}

/// Strict comparison workspace with its durable trial namespace bound before
/// any blind presentation occurs.
#[derive(Debug, PartialEq)]
pub struct PreboundArtifactComparisonTrial {
    trial_instance_id: String,
    inner: ArtifactBoundBlindComparisonWorkspace,
}

impl PreboundArtifactComparisonTrial {
    pub fn new(
        trial_instance_id: String,
        session: ComparisonSession,
        runtime_trial_id: BlindTrialId,
        swapped: bool,
        exposure_policy: BlindExposurePolicy,
        assignment_provenance: BlindAssignmentProvenanceV1,
    ) -> Result<Self, ComparisonEvidenceEnvelopeError> {
        validate_trial_instance_id(&trial_instance_id)?;
        let inner = ArtifactBoundBlindComparisonWorkspace::new(
            session,
            runtime_trial_id,
            swapped,
            exposure_policy,
            assignment_provenance,
        )?;
        Ok(Self {
            trial_instance_id,
            inner,
        })
    }

    pub fn trial_instance_id(&self) -> &str {
        &self.trial_instance_id
    }

    pub fn status(&self) -> BlindComparisonWorkspaceStatus {
        self.inner.status()
    }

    pub fn visible_subjects(&self) -> [BlindSubjectView; 2] {
        self.inner.visible_subjects()
    }

    pub fn set_transport_intent(
        &mut self,
        intent: ComparisonTransportIntent,
    ) -> Result<(), ComparisonEvidenceEnvelopeError> {
        Ok(self.inner.set_transport_intent(intent)?)
    }

    pub fn begin_visible_switch(
        &mut self,
        visible_label: BlindLabel,
        resolved_target: ResolvedComparisonAnchor,
        playback: &mut PlaybackState,
    ) -> Result<Vec<PlaybackEffect>, ComparisonEvidenceEnvelopeError> {
        Ok(self
            .inner
            .begin_visible_switch(visible_label, resolved_target, playback)?)
    }

    pub fn handle_browser_event(
        &mut self,
        playback: &mut PlaybackState,
        event: ComparisonBrowserEvent,
    ) -> Result<ComparisonBrowserUpdate, ComparisonEvidenceEnvelopeError> {
        Ok(self.inner.handle_browser_event(playback, event)?)
    }

    pub fn record_judgment(
        &mut self,
        choice: BlindComparisonChoice,
        self_reported_confidence: Option<f32>,
        note: String,
    ) -> Result<&BlindComparisonJudgment, ComparisonEvidenceEnvelopeError> {
        Ok(self
            .inner
            .record_judgment(choice, self_reported_confidence, note)?)
    }

    pub fn reveal(
        &mut self,
    ) -> Result<&BlindComparisonRevealReport, ComparisonEvidenceEnvelopeError> {
        Ok(self.inner.reveal()?)
    }

    /// Construct the ordinary Serde envelope from the trial-instance identifier
    /// frozen at workspace creation and the evidence produced by that same
    /// owned strict comparison runtime.
    pub fn build_envelope(
        &self,
        recorded_at_unix_ms: u64,
    ) -> Result<ArtifactBoundBlindComparisonEnvelopeV1, ComparisonEvidenceEnvelopeError> {
        Ok(ArtifactBoundBlindComparisonEnvelopeV1 {
            schema_version: ARTIFACT_BOUND_COMPARISON_ENVELOPE_SCHEMA_VERSION,
            trial_instance_id: self.trial_instance_id.clone(),
            evidence: self.inner.build_durable_evidence(recorded_at_unix_ms)?,
        })
    }
}

/// Opaque in-process admission result for one deserialized envelope.
///
/// This wrapper is deliberately non-Serde. It establishes only structural
/// validity and trial-instance syntax; it does not authenticate the producer or
/// prove identifier uniqueness/randomness.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedArtifactBoundBlindComparisonEnvelopeV1 {
    trial_instance_id: String,
    evidence: ValidatedArtifactBoundBlindComparisonEvidenceV1,
}

impl ValidatedArtifactBoundBlindComparisonEnvelopeV1 {
    pub fn trial_instance_id(&self) -> &str {
        &self.trial_instance_id
    }

    pub fn evidence(&self) -> &ValidatedArtifactBoundBlindComparisonEvidenceV1 {
        &self.evidence
    }
}

pub fn validate_artifact_bound_comparison_envelope(
    envelope: ArtifactBoundBlindComparisonEnvelopeV1,
) -> Result<ValidatedArtifactBoundBlindComparisonEnvelopeV1, ComparisonEvidenceEnvelopeError> {
    if envelope.schema_version != ARTIFACT_BOUND_COMPARISON_ENVELOPE_SCHEMA_VERSION {
        return Err(ComparisonEvidenceEnvelopeError::UnsupportedEnvelopeSchema {
            found: envelope.schema_version,
        });
    }
    validate_trial_instance_id(&envelope.trial_instance_id)?;
    let evidence = validate_artifact_bound_comparison_evidence(envelope.evidence)?;
    Ok(ValidatedArtifactBoundBlindComparisonEnvelopeV1 {
        trial_instance_id: envelope.trial_instance_id,
        evidence,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonEvidenceReplayDisposition {
    /// No record exists under this durable trial-instance identifier.
    NewTrial,
    /// The same durable identifier already names byte/field-equivalent evidence.
    ExactReplay,
    /// The same durable identifier is being reused for different evidence.
    Conflict,
}

/// Pure create-once admission classification for a future storage backend.
///
/// Both operands must already have crossed the structural validation boundary.
/// Different trial-instance identifiers are distinct observations even when the
///ir evidence happens to be identical. Reusing one identifier is idempotent only
/// when the complete evidence record is exactly equal.
pub fn classify_validated_replay(
    existing: Option<&ValidatedArtifactBoundBlindComparisonEnvelopeV1>,
    candidate: &ValidatedArtifactBoundBlindComparisonEnvelopeV1,
) -> ComparisonEvidenceReplayDisposition {
    let Some(existing) = existing else {
        return ComparisonEvidenceReplayDisposition::NewTrial;
    };
    if existing.trial_instance_id != candidate.trial_instance_id {
        return ComparisonEvidenceReplayDisposition::NewTrial;
    }
    if existing.evidence.record() == candidate.evidence.record() {
        ComparisonEvidenceReplayDisposition::ExactReplay
    } else {
        ComparisonEvidenceReplayDisposition::Conflict
    }
}

fn validate_trial_instance_id(value: &str) -> Result<(), ComparisonEvidenceEnvelopeError> {
    if value.len() == TRIAL_INSTANCE_ID_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(ComparisonEvidenceEnvelopeError::InvalidTrialInstanceId)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn identity(score: char, composition: char, rendition: char) -> ArtifactIdentity {
        ArtifactIdentity {
            score_content: ScoreContentArtifactId(score.to_string().repeat(64)),
            composition: CompositionArtifactId(composition.to_string().repeat(64)),
            rendition: RenditionArtifactId(rendition.to_string().repeat(64)),
        }
    }

    fn evidence(note: &str) -> ArtifactBoundBlindComparisonEvidenceV1 {
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
                note: note.into(),
            },
            identity_relation: ComparisonIdentityRelationV1::DifferentComposition,
            assignment_provenance: BlindAssignmentProvenanceV1::Unspecified,
            authority: ComparisonEvidenceAuthorityV1::HumanSelfReport,
        }
    }

    fn envelope(id: &str, note: &str) -> ArtifactBoundBlindComparisonEnvelopeV1 {
        ArtifactBoundBlindComparisonEnvelopeV1 {
            schema_version: ARTIFACT_BOUND_COMPARISON_ENVELOPE_SCHEMA_VERSION,
            trial_instance_id: id.into(),
            evidence: evidence(note),
        }
    }

    #[test]
    fn durable_trial_instance_id_is_strictly_canonical() {
        let valid = "0123456789abcdef0123456789abcdef";
        assert!(validate_trial_instance_id(valid).is_ok());
        for invalid in [
            "",
            "0123456789abcdef",
            "0123456789abcdef0123456789abcdeg",
            "0123456789ABCDEF0123456789ABCDEF",
        ] {
            assert_eq!(
                validate_trial_instance_id(invalid),
                Err(ComparisonEvidenceEnvelopeError::InvalidTrialInstanceId)
            );
        }
    }

    #[test]
    fn invalid_trial_instance_id_fails_before_strict_workspace_construction() {
        // The ID validator is intentionally the first operation in
        // `PreboundArtifactComparisonTrial::new`; this focused unit assertion
        // freezes the same primitive without constructing browser/playback fixtures.
        assert_eq!(
            validate_trial_instance_id("not-an-id"),
            Err(ComparisonEvidenceEnvelopeError::InvalidTrialInstanceId)
        );
    }

    #[test]
    fn deserialized_envelope_revalidates_both_envelope_and_inner_evidence() {
        let raw = envelope("0123456789abcdef0123456789abcdef", "blind report");
        let bytes = serde_json::to_vec(&raw).unwrap();
        let decoded: ArtifactBoundBlindComparisonEnvelopeV1 =
            serde_json::from_slice(&bytes).unwrap();
        let validated = validate_artifact_bound_comparison_envelope(decoded).unwrap();
        assert_eq!(
            validated.trial_instance_id(),
            "0123456789abcdef0123456789abcdef"
        );
        assert_eq!(validated.evidence().record(), &raw.evidence);
    }

    #[test]
    fn unsupported_envelope_schema_does_not_reach_inner_validation() {
        let mut raw = envelope("0123456789abcdef0123456789abcdef", "blind report");
        raw.schema_version = 2;
        assert_eq!(
            validate_artifact_bound_comparison_envelope(raw),
            Err(ComparisonEvidenceEnvelopeError::UnsupportedEnvelopeSchema { found: 2 })
        );
    }

    #[test]
    fn create_once_replay_is_idempotent_only_for_exact_evidence() {
        let existing = validate_artifact_bound_comparison_envelope(envelope(
            "0123456789abcdef0123456789abcdef",
            "blind report",
        ))
        .unwrap();
        let exact = validate_artifact_bound_comparison_envelope(envelope(
            "0123456789abcdef0123456789abcdef",
            "blind report",
        ))
        .unwrap();
        let conflict = validate_artifact_bound_comparison_envelope(envelope(
            "0123456789abcdef0123456789abcdef",
            "rewritten report",
        ))
        .unwrap();
        let distinct = validate_artifact_bound_comparison_envelope(envelope(
            "fedcba9876543210fedcba9876543210",
            "blind report",
        ))
        .unwrap();

        assert_eq!(
            classify_validated_replay(None, &exact),
            ComparisonEvidenceReplayDisposition::NewTrial
        );
        assert_eq!(
            classify_validated_replay(Some(&existing), &exact),
            ComparisonEvidenceReplayDisposition::ExactReplay
        );
        assert_eq!(
            classify_validated_replay(Some(&existing), &conflict),
            ComparisonEvidenceReplayDisposition::Conflict
        );
        assert_eq!(
            classify_validated_replay(Some(&existing), &distinct),
            ComparisonEvidenceReplayDisposition::NewTrial
        );
    }
}
