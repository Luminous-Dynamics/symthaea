// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Durable, artifact-bound evidence for completed blind A/B comparisons.
//!
//! The ordinary comparison runtime intentionally permits legacy/imported material
//! whose full `ArtifactIdentity` is unavailable. This stricter wrapper is for a
//! different claim: preserving a human self-report that can still be attributed
//! to the exact two audible renditions after restart.
//!
//! The wrapper therefore captures and validates, before blinding begins:
//!
//! - full score/composition/rendition identity for both subjects;
//! - an explicit playback-source rendition commitment matching each identity;
//! - the externally supplied blind assignment and its descriptive provenance;
//! - the fixed anchor and exposure policy owned by the lower runtime.
//!
//! It owns the lower runtime through reveal and only serializes the reveal report
//! emitted by that exact runtime. The resulting record remains human self-report
//! evidence: preference is not musical quality, attention proof, causal evidence,
//! study validity, or population-level evidence.

use std::fmt;

use serde::{Deserialize, Serialize};
use symthaea_muse_protocol::ArtifactIdentity;

use crate::comparison::{ComparisonSession, ComparisonSide, ComparisonTransportIntent};
use crate::comparison_blind::{BlindLabel, BlindSubjectView, BlindTrialId, RevealedBlindAssignment};
use crate::comparison_browser_events::{ComparisonBrowserEvent, ComparisonBrowserUpdate};
use crate::comparison_duration_exposure::{BlindDurationEvidence, BlindExposurePolicy};
use crate::comparison_identity::{
    ComparisonIdentityDiff, ComparisonIdentityRelation, compare_subject_identity,
};
use crate::comparison_judgment::{
    BlindComparisonChoice, BlindComparisonJudgment, ResolvedBlindComparisonJudgment,
    ResolvedComparisonChoice,
};
use crate::comparison_timeline::ResolvedComparisonAnchor;
use crate::comparison_workspace_runtime::{
    BlindComparisonRevealReport, BlindComparisonWorkspaceRuntime, BlindComparisonWorkspaceStatus,
    ComparisonWorkspaceError,
};
use crate::playback::{PlaybackEffect, PlaybackState};

pub const ARTIFACT_BOUND_COMPARISON_EVIDENCE_SCHEMA_VERSION: u32 = 1;
pub const MAX_COMPARISON_NOTE_UTF8_BYTES: usize = 4096;
pub const MAX_ASSIGNMENT_METHOD_UTF8_BYTES: usize = 256;
pub const MAX_ASSIGNMENT_REFERENCE_UTF8_BYTES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSideV1 {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevealedAssignmentEvidenceV1 {
    pub visible_a_side: EvidenceSideV1,
    pub visible_b_side: EvidenceSideV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MusicalAnchorEvidenceV1 {
    pub bar_index: u32,
    pub beat_offset: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExposurePolicyEvidenceV1 {
    pub minimum_seconds_per_side: f64,
    pub max_contiguous_step_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DurationEvidenceV1 {
    pub visible_a_seconds: f64,
    pub visible_b_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListeningExposureEvidenceV1 {
    pub visible_a_auditions: u32,
    pub visible_b_auditions: u32,
}

/// The response the listener actually made while identities were still hidden.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlindHumanComparisonChoiceV1 {
    PreferVisibleA,
    PreferVisibleB,
    NoPreference,
}

/// The same response resolved through the revealed assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedHumanComparisonChoiceV1 {
    PreferSideA,
    PreferSideB,
    NoPreference,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HumanComparisonJudgmentV1 {
    pub blind_choice: BlindHumanComparisonChoiceV1,
    pub resolved_choice: ResolvedHumanComparisonChoiceV1,
    pub listening_exposure: ListeningExposureEvidenceV1,
    pub self_reported_confidence: Option<f32>,
    pub note: String,
}

/// Descriptive provenance for the externally supplied blind assignment.
///
/// This field is bound when the strict workspace is constructed, before reveal.
/// It still does not prove randomness quality, unbiased allocation,
/// preregistration, or suitability for inferential statistics.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BlindAssignmentProvenanceV1 {
    Unspecified,
    ExternallySupplied {
        method: String,
        reference: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonIdentityRelationV1 {
    AlternateRendition,
    SameScoreDifferentComposition,
    DifferentComposition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonEvidenceAuthorityV1 {
    HumanSelfReport,
}

/// Versioned restart-stable evidence for one exact artifact-bound blind trial.
///
/// URLs, titles, styles, renderer presentation metadata, and mutable transport
/// state are intentionally absent. The persisted subject identity is the exact
/// content identity whose rendition hash was also carried by the playback source
/// admitted into the trial.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtifactBoundBlindComparisonEvidenceV1 {
    pub schema_version: u32,
    pub trial_id: u64,
    pub recorded_at_unix_ms: u64,
    pub side_a_identity: ArtifactIdentity,
    pub side_b_identity: ArtifactIdentity,
    pub revealed_assignment: RevealedAssignmentEvidenceV1,
    pub anchor: MusicalAnchorEvidenceV1,
    pub exposure_policy: ExposurePolicyEvidenceV1,
    pub duration_evidence_at_judgment: DurationEvidenceV1,
    pub judgment: HumanComparisonJudgmentV1,
    pub identity_relation: ComparisonIdentityRelationV1,
    pub assignment_provenance: BlindAssignmentProvenanceV1,
    pub authority: ComparisonEvidenceAuthorityV1,
}

#[derive(Debug, PartialEq)]
pub enum ComparisonEvidenceRecordError {
    Workspace(ComparisonWorkspaceError),
    WorkspaceNotFinalized,
    SwitchStillInFlight,
    TrialMismatch,
    AnchorMismatch,
    DurationEvidenceMismatch,
    ExposureNotQualified,
    InvalidExposurePolicy,
    InvalidDurationEvidence,
    InvalidConfidence,
    MissingListeningExposure,
    NoteTooLarge,
    InvalidAssignment,
    IdentityDiffMismatch,
    MissingArtifactIdentity(ComparisonSide),
    MissingSourceRendition(ComparisonSide),
    InvalidArtifactIdentity(ComparisonSide),
    SubjectRenditionMismatch(ComparisonSide),
    InadmissibleIdentityRelation,
    InvalidAssignmentProvenance,
}

impl fmt::Display for ComparisonEvidenceRecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workspace(error) => error.fmt(f),
            Self::WorkspaceNotFinalized => write!(
                f,
                "artifact-bound evidence requires a revealed workspace with a final judgment"
            ),
            Self::SwitchStillInFlight => write!(
                f,
                "artifact-bound evidence cannot be recorded while a switch is in flight"
            ),
            Self::TrialMismatch => write!(f, "comparison evidence trial identifiers do not agree"),
            Self::AnchorMismatch => write!(
                f,
                "comparison judgment is not bound to the workspace's fixed musical anchor"
            ),
            Self::DurationEvidenceMismatch => write!(
                f,
                "duration evidence differs from the workspace's frozen judgment snapshot"
            ),
            Self::ExposureNotQualified => write!(
                f,
                "comparison evidence does not satisfy the recorded exposure policy"
            ),
            Self::InvalidExposurePolicy => write!(
                f,
                "comparison exposure policy must be finite and positive"
            ),
            Self::InvalidDurationEvidence => write!(
                f,
                "comparison duration evidence must be finite and non-negative"
            ),
            Self::InvalidConfidence => write!(
                f,
                "self-reported confidence must be finite and within [0, 1]"
            ),
            Self::MissingListeningExposure => write!(
                f,
                "artifact-bound evidence requires admitted auditions of both visible sides"
            ),
            Self::NoteTooLarge => write!(f, "comparison note exceeds the V1 UTF-8 byte budget"),
            Self::InvalidAssignment => write!(
                f,
                "revealed assignment must map visible A/B bijectively onto underlying A/B"
            ),
            Self::IdentityDiffMismatch => write!(
                f,
                "owned trial identity relation changed between construction and reveal"
            ),
            Self::MissingArtifactIdentity(side) => write!(
                f,
                "comparison side {side:?} lacks full ArtifactIdentity required for durable binding"
            ),
            Self::MissingSourceRendition(side) => write!(
                f,
                "comparison side {side:?} has full identity but no playback-source rendition commitment"
            ),
            Self::InvalidArtifactIdentity(side) => write!(
                f,
                "comparison side {side:?} contains a non-canonical SHA-256 artifact identity"
            ),
            Self::SubjectRenditionMismatch(side) => write!(
                f,
                "comparison side {side:?} playback rendition conflicts with its full artifact identity"
            ),
            Self::InadmissibleIdentityRelation => write!(
                f,
                "subjects are conflicting, insufficiently identified, or not two distinct artifacts"
            ),
            Self::InvalidAssignmentProvenance => write!(
                f,
                "blind-assignment provenance is empty or exceeds the V1 byte budget"
            ),
        }
    }
}

impl std::error::Error for ComparisonEvidenceRecordError {}

impl From<ComparisonWorkspaceError> for ComparisonEvidenceRecordError {
    fn from(value: ComparisonWorkspaceError) -> Self {
        Self::Workspace(value)
    }
}

/// Strict workspace for trials intended to become durable artifact-bound
/// evidence. The ordinary runtime remains available for legacy/unidentified
/// listening comparisons.
#[derive(Debug, PartialEq)]
pub struct ArtifactBoundBlindComparisonWorkspace {
    runtime: BlindComparisonWorkspaceRuntime,
    side_a_identity: ArtifactIdentity,
    side_b_identity: ArtifactIdentity,
    identity_diff_at_start: ComparisonIdentityDiff,
    assignment_provenance: BlindAssignmentProvenanceV1,
    reveal_report: Option<BlindComparisonRevealReport>,
}

impl ArtifactBoundBlindComparisonWorkspace {
    pub fn new(
        session: ComparisonSession,
        trial_id: BlindTrialId,
        swapped: bool,
        exposure_policy: BlindExposurePolicy,
        assignment_provenance: BlindAssignmentProvenanceV1,
    ) -> Result<Self, ComparisonEvidenceRecordError> {
        validate_assignment_provenance(&assignment_provenance)?;
        let side_a_identity = validated_subject_identity(&session.a, ComparisonSide::A)?.clone();
        let side_b_identity = validated_subject_identity(&session.b, ComparisonSide::B)?.clone();
        let identity_diff_at_start = compare_subject_identity(&session.a, &session.b);
        durable_identity_relation(&identity_diff_at_start)?;
        let runtime = BlindComparisonWorkspaceRuntime::new(
            session,
            trial_id,
            swapped,
            exposure_policy,
        )?;
        Ok(Self {
            runtime,
            side_a_identity,
            side_b_identity,
            identity_diff_at_start,
            assignment_provenance,
            reveal_report: None,
        })
    }

    pub fn status(&self) -> BlindComparisonWorkspaceStatus {
        self.runtime.status()
    }

    pub fn visible_subjects(&self) -> [BlindSubjectView; 2] {
        self.runtime.visible_subjects()
    }

    pub fn set_transport_intent(
        &mut self,
        intent: ComparisonTransportIntent,
    ) -> Result<(), ComparisonEvidenceRecordError> {
        Ok(self.runtime.set_transport_intent(intent)?)
    }

    pub fn begin_visible_switch(
        &mut self,
        visible_label: BlindLabel,
        resolved_target: ResolvedComparisonAnchor,
        playback: &mut PlaybackState,
    ) -> Result<Vec<PlaybackEffect>, ComparisonEvidenceRecordError> {
        Ok(self
            .runtime
            .begin_visible_switch(visible_label, resolved_target, playback)?)
    }

    pub fn handle_browser_event(
        &mut self,
        playback: &mut PlaybackState,
        event: ComparisonBrowserEvent,
    ) -> Result<ComparisonBrowserUpdate, ComparisonEvidenceRecordError> {
        Ok(self.runtime.handle_browser_event(playback, event)?)
    }

    pub fn record_judgment(
        &mut self,
        choice: BlindComparisonChoice,
        self_reported_confidence: Option<f32>,
        note: String,
    ) -> Result<&BlindComparisonJudgment, ComparisonEvidenceRecordError> {
        Ok(self
            .runtime
            .record_judgment(choice, self_reported_confidence, note)?)
    }

    /// Reveal once and retain only this owned runtime's report.
    pub fn reveal(
        &mut self,
    ) -> Result<&BlindComparisonRevealReport, ComparisonEvidenceRecordError> {
        let report = self.runtime.reveal()?;
        if report.identity_diff != self.identity_diff_at_start {
            return Err(ComparisonEvidenceRecordError::IdentityDiffMismatch);
        }
        self.reveal_report = Some(report);
        Ok(self
            .reveal_report
            .as_ref()
            .expect("reveal report was just stored"))
    }

    pub fn reveal_report(&self) -> Option<&BlindComparisonRevealReport> {
        self.reveal_report.as_ref()
    }

    /// Construct the serializable V1 evidence object from the identities and
    /// assignment provenance captured before blinding plus this runtime's own
    /// frozen post-judgment reveal report.
    pub fn build_durable_evidence(
        &self,
        recorded_at_unix_ms: u64,
    ) -> Result<ArtifactBoundBlindComparisonEvidenceV1, ComparisonEvidenceRecordError> {
        let report = self
            .reveal_report
            .as_ref()
            .ok_or(ComparisonEvidenceRecordError::WorkspaceNotFinalized)?;
        let status = self.runtime.status();
        validate_workspace_status(&status)?;
        validate_trial_and_anchor(&status, &report.judgment)?;
        validate_duration(&status, report)?;
        validate_judgment(&report.judgment)?;
        validate_assignment(report.assignment)?;
        if report.identity_diff != self.identity_diff_at_start {
            return Err(ComparisonEvidenceRecordError::IdentityDiffMismatch);
        }

        Ok(ArtifactBoundBlindComparisonEvidenceV1 {
            schema_version: ARTIFACT_BOUND_COMPARISON_EVIDENCE_SCHEMA_VERSION,
            trial_id: status.trial_id.0,
            recorded_at_unix_ms,
            side_a_identity: self.side_a_identity.clone(),
            side_b_identity: self.side_b_identity.clone(),
            revealed_assignment: assignment_evidence(report.assignment),
            anchor: MusicalAnchorEvidenceV1 {
                bar_index: status.anchor.bar_index,
                beat_offset: status.anchor.beat_offset,
            },
            exposure_policy: ExposurePolicyEvidenceV1 {
                minimum_seconds_per_side: status.exposure_policy.minimum_seconds_per_side,
                max_contiguous_step_seconds: status.exposure_policy.max_contiguous_step_seconds,
            },
            duration_evidence_at_judgment: DurationEvidenceV1 {
                visible_a_seconds: report.duration_evidence_at_judgment.visible_a_seconds,
                visible_b_seconds: report.duration_evidence_at_judgment.visible_b_seconds,
            },
            judgment: judgment_evidence(&report.judgment, report.assignment),
            identity_relation: durable_identity_relation(&self.identity_diff_at_start)?,
            assignment_provenance: self.assignment_provenance.clone(),
            authority: ComparisonEvidenceAuthorityV1::HumanSelfReport,
        })
    }
}

fn validate_workspace_status(
    status: &BlindComparisonWorkspaceStatus,
) -> Result<(), ComparisonEvidenceRecordError> {
    if !status.revealed || !status.judgment_recorded || !status.qualifies_for_judgment {
        return Err(ComparisonEvidenceRecordError::WorkspaceNotFinalized);
    }
    if status.switch_in_flight {
        return Err(ComparisonEvidenceRecordError::SwitchStillInFlight);
    }
    let policy = status.exposure_policy;
    if !policy.minimum_seconds_per_side.is_finite()
        || policy.minimum_seconds_per_side <= 0.0
        || !policy.max_contiguous_step_seconds.is_finite()
        || policy.max_contiguous_step_seconds <= 0.0
    {
        return Err(ComparisonEvidenceRecordError::InvalidExposurePolicy);
    }
    if !valid_duration(status.duration_evidence) {
        return Err(ComparisonEvidenceRecordError::InvalidDurationEvidence);
    }
    Ok(())
}

fn validate_trial_and_anchor(
    status: &BlindComparisonWorkspaceStatus,
    judgment: &ResolvedBlindComparisonJudgment,
) -> Result<(), ComparisonEvidenceRecordError> {
    if status.trial_id != judgment.trial_id {
        return Err(ComparisonEvidenceRecordError::TrialMismatch);
    }
    if judgment.anchor != Some(status.anchor)
        || !status.anchor.beat_offset.is_finite()
        || status.anchor.beat_offset < 0.0
    {
        return Err(ComparisonEvidenceRecordError::AnchorMismatch);
    }
    Ok(())
}

fn validate_duration(
    status: &BlindComparisonWorkspaceStatus,
    report: &BlindComparisonRevealReport,
) -> Result<(), ComparisonEvidenceRecordError> {
    let duration = report.duration_evidence_at_judgment;
    if !valid_duration(duration) {
        return Err(ComparisonEvidenceRecordError::InvalidDurationEvidence);
    }
    if status.duration_evidence != duration {
        return Err(ComparisonEvidenceRecordError::DurationEvidenceMismatch);
    }
    if !duration.qualifies(status.exposure_policy) {
        return Err(ComparisonEvidenceRecordError::ExposureNotQualified);
    }
    Ok(())
}

fn validate_judgment(
    judgment: &ResolvedBlindComparisonJudgment,
) -> Result<(), ComparisonEvidenceRecordError> {
    if !judgment.exposure.both_sides_auditioned() {
        return Err(ComparisonEvidenceRecordError::MissingListeningExposure);
    }
    if let Some(confidence) = judgment.self_reported_confidence
        && (!confidence.is_finite() || !(0.0..=1.0).contains(&confidence))
    {
        return Err(ComparisonEvidenceRecordError::InvalidConfidence);
    }
    if judgment.note.len() > MAX_COMPARISON_NOTE_UTF8_BYTES {
        return Err(ComparisonEvidenceRecordError::NoteTooLarge);
    }
    Ok(())
}

fn validate_assignment(
    assignment: RevealedBlindAssignment,
) -> Result<(), ComparisonEvidenceRecordError> {
    match (assignment.visible_a_side, assignment.visible_b_side) {
        (ComparisonSide::A, ComparisonSide::B) | (ComparisonSide::B, ComparisonSide::A) => Ok(()),
        _ => Err(ComparisonEvidenceRecordError::InvalidAssignment),
    }
}

fn validate_assignment_provenance(
    provenance: &BlindAssignmentProvenanceV1,
) -> Result<(), ComparisonEvidenceRecordError> {
    match provenance {
        BlindAssignmentProvenanceV1::Unspecified => Ok(()),
        BlindAssignmentProvenanceV1::ExternallySupplied { method, reference } => {
            if method.trim().is_empty() || method.len() > MAX_ASSIGNMENT_METHOD_UTF8_BYTES {
                return Err(ComparisonEvidenceRecordError::InvalidAssignmentProvenance);
            }
            if let Some(reference) = reference
                && (reference.trim().is_empty()
                    || reference.len() > MAX_ASSIGNMENT_REFERENCE_UTF8_BYTES)
            {
                return Err(ComparisonEvidenceRecordError::InvalidAssignmentProvenance);
            }
            Ok(())
        }
    }
}

fn validated_subject_identity(
    subject: &crate::comparison::ComparisonSubject,
    side: ComparisonSide,
) -> Result<&ArtifactIdentity, ComparisonEvidenceRecordError> {
    let identity = subject
        .artifact_identity
        .as_ref()
        .ok_or(ComparisonEvidenceRecordError::MissingArtifactIdentity(side))?;
    if !valid_sha256_hex(&identity.score_content.0)
        || !valid_sha256_hex(&identity.composition.0)
        || !valid_sha256_hex(&identity.rendition.0)
    {
        return Err(ComparisonEvidenceRecordError::InvalidArtifactIdentity(side));
    }
    let source_rendition = subject
        .source
        .rendition_id
        .as_ref()
        .ok_or(ComparisonEvidenceRecordError::MissingSourceRendition(side))?;
    if source_rendition != &identity.rendition {
        return Err(ComparisonEvidenceRecordError::SubjectRenditionMismatch(side));
    }
    Ok(identity)
}

fn durable_identity_relation(
    diff: &ComparisonIdentityDiff,
) -> Result<ComparisonIdentityRelationV1, ComparisonEvidenceRecordError> {
    match diff.relation {
        ComparisonIdentityRelation::AlternateRendition => {
            Ok(ComparisonIdentityRelationV1::AlternateRendition)
        }
        ComparisonIdentityRelation::SameScoreDifferentComposition => {
            Ok(ComparisonIdentityRelationV1::SameScoreDifferentComposition)
        }
        ComparisonIdentityRelation::DifferentComposition => {
            Ok(ComparisonIdentityRelationV1::DifferentComposition)
        }
        ComparisonIdentityRelation::SameArtifact
        | ComparisonIdentityRelation::EvidenceConflict
        | ComparisonIdentityRelation::InsufficientIdentity => {
            Err(ComparisonEvidenceRecordError::InadmissibleIdentityRelation)
        }
    }
}

fn assignment_evidence(assignment: RevealedBlindAssignment) -> RevealedAssignmentEvidenceV1 {
    RevealedAssignmentEvidenceV1 {
        visible_a_side: evidence_side(assignment.visible_a_side),
        visible_b_side: evidence_side(assignment.visible_b_side),
    }
}

fn judgment_evidence(
    judgment: &ResolvedBlindComparisonJudgment,
    assignment: RevealedBlindAssignment,
) -> HumanComparisonJudgmentV1 {
    let (blind_choice, resolved_choice) = match judgment.choice {
        ResolvedComparisonChoice::Prefer(side) => {
            let blind_choice = if assignment.visible_a_side == side {
                BlindHumanComparisonChoiceV1::PreferVisibleA
            } else {
                BlindHumanComparisonChoiceV1::PreferVisibleB
            };
            let resolved_choice = match side {
                ComparisonSide::A => ResolvedHumanComparisonChoiceV1::PreferSideA,
                ComparisonSide::B => ResolvedHumanComparisonChoiceV1::PreferSideB,
            };
            (blind_choice, resolved_choice)
        }
        ResolvedComparisonChoice::NoPreference => (
            BlindHumanComparisonChoiceV1::NoPreference,
            ResolvedHumanComparisonChoiceV1::NoPreference,
        ),
    };
    HumanComparisonJudgmentV1 {
        blind_choice,
        resolved_choice,
        listening_exposure: ListeningExposureEvidenceV1 {
            visible_a_auditions: judgment.exposure.visible_a_auditions,
            visible_b_auditions: judgment.exposure.visible_b_auditions,
        },
        self_reported_confidence: judgment.self_reported_confidence,
        note: judgment.note.clone(),
    }
}

const fn evidence_side(side: ComparisonSide) -> EvidenceSideV1 {
    match side {
        ComparisonSide::A => EvidenceSideV1::A,
        ComparisonSide::B => EvidenceSideV1::B,
    }
}

fn valid_duration(duration: BlindDurationEvidence) -> bool {
    duration.visible_a_seconds.is_finite()
        && duration.visible_a_seconds >= 0.0
        && duration.visible_b_seconds.is_finite()
        && duration.visible_b_seconds >= 0.0
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
    use crate::comparison::{ComparisonSubject, MusicalComparisonAnchor};
    use crate::playback::{PlaybackEvent, PlaybackPresentation, PlaybackSource, PlaybackSubjectKind};
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

    fn subject(
        identity: Option<ArtifactIdentity>,
        bind_source_rendition: bool,
        name: &str,
    ) -> ComparisonSubject {
        let rendition_id = bind_source_rendition
            .then(|| identity.as_ref().map(|identity| identity.rendition.clone()))
            .flatten();
        ComparisonSubject {
            source: PlaybackSource {
                rendition_id,
                audio_url: format!("/audio/{name}"),
                duration_hint_seconds: Some(30.0),
                advance_on_end: false,
                presentation: PlaybackPresentation {
                    kind: PlaybackSubjectKind::Review,
                    title: name.to_string(),
                    subtitle: None,
                    style_hint: None,
                },
            },
            artifact_identity: identity,
        }
    }

    fn session() -> ComparisonSession {
        let a = subject(Some(identity('a', 'b', 'c')), true, "a");
        let b = subject(Some(identity('d', 'e', 'f')), true, "b");
        let mut session = ComparisonSession::new(a, b).unwrap();
        session.set_musical_anchor(Some(MusicalComparisonAnchor::new(2, 0.5).unwrap()));
        session.set_transport_intent(ComparisonTransportIntent::Playing);
        session
    }

    fn resolved(seconds: f64) -> ResolvedComparisonAnchor {
        ResolvedComparisonAnchor {
            bar_index: 2,
            beat_offset: 0.5,
            absolute_beats: 8.5,
            seconds,
            meter_numerator: 4,
            meter_denominator: 4,
            tempo_bpm: 120.0,
        }
    }

    fn advance(
        workspace: &mut ArtifactBoundBlindComparisonWorkspace,
        playback: &mut PlaybackState,
        seconds: f64,
    ) {
        let epoch = playback.load_epoch;
        workspace
            .handle_browser_event(
                playback,
                ComparisonBrowserEvent::TimeAdvanced {
                    load_epoch: epoch,
                    seconds,
                },
            )
            .unwrap();
    }

    fn completed_workspace() -> ArtifactBoundBlindComparisonWorkspace {
        let session = session();
        let initial_source = session.active_subject().source.clone();
        let mut workspace = ArtifactBoundBlindComparisonWorkspace::new(
            session,
            BlindTrialId(202),
            true,
            BlindExposurePolicy::new(0.5, 0.6).unwrap(),
            BlindAssignmentProvenanceV1::ExternallySupplied {
                method: "precomputed balanced assignment".into(),
                reference: Some("study-plan-v1/trial-202".into()),
            },
        )
        .unwrap();

        let mut playback = PlaybackState::default();
        let _ = playback.reduce(PlaybackEvent::LoadRequested {
            source: initial_source,
            autoplay: false,
        });
        let epoch = playback.load_epoch;
        workspace
            .handle_browser_event(
                &mut playback,
                ComparisonBrowserEvent::MetadataLoaded {
                    load_epoch: epoch,
                    duration_seconds: 30.0,
                },
            )
            .unwrap();
        workspace
            .handle_browser_event(
                &mut playback,
                ComparisonBrowserEvent::PlaybackStarted { load_epoch: epoch },
            )
            .unwrap();
        advance(&mut workspace, &mut playback, 0.25);
        advance(&mut workspace, &mut playback, 0.5);

        let effects = workspace
            .begin_visible_switch(BlindLabel::A, resolved(5.0), &mut playback)
            .unwrap();
        assert_eq!(effects.len(), 1);
        let epoch = playback.load_epoch;
        let metadata = workspace
            .handle_browser_event(
                &mut playback,
                ComparisonBrowserEvent::MetadataLoaded {
                    load_epoch: epoch,
                    duration_seconds: 30.0,
                },
            )
            .unwrap();
        assert_eq!(metadata.effects.len(), 1);
        let seeked = workspace
            .handle_browser_event(
                &mut playback,
                ComparisonBrowserEvent::SeekCompleted {
                    load_epoch: epoch,
                    seconds: 5.0,
                },
            )
            .unwrap();
        assert_eq!(seeked.effects.len(), 1);
        workspace
            .handle_browser_event(
                &mut playback,
                ComparisonBrowserEvent::PlaybackStarted { load_epoch: epoch },
            )
            .unwrap();
        advance(&mut workspace, &mut playback, 5.25);
        advance(&mut workspace, &mut playback, 5.5);

        workspace
            .record_judgment(
                BlindComparisonChoice::Prefer(BlindLabel::A),
                Some(0.75),
                "preferred visible A under the fixed anchor".into(),
            )
            .unwrap();
        workspace.reveal().unwrap();
        workspace
    }

    #[test]
    fn strict_trial_requires_full_identity_before_blinding() {
        let a = subject(Some(identity('a', 'b', 'c')), true, "a");
        let b = subject(None, false, "legacy");
        let mut session = ComparisonSession::new(a, b).unwrap();
        session.set_musical_anchor(Some(MusicalComparisonAnchor::new(2, 0.5).unwrap()));
        assert!(matches!(
            ArtifactBoundBlindComparisonWorkspace::new(
                session,
                BlindTrialId(1),
                false,
                BlindExposurePolicy::new(0.5, 0.6).unwrap(),
                BlindAssignmentProvenanceV1::Unspecified,
            ),
            Err(ComparisonEvidenceRecordError::MissingArtifactIdentity(ComparisonSide::B))
        ));
    }

    #[test]
    fn full_identity_without_audible_rendition_binding_is_not_enough() {
        let a = subject(Some(identity('a', 'b', 'c')), false, "a");
        let b = subject(Some(identity('d', 'e', 'f')), true, "b");
        let mut session = ComparisonSession::new(a, b).unwrap();
        session.set_musical_anchor(Some(MusicalComparisonAnchor::new(2, 0.5).unwrap()));
        assert!(matches!(
            ArtifactBoundBlindComparisonWorkspace::new(
                session,
                BlindTrialId(1),
                false,
                BlindExposurePolicy::new(0.5, 0.6).unwrap(),
                BlindAssignmentProvenanceV1::Unspecified,
            ),
            Err(ComparisonEvidenceRecordError::MissingSourceRendition(ComparisonSide::A))
        ));
    }

    #[test]
    fn conflicting_identity_relation_cannot_start_strict_trial() {
        let a = subject(Some(identity('a', 'b', 'c')), true, "a");
        let b = subject(Some(identity('d', 'b', 'e')), true, "b");
        let mut session = ComparisonSession::new(a, b).unwrap();
        session.set_musical_anchor(Some(MusicalComparisonAnchor::new(2, 0.5).unwrap()));
        assert!(matches!(
            ArtifactBoundBlindComparisonWorkspace::new(
                session,
                BlindTrialId(1),
                false,
                BlindExposurePolicy::new(0.5, 0.6).unwrap(),
                BlindAssignmentProvenanceV1::Unspecified,
            ),
            Err(ComparisonEvidenceRecordError::InadmissibleIdentityRelation)
        ));
    }

    #[test]
    fn durable_record_preserves_prebound_assignment_and_blind_choice() {
        let workspace = completed_workspace();
        let record = workspace.build_durable_evidence(1_800_000_000_000).unwrap();

        assert_eq!(record.schema_version, ARTIFACT_BOUND_COMPARISON_EVIDENCE_SCHEMA_VERSION);
        assert_eq!(record.side_a_identity, identity('a', 'b', 'c'));
        assert_eq!(record.side_b_identity, identity('d', 'e', 'f'));
        assert_eq!(
            record.revealed_assignment,
            RevealedAssignmentEvidenceV1 {
                visible_a_side: EvidenceSideV1::B,
                visible_b_side: EvidenceSideV1::A,
            }
        );
        assert_eq!(record.judgment.blind_choice, BlindHumanComparisonChoiceV1::PreferVisibleA);
        assert_eq!(
            record.judgment.resolved_choice,
            ResolvedHumanComparisonChoiceV1::PreferSideB
        );
        assert!(matches!(
            record.assignment_provenance,
            BlindAssignmentProvenanceV1::ExternallySupplied { .. }
        ));
        assert_eq!(record.authority, ComparisonEvidenceAuthorityV1::HumanSelfReport);
        assert_eq!(record.identity_relation, ComparisonIdentityRelationV1::DifferentComposition);

        let bytes = serde_json::to_vec(&record).unwrap();
        let decoded: ArtifactBoundBlindComparisonEvidenceV1 = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded, record);
    }

    #[test]
    fn evidence_cannot_be_built_before_owned_runtime_reveal() {
        let workspace = ArtifactBoundBlindComparisonWorkspace::new(
            session(),
            BlindTrialId(202),
            true,
            BlindExposurePolicy::new(0.5, 0.6).unwrap(),
            BlindAssignmentProvenanceV1::Unspecified,
        )
        .unwrap();
        assert_eq!(
            workspace.build_durable_evidence(1),
            Err(ComparisonEvidenceRecordError::WorkspaceNotFinalized)
        );
    }

    #[test]
    fn assignment_provenance_must_be_valid_before_trial_start() {
        assert!(matches!(
            ArtifactBoundBlindComparisonWorkspace::new(
                session(),
                BlindTrialId(202),
                true,
                BlindExposurePolicy::new(0.5, 0.6).unwrap(),
                BlindAssignmentProvenanceV1::ExternallySupplied {
                    method: String::new(),
                    reference: None,
                },
            ),
            Err(ComparisonEvidenceRecordError::InvalidAssignmentProvenance)
        ));
    }
}
