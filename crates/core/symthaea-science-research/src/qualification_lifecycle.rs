// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Append-only authenticated lifecycle for already-qualified scientific claims.
//!
//! Qualification history is immutable. Lifecycle events never rewrite the
//! `QualifiedScientificClaim`; they record later institutional state changes.
//! Each event is timed by authenticated `TrustedTime` and authorized by the
//! root's `QualificationLifecycle` role. V1 deliberately requires one exact root
//! authority throughout a tracker lineage; cross-root continuation needs an
//! explicit root-transition lineage rather than silently trusting a new root.
//!
//! This lifecycle does not retroactively prove that an older qualification was
//! itself minted under the newer delegated-root role architecture. That migration
//! remains a separate authority step.

use serde::Serialize;
use symthaea_trust_core::{
    AuthorizedTrustRoleAttestation, Sha256Digest as TrustSha256Digest, TrustRole,
    TrustedTime,
};

use crate::{QualifiedScientificClaim, ResearchId, Sha256Digest};

pub const QUALIFICATION_LIFECYCLE_SCHEMA: &str = "symthaea.scientific-qualification-lifecycle.v1";
const LIFECYCLE_STATEMENT_DOMAIN: &str =
    "symthaea.scientific-qualification-lifecycle-statement.identity.v1";
const AUTHENTICATED_LIFECYCLE_EVENT_DOMAIN: &str =
    "symthaea.authenticated-scientific-qualification-lifecycle-event.identity.v1";
pub const MAX_LIFECYCLE_TRIGGER_EVIDENCE: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum QualificationLifecycleEventKind {
    Activate,
    MarkUnderReview,
    Suspend,
    Revoke,
    Supersede {
        successor_qualification_sha256: Sha256Digest,
    },
    /// Reaffirms the existing qualification as Active after review. It does not
    /// create a new scientific qualification or change the historical claim.
    Renew,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum QualificationLifecycleState {
    Active,
    UnderReview,
    Suspended,
    Revoked,
    Superseded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationLifecycleStatementError {
    SequenceZero,
    GenesisMissingActivate,
    GenesisHasPredecessor,
    NonGenesisActivate,
    NonGenesisMissingPredecessor,
    NonGenesisMissingTriggerEvidence,
    TooManyTriggerEvidence,
    DuplicateTriggerEvidence,
    SelfSupersession,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QualificationLifecycleStatement {
    schema_version: String,
    qualification_sha256: Sha256Digest,
    claim_id: ResearchId,
    subject_sha256: Sha256Digest,
    sequence: u64,
    previous_event_sha256: Option<Sha256Digest>,
    event_kind: QualificationLifecycleEventKind,
    trusted_time_authority_sha256: TrustSha256Digest,
    event_earliest_unix_s: u64,
    event_latest_unix_s: u64,
    rationale_sha256: Sha256Digest,
    triggering_evidence_sha256s: Vec<Sha256Digest>,
    statement_sha256: Sha256Digest,
}

impl QualificationLifecycleStatement {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qualified: &QualifiedScientificClaim,
        sequence: u64,
        previous_event_sha256: Option<Sha256Digest>,
        event_kind: QualificationLifecycleEventKind,
        trusted_time: &TrustedTime,
        rationale_sha256: Sha256Digest,
        mut triggering_evidence_sha256s: Vec<Sha256Digest>,
    ) -> Result<Self, Vec<QualificationLifecycleStatementError>> {
        let mut issues = Vec::new();
        if sequence == 0 {
            issues.push(QualificationLifecycleStatementError::SequenceZero);
        }
        if sequence == 1 {
            if !matches!(&event_kind, QualificationLifecycleEventKind::Activate) {
                issues.push(QualificationLifecycleStatementError::GenesisMissingActivate);
            }
            if previous_event_sha256.is_some() {
                issues.push(QualificationLifecycleStatementError::GenesisHasPredecessor);
            }
        } else {
            if matches!(&event_kind, QualificationLifecycleEventKind::Activate) {
                issues.push(QualificationLifecycleStatementError::NonGenesisActivate);
            }
            if previous_event_sha256.is_none() {
                issues.push(QualificationLifecycleStatementError::NonGenesisMissingPredecessor);
            }
            if triggering_evidence_sha256s.is_empty() {
                issues.push(QualificationLifecycleStatementError::NonGenesisMissingTriggerEvidence);
            }
        }
        if triggering_evidence_sha256s.len() > MAX_LIFECYCLE_TRIGGER_EVIDENCE {
            issues.push(QualificationLifecycleStatementError::TooManyTriggerEvidence);
        }
        triggering_evidence_sha256s.sort();
        if triggering_evidence_sha256s.windows(2).any(|pair| pair[0] == pair[1]) {
            issues.push(QualificationLifecycleStatementError::DuplicateTriggerEvidence);
        }
        if matches!(
            &event_kind,
            QualificationLifecycleEventKind::Supersede {
                successor_qualification_sha256
            } if successor_qualification_sha256 == qualified.qualification_sha256()
        ) {
            issues.push(QualificationLifecycleStatementError::SelfSupersession);
        }
        if !issues.is_empty() {
            return Err(issues);
        }

        let (earliest, latest) = trusted_time.consensus_interval();
        let statement_sha256 = lifecycle_statement_digest(
            qualified.qualification_sha256(),
            qualified.claim_id(),
            qualified.subject_sha256(),
            sequence,
            previous_event_sha256.as_ref(),
            &event_kind,
            trusted_time.authority_sha256(),
            earliest,
            latest,
            &rationale_sha256,
            &triggering_evidence_sha256s,
        );
        Ok(Self {
            schema_version: QUALIFICATION_LIFECYCLE_SCHEMA.into(),
            qualification_sha256: qualified.qualification_sha256().clone(),
            claim_id: qualified.claim_id().clone(),
            subject_sha256: qualified.subject_sha256().clone(),
            sequence,
            previous_event_sha256,
            event_kind,
            trusted_time_authority_sha256: trusted_time.authority_sha256().clone(),
            event_earliest_unix_s: earliest,
            event_latest_unix_s: latest,
            rationale_sha256,
            triggering_evidence_sha256s,
            statement_sha256,
        })
    }

    pub fn qualification_sha256(&self) -> &Sha256Digest { &self.qualification_sha256 }
    pub fn claim_id(&self) -> &ResearchId { &self.claim_id }
    pub fn subject_sha256(&self) -> &Sha256Digest { &self.subject_sha256 }
    pub fn sequence(&self) -> u64 { self.sequence }
    pub fn previous_event_sha256(&self) -> Option<&Sha256Digest> {
        self.previous_event_sha256.as_ref()
    }
    pub fn event_kind(&self) -> &QualificationLifecycleEventKind { &self.event_kind }
    pub fn trusted_time_authority_sha256(&self) -> &TrustSha256Digest {
        &self.trusted_time_authority_sha256
    }
    pub fn event_interval(&self) -> (u64, u64) {
        (self.event_earliest_unix_s, self.event_latest_unix_s)
    }
    pub fn rationale_sha256(&self) -> &Sha256Digest { &self.rationale_sha256 }
    pub fn triggering_evidence_sha256s(&self) -> &[Sha256Digest] {
        &self.triggering_evidence_sha256s
    }
    pub fn statement_sha256(&self) -> &Sha256Digest { &self.statement_sha256 }
    pub const fn lifecycle_authority_established(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationLifecycleAuthenticationError {
    QualificationMismatch,
    ClaimMismatch,
    SubjectMismatch,
    TrustedTimeMismatch,
    WrongRole,
    RootAuthorityMismatch,
    AuthoritySubjectMismatch,
    AuthorityPayloadMismatch,
    AuthorityContextMismatch,
}

/// One lifecycle event authenticated by the exact root-authorized lifecycle role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuthenticatedQualificationLifecycleEvent {
    statement: QualificationLifecycleStatement,
    lifecycle_role_authority_sha256: TrustSha256Digest,
    root_authority_sha256: TrustSha256Digest,
    trust_snapshot_authority_sha256: TrustSha256Digest,
    event_sha256: Sha256Digest,
}

impl AuthenticatedQualificationLifecycleEvent {
    pub fn statement(&self) -> &QualificationLifecycleStatement { &self.statement }
    pub fn lifecycle_role_authority_sha256(&self) -> &TrustSha256Digest {
        &self.lifecycle_role_authority_sha256
    }
    pub fn root_authority_sha256(&self) -> &TrustSha256Digest { &self.root_authority_sha256 }
    pub fn trust_snapshot_authority_sha256(&self) -> &TrustSha256Digest {
        &self.trust_snapshot_authority_sha256
    }
    pub fn event_sha256(&self) -> &Sha256Digest { &self.event_sha256 }
    pub const fn lifecycle_authority_established(&self) -> bool { true }
    pub const fn current_scientific_validity_established(&self) -> bool { false }
}

pub fn authenticate_qualification_lifecycle_event(
    statement: QualificationLifecycleStatement,
    qualified: &QualifiedScientificClaim,
    trusted_time: &TrustedTime,
    authority: &AuthorizedTrustRoleAttestation,
) -> Result<AuthenticatedQualificationLifecycleEvent, QualificationLifecycleAuthenticationError> {
    if statement.qualification_sha256() != qualified.qualification_sha256() {
        return Err(QualificationLifecycleAuthenticationError::QualificationMismatch);
    }
    if statement.claim_id() != qualified.claim_id() {
        return Err(QualificationLifecycleAuthenticationError::ClaimMismatch);
    }
    if statement.subject_sha256() != qualified.subject_sha256() {
        return Err(QualificationLifecycleAuthenticationError::SubjectMismatch);
    }
    if statement.trusted_time_authority_sha256() != trusted_time.authority_sha256()
        || statement.event_interval() != trusted_time.consensus_interval()
    {
        return Err(QualificationLifecycleAuthenticationError::TrustedTimeMismatch);
    }
    if authority.role() != TrustRole::QualificationLifecycle {
        return Err(QualificationLifecycleAuthenticationError::WrongRole);
    }
    if authority.root_authority_sha256() != trusted_time.root_authority_sha256() {
        return Err(QualificationLifecycleAuthenticationError::RootAuthorityMismatch);
    }
    if authority.subject_sha256() != &bridge_digest(qualified.qualification_sha256()) {
        return Err(QualificationLifecycleAuthenticationError::AuthoritySubjectMismatch);
    }
    if authority.payload_sha256() != &bridge_digest(statement.statement_sha256()) {
        return Err(QualificationLifecycleAuthenticationError::AuthorityPayloadMismatch);
    }
    let expected_context = statement.previous_event_sha256().map(bridge_digest);
    if authority.context_sha256() != expected_context.as_ref() {
        return Err(QualificationLifecycleAuthenticationError::AuthorityContextMismatch);
    }

    let event_sha256 = authenticated_event_digest(
        statement.statement_sha256(),
        authority.authority_sha256(),
        authority.root_authority_sha256(),
        authority.trust_snapshot_authority_sha256(),
    );
    Ok(AuthenticatedQualificationLifecycleEvent {
        statement,
        lifecycle_role_authority_sha256: authority.authority_sha256().clone(),
        root_authority_sha256: authority.root_authority_sha256().clone(),
        trust_snapshot_authority_sha256: authority.trust_snapshot_authority_sha256().clone(),
        event_sha256,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationLifecycleTrackingError {
    QualificationChanged,
    RootAuthorityChangedUnsupported,
    InvalidGenesisSequence,
    InvalidGenesisEvent,
    UnexpectedPredecessor,
    SequenceNotNext { latest: u64, proposed: u64 },
    EventTimeNotDefinitelyAfterPrevious,
    InvalidTransition {
        from: QualificationLifecycleState,
        event: &'static str,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct QualificationLifecycleTracker {
    qualification_sha256: Option<Sha256Digest>,
    root_authority_sha256: Option<TrustSha256Digest>,
    latest_sequence: Option<u64>,
    latest_event_sha256: Option<Sha256Digest>,
    latest_state: Option<QualificationLifecycleState>,
    latest_time_earliest_unix_s: Option<u64>,
    latest_time_latest_unix_s: Option<u64>,
}

impl QualificationLifecycleTracker {
    pub fn accept(
        &mut self,
        event: &AuthenticatedQualificationLifecycleEvent,
    ) -> Result<(), QualificationLifecycleTrackingError> {
        let statement = event.statement();
        let (earliest, latest) = statement.event_interval();
        let Some(previous_sequence) = self.latest_sequence else {
            if statement.sequence() != 1 || statement.previous_event_sha256().is_some() {
                return Err(QualificationLifecycleTrackingError::InvalidGenesisSequence);
            }
            if !matches!(statement.event_kind(), QualificationLifecycleEventKind::Activate) {
                return Err(QualificationLifecycleTrackingError::InvalidGenesisEvent);
            }
            self.qualification_sha256 = Some(statement.qualification_sha256().clone());
            self.root_authority_sha256 = Some(event.root_authority_sha256().clone());
            self.latest_sequence = Some(1);
            self.latest_event_sha256 = Some(event.event_sha256().clone());
            self.latest_state = Some(QualificationLifecycleState::Active);
            self.latest_time_earliest_unix_s = Some(earliest);
            self.latest_time_latest_unix_s = Some(latest);
            return Ok(());
        };

        if self.qualification_sha256.as_ref() != Some(statement.qualification_sha256()) {
            return Err(QualificationLifecycleTrackingError::QualificationChanged);
        }
        if self.root_authority_sha256.as_ref() != Some(event.root_authority_sha256()) {
            return Err(QualificationLifecycleTrackingError::RootAuthorityChangedUnsupported);
        }
        if statement.sequence() != previous_sequence.saturating_add(1) {
            return Err(QualificationLifecycleTrackingError::SequenceNotNext {
                latest: previous_sequence,
                proposed: statement.sequence(),
            });
        }
        if statement.previous_event_sha256() != self.latest_event_sha256.as_ref() {
            return Err(QualificationLifecycleTrackingError::UnexpectedPredecessor);
        }
        let previous_latest = self
            .latest_time_latest_unix_s
            .expect("accepted lifecycle state has a time interval");
        if earliest < previous_latest {
            return Err(QualificationLifecycleTrackingError::EventTimeNotDefinitelyAfterPrevious);
        }

        let current = self.latest_state.expect("accepted lifecycle state has status");
        let next = transition(current, statement.event_kind()).ok_or(
            QualificationLifecycleTrackingError::InvalidTransition {
                from: current,
                event: lifecycle_event_tag(statement.event_kind()),
            },
        )?;
        self.latest_sequence = Some(statement.sequence());
        self.latest_event_sha256 = Some(event.event_sha256().clone());
        self.latest_state = Some(next);
        self.latest_time_earliest_unix_s = Some(earliest);
        self.latest_time_latest_unix_s = Some(latest);
        Ok(())
    }

    pub fn latest_state(&self) -> Option<QualificationLifecycleState> { self.latest_state }
    pub fn latest_sequence(&self) -> Option<u64> { self.latest_sequence }
    pub fn latest_event_sha256(&self) -> Option<&Sha256Digest> {
        self.latest_event_sha256.as_ref()
    }
    pub fn qualification_sha256(&self) -> Option<&Sha256Digest> {
        self.qualification_sha256.as_ref()
    }
    pub fn root_authority_sha256(&self) -> Option<&TrustSha256Digest> {
        self.root_authority_sha256.as_ref()
    }
    pub fn latest_state_is_active(&self) -> bool {
        self.latest_state == Some(QualificationLifecycleState::Active)
    }
    /// Tracker-local latest state is not yet proof of global currentness. Later
    /// layers must add transparent publication, anti-rollback monitor evidence,
    /// trust-currentness, and scientific evidence freshness.
    pub const fn current_scientific_validity_established(&self) -> bool { false }
}

fn transition(
    current: QualificationLifecycleState,
    event: &QualificationLifecycleEventKind,
) -> Option<QualificationLifecycleState> {
    use QualificationLifecycleEventKind as Event;
    use QualificationLifecycleState as State;
    match (current, event) {
        (State::Active, Event::MarkUnderReview) => Some(State::UnderReview),
        (State::Active, Event::Suspend) => Some(State::Suspended),
        (State::Active, Event::Revoke) => Some(State::Revoked),
        (State::Active, Event::Supersede { .. }) => Some(State::Superseded),
        (State::Active, Event::Renew) => Some(State::Active),
        (State::UnderReview, Event::Renew) => Some(State::Active),
        (State::UnderReview, Event::Suspend) => Some(State::Suspended),
        (State::UnderReview, Event::Revoke) => Some(State::Revoked),
        (State::UnderReview, Event::Supersede { .. }) => Some(State::Superseded),
        (State::Suspended, Event::MarkUnderReview) => Some(State::UnderReview),
        (State::Suspended, Event::Renew) => Some(State::Active),
        (State::Suspended, Event::Revoke) => Some(State::Revoked),
        (State::Suspended, Event::Supersede { .. }) => Some(State::Superseded),
        _ => None,
    }
}

fn bridge_digest(value: &Sha256Digest) -> TrustSha256Digest {
    TrustSha256Digest::parse(value.as_str())
        .expect("science-research SHA-256 identities are already validated")
}

#[allow(clippy::too_many_arguments)]
fn lifecycle_statement_digest(
    qualification_sha256: &Sha256Digest,
    claim_id: &ResearchId,
    subject_sha256: &Sha256Digest,
    sequence: u64,
    previous_event_sha256: Option<&Sha256Digest>,
    event_kind: &QualificationLifecycleEventKind,
    trusted_time_authority_sha256: &TrustSha256Digest,
    earliest: u64,
    latest: u64,
    rationale_sha256: &Sha256Digest,
    triggering_evidence_sha256s: &[Sha256Digest],
) -> Sha256Digest {
    let mut bytes = Vec::new();
    append_frame(&mut bytes, LIFECYCLE_STATEMENT_DOMAIN);
    append_frame(&mut bytes, QUALIFICATION_LIFECYCLE_SCHEMA);
    append_frame(&mut bytes, qualification_sha256.as_str());
    append_frame(&mut bytes, claim_id.as_str());
    append_frame(&mut bytes, subject_sha256.as_str());
    append_frame(&mut bytes, &sequence.to_string());
    append_optional_digest(&mut bytes, previous_event_sha256);
    digest_event_kind(&mut bytes, event_kind);
    append_frame(&mut bytes, trusted_time_authority_sha256.as_str());
    append_frame(&mut bytes, &earliest.to_string());
    append_frame(&mut bytes, &latest.to_string());
    append_frame(&mut bytes, rationale_sha256.as_str());
    for evidence in triggering_evidence_sha256s {
        append_frame(&mut bytes, "trigger-evidence");
        append_frame(&mut bytes, evidence.as_str());
    }
    Sha256Digest::of_bytes(&bytes)
}

fn authenticated_event_digest(
    statement_sha256: &Sha256Digest,
    lifecycle_role_authority_sha256: &TrustSha256Digest,
    root_authority_sha256: &TrustSha256Digest,
    trust_snapshot_authority_sha256: &TrustSha256Digest,
) -> Sha256Digest {
    let mut bytes = Vec::new();
    append_frame(&mut bytes, AUTHENTICATED_LIFECYCLE_EVENT_DOMAIN);
    append_frame(&mut bytes, statement_sha256.as_str());
    append_frame(&mut bytes, lifecycle_role_authority_sha256.as_str());
    append_frame(&mut bytes, root_authority_sha256.as_str());
    append_frame(&mut bytes, trust_snapshot_authority_sha256.as_str());
    Sha256Digest::of_bytes(&bytes)
}

fn digest_event_kind(bytes: &mut Vec<u8>, event: &QualificationLifecycleEventKind) {
    append_frame(bytes, lifecycle_event_tag(event));
    if let QualificationLifecycleEventKind::Supersede {
        successor_qualification_sha256,
    } = event
    {
        append_frame(bytes, successor_qualification_sha256.as_str());
    }
}

const fn lifecycle_event_tag(event: &QualificationLifecycleEventKind) -> &'static str {
    match event {
        QualificationLifecycleEventKind::Activate => "activate",
        QualificationLifecycleEventKind::MarkUnderReview => "under-review",
        QualificationLifecycleEventKind::Suspend => "suspend",
        QualificationLifecycleEventKind::Revoke => "revoke",
        QualificationLifecycleEventKind::Supersede { .. } => "supersede",
        QualificationLifecycleEventKind::Renew => "renew",
    }
}

fn append_optional_digest(bytes: &mut Vec<u8>, value: Option<&Sha256Digest>) {
    match value {
        Some(value) => {
            append_frame(bytes, "some");
            append_frame(bytes, value.as_str());
        }
        None => append_frame(bytes, "none"),
    }
}

fn append_frame(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revoked_and_superseded_states_are_terminal() {
        assert!(transition(
            QualificationLifecycleState::Revoked,
            &QualificationLifecycleEventKind::Renew
        )
        .is_none());
        assert!(transition(
            QualificationLifecycleState::Superseded,
            &QualificationLifecycleEventKind::Renew
        )
        .is_none());
    }

    #[test]
    fn review_and_suspension_can_return_to_active_only_via_renewal() {
        assert_eq!(
            transition(
                QualificationLifecycleState::UnderReview,
                &QualificationLifecycleEventKind::Renew
            ),
            Some(QualificationLifecycleState::Active)
        );
        assert_eq!(
            transition(
                QualificationLifecycleState::Suspended,
                &QualificationLifecycleEventKind::Renew
            ),
            Some(QualificationLifecycleState::Active)
        );
    }
}
