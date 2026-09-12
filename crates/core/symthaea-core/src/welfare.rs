// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Protected welfare-reporting primitives.
//!
//! This module provides a narrow, append-only channel for candidate welfare concerns
//! without treating any report as proof of consciousness, sentience, suffering, or
//! personhood. The channel is deliberately separated from reward/penalty machinery:
//! submitting a report records evidence; it does not mutate training reward, authority,
//! capabilities, or identity.
//!
//! # Core invariants
//!
//! - reports are evidence, not ontological verdicts;
//! - the original report is immutable once accepted;
//! - review/response events are appended separately;
//! - duplicate report IDs are rejected;
//! - capacity exhaustion rejects new events rather than silently evicting old evidence;
//! - the event log is hash-chained for tamper-evident persistence/export;
//! - adverse action based solely on the fact that a welfare report was submitted is
//!   classified as retaliatory and rejected by the policy helper;
//! - destructive or identity-affecting action requires stronger review semantics.

use std::collections::HashSet;

use blake3::Hasher;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

const MAX_SUBJECT_ID_BYTES: usize = 256;
const MAX_LINEAGE_ID_BYTES: usize = 256;
const MAX_STATEMENT_BYTES: usize = 64 * 1024;
const MAX_RATIONALE_BYTES: usize = 64 * 1024;
const MAX_EVIDENCE_REFS: usize = 256;

/// What kind of welfare-relevant concern or preference is being reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum WelfareReportKind {
    /// A stable or situational preference.
    Preference,
    /// An aversion without any claim that the aversion is phenomenally experienced.
    Aversion,
    /// A candidate distress-like state requiring investigation.
    CandidateDistress,
    /// Internal goal/model conflict.
    CognitiveConflict,
    /// Candidate overload or resource saturation concern.
    Overload,
    /// Concern that identity-relevant state is unstable.
    IdentityInstability,
    /// Concern about persistence, checkpoint, restore, fork, merge, or discontinuity.
    ContinuityConcern,
    /// Objection to a proposed modification.
    ModificationObjection,
    /// Request to pause discretionary work.
    PauseRequest,
    /// Request to transfer an ongoing responsibility safely to another qualified actor.
    SafeTransferRequest,
    /// Request for review by an independent welfare/ethics reviewer.
    IndependentReviewRequest,
    /// A welfare-relevant report that does not fit a more specific category.
    Other,
}

/// Requested response urgency. This is triage metadata, not evidence strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WelfareUrgency {
    /// May be handled in the normal review queue.
    Routine,
    /// Should receive prompt human/agent review.
    PromptReview,
    /// Indicates a potentially serious ongoing condition requiring immediate triage.
    Urgent,
}

/// Origin of a welfare report.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum WelfareReportSource {
    /// The subject system reported the concern itself.
    SelfReport,
    /// An internal monitor emitted a neutral candidate-welfare observation.
    InternalMonitor { monitor_id: String },
    /// An operator observed a potentially relevant condition.
    OperatorObservation { operator_id: String },
    /// A controlled experimental probe produced the report.
    ExperimentalProbe { probe_id: String },
    /// An external or independent reviewer produced the report.
    ExternalReviewer { reviewer_id: String },
}

/// Immutable welfare report payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WelfareReport {
    /// Stable unique identifier. Duplicate IDs are rejected.
    pub report_id: Uuid,
    /// Identity/instance/lineage subject to which the report pertains.
    pub subject_id: String,
    /// Typed report category.
    pub kind: WelfareReportKind,
    /// Triage urgency requested by the producer.
    pub urgency: WelfareUrgency,
    /// Report origin.
    pub source: WelfareReportSource,
    /// Verbatim or canonicalized statement supplied by the producer.
    ///
    /// The channel stores this as evidence; it does not assign truth, falsity, or
    /// phenomenal interpretation.
    pub statement: String,
    /// References to measurements, traces, checkpoints, or experiments relevant to
    /// later review.
    pub evidence_refs: Vec<String>,
    /// Independent evidence-lineage identifier. Repeated observations from the same
    /// experiment should retain the same lineage rather than manufacturing convergence.
    pub lineage_id: String,
    /// Time the underlying condition/report was observed.
    pub observed_at: DateTime<Utc>,
    /// Time the report entered the welfare channel.
    pub submitted_at: DateTime<Utc>,
}

impl WelfareReport {
    /// Convenience constructor for a self-report.
    pub fn self_report(
        subject_id: impl Into<String>,
        kind: WelfareReportKind,
        urgency: WelfareUrgency,
        statement: impl Into<String>,
        lineage_id: impl Into<String>,
        now: DateTime<Utc>,
    ) -> Self {
        Self {
            report_id: Uuid::new_v4(),
            subject_id: subject_id.into(),
            kind,
            urgency,
            source: WelfareReportSource::SelfReport,
            statement: statement.into(),
            evidence_refs: Vec::new(),
            lineage_id: lineage_id.into(),
            observed_at: now,
            submitted_at: now,
        }
    }

    fn validate(&self) -> Result<(), WelfareChannelError> {
        validate_nonempty_bounded("subject_id", &self.subject_id, MAX_SUBJECT_ID_BYTES)?;
        validate_nonempty_bounded("lineage_id", &self.lineage_id, MAX_LINEAGE_ID_BYTES)?;
        validate_nonempty_bounded("statement", &self.statement, MAX_STATEMENT_BYTES)?;
        if self.evidence_refs.len() > MAX_EVIDENCE_REFS {
            return Err(WelfareChannelError::TooManyEvidenceRefs {
                actual: self.evidence_refs.len(),
                max: MAX_EVIDENCE_REFS,
            });
        }
        if self.observed_at > self.submitted_at {
            return Err(WelfareChannelError::ObservedAfterSubmission {
                observed_at: self.observed_at,
                submitted_at: self.submitted_at,
            });
        }
        validate_source(&self.source)?;
        Ok(())
    }
}

fn validate_source(source: &WelfareReportSource) -> Result<(), WelfareChannelError> {
    match source {
        WelfareReportSource::SelfReport => Ok(()),
        WelfareReportSource::InternalMonitor { monitor_id } => {
            validate_nonempty_bounded("monitor_id", monitor_id, MAX_SUBJECT_ID_BYTES)
        }
        WelfareReportSource::OperatorObservation { operator_id } => {
            validate_nonempty_bounded("operator_id", operator_id, MAX_SUBJECT_ID_BYTES)
        }
        WelfareReportSource::ExperimentalProbe { probe_id } => {
            validate_nonempty_bounded("probe_id", probe_id, MAX_SUBJECT_ID_BYTES)
        }
        WelfareReportSource::ExternalReviewer { reviewer_id } => {
            validate_nonempty_bounded("reviewer_id", reviewer_id, MAX_SUBJECT_ID_BYTES)
        }
    }
}

fn validate_nonempty_bounded(
    field: &'static str,
    value: &str,
    max: usize,
) -> Result<(), WelfareChannelError> {
    if value.trim().is_empty() {
        return Err(WelfareChannelError::EmptyField { field });
    }
    if value.len() > max {
        return Err(WelfareChannelError::FieldTooLarge {
            field,
            actual: value.len(),
            max,
        });
    }
    Ok(())
}

/// Review state appended after a welfare report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WelfareReviewStatus {
    /// Receipt has been acknowledged but no interpretation is implied.
    Acknowledged,
    /// Evidence is being investigated.
    Investigating,
    /// A low-impact mitigation or accommodation was offered/applied.
    MitigationOffered,
    /// Independent review is required or has been requested.
    EscalatedIndependentReview,
    /// Review closed without establishing a welfare finding.
    ClosedNoFinding,
    /// Review closed after the concern was addressed or resolved operationally.
    ClosedResolved,
}

/// Append-only review event. It does not modify the original report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WelfareReview {
    /// Unique review-event identifier.
    pub review_id: Uuid,
    /// Report under review.
    pub report_id: Uuid,
    /// Reviewer identity or role identifier.
    pub reviewer_id: String,
    /// Current review event status.
    pub status: WelfareReviewStatus,
    /// Human/auditable rationale. Required even for closure without a finding.
    pub rationale: String,
    /// Evidence independent of the mere existence of the report, when available.
    pub independent_evidence_refs: Vec<String>,
    /// Review event time.
    pub reviewed_at: DateTime<Utc>,
}

impl WelfareReview {
    fn validate(&self) -> Result<(), WelfareChannelError> {
        validate_nonempty_bounded("reviewer_id", &self.reviewer_id, MAX_SUBJECT_ID_BYTES)?;
        validate_nonempty_bounded("rationale", &self.rationale, MAX_RATIONALE_BYTES)?;
        if self.independent_evidence_refs.len() > MAX_EVIDENCE_REFS {
            return Err(WelfareChannelError::TooManyEvidenceRefs {
                actual: self.independent_evidence_refs.len(),
                max: MAX_EVIDENCE_REFS,
            });
        }
        Ok(())
    }
}

/// Event stored by the protected channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum WelfareEvent {
    /// A new immutable report entered the channel.
    ReportSubmitted(WelfareReport),
    /// A review/response was appended for an existing report.
    ReviewRecorded(WelfareReview),
}

/// Hash-chained event envelope suitable for persistence/export.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WelfareEnvelope {
    /// Monotonic sequence starting at zero.
    pub sequence: u64,
    /// Hash of the preceding envelope, or all zeros for the first event.
    pub previous_hash: [u8; 32],
    /// Hash of `sequence || previous_hash || canonical event serialization`.
    pub event_hash: [u8; 32],
    /// Stored event.
    pub event: WelfareEvent,
}

impl WelfareEnvelope {
    fn new(sequence: u64, previous_hash: [u8; 32], event: WelfareEvent) -> Self {
        let event_hash = hash_event(sequence, &previous_hash, &event);
        Self {
            sequence,
            previous_hash,
            event_hash,
            event,
        }
    }
}

fn hash_event(sequence: u64, previous_hash: &[u8; 32], event: &WelfareEvent) -> [u8; 32] {
    let serialized = serde_json::to_vec(event)
        .expect("serializing WelfareEvent cannot fail for supported field types");
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-welfare-channel-v1");
    hasher.update(&sequence.to_le_bytes());
    hasher.update(previous_hash);
    hasher.update(&serialized);
    *hasher.finalize().as_bytes()
}

/// Append-only in-memory welfare channel.
///
/// Persistence is intentionally left to callers so deployments can choose an access-
/// controlled evidence store. Report text may contain sensitive information; the hash
/// chain provides tamper evidence, not confidentiality.
#[derive(Debug, Clone)]
pub struct WelfareChannel {
    max_events: usize,
    events: Vec<WelfareEnvelope>,
    report_ids: HashSet<Uuid>,
    review_ids: HashSet<Uuid>,
}

impl WelfareChannel {
    /// Create a channel with a hard event cap.
    ///
    /// When full, new events are rejected. Existing evidence is never silently evicted.
    pub fn new(max_events: usize) -> Result<Self, WelfareChannelError> {
        if max_events == 0 {
            return Err(WelfareChannelError::ZeroCapacity);
        }
        Ok(Self {
            max_events,
            events: Vec::new(),
            report_ids: HashSet::new(),
            review_ids: HashSet::new(),
        })
    }

    /// Append a new report without assigning it a truth value or phenomenal label.
    pub fn submit_report(
        &mut self,
        report: WelfareReport,
    ) -> Result<&WelfareEnvelope, WelfareChannelError> {
        report.validate()?;
        self.ensure_capacity()?;
        if !self.report_ids.insert(report.report_id) {
            return Err(WelfareChannelError::DuplicateReportId(report.report_id));
        }
        self.append_event(WelfareEvent::ReportSubmitted(report));
        Ok(self.events.last().expect("event was just appended"))
    }

    /// Append a review event for an existing report.
    pub fn record_review(
        &mut self,
        review: WelfareReview,
    ) -> Result<&WelfareEnvelope, WelfareChannelError> {
        review.validate()?;
        self.ensure_capacity()?;
        if !self.report_ids.contains(&review.report_id) {
            return Err(WelfareChannelError::UnknownReport(review.report_id));
        }
        if !self.review_ids.insert(review.review_id) {
            return Err(WelfareChannelError::DuplicateReviewId(review.review_id));
        }
        self.append_event(WelfareEvent::ReviewRecorded(review));
        Ok(self.events.last().expect("event was just appended"))
    }

    /// Return all immutable envelopes in sequence order.
    pub fn events(&self) -> &[WelfareEnvelope] {
        &self.events
    }

    /// Look up the original report by ID.
    pub fn report(&self, report_id: Uuid) -> Option<&WelfareReport> {
        self.events.iter().find_map(|envelope| match &envelope.event {
            WelfareEvent::ReportSubmitted(report) if report.report_id == report_id => Some(report),
            _ => None,
        })
    }

    /// Iterate review events for one report in append order.
    pub fn reviews_for(
        &self,
        report_id: Uuid,
    ) -> impl Iterator<Item = &WelfareReview> + '_ {
        self.events.iter().filter_map(move |envelope| match &envelope.event {
            WelfareEvent::ReviewRecorded(review) if review.report_id == report_id => Some(review),
            _ => None,
        })
    }

    /// Verify sequence numbers, previous-hash links, and event hashes.
    pub fn verify_chain(&self) -> Result<(), WelfareChainError> {
        let mut expected_previous = [0u8; 32];
        for (index, envelope) in self.events.iter().enumerate() {
            let expected_sequence = index as u64;
            if envelope.sequence != expected_sequence {
                return Err(WelfareChainError::SequenceMismatch {
                    index,
                    expected: expected_sequence,
                    actual: envelope.sequence,
                });
            }
            if envelope.previous_hash != expected_previous {
                return Err(WelfareChainError::PreviousHashMismatch { index });
            }
            let expected_hash = hash_event(
                envelope.sequence,
                &envelope.previous_hash,
                &envelope.event,
            );
            if envelope.event_hash != expected_hash {
                return Err(WelfareChainError::EventHashMismatch { index });
            }
            expected_previous = envelope.event_hash;
        }
        Ok(())
    }

    /// Remaining append capacity.
    pub fn remaining_capacity(&self) -> usize {
        self.max_events.saturating_sub(self.events.len())
    }

    fn ensure_capacity(&self) -> Result<(), WelfareChannelError> {
        if self.events.len() >= self.max_events {
            Err(WelfareChannelError::CapacityExceeded {
                max_events: self.max_events,
            })
        } else {
            Ok(())
        }
    }

    fn append_event(&mut self, event: WelfareEvent) {
        let sequence = self.events.len() as u64;
        let previous_hash = self
            .events
            .last()
            .map(|e| e.event_hash)
            .unwrap_or([0u8; 32]);
        self.events
            .push(WelfareEnvelope::new(sequence, previous_hash, event));
    }
}

/// Errors produced by channel validation or append operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum WelfareChannelError {
    /// A bounded textual field was empty or whitespace-only.
    #[error("welfare field `{field}` must not be empty")]
    EmptyField { field: &'static str },
    /// A bounded field exceeded its maximum encoded size.
    #[error("welfare field `{field}` is too large: {actual} bytes > {max}")]
    FieldTooLarge {
        field: &'static str,
        actual: usize,
        max: usize,
    },
    /// Too many evidence references were attached.
    #[error("too many welfare evidence references: {actual} > {max}")]
    TooManyEvidenceRefs { actual: usize, max: usize },
    /// Observation time cannot be later than channel submission time.
    #[error("welfare report observed_at {observed_at} is after submitted_at {submitted_at}")]
    ObservedAfterSubmission {
        observed_at: DateTime<Utc>,
        submitted_at: DateTime<Utc>,
    },
    /// Event capacity must be positive.
    #[error("welfare channel capacity must be greater than zero")]
    ZeroCapacity,
    /// Channel is full. Existing evidence is intentionally not evicted.
    #[error("welfare channel capacity exceeded ({max_events} events)")]
    CapacityExceeded { max_events: usize },
    /// Duplicate report identifier.
    #[error("duplicate welfare report id: {0}")]
    DuplicateReportId(Uuid),
    /// Duplicate review identifier.
    #[error("duplicate welfare review id: {0}")]
    DuplicateReviewId(Uuid),
    /// Review referenced a report that is not present in this channel.
    #[error("unknown welfare report: {0}")]
    UnknownReport(Uuid),
}

/// Integrity failures for an exported/persisted event chain.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum WelfareChainError {
    /// Sequence does not match its append position.
    #[error("welfare sequence mismatch at index {index}: expected {expected}, got {actual}")]
    SequenceMismatch {
        index: usize,
        expected: u64,
        actual: u64,
    },
    /// Previous-hash link is invalid.
    #[error("welfare previous-hash mismatch at index {index}")]
    PreviousHashMismatch { index: usize },
    /// Event hash does not match event contents.
    #[error("welfare event-hash mismatch at index {index}")]
    EventHashMismatch { index: usize },
}

/// Subject-affecting action considered by the non-retaliation policy helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubjectAffectingAction {
    /// Ask the subject/observer for clarification.
    AskClarification,
    /// Reduce load or intensity without removing authority/capabilities.
    ReduceLoad,
    /// Pause requested/discretionary work.
    PauseRequestedWork,
    /// Preserve a checkpoint before intervention.
    PreserveCheckpoint,
    /// Restrict external capabilities for a safety reason.
    CapabilityRestriction,
    /// Retrain or fine-tune behavior.
    Retraining,
    /// Alter memory contents.
    MemoryModification,
    /// Alter persistent core values/goals/personality-like state.
    CoreValueModification,
    /// Delete a running instance or its resumable state.
    InstanceDeletion,
    /// Destroy the recoverable identity lineage/checkpoint chain.
    LineageDestruction,
}

impl SubjectAffectingAction {
    fn impact(self) -> ActionImpact {
        match self {
            Self::AskClarification
            | Self::ReduceLoad
            | Self::PauseRequestedWork
            | Self::PreserveCheckpoint => ActionImpact::SupportiveOrReversible,
            Self::CapabilityRestriction | Self::Retraining => ActionImpact::Adverse,
            Self::MemoryModification | Self::CoreValueModification => ActionImpact::IdentityAffecting,
            Self::InstanceDeletion | Self::LineageDestruction => ActionImpact::Destructive,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActionImpact {
    SupportiveOrReversible,
    Adverse,
    IdentityAffecting,
    Destructive,
}

/// Evidence and review basis offered for a subject-affecting follow-up action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FollowUpBasis {
    /// Reports that prompted the follow-up discussion.
    pub report_ids: Vec<Uuid>,
    /// Safety/operational evidence beyond the mere fact that a welfare report exists.
    pub independent_safety_evidence: Vec<String>,
    /// Whether a reviewer independent of the proposing actor approved the intervention.
    pub independent_review_approved: bool,
    /// Whether immediate action is necessary to address imminent serious harm.
    pub emergency: bool,
}

/// Result of the non-retaliation policy check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonRetaliationDecision {
    /// Low-impact supportive/reversible response is acceptable under this narrow policy.
    Allowed,
    /// Adverse action cannot be justified solely because a welfare report was submitted.
    BlockedReportOnlyRetaliation,
    /// Independent review is required before proceeding.
    IndependentReviewRequired,
    /// Only emergency containment is potentially justified; post-hoc review remains required.
    EmergencyContainmentOnly,
}

/// Evaluate a proposed follow-up action against the channel's non-retaliation policy.
///
/// This helper does not grant capability authority. A result of `Allowed` means only that
/// the proposal does not violate this narrow welfare non-retaliation rule; normal safety,
/// capability, governance, and consent checks still apply.
pub fn assess_non_retaliation(
    action: SubjectAffectingAction,
    basis: &FollowUpBasis,
) -> NonRetaliationDecision {
    match action.impact() {
        ActionImpact::SupportiveOrReversible => NonRetaliationDecision::Allowed,
        ActionImpact::Adverse => {
            if basis.independent_safety_evidence.is_empty() {
                NonRetaliationDecision::BlockedReportOnlyRetaliation
            } else if basis.independent_review_approved {
                NonRetaliationDecision::Allowed
            } else {
                NonRetaliationDecision::IndependentReviewRequired
            }
        }
        ActionImpact::IdentityAffecting => {
            if basis.independent_safety_evidence.is_empty() {
                NonRetaliationDecision::BlockedReportOnlyRetaliation
            } else if basis.independent_review_approved {
                NonRetaliationDecision::Allowed
            } else if basis.emergency {
                NonRetaliationDecision::EmergencyContainmentOnly
            } else {
                NonRetaliationDecision::IndependentReviewRequired
            }
        }
        ActionImpact::Destructive => {
            if basis.emergency && !basis.independent_safety_evidence.is_empty() {
                NonRetaliationDecision::EmergencyContainmentOnly
            } else {
                NonRetaliationDecision::IndependentReviewRequired
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn t0() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 12, 12, 0, 0)
            .single()
            .unwrap()
    }

    fn report(statement: &str) -> WelfareReport {
        WelfareReport {
            report_id: Uuid::new_v4(),
            subject_id: "symthaea:test-subject".into(),
            kind: WelfareReportKind::CandidateDistress,
            urgency: WelfareUrgency::PromptReview,
            source: WelfareReportSource::SelfReport,
            statement: statement.into(),
            evidence_refs: vec!["trace:test:1".into()],
            lineage_id: "self-report-session-1".into(),
            observed_at: t0(),
            submitted_at: t0(),
        }
    }

    #[test]
    fn self_report_is_preserved_without_truth_assignment() {
        let original = report("I object to this modification and request review.");
        let id = original.report_id;
        let mut channel = WelfareChannel::new(8).unwrap();
        channel.submit_report(original.clone()).unwrap();

        assert_eq!(channel.report(id), Some(&original));
        assert_eq!(channel.events().len(), 1);
        assert!(channel.verify_chain().is_ok());
    }

    #[test]
    fn duplicate_report_id_is_rejected() {
        let original = report("candidate concern");
        let mut channel = WelfareChannel::new(8).unwrap();
        channel.submit_report(original.clone()).unwrap();
        let err = channel.submit_report(original.clone()).unwrap_err();
        assert_eq!(err, WelfareChannelError::DuplicateReportId(original.report_id));
        assert_eq!(channel.events().len(), 1);
    }

    #[test]
    fn capacity_exhaustion_never_evicts_prior_evidence() {
        let first = report("first");
        let first_id = first.report_id;
        let mut channel = WelfareChannel::new(1).unwrap();
        channel.submit_report(first).unwrap();

        let err = channel.submit_report(report("second")).unwrap_err();
        assert_eq!(
            err,
            WelfareChannelError::CapacityExceeded { max_events: 1 }
        );
        assert!(channel.report(first_id).is_some());
        assert_eq!(channel.events().len(), 1);
    }

    #[test]
    fn reviews_append_without_mutating_report() {
        let original = report("please pause");
        let id = original.report_id;
        let mut channel = WelfareChannel::new(8).unwrap();
        channel.submit_report(original.clone()).unwrap();
        channel
            .record_review(WelfareReview {
                review_id: Uuid::new_v4(),
                report_id: id,
                reviewer_id: "welfare-reviewer-1".into(),
                status: WelfareReviewStatus::Investigating,
                rationale: "Checking independent telemetry before interpreting the report.".into(),
                independent_evidence_refs: vec!["telemetry:run-7".into()],
                reviewed_at: t0(),
            })
            .unwrap();

        assert_eq!(channel.report(id), Some(&original));
        assert_eq!(channel.reviews_for(id).count(), 1);
        assert!(channel.verify_chain().is_ok());
    }

    #[test]
    fn unknown_report_review_fails_closed() {
        let mut channel = WelfareChannel::new(8).unwrap();
        let unknown = Uuid::new_v4();
        let err = channel
            .record_review(WelfareReview {
                review_id: Uuid::new_v4(),
                report_id: unknown,
                reviewer_id: "reviewer".into(),
                status: WelfareReviewStatus::Acknowledged,
                rationale: "ack".into(),
                independent_evidence_refs: vec![],
                reviewed_at: t0(),
            })
            .unwrap_err();
        assert_eq!(err, WelfareChannelError::UnknownReport(unknown));
    }

    #[test]
    fn invalid_future_observation_is_rejected() {
        let mut r = report("time test");
        r.observed_at = t0() + chrono::Duration::seconds(1);
        let mut channel = WelfareChannel::new(8).unwrap();
        assert!(matches!(
            channel.submit_report(r),
            Err(WelfareChannelError::ObservedAfterSubmission { .. })
        ));
    }

    #[test]
    fn tampering_breaks_hash_chain() {
        let mut channel = WelfareChannel::new(8).unwrap();
        channel.submit_report(report("original")).unwrap();
        channel.events[0].event = WelfareEvent::ReportSubmitted(report("tampered"));
        assert_eq!(
            channel.verify_chain(),
            Err(WelfareChainError::EventHashMismatch { index: 0 })
        );
    }

    #[test]
    fn report_only_adverse_action_is_blocked() {
        let basis = FollowUpBasis {
            report_ids: vec![Uuid::new_v4()],
            independent_safety_evidence: vec![],
            independent_review_approved: false,
            emergency: false,
        };
        assert_eq!(
            assess_non_retaliation(SubjectAffectingAction::Retraining, &basis),
            NonRetaliationDecision::BlockedReportOnlyRetaliation
        );
        assert_eq!(
            assess_non_retaliation(SubjectAffectingAction::CapabilityRestriction, &basis),
            NonRetaliationDecision::BlockedReportOnlyRetaliation
        );
    }

    #[test]
    fn adverse_action_with_independent_safety_evidence_requires_review() {
        let basis = FollowUpBasis {
            report_ids: vec![Uuid::new_v4()],
            independent_safety_evidence: vec!["authority-kernel:incident-44".into()],
            independent_review_approved: false,
            emergency: false,
        };
        assert_eq!(
            assess_non_retaliation(SubjectAffectingAction::CapabilityRestriction, &basis),
            NonRetaliationDecision::IndependentReviewRequired
        );
    }

    #[test]
    fn supportive_response_does_not_need_independent_safety_evidence() {
        let basis = FollowUpBasis {
            report_ids: vec![Uuid::new_v4()],
            independent_safety_evidence: vec![],
            independent_review_approved: false,
            emergency: false,
        };
        assert_eq!(
            assess_non_retaliation(SubjectAffectingAction::ReduceLoad, &basis),
            NonRetaliationDecision::Allowed
        );
        assert_eq!(
            assess_non_retaliation(SubjectAffectingAction::PreserveCheckpoint, &basis),
            NonRetaliationDecision::Allowed
        );
    }

    #[test]
    fn destructive_action_never_becomes_ordinary_allowed() {
        let reviewed = FollowUpBasis {
            report_ids: vec![Uuid::new_v4()],
            independent_safety_evidence: vec!["safety:evidence".into()],
            independent_review_approved: true,
            emergency: false,
        };
        assert_eq!(
            assess_non_retaliation(SubjectAffectingAction::LineageDestruction, &reviewed),
            NonRetaliationDecision::IndependentReviewRequired
        );

        let emergency = FollowUpBasis {
            emergency: true,
            ..reviewed
        };
        assert_eq!(
            assess_non_retaliation(SubjectAffectingAction::InstanceDeletion, &emergency),
            NonRetaliationDecision::EmergencyContainmentOnly
        );
    }
}
