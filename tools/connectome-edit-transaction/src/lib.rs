#![forbid(unsafe_code)]

//! Deterministic audit ledger for replayable connectome proofreading edit transactions.
//!
//! The core validates transaction state, snapshot ancestry, independent-role separation,
//! invariant attestations, idempotency and rollback lineage. It does not apply graph patches,
//! compute SHA-256 over external bytes, or decide whether an edit is biologically correct.
//! A separately pinned adapter must hash the bytes and independently inspect graph topology
//! and annotation propagation before submitting the resulting attestations here.

use std::collections::BTreeSet;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditKind {
    Merge,
    Split,
    AttachFragment,
    DetachFragment,
    AnnotationRemap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewDisposition {
    Approved,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationOutcome {
    Passed,
    Failed,
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DigestAssertion {
    pub invariant_id: String,
    pub expected_sha256: String,
    pub observed_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnotationPropagationReceipt {
    pub manifest_sha256: String,
    pub expected_annotation_state_sha256: String,
    pub observed_annotation_state_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditSubmission {
    pub transaction_id: String,
    pub idempotency_key: String,
    pub parent_transaction_id: Option<String>,
    pub operation: EditKind,
    pub affected_object_ids: Vec<String>,
    /// Canonical digest of frozen operation parameters (location, split/merge method, options).
    pub operation_parameters_sha256: String,
    pub base_snapshot_sha256: String,
    pub patch_sha256: String,
    /// Present only when the adapter successfully applied the patch.
    pub resulting_snapshot_sha256: Option<String>,
    pub reversible: bool,
    pub inverse_patch_sha256: Option<String>,
    pub preconditions: Vec<DigestAssertion>,
    pub postconditions: Vec<DigestAssertion>,
    pub proposer_id: String,
    pub reviewer_id: String,
    pub verifier_id: String,
    pub review_disposition: ReviewDisposition,
    pub verification_outcome: VerificationOutcome,
    pub verifier_artifact_sha256: String,
    pub verifier_config_sha256: String,
    pub annotation_propagation: Option<AnnotationPropagationReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RollbackSubmission {
    pub rollback_id: String,
    pub idempotency_key: String,
    pub target_transaction_id: String,
    pub base_snapshot_sha256: String,
    pub inverse_patch_sha256: String,
    pub restored_snapshot_sha256: String,
    pub proposer_id: String,
    pub reviewer_id: String,
    pub verifier_id: String,
    pub review_disposition: ReviewDisposition,
    pub verification_outcome: VerificationOutcome,
    pub verifier_artifact_sha256: String,
    pub verifier_config_sha256: String,
    pub preconditions: Vec<DigestAssertion>,
    pub postconditions: Vec<DigestAssertion>,
    pub annotation_propagation: Option<AnnotationPropagationReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    StaleBaseSnapshot,
    ParentTransactionMismatch,
    PreconditionsFailed,
    ReviewRejected,
    ApplyReceiptMissing,
    NoopSnapshot,
    TargetNotCurrentHead,
    TargetNotReversible,
    RollbackBaseMismatch,
    InversePatchMismatch,
    RestoredSnapshotMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuarantineReason {
    VerifierFailed,
    VerifierInconclusive,
    PostconditionsFailed,
    AnnotationPropagationMissing,
    AnnotationPropagationMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventDecision {
    Committed {
        resulting_snapshot_sha256: String,
    },
    Rejected {
        reason: RejectReason,
    },
    Quarantined {
        reason: QuarantineReason,
    },
    RolledBack {
        restored_snapshot_sha256: String,
        target_transaction_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditAuditEvent {
    pub sequence: u64,
    pub submission: EditSubmission,
    pub decision: EventDecision,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RollbackAuditEvent {
    pub sequence: u64,
    pub submission: RollbackSubmission,
    pub decision: EventDecision,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerEvent {
    Edit(EditAuditEvent),
    Rollback(RollbackAuditEvent),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CommittedEdit {
    transaction_id: String,
    base_snapshot_sha256: String,
    resulting_snapshot_sha256: String,
    inverse_patch_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerSnapshot {
    pub initial_snapshot_sha256: String,
    pub current_snapshot_sha256: String,
    pub next_event_sequence: u64,
    pub committed_transaction_ids: Vec<String>,
    pub event_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerError {
    InvalidIdentifier {
        kind: &'static str,
        value: String,
    },
    DuplicateIdentifier {
        kind: &'static str,
        value: String,
    },
    InvalidSha256 {
        kind: &'static str,
        value: String,
    },
    EmptyCollection(&'static str),
    DuplicateInvariant {
        invariant_id: String,
    },
    ReversibleRequiresInverse(String),
    IrreversibleHasInverse(String),
    InvalidActorSeparation,
    DuplicateEventId(String),
    DuplicateIdempotencyKey(String),
    EventSequenceOverflow,
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier { kind, value } => {
                write!(f, "invalid {kind} identifier: {value:?}")
            }
            Self::DuplicateIdentifier { kind, value } => {
                write!(f, "duplicate {kind} identifier: {value}")
            }
            Self::InvalidSha256 { kind, value } => {
                write!(f, "invalid SHA-256 for {kind}: {value:?}")
            }
            Self::EmptyCollection(name) => write!(f, "{name} must not be empty"),
            Self::DuplicateInvariant { invariant_id } => {
                write!(f, "duplicate invariant assertion: {invariant_id}")
            }
            Self::ReversibleRequiresInverse(id) => {
                write!(f, "reversible transaction {id} requires an inverse patch digest")
            }
            Self::IrreversibleHasInverse(id) => {
                write!(f, "irreversible transaction {id} must not declare an inverse patch")
            }
            Self::InvalidActorSeparation => {
                write!(f, "proposer, reviewer and verifier must be three distinct identities")
            }
            Self::DuplicateEventId(id) => write!(f, "duplicate ledger event ID: {id}"),
            Self::DuplicateIdempotencyKey(key) => {
                write!(f, "duplicate idempotency key: {key}")
            }
            Self::EventSequenceOverflow => write!(f, "ledger event sequence overflow"),
        }
    }
}

impl std::error::Error for LedgerError {}

/// In-memory append-only ledger state.
///
/// Persisted deployments must serialize events durably and atomically update the canonical
/// head. This type provides deterministic state transitions; it is not a distributed
/// consensus protocol or a concurrent database implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditTransactionLedger {
    initial_snapshot_sha256: String,
    current_snapshot_sha256: String,
    next_event_sequence: u64,
    committed_stack: Vec<CommittedEdit>,
    seen_event_ids: BTreeSet<String>,
    seen_idempotency_keys: BTreeSet<String>,
    events: Vec<LedgerEvent>,
}

impl EditTransactionLedger {
    pub fn new(initial_snapshot_sha256: impl Into<String>) -> Result<Self, LedgerError> {
        let initial = initial_snapshot_sha256.into();
        validate_sha("initial graph snapshot", &initial)?;
        Ok(Self {
            initial_snapshot_sha256: initial.clone(),
            current_snapshot_sha256: initial,
            next_event_sequence: 1,
            committed_stack: Vec::new(),
            seen_event_ids: BTreeSet::new(),
            seen_idempotency_keys: BTreeSet::new(),
            events: Vec::new(),
        })
    }

    pub fn current_snapshot_sha256(&self) -> &str {
        &self.current_snapshot_sha256
    }

    pub fn events(&self) -> &[LedgerEvent] {
        &self.events
    }

    pub fn snapshot(&self) -> LedgerSnapshot {
        LedgerSnapshot {
            initial_snapshot_sha256: self.initial_snapshot_sha256.clone(),
            current_snapshot_sha256: self.current_snapshot_sha256.clone(),
            next_event_sequence: self.next_event_sequence,
            committed_transaction_ids: self
                .committed_stack
                .iter()
                .map(|edit| edit.transaction_id.clone())
                .collect(),
            event_count: self.events.len(),
        }
    }

    /// Validate and submit an already-applied edit receipt.
    ///
    /// Rejected and quarantined submissions are retained as append-only audit events but
    /// never advance the canonical graph head. A committed transaction must match the
    /// current snapshot and parent transaction, pass all preconditions, have approval,
    /// pass independent verification, pass postconditions, and verify annotation propagation.
    pub fn submit_edit(
        &mut self,
        submission: EditSubmission,
    ) -> Result<EventDecision, LedgerError> {
        validate_edit_submission(&submission)?;
        let sequence = self.reserve_event_sequence(
            &submission.transaction_id,
            &submission.idempotency_key,
        )?;
        let decision = self.evaluate_edit(&submission);

        if let EventDecision::Committed {
            resulting_snapshot_sha256,
        } = &decision
        {
            self.current_snapshot_sha256 = resulting_snapshot_sha256.clone();
            self.committed_stack.push(CommittedEdit {
                transaction_id: submission.transaction_id.clone(),
                base_snapshot_sha256: submission.base_snapshot_sha256.clone(),
                resulting_snapshot_sha256: resulting_snapshot_sha256.clone(),
                inverse_patch_sha256: submission.inverse_patch_sha256.clone(),
            });
        }

        self.events.push(LedgerEvent::Edit(EditAuditEvent {
            sequence,
            submission,
            decision: decision.clone(),
        }));
        Ok(decision)
    }

    /// Submit a rollback as a new, independently verified ledger event.
    ///
    /// Only the current committed stack head can be rolled back. The rollback must start
    /// from that transaction's resulting snapshot and restore exactly its base snapshot.
    /// A successful rollback pops the committed stack but leaves both the original edit
    /// and rollback event in the immutable event history.
    pub fn submit_rollback(
        &mut self,
        submission: RollbackSubmission,
    ) -> Result<EventDecision, LedgerError> {
        validate_rollback_submission(&submission)?;
        let sequence = self.reserve_event_sequence(
            &submission.rollback_id,
            &submission.idempotency_key,
        )?;
        let decision = self.evaluate_rollback(&submission);

        if let EventDecision::RolledBack {
            restored_snapshot_sha256,
            target_transaction_id: _,
        } = &decision
        {
            self.current_snapshot_sha256 = restored_snapshot_sha256.clone();
            self.committed_stack.pop();
        }

        self.events.push(LedgerEvent::Rollback(RollbackAuditEvent {
            sequence,
            submission,
            decision: decision.clone(),
        }));
        Ok(decision)
    }

    fn reserve_event_sequence(
        &mut self,
        event_id: &str,
        idempotency_key: &str,
    ) -> Result<u64, LedgerError> {
        if self.seen_event_ids.contains(event_id) {
            return Err(LedgerError::DuplicateEventId(event_id.to_owned()));
        }
        if self.seen_idempotency_keys.contains(idempotency_key) {
            return Err(LedgerError::DuplicateIdempotencyKey(
                idempotency_key.to_owned(),
            ));
        }

        let next = self
            .next_event_sequence
            .checked_add(1)
            .ok_or(LedgerError::EventSequenceOverflow)?;
        let sequence = self.next_event_sequence;
        self.seen_event_ids.insert(event_id.to_owned());
        self.seen_idempotency_keys
            .insert(idempotency_key.to_owned());
        self.next_event_sequence = next;
        Ok(sequence)
    }

    fn evaluate_edit(&self, submission: &EditSubmission) -> EventDecision {
        if submission.base_snapshot_sha256 != self.current_snapshot_sha256 {
            return EventDecision::Rejected {
                reason: RejectReason::StaleBaseSnapshot,
            };
        }

        let expected_parent = self
            .committed_stack
            .last()
            .map(|edit| edit.transaction_id.as_str());
        if submission.parent_transaction_id.as_deref() != expected_parent {
            return EventDecision::Rejected {
                reason: RejectReason::ParentTransactionMismatch,
            };
        }

        if !assertions_match(&submission.preconditions) {
            return EventDecision::Rejected {
                reason: RejectReason::PreconditionsFailed,
            };
        }

        if submission.review_disposition == ReviewDisposition::Rejected {
            return EventDecision::Rejected {
                reason: RejectReason::ReviewRejected,
            };
        }

        let Some(resulting_snapshot_sha256) =
            submission.resulting_snapshot_sha256.as_ref()
        else {
            return EventDecision::Rejected {
                reason: RejectReason::ApplyReceiptMissing,
            };
        };

        if resulting_snapshot_sha256 == &submission.base_snapshot_sha256 {
            return EventDecision::Rejected {
                reason: RejectReason::NoopSnapshot,
            };
        }

        match submission.verification_outcome {
            VerificationOutcome::Failed => {
                return EventDecision::Quarantined {
                    reason: QuarantineReason::VerifierFailed,
                };
            }
            VerificationOutcome::Inconclusive => {
                return EventDecision::Quarantined {
                    reason: QuarantineReason::VerifierInconclusive,
                };
            }
            VerificationOutcome::Passed => {}
        }

        if !assertions_match(&submission.postconditions) {
            return EventDecision::Quarantined {
                reason: QuarantineReason::PostconditionsFailed,
            };
        }

        match &submission.annotation_propagation {
            None => {
                return EventDecision::Quarantined {
                    reason: QuarantineReason::AnnotationPropagationMissing,
                };
            }
            Some(receipt)
                if receipt.expected_annotation_state_sha256
                    != receipt.observed_annotation_state_sha256 =>
            {
                return EventDecision::Quarantined {
                    reason: QuarantineReason::AnnotationPropagationMismatch,
                };
            }
            Some(_) => {}
        }

        EventDecision::Committed {
            resulting_snapshot_sha256: resulting_snapshot_sha256.clone(),
        }
    }

    fn evaluate_rollback(&self, submission: &RollbackSubmission) -> EventDecision {
        let Some(target) = self.committed_stack.last() else {
            return EventDecision::Rejected {
                reason: RejectReason::TargetNotCurrentHead,
            };
        };

        if target.transaction_id != submission.target_transaction_id {
            return EventDecision::Rejected {
                reason: RejectReason::TargetNotCurrentHead,
            };
        }
        if target.inverse_patch_sha256.is_none() {
            return EventDecision::Rejected {
                reason: RejectReason::TargetNotReversible,
            };
        }
        if submission.base_snapshot_sha256 != self.current_snapshot_sha256
            || submission.base_snapshot_sha256 != target.resulting_snapshot_sha256
        {
            return EventDecision::Rejected {
                reason: RejectReason::RollbackBaseMismatch,
            };
        }
        if Some(&submission.inverse_patch_sha256) != target.inverse_patch_sha256.as_ref() {
            return EventDecision::Rejected {
                reason: RejectReason::InversePatchMismatch,
            };
        }
        if submission.restored_snapshot_sha256 != target.base_snapshot_sha256 {
            return EventDecision::Rejected {
                reason: RejectReason::RestoredSnapshotMismatch,
            };
        }
        if !assertions_match(&submission.preconditions) {
            return EventDecision::Rejected {
                reason: RejectReason::PreconditionsFailed,
            };
        }
        if submission.review_disposition == ReviewDisposition::Rejected {
            return EventDecision::Rejected {
                reason: RejectReason::ReviewRejected,
            };
        }

        match submission.verification_outcome {
            VerificationOutcome::Failed => {
                return EventDecision::Quarantined {
                    reason: QuarantineReason::VerifierFailed,
                };
            }
            VerificationOutcome::Inconclusive => {
                return EventDecision::Quarantined {
                    reason: QuarantineReason::VerifierInconclusive,
                };
            }
            VerificationOutcome::Passed => {}
        }

        if !assertions_match(&submission.postconditions) {
            return EventDecision::Quarantined {
                reason: QuarantineReason::PostconditionsFailed,
            };
        }
        match &submission.annotation_propagation {
            None => {
                return EventDecision::Quarantined {
                    reason: QuarantineReason::AnnotationPropagationMissing,
                };
            }
            Some(receipt)
                if receipt.expected_annotation_state_sha256
                    != receipt.observed_annotation_state_sha256 =>
            {
                return EventDecision::Quarantined {
                    reason: QuarantineReason::AnnotationPropagationMismatch,
                };
            }
            Some(_) => {}
        }

        EventDecision::RolledBack {
            restored_snapshot_sha256: submission.restored_snapshot_sha256.clone(),
            target_transaction_id: submission.target_transaction_id.clone(),
        }
    }
}

fn validate_edit_submission(submission: &EditSubmission) -> Result<(), LedgerError> {
    validate_id("transaction", &submission.transaction_id)?;
    validate_id("idempotency key", &submission.idempotency_key)?;
    validate_id("proposer", &submission.proposer_id)?;
    validate_id("reviewer", &submission.reviewer_id)?;
    validate_id("verifier", &submission.verifier_id)?;
    if submission.proposer_id == submission.reviewer_id
        || submission.proposer_id == submission.verifier_id
        || submission.reviewer_id == submission.verifier_id
    {
        return Err(LedgerError::InvalidActorSeparation);
    }
    if let Some(parent) = &submission.parent_transaction_id {
        validate_id("parent transaction", parent)?;
    }

    if submission.affected_object_ids.is_empty() {
        return Err(LedgerError::EmptyCollection("affected_object_ids"));
    }
    validate_unique_ids(
        "affected object",
        submission.affected_object_ids.iter().map(String::as_str),
    )?;

    validate_sha("base snapshot", &submission.base_snapshot_sha256)?;
    validate_sha("edit patch", &submission.patch_sha256)?;
    validate_sha(
        "operation parameters",
        &submission.operation_parameters_sha256,
    )?;
    validate_sha("verifier artifact", &submission.verifier_artifact_sha256)?;
    validate_sha("verifier config", &submission.verifier_config_sha256)?;

    match (&submission.reversible, &submission.inverse_patch_sha256) {
        (true, None) => {
            return Err(LedgerError::ReversibleRequiresInverse(
                submission.transaction_id.clone(),
            ));
        }
        (false, Some(_)) => {
            return Err(LedgerError::IrreversibleHasInverse(
                submission.transaction_id.clone(),
            ));
        }
        (_, Some(digest)) => validate_sha("inverse patch", digest)?,
        (_, None) => {}
    }
    if let Some(resulting) = &submission.resulting_snapshot_sha256 {
        validate_sha("resulting snapshot", resulting)?;
    }

    validate_assertions(&submission.preconditions, "preconditions")?;
    validate_assertions(&submission.postconditions, "postconditions")?;
    if let Some(receipt) = &submission.annotation_propagation {
        validate_annotation_receipt(receipt)?;
    }
    Ok(())
}

fn validate_rollback_submission(
    submission: &RollbackSubmission,
) -> Result<(), LedgerError> {
    validate_id("rollback", &submission.rollback_id)?;
    validate_id("idempotency key", &submission.idempotency_key)?;
    validate_id("target transaction", &submission.target_transaction_id)?;
    validate_id("rollback proposer", &submission.proposer_id)?;
    validate_id("rollback reviewer", &submission.reviewer_id)?;
    validate_id("rollback verifier", &submission.verifier_id)?;
    if submission.proposer_id == submission.reviewer_id
        || submission.proposer_id == submission.verifier_id
        || submission.reviewer_id == submission.verifier_id
    {
        return Err(LedgerError::InvalidActorSeparation);
    }

    validate_sha("rollback base snapshot", &submission.base_snapshot_sha256)?;
    validate_sha("rollback inverse patch", &submission.inverse_patch_sha256)?;
    validate_sha("restored snapshot", &submission.restored_snapshot_sha256)?;
    validate_sha("rollback verifier artifact", &submission.verifier_artifact_sha256)?;
    validate_sha("rollback verifier config", &submission.verifier_config_sha256)?;
    validate_assertions(&submission.preconditions, "rollback preconditions")?;
    validate_assertions(&submission.postconditions, "rollback postconditions")?;
    if let Some(receipt) = &submission.annotation_propagation {
        validate_annotation_receipt(receipt)?;
    }
    Ok(())
}

fn validate_assertions(
    assertions: &[DigestAssertion],
    name: &'static str,
) -> Result<(), LedgerError> {
    if assertions.is_empty() {
        return Err(LedgerError::EmptyCollection(name));
    }
    validate_unique_ids(
        "invariant",
        assertions.iter().map(|assertion| assertion.invariant_id.as_str()),
    )?;
    for assertion in assertions {
        validate_sha("expected invariant", &assertion.expected_sha256)?;
        validate_sha("observed invariant", &assertion.observed_sha256)?;
    }
    Ok(())
}

fn validate_annotation_receipt(
    receipt: &AnnotationPropagationReceipt,
) -> Result<(), LedgerError> {
    validate_sha("annotation manifest", &receipt.manifest_sha256)?;
    validate_sha(
        "expected annotation state",
        &receipt.expected_annotation_state_sha256,
    )?;
    validate_sha(
        "observed annotation state",
        &receipt.observed_annotation_state_sha256,
    )?;
    Ok(())
}

fn assertions_match(assertions: &[DigestAssertion]) -> bool {
    assertions
        .iter()
        .all(|assertion| assertion.expected_sha256 == assertion.observed_sha256)
}

fn validate_sha(kind: &'static str, value: &str) -> Result<(), LedgerError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(LedgerError::InvalidSha256 {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_id(kind: &'static str, value: &str) -> Result<(), LedgerError> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(LedgerError::InvalidIdentifier {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_unique_ids<'a>(
    kind: &'static str,
    values: impl Iterator<Item = &'a str>,
) -> Result<(), LedgerError> {
    let mut seen = BTreeSet::new();
    for value in values {
        validate_id(kind, value)?;
        if !seen.insert(value.to_owned()) {
            return Err(LedgerError::DuplicateIdentifier {
                kind,
                value: value.to_owned(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const SHA_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn assertion(id: &str, expected: &str, observed: &str) -> DigestAssertion {
        DigestAssertion {
            invariant_id: id.to_owned(),
            expected_sha256: expected.to_owned(),
            observed_sha256: observed.to_owned(),
        }
    }

    fn annotation() -> AnnotationPropagationReceipt {
        AnnotationPropagationReceipt {
            manifest_sha256: SHA_C.to_owned(),
            expected_annotation_state_sha256: SHA_D.to_owned(),
            observed_annotation_state_sha256: SHA_D.to_owned(),
        }
    }

    fn edit(id: &str, base: &str, result: &str) -> EditSubmission {
        EditSubmission {
            transaction_id: id.to_owned(),
            idempotency_key: format!("idem-{id}"),
            parent_transaction_id: None,
            operation: EditKind::Merge,
            affected_object_ids: vec!["segment-1".to_owned(), "segment-2".to_owned()],
            operation_parameters_sha256: SHA_E.to_owned(),
            base_snapshot_sha256: base.to_owned(),
            patch_sha256: SHA_B.to_owned(),
            resulting_snapshot_sha256: Some(result.to_owned()),
            reversible: true,
            inverse_patch_sha256: Some(SHA_E.to_owned()),
            preconditions: vec![assertion("base-topology", SHA_A, SHA_A)],
            postconditions: vec![assertion("merged-topology", SHA_B, SHA_B)],
            proposer_id: "model-1".to_owned(),
            reviewer_id: "reviewer-1".to_owned(),
            verifier_id: "verifier-1".to_owned(),
            review_disposition: ReviewDisposition::Approved,
            verification_outcome: VerificationOutcome::Passed,
            verifier_artifact_sha256: SHA_C.to_owned(),
            verifier_config_sha256: SHA_D.to_owned(),
            annotation_propagation: Some(annotation()),
        }
    }

    fn rollback(id: &str, target: &str, base: &str, restored: &str) -> RollbackSubmission {
        RollbackSubmission {
            rollback_id: id.to_owned(),
            idempotency_key: format!("idem-{id}"),
            target_transaction_id: target.to_owned(),
            base_snapshot_sha256: base.to_owned(),
            inverse_patch_sha256: SHA_E.to_owned(),
            restored_snapshot_sha256: restored.to_owned(),
            proposer_id: "rollback-actor".to_owned(),
            reviewer_id: "rollback-reviewer".to_owned(),
            verifier_id: "rollback-verifier".to_owned(),
            review_disposition: ReviewDisposition::Approved,
            verification_outcome: VerificationOutcome::Passed,
            verifier_artifact_sha256: SHA_C.to_owned(),
            verifier_config_sha256: SHA_D.to_owned(),
            preconditions: vec![assertion("current-head", SHA_B, SHA_B)],
            postconditions: vec![assertion("restored-head", SHA_A, SHA_A)],
            annotation_propagation: Some(annotation()),
        }
    }

    fn committed_ledger() -> EditTransactionLedger {
        let mut ledger = EditTransactionLedger::new(SHA_A).unwrap();
        assert_eq!(
            ledger.submit_edit(edit("tx-1", SHA_A, SHA_B)).unwrap(),
            EventDecision::Committed {
                resulting_snapshot_sha256: SHA_B.to_owned()
            }
        );
        ledger
    }

    #[test]
    fn commits_valid_edit_and_advances_canonical_head() {
        let mut ledger = EditTransactionLedger::new(SHA_A).unwrap();
        let decision = ledger.submit_edit(edit("tx-1", SHA_A, SHA_B)).unwrap();

        assert_eq!(
            decision,
            EventDecision::Committed {
                resulting_snapshot_sha256: SHA_B.to_owned()
            }
        );
        assert_eq!(ledger.current_snapshot_sha256(), SHA_B);
        assert_eq!(ledger.snapshot().committed_transaction_ids, vec!["tx-1"]);
        assert_eq!(ledger.events().len(), 1);
    }

    #[test]
    fn stale_base_snapshot_is_rejected_without_advancing_head() {
        let mut ledger = EditTransactionLedger::new(SHA_A).unwrap();
        let decision = ledger.submit_edit(edit("tx-stale", SHA_B, SHA_C)).unwrap();

        assert_eq!(
            decision,
            EventDecision::Rejected {
                reason: RejectReason::StaleBaseSnapshot
            }
        );
        assert_eq!(ledger.current_snapshot_sha256(), SHA_A);
        assert_eq!(ledger.events().len(), 1);
    }

    #[test]
    fn failed_precondition_is_rejected_before_commit() {
        let mut ledger = EditTransactionLedger::new(SHA_A).unwrap();
        let mut submission = edit("tx-pre", SHA_A, SHA_B);
        submission.preconditions[0].observed_sha256 = SHA_C.to_owned();

        assert_eq!(
            ledger.submit_edit(submission).unwrap(),
            EventDecision::Rejected {
                reason: RejectReason::PreconditionsFailed
            }
        );
        assert_eq!(ledger.current_snapshot_sha256(), SHA_A);
    }

    #[test]
    fn failed_postcondition_is_quarantined() {
        let mut ledger = EditTransactionLedger::new(SHA_A).unwrap();
        let mut submission = edit("tx-post", SHA_A, SHA_B);
        submission.postconditions[0].observed_sha256 = SHA_C.to_owned();

        assert_eq!(
            ledger.submit_edit(submission).unwrap(),
            EventDecision::Quarantined {
                reason: QuarantineReason::PostconditionsFailed
            }
        );
        assert_eq!(ledger.current_snapshot_sha256(), SHA_A);
        assert_eq!(ledger.snapshot().committed_transaction_ids.len(), 0);
    }

    #[test]
    fn verifier_failure_and_inconclusive_never_commit() {
        for (id, outcome, expected) in [
            (
                "tx-failed",
                VerificationOutcome::Failed,
                QuarantineReason::VerifierFailed,
            ),
            (
                "tx-inconclusive",
                VerificationOutcome::Inconclusive,
                QuarantineReason::VerifierInconclusive,
            ),
        ] {
            let mut ledger = EditTransactionLedger::new(SHA_A).unwrap();
            let mut submission = edit(id, SHA_A, SHA_B);
            submission.verification_outcome = outcome;

            assert_eq!(
                ledger.submit_edit(submission).unwrap(),
                EventDecision::Quarantined { reason: expected }
            );
            assert_eq!(ledger.current_snapshot_sha256(), SHA_A);
        }
    }

    #[test]
    fn missing_or_mismatched_annotation_propagation_is_quarantined() {
        let mut missing = EditTransactionLedger::new(SHA_A).unwrap();
        let mut submission = edit("tx-no-annotation", SHA_A, SHA_B);
        submission.annotation_propagation = None;
        assert_eq!(
            missing.submit_edit(submission).unwrap(),
            EventDecision::Quarantined {
                reason: QuarantineReason::AnnotationPropagationMissing
            }
        );

        let mut mismatch = EditTransactionLedger::new(SHA_A).unwrap();
        let mut submission = edit("tx-bad-annotation", SHA_A, SHA_B);
        submission
            .annotation_propagation
            .as_mut()
            .unwrap()
            .observed_annotation_state_sha256 = SHA_E.to_owned();
        assert_eq!(
            mismatch.submit_edit(submission).unwrap(),
            EventDecision::Quarantined {
                reason: QuarantineReason::AnnotationPropagationMismatch
            }
        );
    }

    #[test]
    fn rejected_review_never_commits() {
        let mut ledger = EditTransactionLedger::new(SHA_A).unwrap();
        let mut submission = edit("tx-review-rejected", SHA_A, SHA_B);
        submission.review_disposition = ReviewDisposition::Rejected;

        assert_eq!(
            ledger.submit_edit(submission).unwrap(),
            EventDecision::Rejected {
                reason: RejectReason::ReviewRejected
            }
        );
        assert_eq!(ledger.current_snapshot_sha256(), SHA_A);
    }

    #[test]
    fn proposer_reviewer_and_verifier_must_be_distinct() {
        let mut ledger = EditTransactionLedger::new(SHA_A).unwrap();
        let mut submission = edit("tx-roles", SHA_A, SHA_B);
        submission.verifier_id = submission.proposer_id.clone();

        assert_eq!(
            ledger.submit_edit(submission),
            Err(LedgerError::InvalidActorSeparation)
        );
        assert!(ledger.events().is_empty());
    }

    #[test]
    fn duplicate_idempotency_key_is_rejected_across_event_kinds() {
        let mut ledger = committed_ledger();
        let mut duplicate = edit("tx-2", SHA_B, SHA_C);
        duplicate.idempotency_key = "idem-tx-1".to_owned();

        assert_eq!(
            ledger.submit_edit(duplicate),
            Err(LedgerError::DuplicateIdempotencyKey(
                "idem-tx-1".to_owned()
            ))
        );

        let mut rollback = rollback("rb-1", "tx-1", SHA_B, SHA_A);
        rollback.idempotency_key = "idem-tx-1".to_owned();
        assert_eq!(
            ledger.submit_rollback(rollback),
            Err(LedgerError::DuplicateIdempotencyKey(
                "idem-tx-1".to_owned()
            ))
        );
    }

    #[test]
    fn duplicate_transaction_id_is_rejected_even_with_a_different_patch() {
        let mut ledger = committed_ledger();
        let mut duplicate = edit("tx-1", SHA_B, SHA_C);
        duplicate.patch_sha256 = SHA_E.to_owned();

        assert_eq!(
            ledger.submit_edit(duplicate),
            Err(LedgerError::DuplicateEventId("tx-1".to_owned()))
        );
        assert_eq!(ledger.current_snapshot_sha256(), SHA_B);
    }

    #[test]
    fn same_transaction_cannot_be_applied_twice_after_commit() {
        let mut ledger = committed_ledger();
        let mut duplicate = edit("tx-1-retry", SHA_B, SHA_C);
        duplicate.idempotency_key = "retry-with-same-logical-operation".to_owned();
        duplicate.patch_sha256 = SHA_B.to_owned();

        // A new transaction ID does not bypass the current-head precondition.
        assert_eq!(
            ledger.submit_edit(duplicate).unwrap(),
            EventDecision::Rejected {
                reason: RejectReason::ParentTransactionMismatch
            }
        );
        assert_eq!(ledger.current_snapshot_sha256(), SHA_B);
    }

    #[test]
    fn reversible_and_inverse_patch_declarations_must_agree() {
        let mut ledger = EditTransactionLedger::new(SHA_A).unwrap();
        let mut reversible_missing = edit("tx-no-inverse", SHA_A, SHA_B);
        reversible_missing.inverse_patch_sha256 = None;
        assert_eq!(
            ledger.submit_edit(reversible_missing),
            Err(LedgerError::ReversibleRequiresInverse(
                "tx-no-inverse".to_owned()
            ))
        );

        let mut irreversible_with_inverse = edit("tx-stray-inverse", SHA_A, SHA_B);
        irreversible_with_inverse.reversible = false;
        assert_eq!(
            ledger.submit_edit(irreversible_with_inverse),
            Err(LedgerError::IrreversibleHasInverse(
                "tx-stray-inverse".to_owned()
            ))
        );
    }

    #[test]
    fn successful_rollback_restores_base_but_preserves_both_events() {
        let mut ledger = committed_ledger();
        let decision = ledger
            .submit_rollback(rollback("rb-1", "tx-1", SHA_B, SHA_A))
            .unwrap();

        assert_eq!(
            decision,
            EventDecision::RolledBack {
                restored_snapshot_sha256: SHA_A.to_owned(),
                target_transaction_id: "tx-1".to_owned(),
            }
        );
        assert_eq!(ledger.current_snapshot_sha256(), SHA_A);
        assert!(ledger.snapshot().committed_transaction_ids.is_empty());
        assert_eq!(ledger.events().len(), 2);
        assert!(matches!(ledger.events()[0], LedgerEvent::Edit(_)));
        assert!(matches!(ledger.events()[1], LedgerEvent::Rollback(_)));
    }

    #[test]
    fn rollback_must_target_current_head_and_cannot_be_repeated() {
        let mut ledger = committed_ledger();
        let decision = ledger
            .submit_rollback(rollback("rb-not-top", "other-tx", SHA_B, SHA_A))
            .unwrap();
        assert_eq!(
            decision,
            EventDecision::Rejected {
                reason: RejectReason::TargetNotCurrentHead
            }
        );
        assert_eq!(ledger.current_snapshot_sha256(), SHA_B);

        let valid = rollback("rb-valid", "tx-1", SHA_B, SHA_A);
        assert!(matches!(
            ledger.submit_rollback(valid).unwrap(),
            EventDecision::RolledBack { .. }
        ));
        let duplicate = rollback("rb-duplicate", "tx-1", SHA_B, SHA_A);
        assert_eq!(
            ledger.submit_rollback(duplicate).unwrap(),
            EventDecision::Rejected {
                reason: RejectReason::TargetNotCurrentHead
            }
        );
        assert_eq!(ledger.current_snapshot_sha256(), SHA_A);
    }

    #[test]
    fn rollback_digest_and_restored_base_must_match_committed_transaction() {
        let mut wrong_inverse = committed_ledger();
        let mut rollback_submission = rollback("rb-wrong-inverse", "tx-1", SHA_B, SHA_A);
        rollback_submission.inverse_patch_sha256 = SHA_D.to_owned();
        assert_eq!(
            wrong_inverse.submit_rollback(rollback_submission).unwrap(),
            EventDecision::Rejected {
                reason: RejectReason::InversePatchMismatch
            }
        );

        let mut wrong_restored = committed_ledger();
        let rollback_submission = rollback("rb-wrong-result", "tx-1", SHA_B, SHA_C);
        assert_eq!(
            wrong_restored.submit_rollback(rollback_submission).unwrap(),
            EventDecision::Rejected {
                reason: RejectReason::RestoredSnapshotMismatch
            }
        );
    }

    #[test]
    fn non_monotonic_or_broken_parent_chain_cannot_advance_head() {
        let mut ledger = committed_ledger();
        let mut submission = edit("tx-2", SHA_B, SHA_C);
        submission.parent_transaction_id = None;

        assert_eq!(
            ledger.submit_edit(submission).unwrap(),
            EventDecision::Rejected {
                reason: RejectReason::ParentTransactionMismatch
            }
        );
        assert_eq!(ledger.current_snapshot_sha256(), SHA_B);
    }

    #[test]
    fn event_sequence_is_monotonic_across_rejected_quarantined_and_committed_events() {
        let mut ledger = EditTransactionLedger::new(SHA_A).unwrap();

        let mut stale = edit("tx-stale", SHA_B, SHA_C);
        stale.idempotency_key = "idem-stale".to_owned();
        assert!(matches!(
            ledger.submit_edit(stale).unwrap(),
            EventDecision::Rejected { .. }
        ));

        let mut quarantined = edit("tx-quarantine", SHA_A, SHA_B);
        quarantined.idempotency_key = "idem-quarantine".to_owned();
        quarantined.verification_outcome = VerificationOutcome::Inconclusive;
        assert!(matches!(
            ledger.submit_edit(quarantined).unwrap(),
            EventDecision::Quarantined { .. }
        ));

        let mut good = edit("tx-good", SHA_A, SHA_B);
        good.idempotency_key = "idem-good".to_owned();
        assert!(matches!(
            ledger.submit_edit(good).unwrap(),
            EventDecision::Committed { .. }
        ));

        let sequences: Vec<u64> = ledger
            .events()
            .iter()
            .map(|event| match event {
                LedgerEvent::Edit(event) => event.sequence,
                LedgerEvent::Rollback(event) => event.sequence,
            })
            .collect();
        assert_eq!(sequences, vec![1, 2, 3]);
        assert_eq!(ledger.snapshot().next_event_sequence, 4);
    }

    #[test]
    fn malformed_digest_and_duplicate_invariant_fail_closed() {
        let mut ledger = EditTransactionLedger::new(SHA_A).unwrap();
        let mut malformed = edit("tx-bad-sha", SHA_A, SHA_B);
        malformed.verifier_artifact_sha256 = "not-a-digest".to_owned();
        assert!(matches!(
            ledger.submit_edit(malformed),
            Err(LedgerError::InvalidSha256 { .. })
        ));

        let mut duplicate = edit("tx-duplicate-invariant", SHA_A, SHA_B);
        duplicate.preconditions.push(duplicate.preconditions[0].clone());
        assert!(matches!(
            ledger.submit_edit(duplicate),
            Err(LedgerError::DuplicateIdentifier { kind: "invariant", .. })
        ));
        assert!(ledger.events().is_empty());
    }

    #[test]
    fn no_op_snapshot_is_rejected() {
        let mut ledger = EditTransactionLedger::new(SHA_A).unwrap();
        assert_eq!(
            ledger.submit_edit(edit("tx-noop", SHA_A, SHA_A)).unwrap(),
            EventDecision::Rejected {
                reason: RejectReason::NoopSnapshot
            }
        );
        assert_eq!(ledger.current_snapshot_sha256(), SHA_A);
    }
}
