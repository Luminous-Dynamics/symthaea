// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Backend-neutral monotonic continuity for the generic intervention execution journal.
//!
//! The execution journal's internal hash chain is tamper-evident but is not, by itself, rollback
//! resistant. This module binds one fully validated journal state to an independently retained
//! monotonic anchor. The anchor is continuity evidence only: it grants no intervention, consent,
//! welfare, capability, or actuator authority.
//!
//! A deployment must supply a stable `journal_namespace` from trusted configuration. It must not be
//! regenerated per process or inferred from attacker-controlled recovered journal bytes.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use serde::{Deserialize, Serialize};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use thiserror::Error;

use crate::execution_recovery::{
    EXECUTION_JOURNAL_SCHEMA, ExecutionJournalEnvelope, ExecutionJournalError,
    InterventionExecutionJournal,
};

pub const EXECUTION_JOURNAL_ANCHOR_SCHEMA: &str =
    "symthaea.welfare.execution-journal-anchor.v1";
const ANCHOR_COMMITMENT_DOMAIN: &[u8] =
    b"symthaea.welfare.execution-journal-anchor.commitment.v1\0";
const MAX_NAMESPACE_BYTES: usize = 256;
const MAX_ANCHOR_REF_BYTES: usize = 2048;

/// Independently retained identity of one exact generic execution-journal state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionJournalAnchorSnapshot {
    pub schema_version: String,
    pub journal_namespace: String,
    pub execution_journal_schema: String,
    pub event_count: u64,
    pub head_hash: Sha256Digest,
    pub revision: u64,
    pub previous_commitment: Sha256Digest,
    pub committed_at_unix_s: u64,
}

impl ExecutionJournalAnchorSnapshot {
    pub fn try_new(
        journal_namespace: impl Into<String>,
        event_count: u64,
        head_hash: Sha256Digest,
        revision: u64,
        previous_commitment: Sha256Digest,
        committed_at_unix_s: u64,
    ) -> Result<Self, ExecutionJournalAnchorError> {
        let snapshot = Self {
            schema_version: EXECUTION_JOURNAL_ANCHOR_SCHEMA.into(),
            journal_namespace: journal_namespace.into(),
            execution_journal_schema: EXECUTION_JOURNAL_SCHEMA.into(),
            event_count,
            head_hash,
            revision,
            previous_commitment,
            committed_at_unix_s,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn validate(&self) -> Result<(), ExecutionJournalAnchorError> {
        if self.schema_version != EXECUTION_JOURNAL_ANCHOR_SCHEMA {
            return Err(ExecutionJournalAnchorError::UnsupportedAnchorSchema);
        }
        validate_namespace(&self.journal_namespace)?;
        if self.execution_journal_schema != EXECUTION_JOURNAL_SCHEMA {
            return Err(ExecutionJournalAnchorError::ExecutionJournalSchemaMismatch);
        }
        if self.revision == 0 {
            return Err(ExecutionJournalAnchorError::RevisionZero);
        }
        if self.event_count == 0 {
            if self.head_hash.0 != [0; 32] {
                return Err(ExecutionJournalAnchorError::EmptyJournalNonzeroHead);
            }
        } else if self.head_hash.0 == [0; 32] {
            return Err(ExecutionJournalAnchorError::NonemptyJournalZeroHead);
        }
        if self.revision == 1 {
            if self.previous_commitment.0 != [0; 32] {
                return Err(ExecutionJournalAnchorError::GenesisPreviousCommitmentNonzero);
            }
        } else if self.previous_commitment.0 == [0; 32] {
            return Err(ExecutionJournalAnchorError::SuccessorPreviousCommitmentZero);
        }
        Ok(())
    }

    pub fn commitment(&self) -> Result<Sha256Digest, ExecutionJournalAnchorError> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(ANCHOR_COMMITMENT_DOMAIN);
        hash_text(&mut hasher, &self.schema_version);
        hash_text(&mut hasher, &self.journal_namespace);
        hash_text(&mut hasher, &self.execution_journal_schema);
        hasher.update(&self.event_count.to_le_bytes());
        hasher.update(&self.head_hash.0);
        hasher.update(&self.revision.to_le_bytes());
        hasher.update(&self.previous_commitment.0);
        hasher.update(&self.committed_at_unix_s.to_le_bytes());
        Ok(hasher.finalize())
    }
}

/// Independent monotonic/trusted retention boundary for one stable journal namespace.
pub trait ExecutionJournalHeadAnchor {
    type Error: StdError + Send + Sync + 'static;

    fn load(
        &self,
        journal_namespace: &str,
    ) -> Result<Option<ExecutionJournalAnchorSnapshot>, Self::Error>;

    /// Atomically replace the externally retained snapshot only when its current commitment equals
    /// `expected_current`. `None` is reserved for first-time bootstrap.
    ///
    /// A returned error MUST NOT be interpreted by callers as proof that the write did not commit:
    /// an implementation may commit and then lose its acknowledgement.
    fn compare_and_swap(
        &mut self,
        journal_namespace: &str,
        expected_current: Option<Sha256Digest>,
        next: &ExecutionJournalAnchorSnapshot,
    ) -> Result<String, Self::Error>;
}

/// Fully validated local journal proven equal to one independently retained anchor snapshot.
#[derive(Debug, Clone)]
pub struct AnchoredExecutionJournal {
    pub journal: InterventionExecutionJournal,
    pub anchor: ExecutionJournalAnchorSnapshot,
    pub anchor_commitment: Sha256Digest,
}

/// Result of an anchor CAS attempt.
///
/// `InDoubt` means the external root may already equal `proposed_snapshot`. The caller must inspect
/// the anchor through ordinary recovery and MUST NOT blindly replay the CAS or any intervention.
#[derive(Debug)]
pub enum ExecutionJournalAnchorCommitOutcome<E>
where
    E: StdError + Send + Sync + 'static,
{
    Committed {
        snapshot: ExecutionJournalAnchorSnapshot,
        anchor_reference: String,
    },
    InDoubt {
        proposed_snapshot: ExecutionJournalAnchorSnapshot,
        reason: ExecutionJournalAnchorCommitInDoubt<E>,
    },
}

impl<E> ExecutionJournalAnchorCommitOutcome<E>
where
    E: StdError + Send + Sync + 'static,
{
    pub fn requires_reconciliation(&self) -> bool {
        matches!(self, Self::InDoubt { .. })
    }
}

#[derive(Debug, Error)]
pub enum ExecutionJournalAnchorCommitInDoubt<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("execution-journal anchor CAS acknowledgement failed: {0}")]
    CompareAndSwap(#[source] E),
    #[error("execution-journal anchor CAS succeeded but returned an invalid evidence reference")]
    InvalidReference,
    #[error("execution-journal anchor post-CAS readback failed: {0}")]
    Readback(#[source] E),
    #[error("execution-journal anchor disappeared during post-CAS readback")]
    MissingReadback,
    #[error("execution-journal anchor post-CAS readback differs from the proposed snapshot")]
    MismatchedReadback,
}

/// Bootstrap an independently retained root for the exact current journal state.
///
/// An empty journal is valid at bootstrap. This is a provisioning decision, not ordinary recovery.
pub fn bootstrap_execution_journal_anchor<A>(
    journal: &InterventionExecutionJournal,
    anchor: &mut A,
    journal_namespace: &str,
    committed_at_unix_s: u64,
) -> Result<ExecutionJournalAnchorCommitOutcome<A::Error>, ExecutionJournalAnchorProtocolError<A::Error>>
where
    A: ExecutionJournalHeadAnchor,
{
    validate_namespace(journal_namespace)?;
    if anchor
        .load(journal_namespace)
        .map_err(ExecutionJournalAnchorProtocolError::BackendLoad)?
        .is_some()
    {
        return Err(ExecutionJournalAnchorProtocolError::AlreadyBootstrapped);
    }
    let snapshot = ExecutionJournalAnchorSnapshot::try_new(
        journal_namespace,
        event_count(journal)?,
        journal.head_hash(),
        1,
        Sha256Digest([0; 32]),
        committed_at_unix_s,
    )?;
    attempt_anchor_commit(anchor, None, snapshot)
}

/// Advance an existing independent root to a strictly newer exact journal state.
///
/// Every returned `Err` is proven to occur before CAS is attempted. Once CAS is attempted, any
/// uncertainty is represented as `ExecutionJournalAnchorCommitOutcome::InDoubt`.
pub fn advance_execution_journal_anchor<A>(
    journal: &InterventionExecutionJournal,
    anchor: &mut A,
    previous: &ExecutionJournalAnchorSnapshot,
    committed_at_unix_s: u64,
) -> Result<ExecutionJournalAnchorCommitOutcome<A::Error>, ExecutionJournalAnchorProtocolError<A::Error>>
where
    A: ExecutionJournalHeadAnchor,
{
    previous.validate()?;
    let current = anchor
        .load(&previous.journal_namespace)
        .map_err(ExecutionJournalAnchorProtocolError::BackendLoad)?
        .ok_or(ExecutionJournalAnchorProtocolError::MissingAnchor)?;
    if current != *previous {
        return Err(ExecutionJournalAnchorProtocolError::StalePreviousSnapshot);
    }
    let local_count = event_count(journal)?;
    if local_count <= previous.event_count {
        return Err(ExecutionJournalAnchorProtocolError::JournalDidNotAdvance {
            previous: previous.event_count,
            current: local_count,
        });
    }
    if journal.head_hash() == previous.head_hash {
        return Err(ExecutionJournalAnchorProtocolError::HeadDidNotAdvance);
    }
    if committed_at_unix_s < previous.committed_at_unix_s {
        return Err(ExecutionJournalAnchorProtocolError::CommitTimeRegression {
            previous: previous.committed_at_unix_s,
            next: committed_at_unix_s,
        });
    }
    let previous_commitment = previous.commitment()?;
    let next_revision = previous
        .revision
        .checked_add(1)
        .ok_or(ExecutionJournalAnchorProtocolError::RevisionOverflow)?;
    let next = ExecutionJournalAnchorSnapshot::try_new(
        previous.journal_namespace.clone(),
        local_count,
        journal.head_hash(),
        next_revision,
        previous_commitment,
        committed_at_unix_s,
    )?;
    attempt_anchor_commit(anchor, Some(previous_commitment), next)
}

/// Recover one exact event stream only when it equals the independently retained current root.
///
/// This intentionally accepts no prefix, nearest revision, majority branch, or caller-selected
/// fallback. A locally valid but stale/forked journal therefore fails closed.
pub fn recover_execution_journal_with_anchor<A>(
    events: Vec<ExecutionJournalEnvelope>,
    anchor: &A,
    journal_namespace: &str,
) -> Result<AnchoredExecutionJournal, ExecutionJournalAnchorProtocolError<A::Error>>
where
    A: ExecutionJournalHeadAnchor,
{
    validate_namespace(journal_namespace)?;
    let journal = InterventionExecutionJournal::from_events(events)?;
    let snapshot = anchor
        .load(journal_namespace)
        .map_err(ExecutionJournalAnchorProtocolError::BackendLoad)?
        .ok_or(ExecutionJournalAnchorProtocolError::MissingAnchor)?;
    snapshot.validate()?;
    if snapshot.journal_namespace != journal_namespace {
        return Err(ExecutionJournalAnchorProtocolError::NamespaceMismatch);
    }
    let actual_count = event_count(&journal)?;
    if snapshot.event_count != actual_count {
        return Err(ExecutionJournalAnchorProtocolError::EventCountMismatch {
            anchored: snapshot.event_count,
            actual: actual_count,
        });
    }
    if snapshot.head_hash != journal.head_hash() {
        return Err(ExecutionJournalAnchorProtocolError::HeadMismatch {
            anchored: snapshot.head_hash,
            actual: journal.head_hash(),
        });
    }
    let anchor_commitment = snapshot.commitment()?;
    Ok(AnchoredExecutionJournal {
        journal,
        anchor: snapshot,
        anchor_commitment,
    })
}

fn attempt_anchor_commit<A>(
    anchor: &mut A,
    expected_current: Option<Sha256Digest>,
    proposed: ExecutionJournalAnchorSnapshot,
) -> Result<ExecutionJournalAnchorCommitOutcome<A::Error>, ExecutionJournalAnchorProtocolError<A::Error>>
where
    A: ExecutionJournalHeadAnchor,
{
    let namespace = proposed.journal_namespace.clone();
    let reference = match anchor.compare_and_swap(&namespace, expected_current, &proposed) {
        Ok(reference) => reference,
        Err(error) => {
            return Ok(ExecutionJournalAnchorCommitOutcome::InDoubt {
                proposed_snapshot: proposed,
                reason: ExecutionJournalAnchorCommitInDoubt::CompareAndSwap(error),
            });
        }
    };
    if validate_anchor_reference(&reference).is_err() {
        return Ok(ExecutionJournalAnchorCommitOutcome::InDoubt {
            proposed_snapshot: proposed,
            reason: ExecutionJournalAnchorCommitInDoubt::InvalidReference,
        });
    }
    let observed = match anchor.load(&namespace) {
        Ok(Some(snapshot)) => snapshot,
        Ok(None) => {
            return Ok(ExecutionJournalAnchorCommitOutcome::InDoubt {
                proposed_snapshot: proposed,
                reason: ExecutionJournalAnchorCommitInDoubt::MissingReadback,
            });
        }
        Err(error) => {
            return Ok(ExecutionJournalAnchorCommitOutcome::InDoubt {
                proposed_snapshot: proposed,
                reason: ExecutionJournalAnchorCommitInDoubt::Readback(error),
            });
        }
    };
    if observed != proposed {
        return Ok(ExecutionJournalAnchorCommitOutcome::InDoubt {
            proposed_snapshot: proposed,
            reason: ExecutionJournalAnchorCommitInDoubt::MismatchedReadback,
        });
    }
    Ok(ExecutionJournalAnchorCommitOutcome::Committed {
        snapshot: proposed,
        anchor_reference: reference,
    })
}

fn event_count<E>(
    journal: &InterventionExecutionJournal,
) -> Result<u64, ExecutionJournalAnchorProtocolError<E>>
where
    E: StdError + Send + Sync + 'static,
{
    u64::try_from(journal.events().len())
        .map_err(|_| ExecutionJournalAnchorProtocolError::EventCountOverflow)
}

fn validate_namespace(value: &str) -> Result<(), ExecutionJournalAnchorError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_NAMESPACE_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(ExecutionJournalAnchorError::InvalidNamespace(value.to_string()));
    }
    Ok(())
}

fn validate_anchor_reference(value: &str) -> Result<(), ExecutionJournalAnchorError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_ANCHOR_REF_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(ExecutionJournalAnchorError::InvalidAnchorReference);
    }
    Ok(())
}

fn hash_text(hasher: &mut Sha256, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ExecutionJournalAnchorError {
    #[error("unsupported execution-journal anchor schema")]
    UnsupportedAnchorSchema,
    #[error("invalid execution-journal namespace: {0:?}")]
    InvalidNamespace(String),
    #[error("execution-journal schema bound by the anchor does not match the runtime schema")]
    ExecutionJournalSchemaMismatch,
    #[error("execution-journal anchor revision must be positive")]
    RevisionZero,
    #[error("empty execution journal must have the zero head")]
    EmptyJournalNonzeroHead,
    #[error("nonempty execution journal must not have the zero head")]
    NonemptyJournalZeroHead,
    #[error("genesis execution-journal anchor must have zero previous commitment")]
    GenesisPreviousCommitmentNonzero,
    #[error("successor execution-journal anchor must bind a nonzero previous commitment")]
    SuccessorPreviousCommitmentZero,
    #[error("execution-journal anchor backend returned an invalid reference")]
    InvalidAnchorReference,
}

#[derive(Debug, Error)]
pub enum ExecutionJournalAnchorProtocolError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Snapshot(#[from] ExecutionJournalAnchorError),
    #[error(transparent)]
    Journal(#[from] ExecutionJournalError),
    #[error("execution-journal anchor load failed before CAS: {0}")]
    BackendLoad(#[source] E),
    #[error("execution-journal namespace already has an external anchor")]
    AlreadyBootstrapped,
    #[error("execution-journal external anchor is missing")]
    MissingAnchor,
    #[error("supplied previous execution-journal anchor is stale or substituted")]
    StalePreviousSnapshot,
    #[error("execution journal did not advance: previous events={previous}, current events={current}")]
    JournalDidNotAdvance { previous: u64, current: u64 },
    #[error("execution-journal head did not advance")]
    HeadDidNotAdvance,
    #[error("execution-journal anchor commit time regressed: previous={previous}, next={next}")]
    CommitTimeRegression { previous: u64, next: u64 },
    #[error("execution-journal anchor revision overflow")]
    RevisionOverflow,
    #[error("execution-journal event count cannot be represented as u64")]
    EventCountOverflow,
    #[error("execution-journal anchor namespace does not match trusted deployment namespace")]
    NamespaceMismatch,
    #[error("execution-journal event count differs from external anchor: anchored={anchored}, actual={actual}")]
    EventCountMismatch { anchored: u64, actual: u64 },
    #[error("execution-journal head differs from external anchor")]
    HeadMismatch {
        anchored: Sha256Digest,
        actual: Sha256Digest,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    use symthaea_core::intervention_interlock::WelfareConstraintLevel;
    use symthaea_core::welfare::SubjectAffectingAction;
    use symthaea_psych_bench::moral_patient::ProtectionDisposition;

    use crate::execution_recovery::PreparedInterventionExecution;

    const NS: &str = "symthaea:self:welfare-execution-journal";

    #[derive(Default)]
    struct MemoryAnchor {
        snapshot: Option<ExecutionJournalAnchorSnapshot>,
        lose_cas_ack_after_commit: bool,
        invalid_reference_after_commit: bool,
    }

    impl ExecutionJournalHeadAnchor for MemoryAnchor {
        type Error = io::Error;

        fn load(
            &self,
            _journal_namespace: &str,
        ) -> Result<Option<ExecutionJournalAnchorSnapshot>, Self::Error> {
            Ok(self.snapshot.clone())
        }

        fn compare_and_swap(
            &mut self,
            _journal_namespace: &str,
            expected_current: Option<Sha256Digest>,
            next: &ExecutionJournalAnchorSnapshot,
        ) -> Result<String, Self::Error> {
            let actual = self
                .snapshot
                .as_ref()
                .map(ExecutionJournalAnchorSnapshot::commitment)
                .transpose()
                .map_err(|error| io::Error::other(error.to_string()))?;
            if actual != expected_current {
                return Err(io::Error::other("stale anchor writer"));
            }
            self.snapshot = Some(next.clone());
            if self.lose_cas_ack_after_commit {
                return Err(io::Error::other("lost CAS acknowledgement after commit"));
            }
            if self.invalid_reference_after_commit {
                return Ok(" bad anchor ref ".into());
            }
            Ok(format!("memory:execution-journal-anchor:{}", next.revision))
        }
    }

    fn digest(seed: u8) -> Sha256Digest {
        Sha256Digest([seed; 32])
    }

    fn prepared(execution: &str, authority: &str) -> PreparedInterventionExecution {
        PreparedInterventionExecution {
            schema_version: EXECUTION_JOURNAL_SCHEMA.into(),
            execution_id: execution.into(),
            authority_id: authority.into(),
            target_id: "symthaea:self:episodic-memory:instance:test".into(),
            action: SubjectAffectingAction::MemoryModification,
            rationale_digest: digest(1),
            welfare_profile_digest: digest(2),
            precaution_policy_digest: digest(3),
            protection_disposition: ProtectionDisposition::Baseline,
            welfare_constraint: WelfareConstraintLevel::Baseline,
            replay_generation: 1,
            replay_snapshot_digest: digest(4),
            replay_persistence_ref: "replay:test".into(),
            prepared_at_unix_s: 100,
            permit_not_after_unix_s: 200,
        }
    }

    fn one_event_journal(execution: &str, authority: &str) -> InterventionExecutionJournal {
        let mut journal = InterventionExecutionJournal::new();
        journal.append_prepared(prepared(execution, authority)).unwrap();
        journal
    }

    fn committed<E>(
        outcome: ExecutionJournalAnchorCommitOutcome<E>,
    ) -> ExecutionJournalAnchorSnapshot
    where
        E: StdError + Send + Sync + 'static,
    {
        match outcome {
            ExecutionJournalAnchorCommitOutcome::Committed { snapshot, .. } => snapshot,
            ExecutionJournalAnchorCommitOutcome::InDoubt { .. } => panic!("expected committed anchor"),
        }
    }

    #[test]
    fn bootstrap_and_exact_recovery_accept_identical_cloned_history() {
        let journal = one_event_journal("exec:1", "authority:1");
        let events = journal.events().to_vec();
        let mut anchor = MemoryAnchor::default();
        let snapshot = committed(
            bootstrap_execution_journal_anchor(&journal, &mut anchor, NS, 110).unwrap(),
        );
        assert_eq!(snapshot.event_count, 1);

        let recovered =
            recover_execution_journal_with_anchor(events.clone(), &anchor, NS).unwrap();
        assert_eq!(recovered.journal.head_hash(), journal.head_hash());
        assert_eq!(recovered.anchor, snapshot);
        let cloned = recover_execution_journal_with_anchor(events, &anchor, NS).unwrap();
        assert_eq!(cloned.anchor_commitment, recovered.anchor_commitment);
    }

    #[test]
    fn rollback_and_local_ahead_state_both_fail_closed() {
        let mut journal = one_event_journal("exec:1", "authority:1");
        let first_events = journal.events().to_vec();
        let mut anchor = MemoryAnchor::default();
        let first = committed(
            bootstrap_execution_journal_anchor(&journal, &mut anchor, NS, 110).unwrap(),
        );

        journal
            .append_prepared(prepared("exec:2", "authority:2"))
            .unwrap();
        let second_events = journal.events().to_vec();
        let second = committed(
            advance_execution_journal_anchor(&journal, &mut anchor, &first, 120).unwrap(),
        );
        assert_eq!(second.event_count, 2);

        let rollback = recover_execution_journal_with_anchor(first_events, &anchor, NS).unwrap_err();
        assert!(matches!(
            rollback,
            ExecutionJournalAnchorProtocolError::EventCountMismatch { .. }
        ));

        let mut ahead = InterventionExecutionJournal::from_events(second_events).unwrap();
        ahead
            .append_prepared(prepared("exec:3", "authority:3"))
            .unwrap();
        let ahead_error =
            recover_execution_journal_with_anchor(ahead.events().to_vec(), &anchor, NS).unwrap_err();
        assert!(matches!(
            ahead_error,
            ExecutionJournalAnchorProtocolError::EventCountMismatch { .. }
        ));
    }

    #[test]
    fn same_length_fork_fails_on_head_hash() {
        let journal = one_event_journal("exec:1", "authority:1");
        let mut anchor = MemoryAnchor::default();
        committed(bootstrap_execution_journal_anchor(&journal, &mut anchor, NS, 110).unwrap());

        let fork = one_event_journal("exec:fork", "authority:fork");
        assert_eq!(fork.events().len(), journal.events().len());
        assert_ne!(fork.head_hash(), journal.head_hash());
        let error =
            recover_execution_journal_with_anchor(fork.events().to_vec(), &anchor, NS).unwrap_err();
        assert!(matches!(error, ExecutionJournalAnchorProtocolError::HeadMismatch { .. }));
    }

    #[test]
    fn stale_previous_snapshot_cannot_advance_anchor() {
        let mut journal = one_event_journal("exec:1", "authority:1");
        let mut anchor = MemoryAnchor::default();
        let first = committed(
            bootstrap_execution_journal_anchor(&journal, &mut anchor, NS, 110).unwrap(),
        );
        journal
            .append_prepared(prepared("exec:2", "authority:2"))
            .unwrap();
        committed(advance_execution_journal_anchor(&journal, &mut anchor, &first, 120).unwrap());
        journal
            .append_prepared(prepared("exec:3", "authority:3"))
            .unwrap();
        let error =
            advance_execution_journal_anchor(&journal, &mut anchor, &first, 130).unwrap_err();
        assert!(matches!(
            error,
            ExecutionJournalAnchorProtocolError::StalePreviousSnapshot
        ));
    }

    #[test]
    fn wrong_namespace_snapshot_is_rejected_even_if_backend_misroutes_it() {
        let journal = one_event_journal("exec:1", "authority:1");
        let mut anchor = MemoryAnchor::default();
        committed(bootstrap_execution_journal_anchor(&journal, &mut anchor, NS, 110).unwrap());
        let error = recover_execution_journal_with_anchor(
            journal.events().to_vec(),
            &anchor,
            "symthaea:other:journal",
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ExecutionJournalAnchorProtocolError::NamespaceMismatch
        ));
    }

    #[test]
    fn lost_cas_ack_is_in_doubt_and_recovery_discovers_committed_root() {
        let journal = one_event_journal("exec:1", "authority:1");
        let mut anchor = MemoryAnchor {
            lose_cas_ack_after_commit: true,
            ..Default::default()
        };
        let outcome =
            bootstrap_execution_journal_anchor(&journal, &mut anchor, NS, 110).unwrap();
        assert!(outcome.requires_reconciliation());
        assert!(matches!(
            outcome,
            ExecutionJournalAnchorCommitOutcome::InDoubt {
                reason: ExecutionJournalAnchorCommitInDoubt::CompareAndSwap(_),
                ..
            }
        ));
        let recovered = recover_execution_journal_with_anchor(
            journal.events().to_vec(),
            &anchor,
            NS,
        )
        .unwrap();
        assert_eq!(recovered.journal.head_hash(), journal.head_hash());
        let retry = bootstrap_execution_journal_anchor(&journal, &mut anchor, NS, 111).unwrap_err();
        assert!(matches!(retry, ExecutionJournalAnchorProtocolError::AlreadyBootstrapped));
    }

    #[test]
    fn invalid_post_cas_reference_is_in_doubt_not_retryable_failure() {
        let journal = one_event_journal("exec:1", "authority:1");
        let mut anchor = MemoryAnchor {
            invalid_reference_after_commit: true,
            ..Default::default()
        };
        let outcome =
            bootstrap_execution_journal_anchor(&journal, &mut anchor, NS, 110).unwrap();
        assert!(matches!(
            outcome,
            ExecutionJournalAnchorCommitOutcome::InDoubt {
                reason: ExecutionJournalAnchorCommitInDoubt::InvalidReference,
                ..
            }
        ));
        recover_execution_journal_with_anchor(journal.events().to_vec(), &anchor, NS).unwrap();
    }

    #[test]
    fn snapshot_commitment_binds_namespace_head_revision_and_predecessor() {
        let base = ExecutionJournalAnchorSnapshot::try_new(
            NS,
            1,
            digest(10),
            1,
            Sha256Digest([0; 32]),
            100,
        )
        .unwrap();
        let base_commitment = base.commitment().unwrap();
        let successor = ExecutionJournalAnchorSnapshot::try_new(
            NS,
            2,
            digest(11),
            2,
            base_commitment,
            101,
        )
        .unwrap();
        assert_ne!(base_commitment, successor.commitment().unwrap());

        let other_namespace = ExecutionJournalAnchorSnapshot::try_new(
            "symthaea:other:journal",
            1,
            digest(10),
            1,
            Sha256Digest([0; 32]),
            100,
        )
        .unwrap();
        assert_ne!(base_commitment, other_namespace.commitment().unwrap());
    }
}
