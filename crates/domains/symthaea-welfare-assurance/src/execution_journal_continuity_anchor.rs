// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Purpose-separated external continuity anchor for the welfare intervention execution journal.
//!
//! The execution journal's hash chain proves internal integrity of one supplied history. It cannot,
//! by itself, detect a rollback to an older self-consistent prefix or distinguish two byte-identical
//! cloned histories. This module makes the missing external trust boundary explicit without turning
//! anchor evidence into action authority.
//!
//! The externally provisioned `journal_scope_id` is a namespace, not a proof of global uniqueness.
//! Rollback resistance and scope uniqueness are only as strong as the independent implementation of
//! `ExecutionJournalHeadAnchor` (for example TPM NV, TEE/BMC state, Xenia/Mycelix consensus, or
//! another protected monotonic store).

#![deny(unsafe_code)]

use std::error::Error as StdError;

use serde::{Deserialize, Serialize};
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use thiserror::Error;

use crate::execution_recovery::InterventionExecutionJournal;

pub const EXECUTION_JOURNAL_CONTINUITY_ANCHOR_SCHEMA: &str =
    "symthaea.welfare.execution-journal-continuity-anchor.v1";
const ANCHOR_COMMITMENT_DOMAIN: &[u8] =
    b"symthaea.welfare.execution-journal-continuity-anchor.commitment.v1\0";
const MAX_SCOPE_ID_BYTES: usize = 256;
const MAX_REF_BYTES: usize = 2048;

/// External monotonic commitment to one exact welfare execution-journal state.
///
/// This is audit/provenance evidence. Possessing or serializing it grants no intervention authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionJournalContinuityAnchorSnapshot {
    pub schema_version: String,
    pub journal_scope_id: String,
    pub revision: u64,
    pub event_count: u64,
    pub journal_head: Sha256Digest,
    pub committed_at_unix_s: u64,
    pub previous_anchor_commitment: Option<Sha256Digest>,
}

impl ExecutionJournalContinuityAnchorSnapshot {
    pub fn validate(&self) -> Result<(), ExecutionJournalContinuityAnchorError> {
        if self.schema_version != EXECUTION_JOURNAL_CONTINUITY_ANCHOR_SCHEMA {
            return Err(ExecutionJournalContinuityAnchorError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        validate_scope_id(&self.journal_scope_id)?;
        if self.revision == 0 {
            return Err(ExecutionJournalContinuityAnchorError::ZeroRevision);
        }
        if self.event_count == 0 {
            return Err(ExecutionJournalContinuityAnchorError::ZeroEventCount);
        }
        if self.journal_head.0 == [0; 32] {
            return Err(ExecutionJournalContinuityAnchorError::ZeroJournalHead);
        }
        match (self.revision, self.previous_anchor_commitment) {
            (1, Some(_)) => {
                return Err(ExecutionJournalContinuityAnchorError::UnexpectedPreviousCommitment);
            }
            (revision, None) if revision > 1 => {
                return Err(ExecutionJournalContinuityAnchorError::MissingPreviousCommitment);
            }
            _ => {}
        }
        Ok(())
    }

    pub fn commitment(&self) -> Result<Sha256Digest, ExecutionJournalContinuityAnchorError> {
        self.validate()?;
        let encoded = bincode::serialize(self)
            .map_err(|error| ExecutionJournalContinuityAnchorError::Encoding(error.to_string()))?;
        let mut hasher = Sha256::new();
        hasher.update(ANCHOR_COMMITMENT_DOMAIN);
        hasher.update(&(encoded.len() as u64).to_le_bytes());
        hasher.update(&encoded);
        Ok(hasher.finalize())
    }
}

/// Independent monotonic/trusted anchor boundary for the welfare execution journal.
///
/// `compare_and_swap` must atomically reject a stale `expected_current`. An implementation may use
/// protected hardware or distributed consensus; an ordinary mutable file does not establish the
/// rollback-resistance theorem merely by implementing this trait.
pub trait ExecutionJournalHeadAnchor {
    type Error: StdError + Send + Sync + 'static;

    fn load(
        &self,
        journal_scope_id: &str,
    ) -> Result<Option<ExecutionJournalContinuityAnchorSnapshot>, Self::Error>;

    fn compare_and_swap(
        &mut self,
        journal_scope_id: &str,
        expected_current: Option<Sha256Digest>,
        next: &ExecutionJournalContinuityAnchorSnapshot,
    ) -> Result<String, Self::Error>;
}

/// Exact journal state accepted against an independently loaded external anchor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedExecutionJournalContinuityAnchor {
    snapshot: ExecutionJournalContinuityAnchorSnapshot,
    commitment: Sha256Digest,
}

impl VerifiedExecutionJournalContinuityAnchor {
    pub fn snapshot(&self) -> &ExecutionJournalContinuityAnchorSnapshot {
        &self.snapshot
    }

    pub fn commitment(&self) -> Sha256Digest {
        self.commitment
    }

    pub fn journal_scope_id(&self) -> &str {
        &self.snapshot.journal_scope_id
    }

    pub fn revision(&self) -> u64 {
        self.snapshot.revision
    }

    pub fn event_count(&self) -> u64 {
        self.snapshot.event_count
    }

    pub fn journal_head(&self) -> Sha256Digest {
        self.snapshot.journal_head
    }
}

/// Explicit one-time provisioning ceremony for the first non-empty durable journal state.
///
/// This function cannot prove that the supplied journal was durably persisted before invocation;
/// the deployment must enforce that ordering. It does prove that the external anchor accepted the
/// exact non-empty head/count under the supplied scope through a compare-and-swap from `None`.
pub fn bootstrap_execution_journal_continuity_anchor<A: ExecutionJournalHeadAnchor>(
    journal: &InterventionExecutionJournal,
    anchor: &mut A,
    journal_scope_id: &str,
    committed_at_unix_s: u64,
) -> Result<
    (VerifiedExecutionJournalContinuityAnchor, String),
    ExecutionJournalContinuityAnchorProtocolError<A::Error>,
> {
    validate_scope_id(journal_scope_id)?;
    if journal.events().is_empty() {
        return Err(ExecutionJournalContinuityAnchorError::EmptyJournal.into());
    }
    if anchor
        .load(journal_scope_id)
        .map_err(ExecutionJournalContinuityAnchorProtocolError::Anchor)?
        .is_some()
    {
        return Err(ExecutionJournalContinuityAnchorProtocolError::AlreadyAnchored);
    }

    let snapshot = snapshot_for(journal, journal_scope_id, 1, committed_at_unix_s, None)?;
    let commitment = snapshot.commitment()?;
    let reference = anchor
        .compare_and_swap(journal_scope_id, None, &snapshot)
        .map_err(ExecutionJournalContinuityAnchorProtocolError::Anchor)?;
    validate_reference(&reference)?;
    Ok((
        VerifiedExecutionJournalContinuityAnchor {
            snapshot,
            commitment,
        },
        reference,
    ))
}

/// Verify that the supplied validated journal exactly equals the externally retained anchored state.
///
/// A valid self-consistent prefix is rejected when the anchor remembers a later event count/head.
pub fn verify_execution_journal_continuity_anchor<A: ExecutionJournalHeadAnchor>(
    journal: &InterventionExecutionJournal,
    anchor: &A,
    journal_scope_id: &str,
) -> Result<
    VerifiedExecutionJournalContinuityAnchor,
    ExecutionJournalContinuityAnchorProtocolError<A::Error>,
> {
    validate_scope_id(journal_scope_id)?;
    let snapshot = anchor
        .load(journal_scope_id)
        .map_err(ExecutionJournalContinuityAnchorProtocolError::Anchor)?
        .ok_or(ExecutionJournalContinuityAnchorProtocolError::MissingAnchor)?;
    snapshot.validate()?;
    if snapshot.journal_scope_id != journal_scope_id {
        return Err(ExecutionJournalContinuityAnchorProtocolError::AnchorScopeMismatch {
            expected: journal_scope_id.to_string(),
            actual: snapshot.journal_scope_id,
        });
    }
    validate_exact_journal_state(journal, &snapshot)?;
    let commitment = snapshot.commitment()?;
    Ok(VerifiedExecutionJournalContinuityAnchor {
        snapshot,
        commitment,
    })
}

/// Advance an existing anchor after the caller has durably persisted a strict append-extension.
///
/// The current external commitment must exactly equal `previous`. The new journal must contain the
/// exact old anchored prefix and at least one additional event; same-length replacement and branch
/// divergence fail closed. As with bootstrap, durable-persistence ordering is a deployment
/// responsibility until a concrete persistence+anchor adapter composes both boundaries.
pub fn advance_execution_journal_continuity_anchor<A: ExecutionJournalHeadAnchor>(
    journal: &InterventionExecutionJournal,
    anchor: &mut A,
    previous: &VerifiedExecutionJournalContinuityAnchor,
    committed_at_unix_s: u64,
) -> Result<
    (VerifiedExecutionJournalContinuityAnchor, String),
    ExecutionJournalContinuityAnchorProtocolError<A::Error>,
> {
    previous.snapshot.validate()?;
    if committed_at_unix_s < previous.snapshot.committed_at_unix_s {
        return Err(ExecutionJournalContinuityAnchorProtocolError::AnchorTimeRegression {
            previous: previous.snapshot.committed_at_unix_s,
            next: committed_at_unix_s,
        });
    }

    let scope = &previous.snapshot.journal_scope_id;
    let current = anchor
        .load(scope)
        .map_err(ExecutionJournalContinuityAnchorProtocolError::Anchor)?
        .ok_or(ExecutionJournalContinuityAnchorProtocolError::MissingAnchor)?;
    current.validate()?;
    let current_commitment = current.commitment()?;
    if current_commitment != previous.commitment || current != previous.snapshot {
        return Err(ExecutionJournalContinuityAnchorProtocolError::StaleWriter {
            expected: previous.commitment,
            actual: current_commitment,
        });
    }

    let previous_count = usize::try_from(previous.snapshot.event_count)
        .map_err(|_| ExecutionJournalContinuityAnchorError::EventCountOverflow)?;
    if journal.events().len() <= previous_count {
        return Err(ExecutionJournalContinuityAnchorProtocolError::NoStrictExtension {
            previous: previous.snapshot.event_count,
            actual: journal.events().len() as u64,
        });
    }
    let prefix_head = journal
        .events()
        .get(previous_count - 1)
        .expect("previous anchor has nonzero event count and extension was checked")
        .event_hash;
    if prefix_head != previous.snapshot.journal_head {
        return Err(ExecutionJournalContinuityAnchorProtocolError::PrefixHeadMismatch {
            expected: previous.snapshot.journal_head,
            actual: prefix_head,
        });
    }

    let revision = previous
        .snapshot
        .revision
        .checked_add(1)
        .ok_or(ExecutionJournalContinuityAnchorError::RevisionOverflow)?;
    let snapshot = snapshot_for(
        journal,
        scope,
        revision,
        committed_at_unix_s,
        Some(previous.commitment),
    )?;
    let commitment = snapshot.commitment()?;
    let reference = anchor
        .compare_and_swap(scope, Some(previous.commitment), &snapshot)
        .map_err(ExecutionJournalContinuityAnchorProtocolError::Anchor)?;
    validate_reference(&reference)?;
    Ok((
        VerifiedExecutionJournalContinuityAnchor {
            snapshot,
            commitment,
        },
        reference,
    ))
}

fn snapshot_for(
    journal: &InterventionExecutionJournal,
    journal_scope_id: &str,
    revision: u64,
    committed_at_unix_s: u64,
    previous_anchor_commitment: Option<Sha256Digest>,
) -> Result<ExecutionJournalContinuityAnchorSnapshot, ExecutionJournalContinuityAnchorError> {
    validate_scope_id(journal_scope_id)?;
    let event_count = u64::try_from(journal.events().len())
        .map_err(|_| ExecutionJournalContinuityAnchorError::EventCountOverflow)?;
    let snapshot = ExecutionJournalContinuityAnchorSnapshot {
        schema_version: EXECUTION_JOURNAL_CONTINUITY_ANCHOR_SCHEMA.into(),
        journal_scope_id: journal_scope_id.to_string(),
        revision,
        event_count,
        journal_head: journal.head_hash(),
        committed_at_unix_s,
        previous_anchor_commitment,
    };
    snapshot.validate()?;
    Ok(snapshot)
}

fn validate_exact_journal_state(
    journal: &InterventionExecutionJournal,
    snapshot: &ExecutionJournalContinuityAnchorSnapshot,
) -> Result<(), ExecutionJournalContinuityAnchorProtocolError<std::convert::Infallible>> {
    let actual_count = u64::try_from(journal.events().len())
        .map_err(|_| ExecutionJournalContinuityAnchorError::EventCountOverflow)?;
    if actual_count != snapshot.event_count {
        return Err(ExecutionJournalContinuityAnchorProtocolError::EventCountMismatch {
            anchored: snapshot.event_count,
            actual: actual_count,
        });
    }
    let actual_head = journal.head_hash();
    if actual_head != snapshot.journal_head {
        return Err(ExecutionJournalContinuityAnchorProtocolError::JournalHeadMismatch {
            anchored: snapshot.journal_head,
            actual: actual_head,
        });
    }
    Ok(())
}

fn validate_scope_id(value: &str) -> Result<(), ExecutionJournalContinuityAnchorError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_SCOPE_ID_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(ExecutionJournalContinuityAnchorError::InvalidScopeId(
            value.to_string(),
        ));
    }
    Ok(())
}

fn validate_reference(value: &str) -> Result<(), ExecutionJournalContinuityAnchorError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_REF_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(ExecutionJournalContinuityAnchorError::InvalidAnchorReference);
    }
    Ok(())
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ExecutionJournalContinuityAnchorError {
    #[error("unsupported welfare execution-journal continuity-anchor schema: {0:?}")]
    UnsupportedSchema(String),
    #[error("invalid externally provisioned execution-journal scope id: {0:?}")]
    InvalidScopeId(String),
    #[error("execution-journal anchor revision must be nonzero")]
    ZeroRevision,
    #[error("execution-journal anchor requires a non-empty journal")]
    ZeroEventCount,
    #[error("execution-journal anchor journal head must not be zero")]
    ZeroJournalHead,
    #[error("revision 1 must not name a previous anchor commitment")]
    UnexpectedPreviousCommitment,
    #[error("anchor revisions after revision 1 must name the exact previous anchor commitment")]
    MissingPreviousCommitment,
    #[error("cannot anchor an empty execution journal")]
    EmptyJournal,
    #[error("execution-journal event count cannot be represented")]
    EventCountOverflow,
    #[error("execution-journal anchor revision overflow")]
    RevisionOverflow,
    #[error("execution-journal anchor encoding failed: {0}")]
    Encoding(String),
    #[error("execution-journal anchor returned an invalid durable reference")]
    InvalidAnchorReference,
}

#[derive(Debug, Error)]
pub enum ExecutionJournalContinuityAnchorProtocolError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error(transparent)]
    Local(#[from] ExecutionJournalContinuityAnchorError),
    #[error("execution-journal external anchor failed: {0}")]
    Anchor(#[source] E),
    #[error("execution-journal scope is already externally anchored")]
    AlreadyAnchored,
    #[error("execution-journal external anchor is missing")]
    MissingAnchor,
    #[error("loaded execution-journal anchor scope mismatch: expected={expected:?}, actual={actual:?}")]
    AnchorScopeMismatch { expected: String, actual: String },
    #[error("execution-journal event-count mismatch: anchored={anchored}, actual={actual}")]
    EventCountMismatch { anchored: u64, actual: u64 },
    #[error("execution-journal head disagrees with external anchor")]
    JournalHeadMismatch {
        anchored: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error("execution-journal anchor changed since the supplied predecessor was verified")]
    StaleWriter {
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
    #[error("execution-journal anchor time regressed: previous={previous}, next={next}")]
    AnchorTimeRegression { previous: u64, next: u64 },
    #[error("execution journal is not a strict append-extension: previous_count={previous}, actual_count={actual}")]
    NoStrictExtension { previous: u64, actual: u64 },
    #[error("execution journal does not contain the exact previously anchored prefix head")]
    PrefixHeadMismatch {
        expected: Sha256Digest,
        actual: Sha256Digest,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io;

    use symthaea_core::intervention_interlock::WelfareConstraintLevel;
    use symthaea_core::welfare::SubjectAffectingAction;
    use symthaea_psych_bench::moral_patient::ProtectionDisposition;

    use crate::execution_recovery::{
        CompletedInterventionExecution, EXECUTION_JOURNAL_SCHEMA, PreparedInterventionExecution,
        digest_prepared_execution,
    };

    #[derive(Default)]
    struct MemoryAnchor {
        values: HashMap<String, ExecutionJournalContinuityAnchorSnapshot>,
    }

    impl ExecutionJournalHeadAnchor for MemoryAnchor {
        type Error = io::Error;

        fn load(
            &self,
            journal_scope_id: &str,
        ) -> Result<Option<ExecutionJournalContinuityAnchorSnapshot>, Self::Error> {
            Ok(self.values.get(journal_scope_id).cloned())
        }

        fn compare_and_swap(
            &mut self,
            journal_scope_id: &str,
            expected_current: Option<Sha256Digest>,
            next: &ExecutionJournalContinuityAnchorSnapshot,
        ) -> Result<String, Self::Error> {
            let actual = self
                .values
                .get(journal_scope_id)
                .map(ExecutionJournalContinuityAnchorSnapshot::commitment)
                .transpose()
                .map_err(|error| io::Error::other(error.to_string()))?;
            if actual != expected_current {
                return Err(io::Error::other("stale anchor writer"));
            }
            self.values
                .insert(journal_scope_id.to_string(), next.clone());
            Ok(format!("memory-execution-anchor:{}:{}", journal_scope_id, next.revision))
        }
    }

    fn digest(seed: u8) -> Sha256Digest {
        Sha256Digest([seed; 32])
    }

    fn prepared(execution_id: &str, authority_id: &str) -> PreparedInterventionExecution {
        PreparedInterventionExecution {
            schema_version: EXECUTION_JOURNAL_SCHEMA.into(),
            execution_id: execution_id.into(),
            authority_id: authority_id.into(),
            target_id: "symthaea:self".into(),
            action: SubjectAffectingAction::MemoryModification,
            rationale_digest: digest(1),
            welfare_profile_digest: digest(2),
            precaution_policy_digest: digest(3),
            protection_disposition: ProtectionDisposition::Baseline,
            welfare_constraint: WelfareConstraintLevel::Baseline,
            replay_generation: 1,
            replay_snapshot_digest: digest(4),
            replay_persistence_ref: "replay:anchor:test".into(),
            prepared_at_unix_s: 100,
            permit_not_after_unix_s: 200,
        }
    }

    fn journal_with_prepared(execution_id: &str, authority_id: &str) -> InterventionExecutionJournal {
        let mut journal = InterventionExecutionJournal::new();
        journal
            .append_prepared(prepared(execution_id, authority_id))
            .unwrap();
        journal
    }

    #[test]
    fn empty_journal_cannot_be_bootstrapped() {
        let journal = InterventionExecutionJournal::new();
        let mut anchor = MemoryAnchor::default();
        let error = bootstrap_execution_journal_continuity_anchor(
            &journal,
            &mut anchor,
            "welfare:self",
            100,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ExecutionJournalContinuityAnchorProtocolError::Local(
                ExecutionJournalContinuityAnchorError::EmptyJournal
            )
        ));
    }

    #[test]
    fn bootstrap_and_verify_bind_exact_scope_head_and_count() {
        let journal = journal_with_prepared("exec:1", "authority:1");
        let mut anchor = MemoryAnchor::default();
        let (verified, _) = bootstrap_execution_journal_continuity_anchor(
            &journal,
            &mut anchor,
            "welfare:self",
            100,
        )
        .unwrap();
        assert_eq!(verified.event_count(), 1);
        assert_eq!(verified.journal_head(), journal.head_hash());

        let recovered = verify_execution_journal_continuity_anchor(
            &journal,
            &anchor,
            "welfare:self",
        )
        .unwrap();
        assert_eq!(recovered, verified);
    }

    #[test]
    fn externally_provisioned_scopes_make_identical_histories_distinct_commitments() {
        let journal = journal_with_prepared("exec:1", "authority:1");
        let mut anchor = MemoryAnchor::default();
        let (first, _) = bootstrap_execution_journal_continuity_anchor(
            &journal,
            &mut anchor,
            "welfare:subject:a",
            100,
        )
        .unwrap();
        let (second, _) = bootstrap_execution_journal_continuity_anchor(
            &journal,
            &mut anchor,
            "welfare:subject:b",
            100,
        )
        .unwrap();
        assert_ne!(first.commitment(), second.commitment());
        assert_eq!(first.journal_head(), second.journal_head());
    }

    #[test]
    fn later_anchor_detects_valid_suffix_rollback() {
        let mut journal = journal_with_prepared("exec:1", "authority:1");
        let prepared_only = InterventionExecutionJournal::from_events(journal.events().to_vec()).unwrap();
        let prepared_digest = digest_prepared_execution(&prepared("exec:1", "authority:1")).unwrap();
        let mut anchor = MemoryAnchor::default();
        let (first, _) = bootstrap_execution_journal_continuity_anchor(
            &journal,
            &mut anchor,
            "welfare:self",
            100,
        )
        .unwrap();
        journal
            .append_completed(
                CompletedInterventionExecution::new(
                    "exec:1",
                    prepared_digest,
                    120,
                    digest(8),
                    "executor:test:1",
                )
                .unwrap(),
            )
            .unwrap();
        advance_execution_journal_continuity_anchor(&journal, &mut anchor, &first, 130).unwrap();

        let error = verify_execution_journal_continuity_anchor(
            &prepared_only,
            &anchor,
            "welfare:self",
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ExecutionJournalContinuityAnchorProtocolError::EventCountMismatch {
                anchored: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn divergent_same_length_history_disagrees_with_anchor() {
        let first = journal_with_prepared("exec:1", "authority:1");
        let second = journal_with_prepared("exec:2", "authority:2");
        assert_eq!(first.events().len(), second.events().len());
        let mut anchor = MemoryAnchor::default();
        bootstrap_execution_journal_continuity_anchor(
            &first,
            &mut anchor,
            "welfare:self",
            100,
        )
        .unwrap();
        let error = verify_execution_journal_continuity_anchor(
            &second,
            &anchor,
            "welfare:self",
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ExecutionJournalContinuityAnchorProtocolError::JournalHeadMismatch { .. }
        ));
    }

    #[test]
    fn advancement_requires_exact_anchored_prefix() {
        let first = journal_with_prepared("exec:1", "authority:1");
        let mut anchor = MemoryAnchor::default();
        let (verified, _) = bootstrap_execution_journal_continuity_anchor(
            &first,
            &mut anchor,
            "welfare:self",
            100,
        )
        .unwrap();

        let mut divergent = journal_with_prepared("exec:2", "authority:2");
        divergent
            .append_prepared(prepared("exec:3", "authority:3"))
            .unwrap();
        let error = advance_execution_journal_continuity_anchor(
            &divergent,
            &mut anchor,
            &verified,
            120,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ExecutionJournalContinuityAnchorProtocolError::PrefixHeadMismatch { .. }
        ));
    }

    #[test]
    fn stale_anchor_writer_and_time_regression_fail_closed() {
        let mut journal = journal_with_prepared("exec:1", "authority:1");
        let prepared_digest = digest_prepared_execution(&prepared("exec:1", "authority:1")).unwrap();
        let mut anchor = MemoryAnchor::default();
        let (first, _) = bootstrap_execution_journal_continuity_anchor(
            &journal,
            &mut anchor,
            "welfare:self",
            100,
        )
        .unwrap();
        journal
            .append_completed(
                CompletedInterventionExecution::new(
                    "exec:1",
                    prepared_digest,
                    110,
                    digest(9),
                    "executor:test:2",
                )
                .unwrap(),
            )
            .unwrap();
        let (second, _) = advance_execution_journal_continuity_anchor(
            &journal,
            &mut anchor,
            &first,
            120,
        )
        .unwrap();

        let time_error = advance_execution_journal_continuity_anchor(
            &journal,
            &mut anchor,
            &second,
            119,
        )
        .unwrap_err();
        assert!(matches!(
            time_error,
            ExecutionJournalContinuityAnchorProtocolError::AnchorTimeRegression { .. }
        ));

        let mut extended = journal;
        extended
            .append_prepared(prepared("exec:2", "authority:2"))
            .unwrap();
        let stale = advance_execution_journal_continuity_anchor(
            &extended,
            &mut anchor,
            &first,
            130,
        )
        .unwrap_err();
        assert!(matches!(
            stale,
            ExecutionJournalContinuityAnchorProtocolError::StaleWriter { .. }
        ));
    }
}
