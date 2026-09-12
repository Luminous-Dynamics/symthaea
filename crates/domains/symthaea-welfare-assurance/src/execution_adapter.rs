// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Two-phase journal adapter for welfare-sensitive mutation execution.
//!
//! The adapter is invoked only after the durable evidence-bound permit has passed all live
//! revalidation. It persists a `Prepared` journal state before calling the downstream mutator.
//! A successful mutation whose completion journal cannot be persisted is returned as an explicit
//! in-doubt outcome rather than a generic retryable error.

use std::error::Error as StdError;

use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_fabrication_kernel::trust::TrustSnapshot;
use symthaea_psych_bench::moral_patient::{MoralPatientEvidenceProfile, PrecautionPolicy};
use symthaea_welfare_authority::WelfareAuthorityPolicyManifest;
use symthaea_welfare_consent::{SubjectConsentLedger, SubjectIdentityRegistry};
use thiserror::Error;

use crate::{AssuredInterventionExecutor, AssuredInterventionPermit};
use crate::evidence_context::EvidenceBoundExecutionError;
use crate::execution_recovery::{
    CompletedInterventionExecution, ExecutionJournalEnvelope, ExecutionJournalError,
    InterventionExecutionJournal, PreparedInterventionExecution,
};
use crate::replay_recovery::{
    DurableEvidenceBoundInterventionPermit, execute_durable_evidence_bound_intervention_once,
};

const MAX_EXECUTION_REF_BYTES: usize = 2048;

/// Persistence boundary for the append-only execution journal.
///
/// Implementations must durably commit the exact event stream and head hash before returning.
pub trait ExecutionJournalPersistence {
    type Error: StdError + Send + Sync + 'static;

    fn persist_execution_journal(
        &mut self,
        events: &[ExecutionJournalEnvelope],
        head_hash: Sha256Digest,
    ) -> Result<String, Self::Error>;
}

/// Downstream mutation result with evidence needed for the terminal completion record.
#[derive(Debug)]
pub struct ReceiptedExecution<O> {
    output: O,
    completed_at_unix_s: u64,
    result_digest: Sha256Digest,
    executor_evidence_ref: String,
}

impl<O> ReceiptedExecution<O> {
    pub fn new(
        output: O,
        completed_at_unix_s: u64,
        result_digest: Sha256Digest,
        executor_evidence_ref: impl Into<String>,
    ) -> Result<Self, ExecutionObservationError> {
        if result_digest.0 == [0; 32] {
            return Err(ExecutionObservationError::ZeroResultDigest);
        }
        let executor_evidence_ref = executor_evidence_ref.into();
        validate_ref(&executor_evidence_ref)
            .map_err(|_| ExecutionObservationError::InvalidExecutorEvidenceReference)?;
        Ok(Self {
            output,
            completed_at_unix_s,
            result_digest,
            executor_evidence_ref,
        })
    }
}

/// Mutator contract used by the journal adapter.
///
/// Ordinary executor errors are conservatively treated as in-doubt because this layer cannot know
/// whether a downstream system partially committed before returning the error.
pub trait ReceiptedInterventionExecutor {
    type Output;
    type Error: StdError + Send + Sync + 'static;

    fn execute_receipted(
        &mut self,
        permit: &AssuredInterventionPermit,
    ) -> Result<ReceiptedExecution<Self::Output>, Self::Error>;
}

/// Outcomes after the write-ahead record has been durably persisted and mutation was attempted.
///
/// Only `Completed` has durable terminal evidence. Every other variant requires reconciliation
/// and must not be converted into an automatic retry.
#[derive(Debug)]
pub enum JournaledExecutionOutcome<O, EE, PE> {
    Completed {
        output: O,
        prepared_digest: Sha256Digest,
        prepared_persistence_ref: String,
        completion_persistence_ref: String,
        journal_head_hash: Sha256Digest,
    },
    ExecutorInDoubt {
        error: EE,
        prepared_digest: Sha256Digest,
        prepared_persistence_ref: String,
    },
    CompletionJournalInDoubt {
        output: O,
        error: ExecutionJournalError,
        prepared_digest: Sha256Digest,
        prepared_persistence_ref: String,
    },
    CompletionPersistenceInDoubt {
        output: O,
        completed: CompletedInterventionExecution,
        error: PE,
        prepared_digest: Sha256Digest,
        prepared_persistence_ref: String,
    },
    CompletionPersistenceReferenceInDoubt {
        output: O,
        completed: CompletedInterventionExecution,
        prepared_digest: Sha256Digest,
        prepared_persistence_ref: String,
    },
}

impl<O, EE, PE> JournaledExecutionOutcome<O, EE, PE> {
    pub fn has_durable_terminal_evidence(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }
}

struct JournaledExecutorAdapter<'a, X, P>
where
    X: ReceiptedInterventionExecutor,
    P: ExecutionJournalPersistence,
{
    prepared: PreparedInterventionExecution,
    journal: &'a mut InterventionExecutionJournal,
    persistence: &'a mut P,
    executor: &'a mut X,
}

impl<X, P> AssuredInterventionExecutor for JournaledExecutorAdapter<'_, X, P>
where
    X: ReceiptedInterventionExecutor,
    P: ExecutionJournalPersistence,
{
    type Output = JournaledExecutionOutcome<X::Output, X::Error, P::Error>;
    type Error = PreExecutionJournalError<P::Error>;

    fn execute(
        &mut self,
        permit: &AssuredInterventionPermit,
    ) -> Result<Self::Output, Self::Error> {
        let prepared_digest = self
            .journal
            .append_prepared(self.prepared.clone())
            .map_err(PreExecutionJournalError::Journal)?;

        // This persistence occurs inside the executor boundary: all live permit checks have passed,
        // but the downstream mutator has not yet been called.
        let prepared_persistence_ref = self
            .persistence
            .persist_execution_journal(self.journal.events(), self.journal.head_hash())
            .map_err(PreExecutionJournalError::PreparedPersistence)?;
        validate_ref(&prepared_persistence_ref)
            .map_err(|_| PreExecutionJournalError::InvalidPersistenceReference)?;

        let execution = match self.executor.execute_receipted(permit) {
            Ok(execution) => execution,
            Err(error) => {
                return Ok(JournaledExecutionOutcome::ExecutorInDoubt {
                    error,
                    prepared_digest,
                    prepared_persistence_ref,
                });
            }
        };

        let ReceiptedExecution {
            output,
            completed_at_unix_s,
            result_digest,
            executor_evidence_ref,
        } = execution;

        let completed = match CompletedInterventionExecution::new(
            self.prepared.execution_id.clone(),
            prepared_digest,
            completed_at_unix_s,
            result_digest,
            executor_evidence_ref,
        ) {
            Ok(completed) => completed,
            Err(error) => {
                return Ok(JournaledExecutionOutcome::CompletionJournalInDoubt {
                    output,
                    error,
                    prepared_digest,
                    prepared_persistence_ref,
                });
            }
        };

        if let Err(error) = self.journal.append_completed(completed.clone()) {
            return Ok(JournaledExecutionOutcome::CompletionJournalInDoubt {
                output,
                error,
                prepared_digest,
                prepared_persistence_ref,
            });
        }

        let completion_persistence_ref = match self
            .persistence
            .persist_execution_journal(self.journal.events(), self.journal.head_hash())
        {
            Ok(reference) => reference,
            Err(error) => {
                return Ok(JournaledExecutionOutcome::CompletionPersistenceInDoubt {
                    output,
                    completed,
                    error,
                    prepared_digest,
                    prepared_persistence_ref,
                });
            }
        };

        if validate_ref(&completion_persistence_ref).is_err() {
            return Ok(
                JournaledExecutionOutcome::CompletionPersistenceReferenceInDoubt {
                    output,
                    completed,
                    prepared_digest,
                    prepared_persistence_ref,
                },
            );
        }

        Ok(JournaledExecutionOutcome::Completed {
            output,
            prepared_digest,
            prepared_persistence_ref,
            completion_persistence_ref,
            journal_head_hash: self.journal.head_hash(),
        })
    }
}

/// Consume the strongest permit through live revalidation and a two-phase durable execution
/// journal. The write-ahead `Prepared` record is persisted only after live checks pass and before
/// the downstream mutator is invoked.
#[allow(clippy::too_many_arguments)]
pub fn execute_durable_intervention_journaled<X, P>(
    permit: DurableEvidenceBoundInterventionPermit,
    execution_id: impl Into<String>,
    current_profile: &MoralPatientEvidenceProfile,
    current_precaution_policy: &PrecautionPolicy,
    consent_ledger: &SubjectConsentLedger,
    subject_registry: &SubjectIdentityRegistry,
    current_authority_manifest: &WelfareAuthorityPolicyManifest,
    current_trust_snapshot: &TrustSnapshot,
    unix_s: u64,
    journal: &mut InterventionExecutionJournal,
    persistence: &mut P,
    executor: &mut X,
) -> Result<
    JournaledExecutionOutcome<X::Output, X::Error, P::Error>,
    JournaledExecutionGateError<P::Error>,
>
where
    X: ReceiptedInterventionExecutor,
    P: ExecutionJournalPersistence,
{
    let prepared = PreparedInterventionExecution::from_permit(execution_id, &permit, unix_s)
        .map_err(JournaledExecutionGateError::Prepare)?;
    let mut adapter = JournaledExecutorAdapter {
        prepared,
        journal,
        persistence,
        executor,
    };

    execute_durable_evidence_bound_intervention_once(
        permit,
        current_profile,
        current_precaution_policy,
        consent_ledger,
        subject_registry,
        current_authority_manifest,
        current_trust_snapshot,
        unix_s,
        &mut adapter,
    )
    .map_err(JournaledExecutionGateError::Live)
}

fn validate_ref(value: &str) -> Result<(), ()> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_EXECUTION_REF_BYTES
        || value.chars().any(char::is_control)
    {
        Err(())
    } else {
        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ExecutionObservationError {
    #[error("executor result digest may not be zero")]
    ZeroResultDigest,
    #[error("executor evidence reference is invalid")]
    InvalidExecutorEvidenceReference,
}

#[derive(Debug, Error)]
pub enum PreExecutionJournalError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("could not append write-ahead execution record: {0}")]
    Journal(#[source] ExecutionJournalError),
    #[error("could not persist write-ahead execution journal: {0}")]
    PreparedPersistence(#[source] E),
    #[error("execution-journal persistence returned an invalid durable reference")]
    InvalidPersistenceReference,
}

#[derive(Debug, Error)]
pub enum JournaledExecutionGateError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("could not construct prepared execution record: {0}")]
    Prepare(#[source] ExecutionJournalError),
    #[error(transparent)]
    Live(EvidenceBoundExecutionError<PreExecutionJournalError<E>>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_rejects_zero_result_digest() {
        assert!(matches!(
            ReceiptedExecution::new((), 100, Sha256Digest([0; 32]), "executor:evidence"),
            Err(ExecutionObservationError::ZeroResultDigest)
        ));
    }

    #[test]
    fn observation_rejects_noncanonical_evidence_reference() {
        assert!(matches!(
            ReceiptedExecution::new((), 100, Sha256Digest([7; 32]), " bad "),
            Err(ExecutionObservationError::InvalidExecutorEvidenceReference)
        ));
    }

    #[test]
    fn only_completed_outcome_reports_durable_terminal_evidence() {
        let completed: JournaledExecutionOutcome<(), (), ()> =
            JournaledExecutionOutcome::Completed {
                output: (),
                prepared_digest: Sha256Digest([1; 32]),
                prepared_persistence_ref: "prepared:1".into(),
                completion_persistence_ref: "completed:1".into(),
                journal_head_hash: Sha256Digest([2; 32]),
            };
        assert!(completed.has_durable_terminal_evidence());

        let in_doubt: JournaledExecutionOutcome<(), (), ()> =
            JournaledExecutionOutcome::CompletionPersistenceReferenceInDoubt {
                output: (),
                completed: CompletedInterventionExecution {
                    schema_version: crate::execution_recovery::EXECUTION_JOURNAL_SCHEMA.into(),
                    execution_id: "exec-1".into(),
                    prepared_digest: Sha256Digest([3; 32]),
                    completed_at_unix_s: 100,
                    result_digest: Sha256Digest([4; 32]),
                    executor_evidence_ref: "executor:evidence".into(),
                },
                prepared_digest: Sha256Digest([3; 32]),
                prepared_persistence_ref: "prepared:1".into(),
            };
        assert!(!in_doubt.has_durable_terminal_evidence());
    }
}
