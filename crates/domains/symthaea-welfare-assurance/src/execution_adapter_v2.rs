// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Additive V2 journal adapter for domain executors that must bind the exact generic Prepared digest.
//!
//! V1 executors remain unchanged. V2 is opt-in: after generic live revalidation, domain preflight,
//! append of the exact `PreparedInterventionExecution`, and durable persistence of that write-ahead
//! journal, this adapter constructs `PreparedExecutionContextV2` and passes it to the domain
//! executor. A V2 domain executor therefore receives correlation evidence only after the generic
//! write-ahead record is durably established.

#![deny(unsafe_code)]

use std::error::Error as StdError;

use symthaea_fabrication_kernel::crypto_digest::Sha256Digest;
use symthaea_fabrication_kernel::trust::TrustSnapshot;
use symthaea_psych_bench::moral_patient::{MoralPatientEvidenceProfile, PrecautionPolicy};
use symthaea_welfare_authority::WelfareAuthorityPolicyManifest;
use symthaea_welfare_consent::{SubjectConsentLedger, SubjectIdentityRegistry};
use thiserror::Error;

use crate::{AssuredInterventionExecutor, AssuredInterventionPermit};
use crate::evidence_context::EvidenceBoundExecutionError;
use crate::execution_adapter::{
    ExecutionJournalPersistence, ExecutionObservationError, JournaledExecutionOutcome,
};
use crate::execution_recovery::{
    CompletedInterventionExecution, ExecutionJournalError, InterventionExecutionJournal,
    PreparedInterventionExecution,
};
use crate::prepared_execution_context_v2::{
    PreparedExecutionContextV2, PreparedExecutionContextV2Error,
};
use crate::replay_recovery::{
    DurableEvidenceBoundInterventionPermit, execute_durable_evidence_bound_intervention_once,
};

const MAX_EXECUTION_REF_BYTES: usize = 2048;

/// V2 downstream mutation result with evidence needed for the terminal generic completion record.
#[derive(Debug)]
pub struct ContextualReceiptedExecution<O> {
    output: O,
    completed_at_unix_s: u64,
    result_digest: Sha256Digest,
    executor_evidence_ref: String,
}

impl<O> ContextualReceiptedExecution<O> {
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

/// Opt-in domain executor contract for cryptographically correlated Prepared evidence.
pub trait ReceiptedInterventionExecutorV2 {
    type Output;
    type Error: StdError + Send + Sync + 'static;

    /// Deterministic read-only domain checks before any generic `Prepared` journal record exists.
    fn preflight(&self, _permit: &AssuredInterventionPermit) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Execute only after the generic Prepared record is durably persisted and exactly rebound into
    /// `context`. Errors remain in-doubt because the adapter cannot assume downstream atomicity.
    fn execute_receipted_v2(
        &mut self,
        permit: &AssuredInterventionPermit,
        context: &PreparedExecutionContextV2,
    ) -> Result<ContextualReceiptedExecution<Self::Output>, Self::Error>;
}

struct JournaledExecutorAdapterV2<'a, X, P>
where
    X: ReceiptedInterventionExecutorV2,
    P: ExecutionJournalPersistence,
{
    prepared: PreparedInterventionExecution,
    journal: &'a mut InterventionExecutionJournal,
    persistence: &'a mut P,
    executor: &'a mut X,
}

impl<X, P> AssuredInterventionExecutor for JournaledExecutorAdapterV2<'_, X, P>
where
    X: ReceiptedInterventionExecutorV2,
    P: ExecutionJournalPersistence,
{
    type Output = JournaledExecutionOutcome<X::Output, X::Error, P::Error>;
    type Error = PreExecutionJournalV2Error<P::Error>;

    fn execute(
        &mut self,
        permit: &AssuredInterventionPermit,
    ) -> Result<Self::Output, Self::Error> {
        if let Err(error) = self.executor.preflight(permit) {
            return Ok(JournaledExecutionOutcome::PreflightRejected { error });
        }

        let prepared_digest = self
            .journal
            .append_prepared(self.prepared.clone())
            .map_err(PreExecutionJournalV2Error::Journal)?;

        let prepared_persistence_ref = self
            .persistence
            .persist_execution_journal(self.journal.events(), self.journal.head_hash())
            .map_err(PreExecutionJournalV2Error::PreparedPersistence)?;
        validate_ref(&prepared_persistence_ref)
            .map_err(|_| PreExecutionJournalV2Error::InvalidPersistenceReference)?;

        // This is the only construction point used by the V2 adapter. The context independently
        // recomputes the canonical prepared digest before any domain mutation is invoked.
        let context = PreparedExecutionContextV2::from_exact_prepared(
            &self.prepared,
            prepared_digest,
            prepared_persistence_ref.clone(),
        )
        .map_err(PreExecutionJournalV2Error::Context)?;

        let execution = match self.executor.execute_receipted_v2(permit, &context) {
            Ok(execution) => execution,
            Err(error) => {
                return Ok(JournaledExecutionOutcome::ExecutorInDoubt {
                    error,
                    prepared_digest,
                    prepared_persistence_ref,
                });
            }
        };

        let ContextualReceiptedExecution {
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

/// V2 counterpart of `execute_durable_intervention_journaled` for only those executors that need
/// exact generic-Prepared correlation evidence inside their domain write-ahead history.
#[allow(clippy::too_many_arguments)]
pub fn execute_durable_intervention_journaled_v2<X, P>(
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
    JournaledExecutionV2GateError<P::Error>,
>
where
    X: ReceiptedInterventionExecutorV2,
    P: ExecutionJournalPersistence,
{
    let prepared = PreparedInterventionExecution::from_permit(execution_id, &permit, unix_s)
        .map_err(JournaledExecutionV2GateError::Prepare)?;
    let mut adapter = JournaledExecutorAdapterV2 {
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
    .map_err(JournaledExecutionV2GateError::Live)
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

#[derive(Debug, Error)]
pub enum PreExecutionJournalV2Error<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("could not append write-ahead execution record: {0}")]
    Journal(#[source] ExecutionJournalError),
    #[error("could not persist write-ahead execution journal: {0}")]
    PreparedPersistence(#[source] E),
    #[error("execution-journal persistence returned an invalid durable reference")]
    InvalidPersistenceReference,
    #[error("could not bind exact generic Prepared evidence into V2 execution context: {0}")]
    Context(#[source] PreparedExecutionContextV2Error),
}

#[derive(Debug, Error)]
pub enum JournaledExecutionV2GateError<E>
where
    E: StdError + Send + Sync + 'static,
{
    #[error("could not construct prepared execution record: {0}")]
    Prepare(#[source] ExecutionJournalError),
    #[error(transparent)]
    Live(EvidenceBoundExecutionError<PreExecutionJournalV2Error<E>>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contextual_observation_rejects_zero_result_digest() {
        assert!(matches!(
            ContextualReceiptedExecution::new(
                (),
                100,
                Sha256Digest([0; 32]),
                "executor:evidence:v2"
            ),
            Err(ExecutionObservationError::ZeroResultDigest)
        ));
    }

    #[test]
    fn contextual_observation_rejects_noncanonical_evidence_reference() {
        assert!(matches!(
            ContextualReceiptedExecution::new(
                (),
                100,
                Sha256Digest([7; 32]),
                " bad "
            ),
            Err(ExecutionObservationError::InvalidExecutorEvidenceReference)
        ));
    }
}
